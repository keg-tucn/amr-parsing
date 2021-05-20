from typing import Dict
import time
import os
import shutil
import re

from absl import app
from absl import flags
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode

from data_pipeline.data_reading import get_paths
from data_pipeline.vocab import Vocabs
from data_pipeline.dataset import PAD, EOS, UNK
from data_pipeline.dataset import AMR_ID_KEY, CONCEPTS_KEY, CONCEPTS_LEN_KEY,\
  GLOVE_CONCEPTS_KEY, ADJ_MAT_KEY, AMR_STR_KEY
from data_pipeline.dataset import AMRDataset
from data_pipeline.glove_embeddings import GloVeEmbeddings
from utils.arcs_masking import create_mask
from utils.data_logger import DataLogger

from config import get_default_config
from models import RelationIdentification
from evaluation.tensors_to_amr import get_unlabelled_amr_strings_from_tensors
from evaluation.arcs_evaluation_metrics import SmatchScore, initialize_smatch, \
  calc_edges_scores, calc_labels_scores, compute_smatch

import logging
logger = logging.getLogger("penman")
for handler in logger.handlers.copy():
    logger.removeHandler(handler)
logger.addHandler(logging.NullHandler())
logger.propagate = False

UNK_REL_LABEL = ':unk-label'

FLAGS = flags.FLAGS
flags.DEFINE_string('config',
                    default=None,
                    help=('Config file to overwrite default configs.'))
flags.DEFINE_string('train_subsets',
                    default=None,
                    help=('Train subsets split by comma. Ex: bolt,proxy'))
flags.DEFINE_string('dev_subsets',
                    default=None,
                    help=('Train subsets split by comma. Ex: bolt,proxy'))
flags.DEFINE_integer('batch_size',
                     default=32,
                     short_name='b',
                     help=('Batch size.'))
flags.DEFINE_integer('dev_batch_size',
                     default=32,
                     help=('Dev batch size.'))
flags.DEFINE_integer('no_epochs',
                     short_name='e',
                     default=20,
                     help=('Number of epochs.'))
flags.DEFINE_boolean('use_glove',
                     default=False,
                     help=('Flag which tells whether model should use GloVe Embeddings or not.'))

def get_gold_data(batch: torch.tensor):
  amr_ids = batch[AMR_ID_KEY]
  gold_concepts = batch[CONCEPTS_KEY]
  gold_concepts_length = batch[CONCEPTS_LEN_KEY]
  gold_adj_mat = batch[ADJ_MAT_KEY]
  gold_amr_str = batch[AMR_STR_KEY]
  glove_concepts = batch[GLOVE_CONCEPTS_KEY]

  return amr_ids, gold_concepts, gold_concepts_length, gold_adj_mat, gold_amr_str, glove_concepts

def compute_loss(vocabs: Vocabs,
                 logits: torch.Tensor,
                 rel_logits: torch.Tensor,
                 gold_outputs: torch.Tensor,
                 mask: torch.Tensor,
                 config: CfgNode):
  """
  Args:
    vocabs: Vocabs object (with the 3 vocabs).
    mask: Mask for weighting the loss.
    logits: Concepts edges scores (batch size, seq len, seq len).
    rel_logits: Relation label scores (batch size, seq len, seq len, no_classes)
    gold_outputs: Gold adj mat (with relation labels) of shape
      (batch size, seq len, seq len).
    config: configuration file

  Returns:
    Binary cross entropy loss over batch. (for Heads Selection module)
    Cross entropy loss over batch. (for Arcs Labelling module)
    It can return the loss of Heads Selection module, the loss of Arcs Labelling module,
      or the loss of both modules, in which case the two previously mentioned losses will be summed.
  """
  arcs_loss = 0
  label_loss = 0
  if config.HEADS_SELECTION:
    no_rel_index = vocabs.relation_vocab[None]
    pad_idx = vocabs.relation_vocab[PAD]
    binary_outputs = (gold_outputs != no_rel_index) * (gold_outputs != pad_idx)
    binary_outputs = binary_outputs.type(torch.FloatTensor)
    data_selection_weights = mask.type(torch.FloatTensor)
    class_weights = torch.where(binary_outputs == 1.0, config.POSITIVE_CLASS_WEIGHT, config.NEGATIVE_CLASS_WEIGHT)
    weights = class_weights * data_selection_weights
    flattened_logits = logits.flatten()
    flattened_binary_outputs = binary_outputs.flatten()
    flattened_weights = weights.flatten()
    arcs_loss = binary_cross_entropy_with_logits(
      flattened_logits, flattened_binary_outputs, flattened_weights)

  if config.ARCS_LABELLING:
    rel_outputs = gold_outputs.type(torch.LongTensor)
    flattened_binary_rel_outputs = rel_outputs.flatten()
    flattened_rel_logits = rel_logits.flatten(start_dim=0, end_dim=2)
    label_loss = cross_entropy(
      flattened_rel_logits, flattened_binary_rel_outputs)

  return arcs_loss + label_loss

def replace_all_edge_labels(amr_str: str, new_edge_label: str):
  """
  Replaces all edge labels in an AMR with a given edge and
    returns the new AMR.
  """
  new_amr_str = re.sub(r':[a-zA-Z0-9~.-]+', new_edge_label, amr_str)
  return new_amr_str

def remove_order_of_words(amr_str: str):
  """
    Replaces all word orders from the concepts so that the smatch will be
      more focused on predicting the concepts and relations between them.
  """
  new_amr_str = re.sub(r'~e.[0-9,]+', '', amr_str)
  return new_amr_str

def gather_logged_data(logger: DataLogger, inputs_lengths, logits, mask, gold_adj_mat, concepts_str):
  logged_index = logger.logged_example_index
  if logger.text_scores is None:
    sentence_len = inputs_lengths[logged_index].item()
    scores = torch.sigmoid(logits)
    scores = scores[logged_index][:sentence_len, :sentence_len]
    color_scores = mask[logged_index][:sentence_len, :sentence_len].int()
    gold_relations = gold_adj_mat[logged_index][:sentence_len, :sentence_len]
    gold_relations[gold_relations != 0] = 1
    logger.set_img_info(concepts_str[logged_index], concepts_str[logged_index],
                        scores.cpu().detach().numpy(),
                        color_scores.cpu().detach().numpy(),
                        gold_relations.cpu().detach().numpy())

def compute_results(gold_amr_str, inputs, inputs_lengths, predictions, vocabs, logger: DataLogger):
  gold_outputs = [replace_all_edge_labels(a, UNK_REL_LABEL) for a in gold_amr_str]
  gold_outputs = [remove_order_of_words(a) for a in gold_outputs]
  predictions_strings = get_unlabelled_amr_strings_from_tensors(
    inputs, inputs_lengths, predictions, vocabs, UNK_REL_LABEL)

  smatch_score = compute_smatch(gold_outputs, predictions_strings)
  amr_comparison_text = '  \n'.join([gold_amr_str[logger.logged_example_index], ' VERSUS',
                                     predictions_strings[logger.logged_example_index]])
  return smatch_score, amr_comparison_text

def decode_concepts(inputs, inputs_lengths, vocabs: Vocabs):
  inputs_no_padding = []
  for index in range(inputs_lengths.shape[0]):
    sentence = inputs[:,index]
    pos = inputs_lengths[index].item()
    no_pad_sen = sentence[:pos]
    inputs_no_padding.append(no_pad_sen)

  ids_to_concepts_list = list(vocabs.concept_vocab.keys())
  concepts_as_list = []
  for concepts in inputs_no_padding:
    decoded_sen = []
    for concept_id in concepts:
      decoded_word = ids_to_concepts_list[int(concept_id)]
      decoded_sen.append(decoded_word)
    concepts_as_list.append(decoded_sen)

  return concepts_as_list

def eval_step(model: nn.Module,
              optimizer: nn.Module,
              vocabs: Vocabs,
              device: str,
              batch: torch.tensor,
              eval_logger: DataLogger,
              config: CfgNode,
              log_res: bool=False):
  amr_ids, inputs, inputs_lengths, gold_adj_mat, gold_amr_str, glove_concepts = get_gold_data(batch)

  optimizer.zero_grad()
  inputs_device = inputs.to(device)
  gold_adj_mat_device = gold_adj_mat.to(device)
  logits, predictions, rel_logits, rel_predictions = model(inputs_device, inputs_lengths)
  mask = create_mask(gold_adj_mat_device, inputs_lengths, config)
  concepts_str = decode_concepts(inputs, inputs_lengths, vocabs)
  gather_logged_data(eval_logger, inputs_lengths, logits, mask, gold_adj_mat, concepts_str)
  f_score, precision, recall, accuracy = calc_edges_scores(gold_adj_mat, predictions, inputs_lengths)
  rel_f_score, rel_precision, rel_recall = calc_labels_scores(gold_adj_mat, rel_predictions, inputs_lengths, vocabs)

  loss = compute_loss(vocabs, logits, rel_logits, gold_adj_mat_device, mask, config)

  # Remove the edge labels for the gold AMRs before doing the smatch.
  smatch_score = initialize_smatch()
  amr_comparison_text = ''
  if log_res:
    smatch_score, amr_comparison_text = compute_results(
      gold_amr_str, inputs, inputs_lengths, predictions, vocabs, eval_logger)

  return loss, smatch_score, amr_comparison_text, f_score, precision, recall, accuracy, \
         rel_f_score, rel_precision, rel_recall

def evaluate_model(model: nn.Module,
                   optimizer: nn.Module,
                   vocabs: Vocabs,
                   device: str,
                   data_loader: DataLoader,
                   eval_logger: DataLogger,
                   config: CfgNode,
                   log_res: bool=False):
  model.eval()
  with torch.no_grad():
    epoch_loss = 0
    no_batches = 0
    epoch_smatch = initialize_smatch()
    f_score = 0
    precision = 0
    recall = 0
    accuracy = 0
    rel_f_score = 0
    rel_precision = 0
    rel_recall = 0
    logged_text = "GOLD VS PREDICTED AMRS\n"

    for batch in data_loader:
      loss, smatch_score, amr_comparison_text, aux_f_score, aux_precision, aux_recall, aux_accuracy,\
        aux_rel_f_score, aux_rel_precision, aux_rel_recall = \
        eval_step(model, optimizer, vocabs, device, batch, eval_logger, config, log_res)
      epoch_loss += loss
      f_score += aux_f_score
      precision += aux_precision
      recall += aux_recall
      accuracy += aux_accuracy
      rel_f_score += aux_rel_f_score
      rel_precision += aux_rel_precision
      rel_recall += aux_rel_recall
      epoch_smatch[SmatchScore.PRECISION] += smatch_score[SmatchScore.PRECISION]
      epoch_smatch[SmatchScore.RECALL] += smatch_score[SmatchScore.RECALL]
      epoch_smatch[SmatchScore.F_SCORE] += smatch_score[SmatchScore.F_SCORE]
      logged_text += 'Batch ' + str(no_batches) + ':\n'
      logged_text += amr_comparison_text + '\n----\n'
      no_batches += 1
    epoch_loss = epoch_loss / no_batches
    f_score = f_score / no_batches
    precision = precision / no_batches
    recall = recall / no_batches
    accuracy = accuracy / no_batches
    rel_f_score = rel_f_score / no_batches
    rel_precision = rel_precision / no_batches
    rel_recall = rel_recall / no_batches
    epoch_smatch = {score_name: score_value / no_batches for score_name, score_value in epoch_smatch.items()}
    logged_text = logged_text.replace('\n', '\n\n')
    return epoch_loss, epoch_smatch, logged_text, f_score, precision, recall, accuracy, \
           rel_f_score, rel_precision, rel_recall

def train_step(model: nn.Module,
               optimizer: Optimizer,
               vocabs: Vocabs,
               device: str,
               batch: Dict[str, torch.Tensor],
               train_logger: DataLogger,
               config: CfgNode,
               log_results: bool=False):
  amr_ids, inputs, inputs_lengths, gold_adj_mat, gold_amr_str, glove_concepts = get_gold_data(batch)

  optimizer.zero_grad()
  # Move to trainig device (eg. cuda).
  inputs_device = inputs.to(device)
  gold_adj_mat_device = gold_adj_mat.to(device)
  logits, predictions, rel_logits, rel_predictions = model(inputs_device, inputs_lengths)
  mask = create_mask(gold_adj_mat_device, inputs_lengths, config)
  concepts_str = decode_concepts(inputs, inputs_lengths, vocabs)
  gather_logged_data(train_logger, inputs_lengths, logits, mask, gold_adj_mat, concepts_str)

  f_score, precision, recall, accuracy = calc_edges_scores(gold_adj_mat, predictions, inputs_lengths)
  loss = compute_loss(vocabs, logits, rel_logits, gold_adj_mat_device, mask, config)
  loss.backward()
  optimizer.step()

  smatch_score = initialize_smatch()
  amr_comparison_text = ''
  if log_results:
    smatch_score, amr_comparison_text = compute_results(
      gold_amr_str, inputs, inputs_lengths, predictions, vocabs, train_logger)

  return loss, smatch_score, amr_comparison_text, f_score, precision, recall, accuracy

def train_model(model: nn.Module,
                optimizer: Optimizer,
                no_epochs: int,
                vocabs: Vocabs,
                device: str,
                train_logger: DataLogger,
                eval_logger: DataLogger,
                train_data_loader: DataLoader,
                dev_data_loader: DataLoader,
                config: CfgNode):
  model.train()
  for epoch in range(1, no_epochs + 1):
    start_time = time.time()
    epoch_loss = 0
    no_batches = 0
    train_f_score = 0
    train_precision = 0
    train_recall = 0
    train_accuracy = 0
    train_smatch = initialize_smatch()
    train_text = ''
    # Only generate amr_string & smatch at the end of training (expensive)
    log_res_train = epoch >= config.LOGGING_START_EPOCH_TRAIN
    log_res_dev = epoch >= config.LOGGING_START_EPOCH_DEV
    for batch in train_data_loader:
      batch_loss, smatch_score, aux_text, aux_f_score, aux_precision, aux_recall, aux_accuracy = \
        train_step(model, optimizer, vocabs, device, batch, train_logger, config, log_res_train)
      train_smatch[SmatchScore.PRECISION] += smatch_score[SmatchScore.PRECISION]
      train_smatch[SmatchScore.RECALL] += smatch_score[SmatchScore.RECALL]
      train_smatch[SmatchScore.F_SCORE] += smatch_score[SmatchScore.F_SCORE]
      train_text += 'Batch ' + str(no_batches) + ':\n' + aux_text + '\n----\n'
      epoch_loss += batch_loss
      train_f_score += aux_f_score
      train_precision += aux_precision
      train_recall += aux_recall
      train_accuracy += aux_accuracy
      no_batches += 1
    epoch_loss = epoch_loss / no_batches
    train_accuracy = train_accuracy / no_batches
    train_f_score = train_f_score / no_batches
    train_precision = train_precision / no_batches
    train_recall = train_recall / no_batches
    train_smatch = {score_name: score_value / no_batches for score_name, score_value in train_smatch.items()}
    dev_loss, smatch, logged_text, f_score, precision, recall, accuracy, \
    rel_f_score, rel_precision, rel_recall = evaluate_model(
      model, optimizer, vocabs, device, dev_data_loader, eval_logger, config, log_res_dev)
    model.train()
    end_time = time.time()
    time_passed = end_time - start_time
    write_res_to_tensorboard(train_logger, epoch, epoch_loss, train_smatch, train_text,
                             train_f_score, train_precision, train_recall, train_accuracy)
    write_res_to_tensorboard(eval_logger, epoch, dev_loss, smatch, logged_text,
                             f_score, precision, recall, accuracy)
    print('Epoch {} (took {:.2f} seconds)'.format(epoch, time_passed))
    print('TRAIN loss: {}, accuracy: {}%, f_score: {}%, precision: {}%, recall: {}%, smatch: {}%'.format(
      epoch_loss, round_res(train_accuracy), round_res(train_f_score), round_res(train_precision),
      round_res(train_recall), round_res(train_smatch[SmatchScore.F_SCORE])))
    print('DEV   loss: {}, accuracy: {}%, f_score: {}%, precision: {}%, recall: {}%, smatch: {}%'.format(
      dev_loss, round_res(accuracy), round_res(f_score), round_res(precision),
      round_res(recall), round_res(smatch[SmatchScore.F_SCORE])))
    print('LABEL f_score: {}, precision: {}, recall: {}'.format(
      rel_f_score, rel_precision, rel_recall))

def round_res(result: float):
  result = result * 100
  return result.__round__(2)

def write_res_to_tensorboard(logger: DataLogger, epoch_no, loss, smatch,
                             text, f_score, precision, recall, accuracy):
  logger.set_epoch(epoch_no)
  logger.set_loss(loss)
  logger.set_edge_scores(f_score, precision, recall, accuracy)
  logger.set_smatch(smatch[SmatchScore.F_SCORE],
                    smatch[SmatchScore.PRECISION],
                    smatch[SmatchScore.RECALL])
  logger.set_logged_text(text)
  logger.to_tensorboard()

def main(_):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print('Training on device', device)

  if FLAGS.train_subsets is None:
    train_subsets = ['bolt', 'cctv', 'dfa', 'dfb', 'guidelines',
                      'mt09sdl', 'proxy', 'wb', 'xinhua']
  else:
    # Take subsets from flag passed.
    train_subsets = FLAGS.train_subsets.split(',')
  if FLAGS.dev_subsets is None:
    dev_subsets = ['bolt', 'consensus', 'dfa', 'proxy', 'xinhua']
  else:
    # Take subsets from flag passed.
    dev_subsets = FLAGS.dev_subsets.split(',')

  train_paths = get_paths('training', train_subsets)
  dev_paths = get_paths('dev', dev_subsets)

  # Construct config object.
  cfg = get_default_config()

  special_words = ([PAD, EOS, UNK], [PAD, EOS, UNK], [PAD, UNK, None])
  vocabs = Vocabs(train_paths, UNK, special_words, min_frequencies=(1, 1, 1))
  glove_embeddings = GloVeEmbeddings(cfg.RELATION_IDENTIFICATION.GLOVE_EMB_DIM, UNK, [PAD, EOS, UNK]) \
    if FLAGS.use_glove else None
  train_dataset = AMRDataset(
    train_paths, vocabs, device, seq2seq_setting=False, ordered=True, glove=glove_embeddings)
  dev_dataset = AMRDataset(
    dev_paths, vocabs, device, seq2seq_setting=False, ordered=True, glove=glove_embeddings)

  train_data_loader = DataLoader(
    train_dataset, batch_size=FLAGS.batch_size, collate_fn=train_dataset.collate_fn)
  dev_data_loader = DataLoader(
    dev_dataset, batch_size=FLAGS.dev_batch_size, collate_fn=dev_dataset.collate_fn)

  print('Data loaded')

  if FLAGS.config:
    config_file_name = FLAGS.config
    config_path = os.path.join('configs', config_file_name)
    cfg.merge_from_file(config_path)
    cfg.freeze()

  model = RelationIdentification(vocabs.concept_vocab_size, vocabs.relation_vocab_size, cfg.RELATION_IDENTIFICATION,
                                 glove_embeddings.embeddings_vocab if FLAGS.use_glove else None).to(device)
  optimizer = optim.Adam(model.parameters())

  #Use --logdir temp/relation_identification for tensorboard dev upload
  tensorboard_dir = 'temp/relation_identification'
  if os.path.isdir(tensorboard_dir):
    # Delete any existing tensorboard logs.
    shutil.rmtree(tensorboard_dir)
  # Create the dir again.
  os.makedirs(tensorboard_dir)
  train_writer = SummaryWriter(tensorboard_dir+"/train")
  eval_writer = SummaryWriter(tensorboard_dir+"/eval")
  train_logger = DataLogger(train_writer, on_training_flow=True)
  eval_logger = DataLogger(eval_writer)
  train_model(model,
    optimizer, FLAGS.no_epochs, vocabs,
    device,
    train_logger, eval_logger,
    train_data_loader, dev_data_loader,
    cfg.RELATION_IDENTIFICATION)
  eval_writer.close()
  train_writer.close()

if __name__ == "__main__":
  app.run(main)