from typing import Dict
import time
import os
import shutil
import io
import re

from absl import app
from absl import flags
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from data_pipeline.data_reading import get_paths
from data_pipeline.vocab import Vocabs
from data_pipeline.dataset import PAD, EOS, UNK
from data_pipeline.dataset import AMR_ID_KEY, CONCEPTS_KEY, CONCEPTS_LEN_KEY,\
  GLOVE_CONCEPTS_KEY, ADJ_MAT_KEY, AMR_STR_KEY
from data_pipeline.dataset import AMRDataset
from data_pipeline.glove_embeddings import GloVeEmbeddings
from arcs_masking import create_mask

from config import get_default_config
from models import HeadsSelection
from evaluation.tensors_to_amr import get_unlabelled_amr_strings_from_tensors
from smatch import score_amr_pairs

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

def compute_loss(vocabs: Vocabs, concepts_lengths: torch.Tensor,
                 logits: torch.Tensor, gold_outputs: torch.Tensor):
  """
  Args:
    vocabs: Vocabs object (with the 3 vocabs).
    mask: Mask for weighting the loss.
    logits: Concepts edges scores (batch size, seq len, seq len).
    gold_outputs: Gold adj mat (with relation labels) of shape
      (batch size, seq len, seq len).

  Returns:
    Binary cross entropy loss over batch.
  """
  no_rel_index = vocabs.relation_vocab[None]
  pad_idx = vocabs.relation_vocab[PAD]
  binary_outputs = (gold_outputs != no_rel_index) * (gold_outputs != pad_idx)
  binary_outputs = binary_outputs.type(torch.FloatTensor)
  mask = create_mask(gold_outputs, concepts_lengths)
  weights = mask.type(torch.FloatTensor)
  flattened_logits = logits.flatten()
  flattened_binary_outputs = binary_outputs.flatten()
  flattened_weights = weights.flatten()
  loss = binary_cross_entropy_with_logits(
    flattened_logits, flattened_binary_outputs, flattened_weights)
  return loss

def compute_smatch(gold_outputs, predictions):
  """
  Args:
    gold_outputs: list of gold amrs as strings
    predictions: list of predictions of amrs as strings

  Returns:
    smatch: the smatch score is composed of 3 metrics:
      - precision
      - recall
      - best f-score
  """
  gold_outputs = ' \n\n '.join(gold_outputs)
  predictions = ' \n\n '.join(predictions)

  gold_file = io.StringIO(gold_outputs)
  pred_file = io.StringIO(predictions)

  smatch_score = {}
  smatch_score["precision"], smatch_score["recall"], smatch_score["best_f_score"] = \
    next(score_amr_pairs(gold_file, pred_file))

  return smatch_score

def replace_all_edge_labels(amr_str: str, new_edge_label: str):
  """Replaces all edge labels in an AMR with a given edge and returns the
  new AMR.
  """
  new_amr_str = re.sub(r':[^\s]*', new_edge_label, amr_str)
  return new_amr_str

def eval_step(model: nn.Module,
              optimizer: nn.Module,
              vocabs: Vocabs,
              device: str,
              batch: torch.tensor):
  amr_ids, inputs, inputs_lengths, gold_adj_mat, gold_amr_str, glove_concepts = get_gold_data(batch)

  optimizer.zero_grad()
  inputs_device = inputs.to(device)
  gold_adj_mat_device = gold_adj_mat.to(device)
  logits, predictions = model(inputs_device, inputs_lengths)

  # Remove the edge labels for the gold AMRs before doing the smatch.
  gold_outputs = [replace_all_edge_labels(a, UNK_REL_LABEL) for a in gold_amr_str]
  predictions_strings = get_unlabelled_amr_strings_from_tensors(
    inputs, inputs_lengths, predictions, vocabs, UNK_REL_LABEL)

  loss = compute_loss(vocabs, inputs_lengths, logits, gold_adj_mat_device)
  smatch_score = compute_smatch(gold_outputs, predictions_strings)
  amr_comparison_text = '  \n'.join([gold_amr_str[0], ' VERSUS', predictions_strings[0]])

  return loss, smatch_score, amr_comparison_text

def evaluate_model(model: nn.Module,
                   optimizer: nn.Module,
                   vocabs: Vocabs,
                   device: str,
                   data_loader: DataLoader):
  model.eval()
  with torch.no_grad():
    epoch_loss = 0
    no_batches = 0
    epoch_smatch = {
      "precision": 0.0,
      "recall": 0.0,
      "best_f_score": 0.0
    }
    logged_text = "GOLD VS PREDICTED AMRS\n"

    for batch in data_loader:
      loss, smatch_score, amr_comparison_text = eval_step(
        model, optimizer, vocabs, device, batch)
      epoch_loss += loss
      epoch_smatch["precision"] += smatch_score["precision"]
      epoch_smatch["recall"] += smatch_score["recall"]
      epoch_smatch["best_f_score"] += smatch_score["best_f_score"]
      logged_text += 'Batch ' + str(no_batches) + ':\n'
      logged_text += amr_comparison_text + '\n----\n'
      no_batches += 1
    epoch_loss = epoch_loss / no_batches
    epoch_smatch = {score_name: score_value / no_batches for score_name, score_value in epoch_smatch.items()}
    logged_text = logged_text.replace('\n', '\n\n')
    return epoch_loss, epoch_smatch, logged_text

def train_step(model: nn.Module,
               optimizer: Optimizer,
               vocabs: Vocabs,
               device: str,
               batch: Dict[str, torch.Tensor]):
  amr_ids, inputs, inputs_lengths, gold_adj_mat, _, _ = get_gold_data(batch)

  optimizer.zero_grad()
  # Move to trainig device (eg. cuda).
  inputs_device = inputs.to(device)
  gold_adj_mat_device = gold_adj_mat.to(device)
  logits, predictions = model(inputs_device, inputs_lengths)
  loss = compute_loss(vocabs, inputs_lengths, logits, gold_adj_mat_device)
  loss.backward()
  optimizer.step()
  return loss

def write_results_on_tensorboard(train_writer: SummaryWriter, eval_writer: SummaryWriter,
                                 epoch, losses, smatch, logged_text):
  epoch_loss = losses['train_loss']
  dev_loss = losses['dev_loss']

  train_writer.add_scalar('loss', epoch_loss, epoch)
  eval_writer.add_scalar('loss', dev_loss, epoch)
  eval_writer.add_scalar('smatch_f_score', smatch["best_f_score"], epoch)
  eval_writer.add_scalar('smatch_precision', smatch["precision"], epoch)
  eval_writer.add_scalar('smatch_recall', smatch["recall"], epoch)
  eval_writer.add_text('amr', logged_text, epoch)

def train_model(model: nn.Module,
                optimizer: Optimizer,
                no_epochs: int,
                vocabs: Vocabs,
                device: str,
                train_writer: SummaryWriter,
                eval_writer: SummaryWriter,
                train_data_loader: DataLoader,
                dev_data_loader: DataLoader):
  model.train()
  for epoch in range(no_epochs):
    start_time = time.time()
    epoch_loss = 0
    no_batches = 0
    for batch in train_data_loader:
      batch_loss = train_step(model, optimizer, vocabs, device, batch)
      epoch_loss += batch_loss
      no_batches += 1
    epoch_loss = epoch_loss / no_batches
    dev_loss, smatch, logged_text = evaluate_model(
      model, optimizer, vocabs, device, dev_data_loader)
    model.train()
    end_time = time.time()
    time_passed = end_time - start_time
    losses = {'train_loss': epoch_loss, 'dev_loss': dev_loss}
    write_results_on_tensorboard(train_writer, eval_writer, epoch, losses, smatch, logged_text)
    print('Epoch {} (took {:.2f} seconds)'.format(epoch+1, time_passed))
    print('Train loss: {}, dev loss: {}, smatch_f_score: {} '.format(epoch_loss, dev_loss, smatch["best_f_score"]))

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
  glove_embeddings = GloVeEmbeddings(cfg.HEAD_SELECTION.GLOVE_EMB_DIM, UNK, [PAD, EOS, UNK]) \
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

  model = HeadsSelection(vocabs.concept_vocab_size, cfg.HEAD_SELECTION,
                         glove_embeddings.embeddings_vocab if FLAGS.use_glove else None).to(device)
  optimizer = optim.Adam(model.parameters())

  #Use --logdir temp/heads_selection for tensorboard dev upload
  tensorboard_dir = 'temp/heads_selection'
  if os.path.isdir(tensorboard_dir):
    # Delete any existing tensorboard logs.
    shutil.rmtree(tensorboard_dir)
  # Create the dir again.
  os.makedirs(tensorboard_dir)
  train_writer = SummaryWriter(tensorboard_dir+"/train")
  eval_writer = SummaryWriter(tensorboard_dir+"/eval")
  train_model(model,
    optimizer, FLAGS.no_epochs, vocabs,
    device,
    train_writer, eval_writer,
    train_data_loader, dev_data_loader)
  train_writer.close()
  eval_writer.close()

if __name__ == "__main__":
  app.run(main)