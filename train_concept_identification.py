import os
import os.path
from typing import Dict
import time


from absl import app
from absl import flags
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import pad as torch_pad
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode

from data_pipeline.dummy.dummy_dataset import DummySeq2SeqDataset, build_dummy_vocab
from data_pipeline.copy_sequence.copy_sequence_dataset import CopySequenceDataset, build_copy_vocab
from data_pipeline.data_reading import get_paths
from data_pipeline.vocab import Vocabs
from data_pipeline.dataset import PAD, BOS, EOS, UNK, PAD_IDX
from data_pipeline.dataset import AMRDataset
from data_pipeline.training_entry import ROOT
from config import get_default_config
from models import Seq2seq
from data_pipeline.glove_embeddings import GloVeEmbeddings
from utils.extended_vocab_utils import construct_extended_vocabulary, numericalize_concepts
from model.transformer import TransformerSeq2Seq
from optimizer import NoamOpt
from yacs.config import CfgNode
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer


FLAGS = flags.FLAGS
flags.DEFINE_string('config',
                    default=None,
                    help=('Config file to overwrite default configs.'))
flags.DEFINE_string('train_subsets',
                    default=None,
                    help=('Train subsets split by comma. Ex: bolt,proxy'))
flags.DEFINE_string('dev_subsets',
                    default=None,
                    help=('Dev subsets split by comma. Ex: bolt,proxy'))
flags.DEFINE_integer('batch_size',
                     default=64,
                     short_name='b',
                     help=('Batch size.'))
flags.DEFINE_integer('dev_batch_size',
                     default=64,
                     short_name='db',
                     help=('Dev batch size.'))
flags.DEFINE_integer('no_epochs',
                     short_name='e',
                     default=80,
                     help=('Number of epochs.'))
flags.DEFINE_boolean('use_glove',
                     default=False,
                     help=('Flag which tells whether model should use GloVe Embeddings or not.'))
flags.DEFINE_integer('max_out_len',
                     short_name='len',
                     default=20,
                     help=('Max sentence length'))
flags.DEFINE_boolean('dummy',
                     default=False,
                     help=('Dataset selection - dummy or default.'))
flags.DEFINE_integer('dummy_train_size',
                     short_name='dts',
                     default=14000,
                     help=('Max AMR length'))
flags.DEFINE_integer('dummy_dev_size',
                     short_name='dds',
                     default=4000,
                     help=('Max AMR length'))
flags.DEFINE_boolean('transformer',
                     default=False,
                     help=('Model selection - transformer or LSTM.'))
flags.DEFINE_boolean('train_is_test',
                     default=False,
                     help=('Train and test on same dataset.'))
flags.DEFINE_boolean('beam',
                     default=False,
                     help=('Turn beam search option on.'))

def compute_loss(criterion, logits, gold_outputs):
  """Computes cross entropy loss.

  Args:
    criterion: Cross entropy loss (with softmax).
    logits: network outputs not passed through activation layer (softmax),
      shape (output seq len, batch size, output no of classes).
    gold_outputs: Gold outputs, shape (output seq len, batch size).

  Returns:
    Loss.
  """
  # Flatten predictions to have only two dimensions,
  # batch size * seq len and no of classes.
  flattened_logits = logits.flatten(start_dim=0, end_dim=1)
  # Flatten gold outputs to have length batch size * seq len.
  flattened_gold_outputs = gold_outputs.flatten()
  loss = criterion(flattened_logits, flattened_gold_outputs)
  return loss


def compute_fScore(gold_outputs,
                   predicted_outputs,
                   extended_vocab: Vocabs):
  """Computes f_score, precision, recall.

  Args:
    gold_outputs: Gold outputs, shape (output seq len, batch size)
    predicted_outputs: Predicted outputs, shape (output seq len, batch size)
    vocabs: Vocabs object
  Returns:
    f_score
  """

  eos_index = list(extended_vocab.keys()).index(EOS)
  concepts_as_list_predicted, concepts_as_list_gold = tensor_to_list(gold_outputs, predicted_outputs, eos_index,
                                                                       extended_vocab)
  f_score = 0
  batch_size = len(concepts_as_list_gold)
  for i in range(batch_size):
    f_score_sentence = compute_sequence_fscore(concepts_as_list_gold[i], concepts_as_list_predicted[i])
    f_score += f_score_sentence

  f_score = f_score / batch_size
  return f_score


def compute_sequence_fscore(gold_sequence, predicted_sequence):
  if len(predicted_sequence) == 0:
    return 0

  true_positive = len(set(gold_sequence) & set(predicted_sequence))
  false_positive = len(set(predicted_sequence).difference(set(gold_sequence)))
  false_negative = len(set(gold_sequence).difference(set(predicted_sequence)))

  if true_positive + false_positive != 0:
    precision = true_positive / (true_positive + false_positive)
  if true_positive + false_negative != 0:
    recall = true_positive / (true_positive + false_negative)
  else:
    precision = 0
    recall = 0
  f_score = 0

  if precision + recall != 0:
    f_score = 2 * (precision * recall) / (precision + recall)

  return f_score


def tensor_to_list(gold_outputs,
                   predicted_outputs,
                   eos_index,
                   extended_vocab):
  # Extract padding from original outputs
  gold_list_no_padding = extract_padding(gold_outputs, eos_index)
  predicted_list_no_padding = extract_padding(predicted_outputs, eos_index)

  # Remove UNK from the sequence
  # TODO store the gold data before numericalization and use it here
  concepts_as_list_gold = indices_to_words(gold_list_no_padding, extended_vocab)
  concepts_as_list_predicted = indices_to_words(predicted_list_no_padding, extended_vocab)

  return concepts_as_list_predicted, concepts_as_list_gold


def extract_padding(outputs, eos_index):
  list_with_padding = []
  list_no_padding = []

  # Transpose the tensors, transform them in lists and remove the root
  for sentence in torch.transpose(outputs, 0, 1):
      list_with_padding.append(sentence.tolist()[1:])

  # Remove the padding -> stop at EOS, for both gold and predicted concepts
  for sentence in list_with_padding:
    sentence_no_padding = []
    for word in sentence:
      if int(word) == eos_index:
        break
      else:
        sentence_no_padding.append(word)
    list_no_padding.append(sentence_no_padding)
  return list_no_padding


def indices_to_words(outputs_no_padding,
                     vocab):

  ids_to_concepts_list = list(vocab.keys())
  concepts_as_list = []
  for sentence in outputs_no_padding:
    concepts = []
    for id in sentence:
        if ids_to_concepts_list[int(id)] != UNK: concepts.append(ids_to_concepts_list[int(id)])
    concepts_as_list.append(concepts)
  return concepts_as_list


def eval_step(model: nn.Module,
              criterion: nn.Module,
              max_out_len: int,
              vocabs: Vocabs,
              batch: Dict[str, torch.tensor],
              config: CfgNode, device):
  inputs = batch['sentence'].to(device)
  inputs_lengths = batch['sentence_lengts']
  gold_outputs = batch['concepts'].to(device)
  character_inputs = batch["char_sentence"]
  character_inputs_lengths = batch["char_sentence_length"]

  if config.LSTM_BASED.USE_POINTER_GENERATOR:
    unnumericalized_inputs = batch['initial_sentence']
    unnumericalized_concepts = batch['concepts_string']
    # compute extended vocab
    extended_vocab, extended_vocab_size = construct_extended_vocabulary(unnumericalized_inputs, vocabs)

    # compute indices of the input sentence for the extended vocab
    indices = [[extended_vocab[t] for t in sentence] for sentence in unnumericalized_inputs]

    # numericalized concepts after the new vocabulary and put it on the device
    gold_outputs = numericalize_concepts(extended_vocab, unnumericalized_concepts).to(device)

    logits, predictions = model(inputs, inputs_lengths,
                                    extended_vocab_size, torch.as_tensor(indices),
                                    max_out_length=max_out_len,
                                    character_inputs=character_inputs,
                                    character_inputs_lengths=character_inputs_lengths)
    f_score = compute_fScore(gold_outputs, predictions, extended_vocab)
  else:
    logits, predictions = model(inputs, inputs_lengths,
                                  max_out_length=max_out_len,
                                  character_inputs=character_inputs,
                                  character_inputs_lengths=character_inputs_lengths)

    f_score = compute_fScore(gold_outputs, predictions, vocabs.concept_vocab)

  gold_output_len = gold_outputs.shape[0]
  padded_gold_outputs = torch_pad(
    gold_outputs, (0, 0, 0, max_out_len - gold_output_len))

  if FLAGS.beam == False:
    loss = compute_loss(criterion, logits, padded_gold_outputs)
  else:
    loss = 0.0#vezi ce shape are loss ul si fa aceeasi chestie cu 0.0
  return f_score, loss


def evaluate_model(model: nn.Module,
                   criterion: nn.Module,
                   max_out_len: int,
                   vocabs: Vocabs,
                   data_loader: DataLoader,
                   config: CfgNode,
                   device):
  model.eval()
  with torch.no_grad():
    epoch_f_score = 0
    epoch_loss = 0
    no_batches = 0
    for batch in data_loader:
      f_score_epoch, loss = eval_step(model, criterion, max_out_len, vocabs, batch, config, device)
      epoch_f_score += f_score_epoch
      epoch_loss += loss
      no_batches += 1
    epoch_f_score = epoch_f_score / no_batches
    epoch_loss = epoch_loss / no_batches
    return epoch_f_score, epoch_loss


def train_step(model: nn.Module,
               criterion: nn.Module,
               optimizer: Optimizer,
               batch: Dict[str, torch.Tensor],
               vocabs: Vocabs,
               config: CfgNode,
               device: str,
               teacher_forcing_ratio: float=0.0):
  inputs = batch['sentence'].to(device)
  inputs_lengths = batch['sentence_lengts']
  gold_outputs = batch['concepts'].to(device)
  character_inputs = batch["char_sentence"]
  character_inputs_lengths = batch["char_sentence_length"]

  if config.LSTM_BASED.USE_POINTER_GENERATOR:
    # initial sentence (un-numericalized)
    unnumericalized_inputs = batch['initial_sentence']
    # compute indices of the input sentence for the extended vocab
    indices = [[vocabs.shared_vocab[t] for t in sentence] for sentence in unnumericalized_inputs]

  optimizer.zero_grad()
  if config.LSTM_BASED.USE_POINTER_GENERATOR:
    logits, predictions = model(inputs, inputs_lengths,
                                    vocabs.shared_vocab_size, torch.as_tensor(indices),
                                    teacher_forcing_ratio, gold_outputs,
                                    character_inputs=character_inputs,
                                    character_inputs_lengths=character_inputs_lengths)
    f_score = compute_fScore(gold_outputs, predictions, vocabs.shared_vocab)
  else:
    logits, predictions = model(inputs, inputs_lengths,
                                teacher_forcing_ratio=teacher_forcing_ratio,
                                gold_output_sequence=gold_outputs,
                                character_inputs=character_inputs,
                                character_inputs_lengths=character_inputs_lengths)
    f_score = compute_fScore(gold_outputs, predictions, vocabs.concept_vocab)

  loss = compute_loss(criterion, logits, gold_outputs)
  loss.backward()
  nn.utils.clip_grad_norm_(model.parameters(), 0.5)
  optimizer.step()
  return loss, f_score

def train_model(model: nn.Module,
                criterion: nn.Module,
                optimizer: Optimizer,
                no_epochs: int,
                max_out_len: int,
                vocabs: Vocabs,
                train_data_loader: DataLoader,
                dev_data_loader: DataLoader,
                train_writer: SummaryWriter,
                eval_writer: SummaryWriter,
                config: CfgNode,
                device: str,
                scheduler=None):
  if config.LOAD_PATH:
    model = load_model_weights(model, config.LOAD_PATH, config.PERSISTED_COMPONENT)

  model.train()
  teacher_forcing_ratio = 1
  step_teacher_forcing_ratio = teacher_forcing_ratio / no_epochs
  for epoch in range(no_epochs):
    start_time = time.time()
    epoch_loss = 0
    no_batches = 0
    batch_f_score_train = 0
    for batch in train_data_loader:
        batch_loss, f_score_train = train_step(model, criterion, optimizer, batch, vocabs, config, device, teacher_forcing_ratio)
        batch_f_score_train += f_score_train
        epoch_loss += batch_loss
        no_batches += 1
    epoch_loss = epoch_loss / no_batches
    batch_f_score_train = batch_f_score_train / no_batches
    train_end_time = time.time()
    train_time = train_end_time - start_time
    print('Training took {:.2f} seconds'.format(train_time))
    eval_start_time = time.time()
    fscore, dev_loss = evaluate_model(
        model, criterion, max_out_len, vocabs, dev_data_loader, config, device)
    # gradually decrease the teacher_forcing_racio
    teacher_forcing_ratio -= step_teacher_forcing_ratio
    model.train()
    eval_end_time = time.time()
    eval_time_passed = eval_end_time - eval_start_time
    print('Evaluating took {:.2f} seconds'.format(eval_time_passed))
    end_time = time.time()
    time_passed = end_time - start_time
    print('Epoch {} (took {:.2f} seconds)'.format(epoch + 1, time_passed))
    print('Train loss: {}, dev loss: {}, f_score_train: {}, f-score: {}'.format(epoch_loss, dev_loss, batch_f_score_train, fscore))
    train_writer.add_scalar('loss', epoch_loss, epoch)
    eval_writer.add_scalar('loss', dev_loss, epoch)
    eval_writer.add_scalar('f-score', fscore, epoch)
    train_writer.add_scalar('f-score', batch_f_score_train, epoch)
    # Use scheduler for LR optimization
    if scheduler:
      scheduler.step()

  # Save pretrained model
  if config.SAVE_PATH:
    torch.save(model.state_dict(), config.SAVE_PATH)


def load_model_weights(model: nn.Module, load_path: str, component: str):
  """
    Loads pretrained model weights to current model
    Args:
      model: model whose weights need to be updated
      load_path (str): file path to load from
      component (str): part of pretrained model to load
  """
  if(os.path.exists(load_path)):
    pretrained_dict = torch.load(load_path)
    # If we're not loading a specific component, we're loading everything
    if component:
      # Select weights belonging to component
      pretrained_dict = {k: v for k, v in pretrained_dict.items() if component in k}
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
  return model


def init_optimizer(model: nn.Module,
                   warmup:bool = False):
  """Initialize optimizer to use with Transformer model
     Args:
      model
      warmup (bool): choose optimizer
    Returns:
      optimizer to be used
  """
  if warmup:
    return NoamOpt(512, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
  return torch.optim.Adam(model.parameters(), lr=5e-5)



def pretrain_transformer_model(criterion,
                               scheduler,
                               max_out_len: int,
                               device: str,
                               config: CfgNode,
                               tensorboard_dir: str):
    """
      Pretrain Transformer model on Wikitext2 Dataset
      Args:
        criterion: criterion for loss computation
        scheduler: scheduler for loss computation
        max_out_len: Max sentence length
        device: device to train on
        config: config node from config file
        tensorboard_dir: directory where to load the experiments
    """
    train_writer = SummaryWriter(tensorboard_dir + "/train")
    eval_writer = SummaryWriter(tensorboard_dir + "/eval")
    if config.COPY_SEQUENCE:
      train_iter, val_iter, test_iter = WikiText2()
      pretrain_special_words = [BOS, EOS, "<extra_pad>"]
      pretrain_vocab = build_copy_vocab(train_iter, pretrain_special_words)
      pretrain_bos_index = list(pretrain_vocab.token_vocab.keys()).index(BOS)
      pretrain_dataset = CopySequenceDataset(pretrain_vocab,
                                             train_iter,
                                             max_sen_len=max_out_len)
      pretrain_eval_dataset = CopySequenceDataset(pretrain_vocab,
                                                  test_iter,
                                                  max_sen_len=max_out_len)
      if not FLAGS.max_out_len:
        max_out_len = pretrain_dataset.max_concepts_length
      pretrain_vocab_size = pretrain_vocab.token_vocab_size
      dataloader = DataLoader(pretrain_dataset,
                              batch_size=FLAGS.batch_size,
                              collate_fn=pretrain_dataset.copy_sequence_collate_fn)
      eval_dataloader = DataLoader(pretrain_eval_dataset,
                                   batch_size=FLAGS.batch_size,
                                   collate_fn=pretrain_dataset.copy_sequence_collate_fn)
      pretrain_model = TransformerSeq2Seq(pretrain_vocab_size,
                                          pretrain_vocab_size,
                                          pretrain_bos_index,
                                          config.TRANSF_BASED,
                                          device=device).to(device)
      pretrain_model.init_params()
      optimizer = init_optimizer(pretrain_model)
      train_model(pretrain_model, criterion, optimizer, config.PRETRAIN_STEPS,
                  max_out_len, pretrain_vocab,
                  dataloader, eval_dataloader,
                  train_writer, eval_writer,
                  config,
                  device,
                  scheduler)


def main(_):
  #TODO: move to new file.
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print('Training on device', device)

  # Construct config object.
  cfg = get_default_config()
  if FLAGS.config:
    config_file_name = FLAGS.config
    config_path = os.path.join('configs', config_file_name)
    cfg.merge_from_file(config_path)
    cfg.freeze()

  concept_identification_config = cfg.CONCEPT_IDENTIFICATION

  max_out_len = FLAGS.max_out_len

  if not FLAGS.dummy:
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

    special_words = ([PAD, EOS, UNK], [PAD, EOS, UNK], [PAD, UNK, None])
    vocabs = Vocabs(train_paths, UNK, special_words, min_frequencies=(1, 1, 1))
    glove_embeddings = GloVeEmbeddings(concept_identification_config.LSTM_BASED.GLOVE_EMB_DIM, UNK, [PAD, EOS, UNK]) \
    if FLAGS.use_glove else None

    if concept_identification_config.LSTM_BASED.USE_POINTER_GENERATOR:
      use_shared = True
      input_vocab_size = vocabs.shared_vocab_size
      output_vocab_size = vocabs.shared_vocab_size
    else:
      use_shared = False
      input_vocab_size = vocabs.token_vocab_size
      output_vocab_size = vocabs.concept_vocab_size

    if not FLAGS.transformer:
      train_dataset = AMRDataset(
        train_paths, vocabs, device, seq2seq_setting=True, ordered=True, use_shared=use_shared, glove=glove_embeddings)
      dev_dataset = AMRDataset(
        dev_paths, vocabs, device, seq2seq_setting=True, ordered=True, use_shared=use_shared, glove=glove_embeddings)
    else:
      train_dataset = AMRDataset(
        train_paths, vocabs, device, seq2seq_setting=True, ordered=True)
      dev_dataset = AMRDataset(
        dev_paths, vocabs, device, seq2seq_setting=True, ordered=True)
      bos_index = list(vocabs.concept_vocab.keys()).index(ROOT)

    if not FLAGS.max_out_len:
      max_out_len = train_dataset.max_concepts_length
  else:
    vocabs = build_dummy_vocab()
    train_dataset = DummySeq2SeqDataset(vocabs, FLAGS.dummy_train_size, FLAGS.max_out_len)
    dev_dataset = DummySeq2SeqDataset(vocabs, FLAGS.dummy_dev_size, FLAGS.max_out_len)
    input_vocab_size = vocabs.token_vocab_size
    output_vocab_size = vocabs.concept_vocab_size
    bos_index = list(train_dataset.vocabs.concept_vocab.keys()).index(BOS)

  train_data_loader = DataLoader(
      train_dataset, batch_size=FLAGS.batch_size,
      collate_fn=train_dataset.collate_fn)
  dev_data_loader = DataLoader(
      dev_dataset, batch_size=FLAGS.dev_batch_size,
      collate_fn=dev_dataset.collate_fn)

  if FLAGS.config:
    config_file_name = FLAGS.config
    config_path = os.path.join('configs', config_file_name)
    cfg.merge_from_file(config_path)
    cfg.freeze()

  criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
  scheduler = None

  if FLAGS.transformer:
    tensorboard_dir = 'temp/concept_identification_transf'
    if not FLAGS.dummy:
      bos_index = list(vocabs.concept_vocab.keys()).index(ROOT)
      # Override configs for Copy Sequence and Save Model Path
      opts = ["CONCEPT_IDENTIFICATION.COPY_SEQUENCE", True,
      "CONCEPT_IDENTIFICATION.SAVE_PATH", "temp/transformer.pth",
      "CONCEPT_IDENTIFICATION.PERSISTED_COMPONENT", "encoder"]
      cfg.merge_from_list(opts)
      pretrain_transformer_model(criterion,
                                 scheduler,
                                 max_out_len,
                                 device,
                                 concept_identification_config,
                                 tensorboard_dir)

    model = TransformerSeq2Seq(vocabs.token_vocab_size,
                               vocabs.concept_vocab_size,
                               bos_index,
                               cfg.CONCEPT_IDENTIFICATION.TRANSF_BASED,
                               device=device).to(device)
    model.init_params()
    # Override configs to add Model Load Path
    opts = ["CONCEPT_IDENTIFICATION.COPY_SEQUENCE", True,
    "CONCEPT_IDENTIFICATION.LOAD_PATH", "temp/transformer.pth"]
    cfg.merge_from_list(opts)
    optimizer = init_optimizer(model)
    
  else:
    model = Seq2seq(
      input_vocab_size,
      output_vocab_size,
      FLAGS.beam,# aici am adaugat flag-ul pt beam search(True pt beam search)
      concept_identification_config.LSTM_BASED,
      glove_embeddings.embeddings_vocab if FLAGS.use_glove else None,
      device=device).to(device)
    tensorboard_dir = 'temp/concept_identification'
    optimizer = optim.Adam(model.parameters())

  # Use --logdir temp/heads_selection for tensorboard dev upload
  if not os.path.exists(tensorboard_dir):
      os.makedirs(tensorboard_dir)
  train_writer = SummaryWriter(tensorboard_dir + "/train")
  eval_writer = SummaryWriter(tensorboard_dir + "/eval")

  if FLAGS.train_is_test:
    train_model(
      model, criterion, optimizer, FLAGS.no_epochs,
      max_out_len, vocabs,
      train_data_loader, train_data_loader,
      train_writer, eval_writer,
      concept_identification_config,
      device,
      scheduler)
  else:
    train_model(
      model, criterion, optimizer, FLAGS.no_epochs,
      max_out_len, vocabs,
      train_data_loader, dev_data_loader,
      train_writer, eval_writer, concept_identification_config,
      device)
  train_writer.close()
  eval_writer.close()


if __name__ == "__main__":
  app.run(main)
