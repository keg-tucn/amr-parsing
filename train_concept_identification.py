import os
from typing import Dict
import time
from copy import deepcopy

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

from data_pipeline.data_reading import get_paths
from data_pipeline.vocab import Vocabs
from data_pipeline.dataset import PAD, EOS, UNK, PAD_IDX
from data_pipeline.dataset import AMRDataset
from config import get_default_config
from models import Seq2seq
from data_pipeline.glove_embeddings import GloVeEmbeddings

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
                     default=64,
                     short_name='b',
                     help=('Batch size.'))
flags.DEFINE_integer('dev_batch_size',
                     default=64,
                     help=('Dev batch size.'))
flags.DEFINE_integer('no_epochs',
                     short_name='e',
                     default=80,
                     help=('Number of epochs.'))
flags.DEFINE_boolean('use_glove',
                     default=False,
                     help=('Flag which tells whether model should use GloVe Embeddings or not.'))

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
                   extended_vocab: Vocabs,
                   config: CfgNode):
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
                                                                       extended_vocab, config)

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

  precision = true_positive / (true_positive + false_positive)
  recall = true_positive / (true_positive + false_negative)
  f_score = 0

  if precision + recall != 0:
    f_score = 2 * (precision * recall) / (precision + recall)

  return f_score

def tensor_to_list(gold_outputs,
                   predicted_outputs,
                   eos_index,
                   extended_vocab,
                   config: CfgNode):
  # Extract padding from original outputs
  gold_list_no_padding = extract_padding(gold_outputs, eos_index)
  predicted_list_no_padding = extract_padding(predicted_outputs, eos_index)

  # Remove UNK from the sequence
  # TODO store the gold data before numericalization and use it here
  concepts_as_list_gold = indices_to_words(gold_list_no_padding, extended_vocab, config)
  concepts_as_list_predicted = indices_to_words(predicted_list_no_padding, extended_vocab, config)

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
                     extended_vocab,
                     config: CfgNode):
    # TODO put config and use concept_vocab if not pointer generator
  ids_to_concepts_list = list(extended_vocab.keys())
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
  inputs = batch['sentence']
  inputs_lengths = batch['sentence_lengts']
  gold_outputs = batch['concepts']

  if config.USE_POINTER_GENERATION:
    unnumericalized_inputs = batch['initial_sentence']
    unnumericalized_concepts = batch['concepts_string']
    # compute extended vocab
    extended_vocab = deepcopy(vocabs.shared_vocab)

     # compute extended vocabulary size
    extended_vocab_size = len(extended_vocab.items())

    # add in the extended vocabulary the words from the initial input
    for sentence in unnumericalized_inputs:
        for token in sentence:
            if token not in extended_vocab.keys():
                extended_vocab[token] = extended_vocab_size
                extended_vocab_size += 1

    indices = [[extended_vocab[t] for t in sentence] for sentence in unnumericalized_inputs]

    # numericalized_output
    gold_outputs = torch.transpose(torch.as_tensor([[extended_vocab[word]
                                                         if word in extended_vocab.keys()
                                                         else extended_vocab[UNK] for word in sentence]
                                                        for sentence in unnumericalized_concepts]), 0, 1).to(device)

    logits, predictions = model(inputs, inputs_lengths,
                                    extended_vocab_size, torch.as_tensor(indices),
                                    max_out_length=max_out_len)
    f_score = compute_fScore(gold_outputs, predictions, extended_vocab, config)
  else:
    logits, predictions = model(inputs, inputs_lengths,
                                    max_out_length=max_out_len)

    f_score = compute_fScore(gold_outputs, predictions, vocabs.shared_vocab, config)

  gold_output_len = gold_outputs.shape[0]
  padded_gold_outputs = torch_pad(
    gold_outputs, (0, 0, 0, max_out_len - gold_output_len))
  loss = compute_loss(criterion, logits, padded_gold_outputs)
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
               config: CfgNode):
  inputs = batch['sentence']
  inputs_lengths = batch['sentence_lengts']
  gold_outputs = batch['concepts']

  if config.USE_POINTER_GENERATION:
    # initial sentence (un-numericalized)
    unnumericalized_inputs = batch['initial_sentence']
    # compute indices
    indices = [[vocabs.shared_vocab[t] for t in sentence] for sentence in unnumericalized_inputs]

  optimizer.zero_grad()
  if config.USE_POINTER_GENERATION:
    logits, predictions = model(inputs, inputs_lengths,
                                    vocabs.shared_vocab_size, torch.as_tensor(indices),
                                    gold_outputs)
  else:
    logits, predictions = model(inputs, inputs_lengths, gold_output_sequence=gold_outputs)

  f_score = compute_fScore(gold_outputs, predictions, vocabs.shared_vocab, config)
  loss = compute_loss(criterion, logits, gold_outputs)
  loss.backward()
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
                device):
  model.train()
  for epoch in range(no_epochs):
    start_time = time.time()
    epoch_loss = 0
    no_batches = 0
    batch_f_score_train = 0
    for batch in train_data_loader:
        batch_loss, f_score_train = train_step(model, criterion, optimizer, batch, vocabs, config)
        batch_f_score_train += f_score_train
        epoch_loss += batch_loss
        no_batches += 1
    epoch_loss = epoch_loss / no_batches
    batch_f_score_train = batch_f_score_train / no_batches
    fscore, dev_loss = evaluate_model(
        model, criterion, max_out_len, vocabs, dev_data_loader, config, device)
    model.train()
    end_time = time.time()
    time_passed = end_time - start_time
    print('Epoch {} (took {:.2f} seconds)'.format(epoch + 1, time_passed))
    print('Train loss: {}, dev loss: {}, f_score_train: {}, f-score: {}'.format(epoch_loss, dev_loss,                                                                             batch_f_score_train, fscore))
    train_writer.add_scalar('loss', epoch_loss, epoch)
    eval_writer.add_scalar('loss', dev_loss, epoch)
    eval_writer.add_scalar('f-score', fscore, epoch)
    train_writer.add_scalar('f-score', batch_f_score_train, epoch)

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

  concept_identification_config = cfg.CONCEPT_IDENTIFICATION.LSTM_BASED

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
  glove_embeddings = GloVeEmbeddings(concept_identification_config.GLOVE_EMB_DIM, UNK, [PAD, EOS, UNK]) \
  if FLAGS.use_glove else None

  if concept_identification_config.USE_POINTER_GENERATION:
    use_shared = True
    input_vocab_size = vocabs.shared_vocab_size
    output_vocab_size = vocabs.shared_vocab_size
  else:
    use_shared = False
    input_vocab_size = vocabs.token_vocab_size
    output_vocab_size = vocabs.concept_vocab_size

  train_dataset = AMRDataset(
    train_paths, vocabs, device, seq2seq_setting=True, ordered=True, use_shared=use_shared, glove=glove_embeddings)
  dev_dataset = AMRDataset(
    dev_paths, vocabs, device, seq2seq_setting=True, ordered=True, use_shared=use_shared, glove=glove_embeddings)
  max_out_len = train_dataset.max_concepts_length

  train_data_loader = DataLoader(
    train_dataset, batch_size=FLAGS.batch_size,
    collate_fn=train_dataset.collate_fn)
  dev_data_loader = DataLoader(
    dev_dataset, batch_size=FLAGS.dev_batch_size,
    collate_fn=dev_dataset.collate_fn)

  model = Seq2seq(
    input_vocab_size,
    output_vocab_size,
    concept_identification_config,
    glove_embeddings.embeddings_vocab if FLAGS.use_glove else None,
    device=device).to(device)
  optimizer = optim.Adam(model.parameters())
  criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

  # Use --logdir temp/heads_selection for tensorboard dev upload
  tensorboard_dir = 'temp/concept_identification'
  if not os.path.exists(tensorboard_dir):
      os.makedirs(tensorboard_dir)
  train_writer = SummaryWriter(tensorboard_dir + "/train")
  eval_writer = SummaryWriter(tensorboard_dir + "/eval")
  train_model(
    model, criterion, optimizer, FLAGS.no_epochs,
    max_out_len, vocabs,
    train_data_loader, dev_data_loader,
    train_writer, eval_writer, concept_identification_config, device)
  train_writer.close()
  eval_writer.close()


if __name__ == "__main__":
  app.run(main)
