import os
from typing import Dict
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import pad as torch_pad
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from data_pipeline.data_reading import get_paths
from data_pipeline.vocab import Vocabs
from data_pipeline.dataset import PAD, EOS, UNK, PAD_IDX
from data_pipeline.dataset import AMRDataset

from models import Seq2seq

BATCH_SIZE = 32
DEV_BATCH_SIZE = 32
NO_EPOCHS = 40


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
                   vocabs: Vocabs):
    """Computes f_score, precision, recall.

  Args:
    gold_outputs: Gold outputs, shape (output seq len, batch size)
    predicted_outputs: Predicted outputs, shape (output seq len, batch size)
    vocabs: Vocabs object
  Returns:
    f_score
  """

    eos_index= list(vocabs.concept_vocab.keys()).index(EOS)
    concepts_as_list_predicted, concepts_as_list_gold = tensor_to_list(gold_outputs, predicted_outputs, eos_index, vocabs)

    print("concepts_as_list_gold", concepts_as_list_gold)
    print("concepts_as_list_predicted", concepts_as_list_predicted)

    true_positive = len(set(concepts_as_list_gold) & set(concepts_as_list_predicted))
    false_positive = len(set(concepts_as_list_predicted).difference(set(concepts_as_list_gold)))
    false_negative = len(set(concepts_as_list_gold).difference(set(concepts_as_list_predicted)))
    precision = 0
    recall = 0
    print(f'tp = {true_positive}, fn = {false_negative}, fp = {false_positive}')

    if len(concepts_as_list_predicted) != 0:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
    f_score = 0

    if precision + recall != 0:
        f_score = 2 * (precision * recall) / (precision + recall)

    return f_score


def tensor_to_list(gold_outputs,
                   predicted_outputs,
                   eos_index,
                   vocabs:Vocabs):
    # Extract padding from original outputs
    gold_list_no_padding = extract_padding(gold_outputs, eos_index)
    predicted_list_no_padding = extract_padding(predicted_outputs, eos_index)

    # Remove UNK from the sequence
    # TODO store the gold data before numericalization and use it here
    concepts_as_list_gold = indices_to_words(gold_list_no_padding, vocabs)
    concepts_as_list_predicted = indices_to_words(predicted_list_no_padding, vocabs)

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
                             vocabs: Vocabs):
    ids_to_concepts_list = list(vocabs.concept_vocab.keys())
    concepts_as_list = [ids_to_concepts_list[int(id)] for sentence in outputs_no_padding for id in sentence
                              if ids_to_concepts_list[int(id)] != UNK]
    return concepts_as_list


def eval_step(model: nn.Module,
              criterion: nn.Module,
              max_out_len: int,
              batch: Dict[str, torch.tensor]):
  inputs = batch['sentence']
  inputs_lengths = batch['sentence_lengts']
  gold_outputs = batch['concepts']

  logits, predictions = model(inputs, inputs_lengths, max_out_length=max_out_len)

  f_score = compute_fScore(gold_outputs, predictions, vocabs)

  gold_output_len = gold_outputs.shape[0]
  padded_gold_outputs = torch_pad(
    gold_outputs, (0, 0, 0, max_out_len - gold_output_len))
  loss = compute_loss(criterion, logits, padded_gold_outputs)
  return f_score, loss


def evaluate_model(model: nn.Module,
                   max_out_len: int,
                   data_loader: DataLoader):
  model.eval()
  with torch.no_grad():
    epoch_f_score = 0
    epoch_loss = 0
    no_batches = 0
    for batch in data_loader:
      f_score_epoch, loss = eval_step(model, criterion, max_out_len, batch)
      epoch_f_score += f_score_epoch
      epoch_loss += loss
      no_batches += 1
    epoch_f_score = epoch_f_score / no_batches
    epoch_loss = epoch_loss / no_batches
    return epoch_f_score, epoch_loss


def train_step(model: nn.Module,
               criterion: nn.Module,
               optimizer: Optimizer,
               batch: Dict[str, torch.Tensor]):
  inputs = batch['sentence']
  inputs_lengths = batch['sentence_lengts']
  gold_outputs = batch['concepts']

  optimizer.zero_grad()
  logits, predictions = model(inputs, inputs_lengths, gold_outputs)
  loss = compute_loss(criterion, logits, gold_outputs)
  loss.backward()
  optimizer.step()
  return loss

def train_model(model: nn.Module,
                criterion: nn.Module,
                optimizer: Optimizer,
                max_out_len: int,
                train_data_loader: DataLoader,
                dev_data_loader: DataLoader,
                train_writer: SummaryWriter,
                eval_writer: SummaryWriter):
  model.train()
  for epoch in range(NO_EPOCHS):
    start_time = time.time()
    i = 0
    epoch_loss = 0
    no_batches = 0
    for batch in train_data_loader:
      batch_loss = train_step(model, criterion, optimizer, batch)
      epoch_loss += batch_loss
      no_batches += 1
    epoch_loss = epoch_loss / no_batches
    fscore, dev_loss = evaluate_model(model, max_out_len, dev_data_loader)
    model.train()
    end_time = time.time()
    time_passed = end_time - start_time
    print('Epoch {} (took {:.2f} seconds)'.format(epoch + 1, time_passed))
    print('Train loss: {}, dev loss: {}, f-score: {}'.format(epoch_loss, dev_loss, fscore))
    train_writer.add_scalar('loss', epoch_loss, epoch)
    eval_writer.add_scalar('loss', dev_loss, epoch)
    eval_writer.add_scalar('f-score', fscore, epoch)

if __name__ == "__main__":

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print('Training on device', device)

  train_subsets = ['bolt', 'cctv', 'dfa', 'dfb', 'guidelines',
             'mt09sdl', 'proxy', 'wb', 'xinhua']
  dev_subsets = ['bolt', 'consensus', 'dfa', 'proxy', 'xinhua']
  train_paths = get_paths('training', train_subsets)
  dev_paths = get_paths('dev', dev_subsets)

  special_words = ([PAD, EOS, UNK], [PAD, EOS, UNK], [PAD, UNK, None])
  vocabs = Vocabs(train_paths, UNK, special_words, min_frequencies=(1, 1, 1))
  train_dataset = AMRDataset(
    train_paths, vocabs, device, seq2seq_setting=True, ordered=True)
  dev_dataset = AMRDataset(
    dev_paths, vocabs, device, seq2seq_setting=True, ordered=True)
  max_out_len = train_dataset.max_concepts_length

  train_data_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, collate_fn=train_dataset.collate_fn)
  dev_data_loader = DataLoader(
    dev_dataset, batch_size=DEV_BATCH_SIZE, collate_fn=dev_dataset.collate_fn)

  model = Seq2seq(
    vocabs.token_vocab_size,
    vocabs.concept_vocab_size,
    device=device).to(device)
  optimizer = optim.Adam(model.parameters())
  criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

  # Use --logdir temp/heads_selection for tensorboard dev upload
  tensorboard_dir = 'temp/concept_identification'
  if not os.path.exists(tensorboard_dir):
      os.makedirs(tensorboard_dir)
  train_writer = SummaryWriter(tensorboard_dir + "/train")
  eval_writer = SummaryWriter(tensorboard_dir + "/eval")
  train_model(
      model, criterion, optimizer, max_out_len, train_data_loader, dev_data_loader, train_writer, eval_writer)
  train_writer.close()
  eval_writer.close()
