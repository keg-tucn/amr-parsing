from typing import Dict
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import pad as torch_pad
from torch.optim import Optimizer

from data_pipeline.data_reading import get_paths
from data_pipeline.vocab import Vocabs
import data_pipeline.dataset
from data_pipeline.dataset import PAD, EOS, UNK, PAD_IDX
from data_pipeline.dataset import AMRDataset

from models import Seq2seq

BATCH_SIZE = 32
DEV_BATCH_SIZE = 32
NO_EPOCHS = 3


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

def eval_step(model: nn.Module,
              criterion: nn.Module,
              max_out_len: int,
              batch: Dict[str, torch.tensor]):
  inputs = batch['sentence']
  inputs_lengths = batch['sentence_lengts']
  gold_outputs = batch['concepts']

  logits, predictions = model(inputs, inputs_lengths, max_out_length=max_out_len)
  
  # Pad gold outputs to max len (pad on second dimension).
  gold_output_len = gold_outputs.shape[0]
  padded_gold_outputs = torch_pad(
    gold_outputs, (0, 0, 0, max_out_len - gold_output_len))
  loss = compute_loss(criterion, logits, padded_gold_outputs)
  return loss

def evaluate_model(model: nn.Module,
                   criterion: nn.Module,
                   max_out_len: int,
                   data_loader: DataLoader):
  model.eval()
  with torch.no_grad():
    epoch_loss = 0
    no_batches = 0
    for batch in data_loader:
      loss = eval_step(model, criterion, max_out_len, batch)
      epoch_loss += loss
      no_batches += 1
    epoch_loss = epoch_loss / no_batches
    return epoch_loss

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
                dev_data_loader: DataLoader):
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
    dev_loss = evaluate_model(model, criterion, max_out_len, dev_data_loader)
    model.train()
    end_time = time.time()
    time_passed = end_time - start_time 
    print('Epoch {} (took {:.2f} seconds)'.format(epoch+1, time_passed))
    print('Train loss: {}, dev loss: {} '.format(epoch_loss, dev_loss))

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
  
  train_model(
    model, criterion, optimizer, max_out_len, train_data_loader, dev_data_loader)
