from typing import Dict
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import pad as torch_pad

from data_pipeline.data_reading import get_paths
from data_pipeline.vocab import Vocabs
import data_pipeline.dataset
from data_pipeline.dataset import PAD, EOS, UNK, PAD_IDX
from data_pipeline.dataset import AMRDataset

from models import HeadsSelection

BATCH_SIZE = 32
DEV_BATCH_SIZE = 32
NO_EPOCHS = 3
HIDDEN_SIZE = 40


def compute_loss(criterion, vocabs: Vocabs, logits: torch.Tensor, gold_outputs: torch.Tensor):
  """
  Args:
    criterion Binary loss criterion.
    logits: Concepts edges scores (batch size, seq len, seq len).
    gold_outputs: Gold adj mat (with relation labels) of shape
      (batch size, seq len, seq len).

  Returns:
    loss.
  """
  no_rel_index = vocabs.relation_vocab[None]
  pad_idx = vocabs.relation_vocab[PAD]
  binary_outputs = (gold_outputs != no_rel_index) * (gold_outputs != pad_idx)
  binary_outputs = binary_outputs.type(torch.FloatTensor)
  flattened_logits = logits.flatten()
  flattened_binary_outputs = binary_outputs.flatten()
  loss = criterion(flattened_logits, flattened_binary_outputs)
  return loss

def train_step(model: nn.Module,
               criterion,
               optimizer,
               vocabs,
               batch: Dict[str, torch.Tensor]):
  inputs = batch['concepts']
  inputs_lengths = batch['concepts_lengths']
  gold_adj_mat = batch['adj_mat']

  optimizer.zero_grad()
  logits = model(inputs, inputs_lengths, gold_adj_mat)
  loss = compute_loss(criterion, vocabs, logits, gold_adj_mat)
  loss.backward()
  optimizer.step()
  return loss

def train_model(model: nn.Module,
                criterion,
                optimizer,
                vocabs,
                train_data_loader: DataLoader,
                dev_data_loader: DataLoader):
  model.train()
  for epoch in range(NO_EPOCHS):
    start_time = time.time()
    i = 0
    epoch_loss = 0
    no_batches = 0
    for batch in train_data_loader:
      batch_loss = train_step(model, criterion, optimizer, vocabs, batch)
      epoch_loss += batch_loss
      no_batches += 1
    epoch_loss = epoch_loss / no_batches
    # dev_loss = evaluate_model(model, criterion, dev_data_loader)
    dev_loss = 0
    model.train()
    end_time = time.time()
    time_passed = end_time - start_time 
    print('Epoch {} (took {:.2f} seconds)'.format(epoch+1, time_passed))
    print('Train loss: {}, dev loss: {} '.format(epoch_loss, dev_loss))

if __name__ == "__main__":

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print('Training on device', device)

  # subsets = ['bolt', 'cctv', 'dfa', 'dfb', 'guidelines',
            #  'mt09sdl', 'proxy', 'wb', 'xinhua']
  subsets = ['bolt']
  train_paths = get_paths('training', subsets)
  dev_paths = get_paths('dev', subsets)

  special_words = ([PAD, EOS, UNK], [PAD, EOS, UNK], [PAD, UNK, None])
  vocabs = Vocabs(train_paths, UNK, special_words, min_frequencies=(1, 1, 1))
  train_dataset = AMRDataset(
    train_paths, vocabs, device, seq2seq_setting=False, ordered=True)
  dev_dataset = AMRDataset(
    dev_paths, vocabs, device, seq2seq_setting=False, ordered=True)
  
  train_data_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, collate_fn=train_dataset.collate_fn)
  dev_data_loader = DataLoader(
    dev_dataset, batch_size=DEV_BATCH_SIZE, collate_fn=dev_dataset.collate_fn)

  criterion = nn.BCEWithLogitsLoss()
  model = HeadsSelection(vocabs.concept_vocab_size, HIDDEN_SIZE).to(device)
  optimizer = optim.Adam(model.parameters())
  
  train_model(
    model, criterion, optimizer, vocabs, train_data_loader, dev_data_loader)