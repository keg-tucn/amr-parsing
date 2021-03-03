from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_pipeline.data_reading import get_paths
from data_pipeline.vocab import Vocabs
import data_pipeline.dataset
from data_pipeline.dataset import PAD, UNK, PAD_IDX
from data_pipeline.dataset import AMRDataset

from models import Seq2seq

BATCH_SIZE = 16
NO_EPOCHS = 3

def compute_accuracy(predictions: torch.Tensor, gold_outputs: torch.Tensor):
  """[summary]

  Args:
      predictions: shape (seq len, batch size).
      gold_outputs: shape (seq len, batch size).
  """
  pass

def train_step(model: nn.Module,
               criterion,
               optimizer,
               batch: Dict[str, torch.Tensor]):
  inputs = batch['sentence']
  inputs_lengths = batch['sentence_lengts']
  gold_outputs = batch['concepts']

  optimizer.zero_grad()
  logits, predictions = model(inputs, inputs_lengths, gold_outputs)
  # Flatten predictions to have only to dimensions,
  # batch size * seq len and no of classes.
  flattened_logits = logits.flatten(start_dim=0, end_dim=1)
  # Flatten gold outputs to have length batch size * seq len.
  flattened_gold_outputs = gold_outputs.flatten()
  loss = criterion(flattened_logits, flattened_gold_outputs)
  loss.backward()
  optimizer.step()
  return loss

def train_model(model: nn.Module,
                criterion,
                optimizer,
                train_data_loader: DataLoader):
  model.train()
  for epoch in range(NO_EPOCHS):
    print('Epoch ', epoch+1)
    i = 0
    epoch_loss = 0
    no_batches = 0
    for batch in train_data_loader:
      batch_loss = train_step(model, criterion, optimizer, batch)
      epoch_loss += batch_loss
      no_batches += 1
    epoch_loss = epoch_loss / no_batches
    print('Loss is ', epoch_loss)

if __name__ == "__main__":

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print('Training on device', device)

  subsets = ['bolt', 'cctv', 'dfa', 'dfb', 'guidelines',
             'mt09sdl', 'proxy', 'wb', 'xinhua']
  train_paths = get_paths('training', subsets)

  special_words = ([PAD, UNK], [PAD, UNK], [PAD, UNK, None])
  vocabs = Vocabs(train_paths, UNK, special_words, min_frequencies=(1, 1, 1))
  train_dataset = AMRDataset(train_paths, vocabs, device)
  
  train_data_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, collate_fn=train_dataset.collate_fn)

  model = Seq2seq(
    vocabs.token_vocab_size, vocabs.concept_vocab_size, device=device).to(device)
  optimizer = optim.Adam(model.parameters())
  criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
  
  train_model(model, criterion, optimizer, train_data_loader)