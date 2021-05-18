from typing import Dict

import torch
from torch.utils.data import DataLoader

from data_pipeline.dataset import AMRDataset
from data_pipeline.data_reading import get_paths

from models import Seq2seq

BATCH_SIZE = 16


def train_step(batch: Dict[str, torch.Tensor]):
  inputs = batch['sentence']
  inputs_lengths = batch['sentence_lengts']
  outputs = batch['concepts']

def train_model(train_data_loader: DataLoader):
  
  for batch in dataloader:
    train_step(batch)

if __name__ == "__main__":
  subsets = ['bolt', 'cctv', 'dfa', 'dfb', 'guidelines',
             'mt09sdl', 'proxy', 'wb', 'xinhua']
  train_paths = get_paths('training', subsets)
  train_dataset = AMRDataset(paths)
  
  train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, collate_fn=AMRDataset.collate_fn)

  criterion = nn.CrossEntropyLoss(ignore_index = dataset.PAD_IDX)
  model = Seq2seq()

  i = 0
  for batch in dataloader:
    if i == 2:
      break
    i+=1
    print('Batch ',i)
    print(batch['sentence'].shape)