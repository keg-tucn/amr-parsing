"""This module implements the Concept Identification flow using Transformers.
   The transformer used is from PyTorch, with no custom encoder or decoder.
   Inputs to the transformer must be passed through embedding layer.
"""
from typing import Dict
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
# import torch_xla
# import torch_xla.core.xla_model as xm

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from data_pipeline.data_reading import get_paths
from data_pipeline.vocab import Vocabs
from data_pipeline.dataset import PAD, EOS, UNK, PAD_IDX
from data_pipeline.dataset import AMRDataset
from model.transformer import TransformerSeq2Seq
from train_concept_identification import compute_fScore

BATCH_SIZE = 32
DEV_BATCH_SIZE = 32
NO_EPOCHS = 5
EMB_DIM = 16
HEAD_NUMBER = 4
NUM_LAYERS = 3

BOS_IDX = 1

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
  """Model Evaluation Step
  Args:
      model (nn.Module): Model to be trained.
      criterion (nn.Module): Criterion for loss computation.
      max_out_len (int): Maximum size of the sequence.
      batch (Dict[str, torch.tensor])): sentences and concepts tensors

  Returns:
      float: the total loss for the evaluation

  Observation:
      We use logits for loss computation
  """
  inputs = batch['sentence']
  gold_outputs = batch['concepts']

  logits, predictions = model(inputs, gold_outputs, max_out_length=max_out_len)
  # logits, predictions = model(inputs, gold_outputs)
  f_score = compute_fScore(gold_outputs, predictions, vocabs)

  # Send logits to loss
  loss = compute_loss(criterion, logits, gold_outputs)
  return f_score, loss

def evaluate_model(model: nn.Module,
                   criterion: nn.Module,
                   max_out_len: int,
                   data_loader: DataLoader):
  """Model Evaluation
  Args:
      model (nn.Module): Model to be trained.
      criterion (nn.Module): Criterion for loss computation.
      max_out_len (int): Maximum size of the sequence.
      data_loader (DataLoader): Loads data to the model

  Returns:
      float: the total loss for the evaluation
  Observation:
    We do not use logits for loss computation
  """
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
    epoch_loss = epoch_loss / no_batches
    epoch_f_score = epoch_f_score / no_batches
    return epoch_f_score, epoch_loss


def train_step(model: nn.Module,
               criterion: nn.Module,
               optimizer: Optimizer,
               batch: Dict[str, torch.Tensor]):
  """Model Training Step
  Args:
      model (nn.Module): Model to be trained.
      criterion (nn.Module): Criterion for loss computation.
      optimizer (Optimizer): Optimizer to be used.
      batch (Dict[str, torch.Tensor])): Data dictionary.

  Returns:
      float: the total loss for the evaluation
  Observation:
    We do not use logits for loss computation.
    We must embed inputs and gold_outputs before forwarding.
  """
  # Input and target for transformer must be on CPU (error)
  inputs = batch['sentence']
  gold_outputs = batch['concepts']

  optimizer.zero_grad()

  logits, _ = model(inputs, gold_outputs)

  # Compute LOSS -> This seems to be the problem
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
  """Model Training
  Args:
      model (nn.Module): Model to be trained.
      criterion (nn.Module): Criterion for loss computation.
      optimizer (Optimizer): Optimizer to be used
      max_out_len (int): Maximum size of the sequence.
      train_data_loader (DataLoader): Loads Train data to the model.
      dev_data_loader (DataLoader): Loads Dev data to the model.
      input_vocab_size (int): Input vocab size
      output_vocab_size (int): Output vocab size.
      train_writer (SummaryWriter): Writes train data to TensorBoard.
      eval_writer (SummaryWriter): Writes eval data to TensorBoard.
  """
  model.train()
  for epoch in range(NO_EPOCHS):
    start_time = time.time()
    epoch_loss = 0
    no_batches = 0
    for batch in train_data_loader:
      batch_loss = train_step(model,
                              criterion,
                              optimizer,
                              batch)
      epoch_loss += batch_loss
      no_batches += 1
    epoch_loss = epoch_loss / no_batches
    fscore, dev_loss = evaluate_model(model,
                              criterion,
                              max_out_len,
                              dev_data_loader)
    model.train()
    end_time = time.time()
    time_passed = end_time - start_time
    print('Epoch {} (took {:.2f} seconds)'.format(epoch + 1, time_passed))
    print('Train loss: {}, dev loss: {}, f-score: {}'.format(epoch_loss, dev_loss, fscore))
    train_writer.add_scalar('loss', epoch_loss, epoch)
    eval_writer.add_scalar('loss', dev_loss, epoch)
    eval_writer.add_scalar('f-score', fscore, epoch)

if __name__ == "__main__":

  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # DEVICE = xm.xla_device()

  print('Training on device', DEVICE)
  torch.cuda.empty_cache()

  train_subsets = ['bolt', 'cctv', 'dfa', 'dfb', 'guidelines',
             'mt09sdl', 'proxy', 'wb', 'xinhua']
  dev_subsets = ['bolt', 'consensus', 'dfa', 'proxy', 'xinhua']
  train_paths = get_paths('training', train_subsets)
  dev_paths = get_paths('dev', dev_subsets)

  special_words = ([PAD, EOS, UNK], [PAD, EOS, UNK], [PAD, UNK, None])
  vocabs = Vocabs(train_paths, UNK, special_words, min_frequencies=(1, 1, 1))

  train_dataset = AMRDataset(train_paths,
                             vocabs,
                             DEVICE,
                             seq2seq_setting=True,
                             ordered=True)
  dev_dataset = AMRDataset(dev_paths,
                           vocabs,
                           DEVICE,
                           seq2seq_setting=True,
                           ordered=True)

  max_out_len = train_dataset.max_concepts_length

  train_data_loader = DataLoader(train_dataset,
                                 batch_size=BATCH_SIZE,
                                 collate_fn=train_dataset.collate_fn)
  dev_data_loader = DataLoader(dev_dataset,
                               batch_size=DEV_BATCH_SIZE,
                               collate_fn=dev_dataset.collate_fn)

  input_vocab_size = vocabs.token_vocab_size
  output_vocab_size = vocabs.concept_vocab_size

  model = TransformerSeq2Seq(input_vocab_size,
                             output_vocab_size,
                             embedding_dim=EMB_DIM,
                             device=DEVICE)
  model = model.to(DEVICE)
  optimizer = optim.Adam(model.parameters())
  criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

  #Use --logdir temp/heads_selection for tensorboard dev upload
  tensorboard_dir = 'temp/concept_identification_transformer'
  if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
  train_writer = SummaryWriter(tensorboard_dir+"/train")
  eval_writer = SummaryWriter(tensorboard_dir+"/eval")

  train_model(model,
              criterion,
              optimizer,
              max_out_len,
              train_data_loader,
              dev_data_loader,
              train_writer,
              eval_writer)

  train_writer.close()
  eval_writer.close()
