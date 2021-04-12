"""This module implements the Concept Identification flow using Transformers.
   The transformer used is from PyTorch, with no custom encoder or decoder.
   Inputs to the transformer must be passed through embedding layer.
"""
from absl import app
from absl import flags
from typing import Dict
import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
# import torch_xla
# import torch_xla.core.xla_model as xm

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from data_pipeline.data_reading import get_paths
from data_pipeline.vocab import Vocabs
from data_pipeline.dummy.dummy_vocab import DummyVocabs

from data_pipeline.dataset import PAD, EOS, UNK, PAD_IDX
from data_pipeline.dataset import AMRDataset
from data_pipeline.dummy.dummy_dataset import DummySeq2SeqDataset
from model.transformer import TransformerSeq2Seq
from train_concept_identification import train_model, train_step, evaluate_model, eval_step, compute_fScore, compute_loss
from yacs.config import CfgNode
from config import get_default_config
from torch.autograd import Variable

import string
import random

FLAGS = flags.FLAGS

BATCH_SIZE = 32
DEV_BATCH_SIZE = 32
NO_EPOCHS = 5


BOS_IDX = 1
BOS = '<bos>'

if __name__ == "__main__":

  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # DEVICE = xm.xla_device()

  print('Training on device', DEVICE)
  torch.cuda.empty_cache()

  train_subsets = ['bolt', 'cctv']
  dev_subsets = ['bolt']
  train_paths = get_paths('training', train_subsets)
  dev_paths = get_paths('dev', dev_subsets)

  all_sentences = []
  all_sentences_dev = []
  for i in range(30):
      # Generate random string
      letters = string.ascii_lowercase
      sentence = ''.join(random.choice(letters) for i in range(10))
      all_sentences.append(sentence)
  print("all training sentences", all_sentences)
  for i in range(15):
      # Generate random string
      letters = string.ascii_lowercase
      sentence = ''.join(random.choice(letters) for i in range(10))
      all_sentences_dev.append(sentence)
  print("all dev sentences", all_sentences_dev)

  special_words = ([PAD, BOS, EOS, UNK], [PAD, BOS, EOS, UNK], [PAD, BOS, EOS])
  vocabs = DummyVocabs(all_sentences, UNK, special_words, min_frequencies=(1, 1, 1))

  train_dataset = DummySeq2SeqDataset(
                            #  train_paths,
                             all_sentences,
                             vocabs,
                             DEVICE,
                            #  seq2seq_setting=True,
                            #  ordered=True
                             )
  dev_dataset = DummySeq2SeqDataset(
                          # dev_paths,
                           all_sentences_dev,
                           vocabs,
                           DEVICE,
                          #  seq2seq_setting=True,
                          #  ordered=True
                           )

  max_out_len = train_dataset.max_concepts_length

  train_data_loader = DataLoader(train_dataset,
                                 batch_size=BATCH_SIZE,
                                 collate_fn=train_dataset.collate_fn)
  dev_data_loader = DataLoader(dev_dataset,
                               batch_size=DEV_BATCH_SIZE,
                               collate_fn=dev_dataset.collate_fn)

  input_vocab_size = vocabs.token_vocab_size
  output_vocab_size = vocabs.concept_vocab_size

  cfg = get_default_config()
  # if FLAGS.config:
  #   config_file_name = FLAGS.config
  #   config_path = os.path.join('configs', config_file_name)
  #   cfg.merge_from_file(config_path)
  #   cfg.freeze()

  model = TransformerSeq2Seq(input_vocab_size,
                             output_vocab_size,
                             max_out_len,
                             cfg.CONCEPT_IDENTIFICATION.TRANSF_BASED,
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
              NO_EPOCHS,
              max_out_len,
              vocabs,
              train_data_loader,
              dev_data_loader,
              train_writer,
              eval_writer)

  train_writer.close()
  eval_writer.close()
  