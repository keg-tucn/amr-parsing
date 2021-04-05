from typing import Tuple
import math
import random
import torch
import torch.nn as nn

from torch.nn import TransformerEncoder,\
                     TransformerEncoderLayer,\
                     TransformerDecoderLayer,\
                     TransformerDecoder
from yacs.config import CfgNode

#TODO: move this.
BOS_IDX = 1

class TransformerSeq2Seq(nn.Module):
  """Transformer Architecture using Vanilla Transformer.
  A sequence of tokens are passed to the embedding layer. 
  The logits need to be passed through final linear layer.
  """
  def __init__(self,
               input_vocab_size: int,
               output_vocab_size: int,
               # config CONCEPT_IDENTIFICATION.LSTM_BASED
               config: CfgNode, 
               dropout=0.5,
               device="cpu"):
    super(TransformerSeq2Seq, self).__init__()
    hidden_size = config.HIDDEN_SIZE
    embedding_dim = config.EMB_DIM
    head_number = config.NUM_HEADS
    # Embed inputs
    self.enc_embedding = nn.Embedding(input_vocab_size, embedding_dim).to(device)
    self.dec_embedding = nn.Embedding(output_vocab_size, embedding_dim).to(device)
    # Vanilla Transformer
    self.transformer = nn.Transformer(embedding_dim, head_number, hidden_size, dropout=dropout)
    # Linear Transformation of Transformer Output
    self.dense = nn.Linear(embedding_dim, output_vocab_size).to(device)
    # Device to train on
    self.device = device
  
  def forward(self,
                input_sequence : torch.Tensor,
                # input_lengths: torch.Tensor = None,
                gold_output_sequence: torch.Tensor = None,
                max_out_length: int = None):
      """Forward transformer

      Args:
          input_sequence (torch.Tensor): encoder inputs
          input_lengths (torch.Tensor): inputs length
          gold_output_sequence (torch.Tensor): decoder inputs
          max_out_length (int): maximum output length
      Returns: 
          logits of shape ...
          predictions of shape (out seq len, batch size)  
      """
      # Embed input sequence
      input_sequence = self.enc_embedding(input_sequence)
      
      if self.training:
        # Embed output sequence
        gold_output_sequence = self.dec_embedding(gold_output_sequence) 
        # Transform
        transformer_out = self.transformer(input_sequence, gold_output_sequence)
        # Linear Layer
        logits = self.dense(transformer_out)
        activated_outputs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(activated_outputs, dim=-1)
      else:
        # predictions = torch.clone(gold_output_sequence).type(torch.LongTensor).to(self.device)
        predictions = torch.zeros(gold_output_sequence.shape).type(torch.LongTensor).to(self.device)
        predictions[:,0]= BOS_IDX
        # Apply model max_out_len times; take arg max and push forward
        for i in range(max_out_length):
          # Embed Predictions
          predictions = self.dec_embedding(predictions)
          # Transform
          transformer_out = self.transformer(input_sequence, predictions)
          logits = self.dense(transformer_out)
          activated_outputs = torch.softmax(logits, dim=-1)
          # Take arg max from softmax
          predictions = torch.argmax(activated_outputs, dim=-1)
      return logits, predictions

