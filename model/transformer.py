"""
  Model Class for Transformer
"""

import math
import torch
import torch.nn as nn

from torch.nn import Transformer
from yacs.config import CfgNode

from data_pipeline.dataset import BOS_IDX

class PositionalEncoding(nn.Module):
  """
  Positional encoding for transformer model
  """
  def __init__(self,
               d_model: int,
               dropout=0.1,
               max_len=500):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    positional_encoding = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(
        0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    positional_encoding[:, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 1::2] = torch.cos(position * div_term)
    positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', positional_encoding)

  def forward(self, encoding):
    encoding = encoding + self.pe[:encoding.size(0), :]
    return self.dropout(encoding)


class TransformerSeq2Seq(nn.Module):
  """Transformer Architecture using Vanilla Transformer.
  A sequence of tokens are passed to the Embedding layer.
  After, Embeddings are passed through Positional Encoding layer.
  The logits need to be passed through final Linear layer.
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
    # Positional Encoding
    self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
    self.pos_decoder = PositionalEncoding(embedding_dim, dropout)
    # Vanilla Transformer
    self.transformer = Transformer(embedding_dim, head_number, hidden_size,
                                   dropout=dropout)
    # Linear Transformation of Transformer Output
    self.dense = nn.Linear(embedding_dim, output_vocab_size).to(device)
    # Device to train on
    self.device = device
    # Masks
    self.trg_mask = None


  def make_triangular_mask(self, inp):
    """
    Wrapper for transformer mask (upper triangular)
    """
    return  nn.Transformer().generate_square_subsequent_mask(inp.size(0)).to(self.device)

  def forward(self,
              input_sequence: torch.Tensor,
              input_lengths: torch.Tensor = None,
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
    input_seq_len = input_sequence.shape[0]
    input_seq_size = input_sequence.shape[1]

    input_sequence = self.enc_embedding(input_sequence)
    input_sequence = self.pos_encoder(input_sequence)
    if self.training:
      gold_output_sequence = self.dec_embedding(gold_output_sequence)
      gold_output_sequence = self.pos_decoder(gold_output_sequence)
      self.trg_mask = self.make_triangular_mask(gold_output_sequence)
      transformer_out = self.transformer(input_sequence,
                                         gold_output_sequence,
                                         tgt_mask=self.trg_mask)
      logits = self.dense(transformer_out)
      activated_outputs = torch.softmax(logits, dim=-1)
      predictions = torch.argmax(activated_outputs, dim=-1)
    else:
      predictions = torch.zeros(
          max_out_length, input_sequence.shape[1]).type(torch.LongTensor).to(self.device)
      predictions[0, :] = BOS_IDX
      self.trg_mask = self.make_triangular_mask(predictions)
      # Apply model max_out_len times; take arg max and push forward
      for i in range(max_out_length):
        predictions = self.dec_embedding(predictions)
        predictions = self.pos_decoder(predictions)
        transformer_out = self.transformer(input_sequence, predictions,
                                           tgt_mask=self.trg_mask)
        logits = self.dense(transformer_out)
        activated_outputs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(activated_outputs, dim=-1)
    return logits, predictions
