"""
  Model Class for Transformer
"""

import math
import torch
import torch.nn as nn

from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from yacs.config import CfgNode


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model
    Code from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self,
                 d_model: int,
                 config: CfgNode):
        super(PositionalEncoding, self).__init__()
        self.dropout = config.DROPOUT_RATE
        self.max_len = config.MAX_POS_ENC_LEN
        self.dropout = nn.Dropout(p=self.dropout)

        positional_encoding = torch.zeros(self.max_len, d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
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
                 bos_index: int,
                 # config CONCEPT_IDENTIFICATION.LSTM_BASED
                 config: CfgNode,
                 device="cpu"):
        super(TransformerSeq2Seq, self).__init__()
        hidden_size = config.HIDDEN_SIZE
        embedding_dim = config.EMB_DIM
        head_number = config.NUM_HEADS
        layers_number = config.NUM_LAYERS
        self.dropout = config.DROPOUT_RATE
        self.BOS_IDX = bos_index
        self.device = device
        # Embed inputs
        self.enc_embedding = nn.Embedding(
            input_vocab_size, embedding_dim).to(self.device)
        self.dec_embedding = nn.Embedding(
            output_vocab_size, embedding_dim).to(self.device)
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, config)
        self.pos_decoder = PositionalEncoding(embedding_dim, config)
        # Vanilla Transformer
        encoder_layers = TransformerEncoderLayer(
            embedding_dim, head_number, hidden_size, self.dropout).to(self.device)
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, layers_number).to(self.device)
        decoder_layers = TransformerDecoderLayer(
            embedding_dim, head_number, hidden_size, self.dropout).to(self.device)
        self.transformer_decoder = TransformerDecoder(
            decoder_layers, layers_number).to(self.device)

        # Linear Transformation of Transformer Output
        self.dense = nn.Linear(
            embedding_dim, output_vocab_size).to(self.device)

    def make_triangular_mask(self, size):
        """
        Wrapper for transformer mask (upper triangular)
        """
        return nn.Transformer().generate_square_subsequent_mask(size).to(self.device)

    def apply_layer(self,
                    input_sequence, output_sequence,
                    src_mask=None,
                    trg_mask=None):
        """
        Function for applying the transformer layer

        Arguments:
          input_sequence: input to the decoder
          output_sequence: input to the decoder
          src_mask:
          trg_mask: target mask for attention
        Returns
          logits: tensor of shape ...
          predictions: tensor of shape ...
        """
        output_sequence = self.dec_embedding(output_sequence)
        output_sequence = self.pos_decoder(output_sequence)
        transformer_out = self.transformer_decoder(output_sequence,
                                                   input_sequence,
                                                   tgt_mask=trg_mask)
        logits = self.dense(transformer_out)
        activated_outputs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(activated_outputs, dim=-1)
        return logits, predictions

    def forward(self,
                input_sequence: torch.Tensor,
                input_lengths: torch.Tensor = None,
                gold_output_sequence: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.0,
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
        input_sequence = self.enc_embedding(input_sequence)
        input_sequence = self.pos_encoder(input_sequence)
        input_sequence = self.transformer_encoder(input_sequence)
        if self.training:
            trg_mask = self.make_triangular_mask(gold_output_sequence.shape[0])
            logits, predictions = self.apply_layer(input_sequence,
                                                   gold_output_sequence,
                                                   trg_mask=trg_mask)
        else:
            predictions = torch.zeros(
                max_out_length, input_sequence.shape[1]).type(torch.LongTensor).to(self.device)
            predictions[0, :] = self.BOS_IDX
            trg_mask = self.make_triangular_mask(predictions.shape[0])
            # Apply model max_out_len times
            for i in range(max_out_length):
                logits, predictions = self.apply_layer(
                    input_sequence, predictions, trg_mask=trg_mask)
        return logits, predictions

    def init_params(self):
        """Initialize parameters with Glorot / fan_avg"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
