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
                 bos_index: int,
                 # config CONCEPT_IDENTIFICATION.LSTM_BASED
                 config: CfgNode,
                 dropout=0.5,
                 device="cpu"):
        super(TransformerSeq2Seq, self).__init__()
        hidden_size = config.HIDDEN_SIZE
        embedding_dim = config.EMB_DIM
        head_number = config.NUM_HEADS
        layers_number = config.NUM_LAYERS
        self.BOS_IDX = bos_index
        self.device = device
        # Embed inputs
        print("in sz", input_vocab_size)
        print("out sz", output_vocab_size)
        self.enc_embedding = nn.Embedding(
            input_vocab_size, embedding_dim).to(self.device)
        print(self.enc_embedding)
        self.dec_embedding = nn.Embedding(
            output_vocab_size, embedding_dim).to(self.device)
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        self.pos_decoder = PositionalEncoding(embedding_dim, dropout)
        # Vanilla Transformer
        encoder_layers = TransformerEncoderLayer(
            embedding_dim, head_number, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, layers_number).to(self.device)
        decoder_layers = TransformerDecoderLayer(
            embedding_dim, head_number, hidden_size, dropout)
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
            # Begin with Root token
            predictions = torch.zeros(
                1, input_sequence.shape[1]).type(torch.LongTensor).to(self.device)
            predictions[0, :] = self.BOS_IDX
            # Generate max_out_len tokens
            for i in range(max_out_length):
                trg_mask = self.make_triangular_mask(i+1)
                logits, top_indices = self.apply_layer(input_sequence,
                                                       predictions,
                                                       trg_mask=trg_mask)
                # Take last decoded token
                top_indices_last_token = top_indices[-1:]
                # Add it to previously decoded tokens
                predictions = torch.cat(
                    [predictions, top_indices_last_token], dim=0)
        return logits, predictions
