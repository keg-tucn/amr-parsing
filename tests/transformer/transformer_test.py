from absl.testing import absltest
import numpy as np
import torch
import torch.nn as nn

from model.transformer import TransformerSeq2Seq
from models import EMB_DIM, HIDDEN_SIZE

class TransformerTest(absltest.TestCase):

  def test_transformer_seq2seq(self):
    batch_size = 3
    input_seq_len = 4
    output_seq_len = 5
    input_vocab_size = 10
    output_vocab_size = 20
    EMB_DIM = 512
    N_HEADS = 16
    N_LAYERS = 12
    inputs = torch.zeros((input_seq_len, batch_size)).type(torch.LongTensor)
    input_lengths = torch.Tensor([2, 4, 1])
    max_input_length = torch.argmax(input_lengths)
    gold_outputs = torch.zeros((output_seq_len, batch_size)).type(torch.LongTensor)
    transformer = TransformerSeq2Seq(input_vocab_size,
                                     output_vocab_size,
                                     N_LAYERS,
                                     EMB_DIM,
                                     N_HEADS)
    transformer.train()

    logits, predictions = transformer(inputs, gold_outputs)
    self.assertEqual(predictions.shape,
                     (output_seq_len, batch_size))

  def test_transformer_pytorch_train(self):
    batch_size = 3
    input_seq_len = 4
    output_seq_len = 5
    input_vocab_size = 10
    output_vocab_size = 20
    EMB_DIM = 512
    N_HEADS = 16

    inputs = torch.zeros((input_seq_len, batch_size)).type(torch.LongTensor)
    gold_outputs = torch.zeros((output_seq_len, batch_size)).type(torch.LongTensor)

    input_embedder = nn.Embedding(input_vocab_size, EMB_DIM)
    output_embedder = nn.Embedding(output_vocab_size, EMB_DIM)

    inputs = input_embedder(inputs)
    gold_outputs = output_embedder(gold_outputs)

    transformer_model = nn.Transformer(d_model=EMB_DIM, nhead=N_HEADS, num_encoder_layers=12)
    transformer_model.train()
    out = transformer_model(inputs, gold_outputs)

    self.assertEqual(out.shape,
                     (output_seq_len, batch_size, EMB_DIM))

if __name__ == '__main__':
  absltest.main()
