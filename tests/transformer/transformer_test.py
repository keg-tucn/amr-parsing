from absl.testing import absltest
import torch
import torch.nn as nn

from model.transformer import TransformerSeq2Seq
from config import get_default_config

class TransformerTest(absltest.TestCase):

  def test_transformer_seq2seq(self):
    batch_size = 3
    input_seq_len = 4
    output_seq_len = 5
    input_vocab_size = 10
    output_vocab_size = 20
    inputs = torch.zeros((input_seq_len, batch_size)).type(torch.LongTensor)
    input_lengths = torch.Tensor([2, 4, 1])
    gold_outputs = torch.zeros((output_seq_len, batch_size)).type(torch.LongTensor)
    cfg = get_default_config()
    device = "cpu"

    transformer = TransformerSeq2Seq(input_vocab_size,
                                     output_vocab_size,
                                     cfg.CONCEPT_IDENTIFICATION.TRANSF_BASED,
                                     device=device).to(device)
    transformer.train()

    logits, predictions = transformer(inputs, input_lengths, gold_outputs)
    self.assertEqual(predictions.shape,
                     (output_seq_len, batch_size))

if __name__ == '__main__':
  absltest.main()
