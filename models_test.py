from absl.testing import absltest
import numpy as np
import torch

from models import Encoder, AdditiveAttention, DecoderStep, Decoder
from models import EMB_DIM, HIDDEN_SIZE

class ModelsTest(absltest.TestCase):

  def test_encoder(self):
    num_layers = 1
    token_vocab_size = 10
    inputs = [
      [3, 5],
      [7, 2],
      [9, 0]
    ]
    inputs = torch.tensor(inputs)
    encoder = Encoder(token_vocab_size)
    outputs = encoder(inputs)
    encoder_states = outputs[0]
    last_encoder_state = outputs[1]
    h, c = last_encoder_state
    self.assertEqual(encoder_states.shape, (3, 2, HIDDEN_SIZE))
    self.assertEqual(h.shape, (num_layers, 2, HIDDEN_SIZE))
    self.assertEqual(c.shape, (num_layers, 2, HIDDEN_SIZE))

  def test_additive_attention(self):
    batch_size = 5
    input_seq_len = 7
    decoder_prev_state = torch.zeros((batch_size, HIDDEN_SIZE))
    encoder_outputs = torch.zeros((input_seq_len, batch_size, HIDDEN_SIZE))
    attention_module = AdditiveAttention()
    context = attention_module(decoder_prev_state, encoder_outputs)
    expected_shape = (batch_size, HIDDEN_SIZE)
    self.assertEqual(context.shape, expected_shape)

  def test_decoder_step(self):
    output_vocab_size = 10
    batch_size = 3
    input_seq_len = 5
    decoder_input = torch.tensor([3, 7, 1])
    previous_state_h = torch.zeros(batch_size, HIDDEN_SIZE)
    previous_state_c = torch.zeros(batch_size, HIDDEN_SIZE)
    previous_state = (previous_state_h, previous_state_c)
    encoder_states = torch.zeros(input_seq_len, batch_size, HIDDEN_SIZE)
    decoder_step = DecoderStep(output_vocab_size)
    decoder_state, predictions = decoder_step(
      decoder_input, previous_state, encoder_states)
    self.assertEqual(decoder_state[0].shape, (batch_size, HIDDEN_SIZE))
    self.assertEqual(decoder_state[1].shape, (batch_size, HIDDEN_SIZE))
    self.assertEqual(predictions.shape, (batch_size, output_vocab_size))

  def test_decoder_train(self):
    output_vocab_size = 10
    batch_size = 3
    input_seq_len = 5
    output_seq_len = 7
    num_layers = 1
    encoder_states = torch.zeros(input_seq_len, batch_size, HIDDEN_SIZE)
    encoder_last_state = (
      torch.zeros(num_layers, batch_size, HIDDEN_SIZE),
      torch.zeros(num_layers, batch_size, HIDDEN_SIZE))
    encoder_output = (encoder_states, encoder_last_state)
    decoder_inputs = torch.full((output_seq_len, batch_size), 0)
    decoder_model = Decoder(output_vocab_size)
    decoder_model.train()
    predictions = decoder_model(encoder_output, decoder_inputs)
    self.assertEqual(predictions.shape,
      (output_seq_len, batch_size, output_vocab_size))

  def test_decoder_eval(self):
    output_vocab_size = 10
    batch_size = 3
    input_seq_len = 5
    max_output_seq_len = 7
    num_layers = 1
    encoder_states = torch.zeros(input_seq_len, batch_size, HIDDEN_SIZE)
    encoder_last_state = (
      torch.zeros(num_layers, batch_size, HIDDEN_SIZE),
      torch.zeros(num_layers, batch_size, HIDDEN_SIZE))
    encoder_output = (encoder_states, encoder_last_state)
    decoder_model = Decoder(output_vocab_size)
    decoder_model.eval()
    predictions = decoder_model(encoder_output,
                                max_out_length=max_output_seq_len)
    self.assertEqual(predictions.shape,
      (max_output_seq_len, batch_size, output_vocab_size))

if __name__ == '__main__':
  absltest.main()