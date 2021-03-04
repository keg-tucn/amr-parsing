from absl.testing import absltest
import numpy as np
import torch

from models import Encoder, AdditiveAttention, DecoderStep, Decoder, Seq2seq
from models import DenseMLP, EdgeScoring, HeadsSelection
from models import EMB_DIM, HIDDEN_SIZE

class ModelsTest(absltest.TestCase):
    

  def test_encoder(self):
    num_layers = 1
    token_vocab_size = 10
    inputs = [
      [3, 5],
      [7, 2],
      [9, 0],
      [9, 0]
    ]
    inputs = torch.tensor(inputs)
    seq_lengths = torch.tensor([4,2])
    encoder = Encoder(token_vocab_size)
    outputs = encoder(inputs, seq_lengths)
    encoder_states = outputs[0]
    last_encoder_state = outputs[1]
    h, c = last_encoder_state
    self.assertEqual(encoder_states.shape, (4, 2, HIDDEN_SIZE))
    self.assertEqual(h.shape, (num_layers, 2, HIDDEN_SIZE))
    self.assertEqual(c.shape, (num_layers, 2, HIDDEN_SIZE))

  def test_encoder_cuda(self):
    if torch.cuda.is_available():
      device = "cuda"
      num_layers = 1
      token_vocab_size = 10
      inputs = [
        [3, 5],
        [7, 2],
        [9, 0],
        [9, 0]
      ]
      inputs = torch.tensor(inputs).to(device)
      # This seems to have to stay on the cpu.
      seq_lengths = torch.tensor([4,2])
      encoder = Encoder(token_vocab_size).to(device)
      outputs = encoder(inputs, seq_lengths)
      encoder_states = outputs[0]
      last_encoder_state = outputs[1]
      h, c = last_encoder_state
      self.assertEqual(encoder_states.shape, (4, 2, HIDDEN_SIZE))
      self.assertEqual(h.shape, (num_layers, 2, HIDDEN_SIZE))
      self.assertEqual(c.shape, (num_layers, 2, HIDDEN_SIZE))
      self.assertEqual(encoder_states.device.type, 'cuda')
  def test_encoder_bilstm(self):
    num_layers = 1
    token_vocab_size = 10
    inputs = [
      [3, 5],
      [7, 2],
      [9, 0]
    ]
    inputs = torch.tensor(inputs)
    seq_lengths = torch.tensor([3,2])
    encoder = Encoder(token_vocab_size, use_bilstm=True)
    outputs = encoder(inputs, seq_lengths)
    encoder_states = outputs[0]
    last_encoder_state = outputs[1]
    h, c = last_encoder_state
    self.assertEqual(encoder_states.shape, (3, 2, 2*HIDDEN_SIZE))
    self.assertEqual(h.shape, (2*num_layers, 2, HIDDEN_SIZE))
    self.assertEqual(c.shape, (2*num_layers, 2, HIDDEN_SIZE))

  def test_additive_attention(self):
    batch_size = 5
    input_seq_len = 7
    mask = torch.tensor([
      [True, True, True, False, False, False, False],
      [True, True, True, True, True, True, False],
      [True, True, True, True, True, False, False],
      [True, True, True, False, False, False, False],
      [True, True, True, True, True, True, True]
    ])
    decoder_prev_state = torch.zeros((batch_size, HIDDEN_SIZE))
    encoder_outputs = torch.zeros((input_seq_len, batch_size, HIDDEN_SIZE))
    attention_module = AdditiveAttention()
    context = attention_module(decoder_prev_state, encoder_outputs, mask)
    expected_shape = (batch_size, HIDDEN_SIZE)
    self.assertEqual(context.shape, expected_shape)

  def test_additive_attention_batchsize_1(self):
    batch_size = 1
    input_seq_len = 7
    mask = torch.tensor([
      [True, True, True, False, False, False, False]
    ])
    decoder_prev_state = torch.zeros((batch_size, HIDDEN_SIZE))
    encoder_outputs = torch.zeros((input_seq_len, batch_size, HIDDEN_SIZE))
    attention_module = AdditiveAttention()
    context = attention_module(decoder_prev_state, encoder_outputs, mask)
    expected_shape = (batch_size, HIDDEN_SIZE)
    self.assertEqual(context.shape, expected_shape)

  def test_decoder_step(self):
    output_vocab_size = 10
    batch_size = 3
    input_seq_len = 5
    mask = torch.tensor([
      [True, True, True, False, False],
      [True, True, True, True, True],
      [True, True, True, True, False]
    ])
    decoder_input = torch.tensor([3, 7, 1])
    previous_state_h = torch.zeros(batch_size, HIDDEN_SIZE)
    previous_state_c = torch.zeros(batch_size, HIDDEN_SIZE)
    previous_state = (previous_state_h, previous_state_c)
    encoder_states = torch.zeros(input_seq_len, batch_size, HIDDEN_SIZE)
    decoder_step = DecoderStep(output_vocab_size)
    decoder_state, predictions = decoder_step(
      decoder_input, previous_state, encoder_states, mask)
    self.assertEqual(decoder_state[0].shape, (batch_size, HIDDEN_SIZE))
    self.assertEqual(decoder_state[1].shape, (batch_size, HIDDEN_SIZE))
    self.assertEqual(predictions.shape, (batch_size, output_vocab_size))

  def test_decoder_train(self):
    output_vocab_size = 10
    batch_size = 3
    input_seq_len = 5
    output_seq_len = 7
    num_layers = 1
    mask = torch.tensor([
      [True, True, True, False, False],
      [True, True, True, True, True],
      [True, True, True, True, False]
    ])
    encoder_states = torch.zeros(input_seq_len, batch_size, HIDDEN_SIZE)
    encoder_last_state = (
      torch.zeros(num_layers, batch_size, HIDDEN_SIZE),
      torch.zeros(num_layers, batch_size, HIDDEN_SIZE))
    encoder_output = (encoder_states, encoder_last_state)
    decoder_inputs = torch.full((output_seq_len, batch_size), 0)
    decoder_model = Decoder(output_vocab_size)
    decoder_model.train()
    logits, predictions = decoder_model(encoder_output, mask, decoder_inputs)
    self.assertEqual(logits.shape,
      (output_seq_len, batch_size, output_vocab_size))
    self.assertEqual(predictions.shape, (output_seq_len, batch_size))

  def test_decoder_eval(self):
    output_vocab_size = 10
    batch_size = 3
    input_seq_len = 5
    max_output_seq_len = 7
    num_layers = 1
    mask = torch.tensor([
      [True, True, True, False, False],
      [True, True, True, True, True],
      [True, True, True, True, False]
    ])
    encoder_states = torch.zeros(input_seq_len, batch_size, HIDDEN_SIZE)
    encoder_last_state = (
      torch.zeros(num_layers, batch_size, HIDDEN_SIZE),
      torch.zeros(num_layers, batch_size, HIDDEN_SIZE))
    encoder_output = (encoder_states, encoder_last_state)
    decoder_model = Decoder(output_vocab_size)
    decoder_model.eval()
    logits, predictions = decoder_model(encoder_output,
                                mask,
                                max_out_length=max_output_seq_len)
    self.assertEqual(logits.shape,
      (max_output_seq_len, batch_size, output_vocab_size))
    self.assertEqual(predictions.shape, (max_output_seq_len, batch_size))

  def test_seq2seq_train(self):
    batch_size = 3
    input_seq_len = 4
    output_seq_len = 5
    input_vocab_size = 10
    output_vocab_size = 20
    inputs = torch.zeros((input_seq_len, batch_size)).type(torch.LongTensor)
    input_lengths = torch.tensor([2, 4, 1])
    gold_outputs = torch.zeros((output_seq_len, batch_size)).type(torch.LongTensor)
    seq2seq_model = Seq2seq(input_vocab_size, output_vocab_size)
    seq2seq_model.train()
    logits, predictions = seq2seq_model(inputs, input_lengths, gold_outputs)
    self.assertEqual(logits.shape,
      (output_seq_len, batch_size, output_vocab_size))
    self.assertEqual(predictions.shape, (output_seq_len, batch_size))

  def test_seq2seq_train_cuda(self):
    if torch.cuda.is_available():
      device = "cuda"
      batch_size = 3
      input_seq_len = 4
      output_seq_len = 5
      input_vocab_size = 10
      output_vocab_size = 20
      inputs = torch.zeros((input_seq_len, batch_size)).type(torch.LongTensor).to(device)
      input_lengths = torch.tensor([2, 4, 1])
      gold_outputs = torch.zeros(
        (output_seq_len, batch_size)).type(torch.LongTensor).to(device)
      seq2seq_model = Seq2seq(
        input_vocab_size, output_vocab_size, device=device).to(device)
      seq2seq_model.train()
      logits, predictions = seq2seq_model(inputs, input_lengths, gold_outputs)
      self.assertEqual(logits.shape,
        (output_seq_len, batch_size, output_vocab_size))
      self.assertEqual(predictions.shape, (output_seq_len, batch_size))


  def test_dense_mlp(self):
    node_repr_size = 5
    batch_size = 3
    dense_mlp = DenseMLP(node_repr_size=node_repr_size)
    parent = torch.zeros((batch_size, node_repr_size))
    child = torch.zeros((batch_size, node_repr_size))
    edge_repr = dense_mlp(parent, child)
    self.assertEqual(edge_repr.shape, (batch_size,))

  def test_edge_scoring(self):
    batch_size = 2
    no_of_concepts = 3
    concepts = torch.zeros((batch_size, no_of_concepts, 2*HIDDEN_SIZE))
    edge_scoring = EdgeScoring()
    scores = edge_scoring(concepts)
    self.assertEqual(scores.shape, (batch_size, no_of_concepts, no_of_concepts))

  def test_heads_selection_eval(self):
    batch_size = 2
    concept_vocab_size = 10
    seq_len = 3
    concepts = [
      #batch ex 1 (amr1)
      [1, 3, 3],
      #batch ex 2 (amr2)
      [4, 5, 0]
    ]
    concepts = torch.tensor(concepts)
    concepts = concepts.transpose(0,1)
    concepts_lengths = torch.tensor([3, 2])
    head_selection = HeadsSelection(concept_vocab_size)
    head_selection.eval()
    scores = head_selection(concepts, concepts_lengths)
    self.assertEqual(scores.shape, (batch_size, seq_len, seq_len))

  def test_heads_selection_train(self):
    batch_size = 2
    concept_vocab_size = 10
    seq_len = 3
    concepts = [
      #batch ex 1 (amr1)
      [1, 3, 3],
      #batch ex 2 (amr2)
      [4, 5, 0]
    ]
    adj_mat = torch.ones((batch_size, seq_len, seq_len))
    concepts = torch.tensor(concepts)
    concepts = concepts.transpose(0,1)
    concepts_lengths = torch.tensor([3, 2])
    head_selection = HeadsSelection(concept_vocab_size)
    head_selection.train()
    scores = head_selection(concepts, concepts_lengths, adj_mat)
    self.assertEqual(scores.shape, (batch_size, seq_len, seq_len))

  #TODO: mask tests!!!!!

  def test_dense_mlp(self):
    node_repr_size = 5
    batch_size = 3
    dense_mlp = DenseMLP(node_repr_size=node_repr_size)
    parent = torch.zeros((batch_size, node_repr_size))
    child = torch.zeros((batch_size, node_repr_size))
    edge_repr = dense_mlp(parent, child)
    self.assertEqual(edge_repr.shape, (batch_size,))

  def test_edge_scoring(self):
    batch_size = 2
    no_of_concepts = 3
    concepts = torch.zeros((batch_size, no_of_concepts, 2*HIDDEN_SIZE))
    edge_scoring = EdgeScoring()
    scores = edge_scoring(concepts)
    self.assertEqual(scores.shape, (batch_size, no_of_concepts, no_of_concepts))

  def test_heads_selection(self):
    batch_size = 2
    concept_vocab_size = 10
    seq_len = 3
    concepts = [
      #batch ex 1 (amr1)
      [1, 3, 3],
      #batch ex 2 (amr2)
      [4, 5, 0]
    ]
    concepts = torch.tensor(concepts)
    concepts = concepts.transpose(0,1)
    concepts_lengths = torch.tensor([3, 2])
    head_selection = HeadsSelection(concept_vocab_size)
    scores = head_selection(concepts, concepts_lengths)
    self.assertEqual(scores.shape, (batch_size, seq_len, seq_len))

if __name__ == '__main__':
  absltest.main()