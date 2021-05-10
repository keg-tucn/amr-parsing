from absl.testing import absltest
import torch

from config import get_default_config
from models import Encoder, AdditiveAttention, DecoderStep, Decoder, Seq2seq
from models import DenseMLP, EdgeScoring, HeadsSelection
from yacs.config import CfgNode

class ModelsTest(absltest.TestCase):


  def test_encoder(self):
    num_layers = 1
    token_vocab_size = 10
    hidden_size = 15
    inputs = [
      [3, 5],
      [7, 2],
      [9, 0],
      [9, 0]
    ]
    inputs = torch.tensor(inputs)
    seq_lengths = torch.tensor([4,2])
    encoder = Encoder(token_vocab_size, hidden_size)
    outputs = encoder(inputs, seq_lengths)
    encoder_states = outputs[0]
    last_encoder_state = outputs[1]
    h, c = last_encoder_state
    self.assertEqual(encoder_states.shape, (4, 2, hidden_size))
    self.assertEqual(h.shape, (num_layers, 2, hidden_size))
    self.assertEqual(c.shape, (num_layers, 2, hidden_size))

  def test_encoder_cuda(self):
    if torch.cuda.is_available():
      device = "cuda"
      num_layers = 1
      token_vocab_size = 10
      hidden_size = 15
      inputs = [
        [3, 5],
        [7, 2],
        [9, 0],
        [9, 0]
      ]
      inputs = torch.tensor(inputs).to(device)
      # This seems to have to stay on the cpu.
      seq_lengths = torch.tensor([4,2])
      encoder = Encoder(token_vocab_size, hidden_size).to(device)
      outputs = encoder(inputs, seq_lengths)
      encoder_states = outputs[0]
      last_encoder_state = outputs[1]
      h, c = last_encoder_state
      self.assertEqual(encoder_states.shape, (4, 2, hidden_size))
      self.assertEqual(h.shape, (num_layers, 2, hidden_size))
      self.assertEqual(c.shape, (num_layers, 2, hidden_size))
      self.assertEqual(encoder_states.device.type, 'cuda')

  def test_encoder_bilstm(self):
    num_layers = 1
    token_vocab_size = 10
    hidden_size = 15
    inputs = [
      [3, 5],
      [7, 2],
      [9, 0]
    ]
    inputs = torch.tensor(inputs)
    seq_lengths = torch.tensor([3,2])
    encoder = Encoder(token_vocab_size, hidden_size, use_bilstm=True)
    outputs = encoder(inputs, seq_lengths)
    encoder_states = outputs[0]
    last_encoder_state = outputs[1]
    h, c = last_encoder_state
    self.assertEqual(encoder_states.shape, (3, 2, 2*hidden_size))
    self.assertEqual(h.shape, (2*num_layers, 2, hidden_size))
    self.assertEqual(c.shape, (2*num_layers, 2, hidden_size))

  def test_additive_attention(self):
    batch_size = 5
    input_seq_len = 7
    hidden_size = 15
    mask = torch.tensor([
      [True, True, True, False, False, False, False],
      [True, True, True, True, True, True, False],
      [True, True, True, True, True, False, False],
      [True, True, True, False, False, False, False],
      [True, True, True, True, True, True, True]
    ])
    decoder_prev_state = torch.zeros((batch_size, hidden_size))
    encoder_outputs = torch.zeros((input_seq_len, batch_size, hidden_size))
    attention_module = AdditiveAttention(hidden_size)
    context = attention_module(decoder_prev_state, encoder_outputs, mask)
    expected_shape = (batch_size, hidden_size)
    self.assertEqual(context.shape, expected_shape)

  def test_additive_attention_batchsize_1(self):
    batch_size = 1
    input_seq_len = 7
    hidden_size = 15
    mask = torch.tensor([
      [True, True, True, False, False, False, False]
    ])
    decoder_prev_state = torch.zeros((batch_size, hidden_size))
    encoder_outputs = torch.zeros((input_seq_len, batch_size, hidden_size))
    attention_module = AdditiveAttention(hidden_size)
    context = attention_module(decoder_prev_state, encoder_outputs, mask)
    expected_shape = (batch_size, hidden_size)
    self.assertEqual(context.shape, expected_shape)

  def test_decoder_step(self):
    output_vocab_size = 10
    batch_size = 3
    input_seq_len = 5
    hidden_size = 15
    config: CfgNode
    config = get_default_config();
    opts = ["CONCEPT_IDENTIFICATION.LSTM_BASED.HIDDEN_SIZE", 15,
            "CONCEPT_IDENTIFICATION.LSTM_BASED.EMB_DIM", 50]
    config.merge_from_list(opts)

    mask = torch.tensor([
      [True, True, True, False, False],
      [True, True, True, True, True],
      [True, True, True, True, False]
    ])
    decoder_input = torch.tensor([3, 7, 1])
    previous_state_h = torch.zeros(batch_size, hidden_size)
    previous_state_c = torch.zeros(batch_size, hidden_size)
    previous_state = (previous_state_h, previous_state_c)
    encoder_states = torch.zeros(input_seq_len, batch_size, hidden_size)
    decoder_step = DecoderStep(output_vocab_size, config.CONCEPT_IDENTIFICATION.LSTM_BASED)
    decoder_state, predictions = decoder_step(
      decoder_input, previous_state, encoder_states, mask)
    self.assertEqual(decoder_state[0].shape, (batch_size, hidden_size))
    self.assertEqual(decoder_state[1].shape, (batch_size, hidden_size))
    self.assertEqual(predictions.shape, (batch_size, output_vocab_size))

  def test_decoder_train(self):
    output_vocab_size = 10
    batch_size = 3
    input_seq_len = 5
    output_seq_len = 7
    num_layers = 1
    hidden_size = 15
    config: CfgNode
    mask = torch.tensor([
      [True, True, True, False, False],
      [True, True, True, True, True],
      [True, True, True, True, False]
    ])
    encoder_states = torch.zeros(input_seq_len, batch_size, hidden_size)
    encoder_last_state = (
      torch.zeros(num_layers, batch_size, hidden_size),
      torch.zeros(num_layers, batch_size, hidden_size))
    encoder_output = (encoder_states, encoder_last_state)
    decoder_inputs = torch.full((output_seq_len, batch_size), 0)
    decoder_model = Decoder(output_vocab_size, config)
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
    hidden_size = 15
    mask = torch.tensor([
      [True, True, True, False, False],
      [True, True, True, True, True],
      [True, True, True, True, False]
    ])
    encoder_states = torch.zeros(input_seq_len, batch_size, hidden_size)
    encoder_last_state = (
      torch.zeros(num_layers, batch_size, hidden_size),
      torch.zeros(num_layers, batch_size, hidden_size))
    encoder_output = (encoder_states, encoder_last_state)
    decoder_model = Decoder(output_vocab_size, hidden_size)
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
    config: CfgNode
    config = get_default_config();
    opts = ["CONCEPT_IDENTIFICATION.LSTM_BASED.HIDDEN_SIZE", 15,
            "CONCEPT_IDENTIFICATION.LSTM_BASED.EMB_DIM", 50]
    config.merge_from_list(opts)
    inputs = torch.zeros((input_seq_len, batch_size)).type(torch.LongTensor)
    input_lengths = torch.tensor([2, 4, 1])
    gold_outputs = torch.zeros((output_seq_len, batch_size)).type(torch.LongTensor)
    seq2seq_model = Seq2seq(input_vocab_size, output_vocab_size, config.CONCEPT_IDENTIFICATION.LSTM_BASED)
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
    hidden_size = 15
    concepts = torch.zeros((batch_size, no_of_concepts, 2*hidden_size))
    edge_scoring = EdgeScoring(hidden_size)
    scores = edge_scoring(concepts)
    self.assertEqual(scores.shape, (batch_size, no_of_concepts, no_of_concepts))

  def test_heads_selection_eval(self):
    batch_size = 2
    concept_vocab_size = 10
    seq_len = 3
    hidden_size = 15
    concepts = [
      #batch ex 1 (amr1)
      [1, 3, 3],
      #batch ex 2 (amr2)
      [4, 5, 0]
    ]
    concepts = torch.tensor(concepts)
    concepts = concepts.transpose(0,1)
    concepts_lengths = torch.tensor([3, 2])
    head_selection = HeadsSelection(concept_vocab_size, hidden_size)
    head_selection.eval()
    scores, predictions = head_selection(concepts, concepts_lengths)
    self.assertEqual(scores.shape, (batch_size, seq_len, seq_len))
    self.assertEqual(predictions.shape, (batch_size, seq_len, seq_len))

  def test_edge_prediction(self):
    scores = [[[-0.0740, 0.0122, -0.0039],
              [0.0268, 0.1135, 0.0972],
              [0.0483, 0.1349, 0.1184]],

            [[-0.0467, 0.0541, -0.0448],
              [0.0024, 0.1034, 0.0080],
              [-0.0051, 0.0995, 0.0000]]]
    scores = torch.tensor(scores)

    expected_predictions = [[[0, 1, 0],
                             [1, 1, 1],
                             [1, 1, 1]],
                            [[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 1]]]
    expected_predictions = torch.tensor(expected_predictions)

    predictions = HeadsSelection.get_predictions(scores, 0.5)
    self.assertTrue(torch.equal(expected_predictions, predictions))

  def test_heads_selection_train(self):
    batch_size = 2
    concept_vocab_size = 10
    seq_len = 3
    hidden_size = 15
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
    head_selection = HeadsSelection(concept_vocab_size, hidden_size)
    head_selection.train()
    scores, predictions = head_selection(concepts, concepts_lengths, adj_mat)
    self.assertEqual(scores.shape, (batch_size, seq_len, seq_len))
    self.assertEqual(predictions.shape, (batch_size, seq_len, seq_len))

  def test_create_padding_mask(self):
    max_seq_len = 5
    concept_lengths = torch.tensor([2, 3])
    expected_padding_mask = [
      [
        [True, True, False, False, False],
        [True, True, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False]
      ],
      [
        [True, True, True, False, False],
        [True, True, True, False, False],
        [True, True, True, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False]
      ]
    ]
    expected_padding_mask = torch.tensor(expected_padding_mask)
    padding_mask = HeadsSelection.create_padding_mask(concept_lengths, max_seq_len)
    self.assertTrue(torch.equal(padding_mask, expected_padding_mask))

  def test_create_fake_root_mask(self):
    batch_size = 2
    seq_len = 5
    expected_fake_root_mask = [
      [
        [False, True, True, True, True],
        [False, True, True, True, True],
        [False, True, True, True, True],
        [False, True, True, True, True],
        [False, True, True, True, True]
      ],
      [
        [False, True, True, True, True],
        [False, True, True, True, True],
        [False, True, True, True, True],
        [False, True, True, True, True],
        [False, True, True, True, True]
      ]
    ]
    expected_fake_root_mask = torch.tensor(expected_fake_root_mask)
    fake_root_mask = HeadsSelection.create_fake_root_mask(batch_size, seq_len)
    self.assertTrue(torch.equal(fake_root_mask, expected_fake_root_mask))

  def test_create_mask(self):
    seq_len = 5
    concept_lengths = torch.tensor([2, 3])
    expected_mask = [
      [
        [False, True, False, False, False],
        [False, True, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False]
      ],
      [
        [False, True, True, False, False],
        [False, True, True, False, False],
        [False, True, True, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False]
      ]
    ]
    expected_mask = torch.tensor(expected_mask)
    mask = HeadsSelection.create_mask(seq_len, concept_lengths, False)
    self.assertTrue(torch.equal(mask, expected_mask))

if __name__ == '__main__':
  absltest.main()