from absl.testing import absltest
import torch

from models import CharacterLevelEmbedding, Encoder, AdditiveAttention, DecoderStep, Decoder, Seq2seq
from models import DenseMLP, EdgeScoring, RelationIdentification
from yacs.config import CfgNode
from config import get_default_config

class ModelsTest(absltest.TestCase):

  def test_character_level_embeddings(self):
    word_len = 5
    input_seq_len = 3
    batch_size = 10
    inputs = torch.zeros(word_len, input_seq_len, batch_size, dtype=torch.long)
    inputs_lengths = torch.full((input_seq_len, batch_size), 4)
    cfg = get_default_config()
    character_level_embedding = CharacterLevelEmbedding(cfg.CONCEPT_IDENTIFICATION.LSTM_BASED)
    outputs = character_level_embedding(inputs, inputs_lengths)
    expected_output_shape = (input_seq_len, batch_size, cfg.CONCEPT_IDENTIFICATION.LSTM_BASED.CHAR_HIDDEN_SIZE)

    self.assertEqual(outputs.shape, expected_output_shape)

  def test_character_level_embeddings_encoder(self):
    num_layers = 1
    token_vocab_size = 10
    cfg = get_default_config()
    hidden_size = cfg.CONCEPT_IDENTIFICATION.LSTM_BASED.HIDDEN_SIZE
    inputs = [
      [3, 5],
      [7, 2],
      [9, 0],
      [9, 0]
    ]

    character_inputs = [
      [[3, 5],
      [7, 2],
      [9, 0],
      [9, 0]],

     [[4, 6],
      [5, 1],
      [2, 0],
      [0, 0]],

     [[0, 9],
      [7, 0],
      [9, 0],
      [0, 0]]
    ]

    inputs = torch.tensor(inputs)
    character_inputs = torch.tensor(character_inputs)
    inputs_lengths = torch.tensor([4, 2])
    character_inputs_lengths = torch.tensor(
      [
        [2, 3],
        [3, 2],
        [3, 1],
        [1, 1]
      ])

    encoder = Encoder(token_vocab_size, cfg.CONCEPT_IDENTIFICATION.LSTM_BASED)
    outputs = encoder(inputs, inputs_lengths, character_inputs, character_inputs_lengths)
    encoder_states = outputs[0]
    last_encoder_state = outputs[1]
    h, c = last_encoder_state
    self.assertEqual(encoder_states.shape, (4, 2, hidden_size))
    self.assertEqual(h.shape, (num_layers, 2, hidden_size))
    self.assertEqual(c.shape, (num_layers, 2, hidden_size))


  def test_encoder(self):
    num_layers = 1
    token_vocab_size = 10
    cfg = get_default_config()
    hidden_size = cfg.CONCEPT_IDENTIFICATION.LSTM_BASED.HIDDEN_SIZE
    inputs = [
      [3, 5],
      [7, 2],
      [9, 0],
      [9, 0]
    ]
    inputs = torch.tensor(inputs)
    seq_lengths = torch.tensor([4,2])
    encoder = Encoder(token_vocab_size, cfg.CONCEPT_IDENTIFICATION.LSTM_BASED)
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
      cfg = get_default_config()
      hidden_size = cfg.CONCEPT_IDENTIFICATION.LSTM_BASED.HIDDEN_SIZE
      inputs = [
        [3, 5],
        [7, 2],
        [9, 0],
        [9, 0]
      ]
      inputs = torch.tensor(inputs).to(device)
      # This seems to have to stay on the cpu.
      seq_lengths = torch.tensor([4,2])
      encoder = Encoder(token_vocab_size, cfg.CONCEPT_IDENTIFICATION.LSTM_BASED).to(device)
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
    cfg = get_default_config()
    hidden_size = cfg.CONCEPT_IDENTIFICATION.LSTM_BASED.HIDDEN_SIZE
    inputs = [
      [3, 5],
      [7, 2],
      [9, 0]
    ]
    inputs = torch.tensor(inputs)
    seq_lengths = torch.tensor([3,2])
    encoder = Encoder(token_vocab_size, cfg.CONCEPT_IDENTIFICATION.LSTM_BASED, use_bilstm=True)
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
    cfg = get_default_config()
    hidden_size = cfg.CONCEPT_IDENTIFICATION.LSTM_BASED.HIDDEN_SIZE
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
    decoder_model = Decoder(output_vocab_size, cfg.CONCEPT_IDENTIFICATION.LSTM_BASED)
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
    cfg = get_default_config()
    hidden_size = cfg.CONCEPT_IDENTIFICATION.LSTM_BASED.HIDDEN_SIZE
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
    decoder_model = Decoder(output_vocab_size, cfg.CONCEPT_IDENTIFICATION.LSTM_BASED)
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
    cfg = get_default_config()
    opts = ["CONCEPT_IDENTIFICATION.LSTM_BASED.HIDDEN_SIZE", 15,
            "CONCEPT_IDENTIFICATION.LSTM_BASED.EMB_DIM", 50]
    cfg.merge_from_list(opts)
    inputs = torch.zeros((input_seq_len, batch_size)).type(torch.LongTensor)
    input_lengths = torch.tensor([2, 4, 1])
    gold_outputs = torch.zeros((output_seq_len, batch_size)).type(torch.LongTensor)
    seq2seq_model = Seq2seq(input_vocab_size, output_vocab_size, cfg.CONCEPT_IDENTIFICATION.LSTM_BASED)
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
      cfg = get_default_config()
      inputs = torch.zeros((input_seq_len, batch_size)).type(torch.LongTensor).to(device)
      input_lengths = torch.tensor([2, 4, 1])
      gold_outputs = torch.zeros(
        (output_seq_len, batch_size)).type(torch.LongTensor).to(device)
      seq2seq_model = Seq2seq(
        input_vocab_size, output_vocab_size, cfg.CONCEPT_IDENTIFICATION.LSTM_BASED, device=device).to(device)
      seq2seq_model.train()
      logits, predictions = seq2seq_model(inputs, input_lengths, gold_outputs)
      self.assertEqual(logits.shape,
        (output_seq_len, batch_size, output_vocab_size))
      self.assertEqual(predictions.shape, (output_seq_len, batch_size))


  def test_dense_mlp(self):
    no_labels = 5
    cfg = get_default_config()
    dense_mlp = DenseMLP(no_labels, cfg.RELATION_IDENTIFICATION)
    classifier_input = torch.zeros((no_labels, cfg.RELATION_IDENTIFICATION.DENSE_MLP_HIDDEN_SIZE))
    edge_repr, arc_repr = dense_mlp(classifier_input)
    self.assertEqual(edge_repr.shape, (no_labels,))

  def test_edge_scoring(self):
    batch_size = 2
    no_of_concepts = 3
    no_labels = 5
    cfg = get_default_config()
    hidden_size = cfg.RELATION_IDENTIFICATION.HIDDEN_SIZE
    concepts = torch.zeros((batch_size, no_of_concepts, 2*hidden_size))
    edge_scoring = EdgeScoring(no_labels, cfg.RELATION_IDENTIFICATION)
    scores, label_scores = edge_scoring(concepts)
    self.assertEqual(scores.shape, (batch_size, no_of_concepts, no_of_concepts))
    self.assertEqual(label_scores.shape, (batch_size, no_of_concepts, no_of_concepts, no_labels))

  def test_heads_selection_eval(self):
    batch_size = 2
    concept_vocab_size = 10
    relation_vocab_size = 50
    lemmas_vocab_size = 10
    seq_len = 3
    cfg = get_default_config()
    concepts = [
      #batch ex 1 (amr1)
      [1, 3, 3],
      #batch ex 2 (amr2)
      [4, 5, 0]
    ]
    concepts = torch.tensor(concepts)
    concepts = concepts.transpose(0,1)
    concepts_lengths = torch.tensor([3, 2])
    relation_identification = RelationIdentification(concept_vocab_size, relation_vocab_size, lemmas_vocab_size,
                                            cfg.RELATION_IDENTIFICATION)
    relation_identification.eval()
    scores, predictions, rel_scores, rel_predictions = relation_identification(concepts, concepts_lengths)
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

    predictions = RelationIdentification.get_predictions(scores, 0.5)
    self.assertTrue(torch.equal(expected_predictions, predictions))

  def test_heads_selection_train(self):
    batch_size = 2
    concept_vocab_size = 10
    relation_vocab_size = 50
    lemmas_vocab_size = 10
    seq_len = 3
    cfg = get_default_config()
    concepts = [
      #batch ex 1 (amr1)
      [1, 3, 3],
      #batch ex 2 (amr2)
      [4, 5, 0]
    ]
    concepts = torch.tensor(concepts)
    concepts = concepts.transpose(0,1)
    concepts_lengths = torch.tensor([3, 2])
    relation_identification = RelationIdentification(concept_vocab_size, relation_vocab_size, lemmas_vocab_size,
                                            cfg.RELATION_IDENTIFICATION)
    relation_identification.train()
    scores, predictions, rel_scores, rel_predictions = relation_identification(concepts, concepts_lengths)
    self.assertEqual(scores.shape, (batch_size, seq_len, seq_len))
    self.assertEqual(predictions.shape, (batch_size, seq_len, seq_len))


if __name__ == '__main__':
  absltest.main()