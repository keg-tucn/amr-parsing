from typing import Tuple, Dict
import random
import torch
import torch.nn as nn
from yacs.config import CfgNode

from data_pipeline.glove_embeddings import get_weight_matrix

# this size came from the highest code of a character that was found in the data set
# の has the code 12398
NUMBER_OF_ASCII_CHARACTERS = 12400

#TODO: move this.
BOS_IDX = 1

class CharacterLevelEmbedding(nn.Module):


  def __init__(self, config: CfgNode):
    super(CharacterLevelEmbedding, self).__init__()
    self.embedding = nn.Embedding(NUMBER_OF_ASCII_CHARACTERS, config.CHAR_EMB_DIM)
    self.gru = nn.GRU(config.CHAR_EMB_DIM, config.CHAR_HIDDEN_SIZE)
    self.char_hidden_size = config.CHAR_HIDDEN_SIZE

  def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
    """
    Compute the learned embeddings at character level
    :param inputs: (torch.Tensor): Inputs (word len, input seq len, batch size).
           input_lengths: (torch.Tensor): Inputs Length (input seq len, batch size)
    :return:
    """
    # word_len x input_seq_len x batch_size x char_emb_dim
    embedded_inputs = self.embedding(inputs)
    word_len, input_seq_len, batch_size, char_emb_size = embedded_inputs.shape
    packable_embedded_inputs = embedded_inputs.view(word_len, input_seq_len * batch_size, char_emb_size)
    packable_input_lengths = input_lengths.reshape(input_seq_len * batch_size)
    packed_embedded = nn.utils.rnn.pack_padded_sequence(
      packable_embedded_inputs, packable_input_lengths, enforce_sorted=False)

    # shape 1 x input_seq_len * batch_size x gru_hidden_size -- 1 is from the number of layers
    _, gru_states = self.gru(packed_embedded)

    # shape input_seq_len x batch_size x gru_hidden_size
    output = gru_states.view(input_seq_len, batch_size, self.char_hidden_size)

    return output

class Encoder(nn.Module):


  def __init__(self, input_vocab_size, config: CfgNode, use_bilstm=False, glove_embeddings: Dict=None):
    super(Encoder, self).__init__()
    self.embedding = nn.Embedding(input_vocab_size, config.EMB_DIM)
    self.use_glove = config.GLOVE_EMB_DIM != 0 and glove_embeddings is not None
    self.use_trainable_embeddings = config.EMB_DIM != 0
    # If config.CHAR_EMB_DIM = 0, then the CharacterEmbedding will not be used char_emb from config.
    self.use_character_level_embeddings = config.CHAR_EMB_DIM != 0

    if self.use_glove:
      self.glove = nn.Embedding(len(glove_embeddings.keys()), config.GLOVE_EMB_DIM)
      weight_matrix = get_weight_matrix(glove_embeddings, config.GLOVE_EMB_DIM)
      self.glove.load_state_dict({'weight': weight_matrix})
      self.glove.weight.requires_grad = False
    emb_dim = config.GLOVE_EMB_DIM + config.CHAR_HIDDEN_SIZE if self.use_glove & self.use_character_level_embeddings \
      else config.GLOVE_EMB_DIM if self.use_glove\
      else config.CHAR_HIDDEN_SIZE if self.use_character_level_embeddings \
      else 0
    if self.use_trainable_embeddings:
      emb_dim += config.EMB_DIM
    self.lstm = nn.LSTM(
      emb_dim, config.HIDDEN_SIZE, config.NUM_LAYERS,
      bidirectional=use_bilstm, dropout=config.DROPOUT_RATE)
    if self.use_character_level_embeddings:
      self.character_level_embeddings = CharacterLevelEmbedding(config)

  def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor,
              character_inputs: torch.Tensor=None, character_inputs_lengths: torch.Tensor=None):
    """
    Args:
        inputs (torch.Tensor): Inputs (input seq len, batch size).
        input_lengths (torch.Tensor): (batch size).

    Returns:
        [type]: [description]
    Observation:
      During packing the input lengths are taken into consideration and the lstm
      will return a sequence with seq length the maximum length in the
      input_lengths field, regardless of what the length with padding actually
      is.
    """
    embedded_inputs = self.embedding(inputs) if self.use_trainable_embeddings else None
    if self.use_glove and self.use_trainable_embeddings:
      glove_embedded_inputs = self.glove(inputs)
      embedded_inputs = torch.cat((embedded_inputs, glove_embedded_inputs), dim=-1)
    if self.use_glove and not self.use_trainable_embeddings:
      embedded_inputs = self.glove(inputs)
    if self.use_character_level_embeddings:
      # compute character level embeddings
      character_embedded_inputs = self.character_level_embeddings(character_inputs, character_inputs_lengths)
      # concatenate embedded_inputs with character lever embeddings on the las dimention
      embedded_inputs = torch.cat((embedded_inputs, character_embedded_inputs), dim=-1)
    #TODO: see if enforce_sorted would help to be True (eg efficiency).
    packed_embedded = nn.utils.rnn.pack_padded_sequence(
      embedded_inputs, input_lengths, enforce_sorted = False)
    packed_lstm_states, final_states = self.lstm(packed_embedded)
    lstm_states, _ = nn.utils.rnn.pad_packed_sequence(
      packed_lstm_states)
    return lstm_states, final_states

class AdditiveAttention(nn.Module):
  """
  Takes as input the previous decoder state and the encoder hidden states and
  returns the context vector.
  """

  def __init__(self, hidden_size: int):
    super(AdditiveAttention, self).__init__()
    self.previous_state_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    self.encoder_states_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    self.attention_scores_proj = nn.Linear(hidden_size, 1, bias=False)

  def forward(self,
              decoder_prev_state: torch.Tensor,
              encoder_states: torch.Tensor,
              mask: torch.Tensor):
    """

    Args:
        decoder_prev_state: shape (batch size, hidden size).
        encoder_states: shape (input_seq_len, batch_size, HIDDEN_SIZE).
        mask: shape (batch size, input seq len)

    Returns:
        Context vector: (batch size, HIDDEN_SIZE).
    """

    seq_len = encoder_states.shape[0]
    # [ input seq len, batch_size, lstm size] -> [batch_size, input seq len, lstm size]
    encoder_states = encoder_states.transpose(0, 1)
    # [batch_size, hidden size]
    projected_prev_state = self.previous_state_proj(decoder_prev_state)
    # [batch_size, input seq len, hidden size]
    projected_encoder_states = self.encoder_states_proj(encoder_states)
     # [batch_size, hidden size] -> [batch_size, input seq len, hidden size]
    repeated_projected_prev_state  = projected_prev_state.unsqueeze(1).repeat(1, seq_len, 1)
    # [batch_size, input seq len]
    tanh_out = torch.tanh(
      repeated_projected_prev_state + projected_encoder_states)
    attention_scores = self.attention_scores_proj(tanh_out)
    # From [batch size, seq len, 1] -> [batch size, seq len]
    attention_scores = torch.squeeze(attention_scores, dim=-1)
    attention_scores = attention_scores.masked_fill(mask == 0, -float('inf'))
    attention_scores = torch.softmax(attention_scores, dim=-1)
    # [batch_size, input seq len, hidden size] -> [batch_size, hidden size, input seq len]
    encoder_states = encoder_states.transpose(1, 2)
    context_vector = torch.bmm(encoder_states, torch.unsqueeze(attention_scores, -1))
    context_vector = context_vector.squeeze(dim=-1)
    return context_vector, attention_scores

class DecoderClassifier(nn.Module):

  def __init__(self, output_vocab_size:int, config: CfgNode, use_glove: False):
    super(DecoderClassifier, self).__init__()
    # Classifier input is a concatenation of previous embedding - EMB_DIM + GLOVE_EMB_DIM if use_glove,
    # decoder state - HIDDEN_SIZE and context vector - HIDDEN_SIZE.
    emb_dim = config.EMB_DIM + 2 * config.HIDDEN_SIZE + config.GLOVE_EMB_DIM if use_glove \
      else config.EMB_DIM + 2 * config.HIDDEN_SIZE

    self.linear_layer = nn.Linear(
      emb_dim, output_vocab_size)

  def forward(self, classifier_input):
    logits = self.linear_layer(classifier_input)
    # Use softmax for now, it can be experimented with other activation
    # functions.
    # predictions = torch.softmax(logits, dim=-1) -- softmax used in the loss fct
    return logits

class DecoderStep(nn.Module):
  """
  Module contains the logic for one decoding (time) step. It follow's Bahdanau's
  decoding strategy.
  """

  def __init__(self, output_vocab_size: int, config: CfgNode, glove_embeddings: Dict=None):
    super(DecoderStep, self).__init__()
    # Flags
    self.use_glove = config.GLOVE_EMB_DIM != 0 and glove_embeddings is not None
    self.use_trainable_embeddings = config.EMB_DIM != 0
    self.use_pointer_generator = config.USE_POINTER_GENERATOR

    self.additive_attention = AdditiveAttention(config.HIDDEN_SIZE)
    self.output_embedding = nn.Embedding(output_vocab_size, config.EMB_DIM)
    # Classifier input: previous embedding, decoder state, context vector.
    self.classifier = DecoderClassifier(output_vocab_size, config, use_glove=self.use_glove)
    self.drop_out = config.DROPOUT_RATE
    if self.use_pointer_generator:
        emb_dim_pointer_generator = config.EMB_DIM + config.HIDDEN_SIZE * 3 + config.GLOVE_EMB_DIM if self.use_glove \
          else config.EMB_DIM + config.HIDDEN_SIZE * 3
        self.p_gen_linear = nn.Linear(emb_dim_pointer_generator, 1, bias=True)
    self.output_vocab_size = output_vocab_size

    if self.use_glove:
      self.glove = nn.Embedding(len(glove_embeddings.keys()), config.GLOVE_EMB_DIM)
      weight_matrix = get_weight_matrix(glove_embeddings, config.GLOVE_EMB_DIM)
      self.glove.load_state_dict({'weight': weight_matrix})
      self.glove.weight.requires_grad = False

    emb_dim = config.EMB_DIM + config.HIDDEN_SIZE + config.GLOVE_EMB_DIM if self.use_glove \
      else config.EMB_DIM + config.HIDDEN_SIZE
    # Lstm input has size EMB_DIM + HIDDEN_SIZE (will take as input the
    # previous embedding - EMB_DIM and the context vector - HIDDEN_SIZE).
    self.lstm_cell = nn.LSTMCell(
      emb_dim, config.HIDDEN_SIZE)

  def forward(self,
              input: torch.Tensor,
              previous_state: Tuple[torch.Tensor, torch.Tensor],
              encoder_states: torch.Tensor,
              attention_mask: torch.Tensor,
              indices : torch.Tensor,
              extended_vocab_size: int,
              batch_size: int,
              device):
    """
    Args:
      input: Decoder input (previous prediction or gold label), with shape
        (batch size).
      previous_state: LSTM previous state (c,h) where each tuple element has
        shape (batch size, hidden size).
      encoder_states: Encoder outputs with shape
        (input seq len, batch size, hidden size).
      attention_mask: Attention mask (batch size, input seq len).
      indices: the index of each word in the input sequence corresponding to the
        extended vocab. This is needed in order to compute the probability for oov
        words that appeared in source sequence.
      extended_vocab_size: the size of the extended vocab.
      batch_size: the batch size.
      device: the device the model is running on.
    Returns:
       A tuple of:
         decoder state with shape (batch size, hidden size)
         predictions: probability distribution over output classes, with shape
           (batch size, number of output classes).
    """
    # Embed input.
    previous_embedding = self.output_embedding(input) if self.use_trainable_embeddings else None
    if self.use_glove and self.use_trainable_embeddings:
      glove_previous_embedding = self.glove(input)
      previous_embedding = torch.cat((previous_embedding, glove_previous_embedding), dim=-1)
    if self.use_glove and not self.use_trainable_embeddings:
      previous_embedding = self.glove(input)
    # Compute context vector.
    h = previous_state[0]
    context_vector, attention = self.additive_attention(h, encoder_states, attention_mask)
    # Compute lstm input.
    lstm_input =  torch.cat((previous_embedding, context_vector), dim=-1)
    if self.training:
      # Add dropout to lstm input.
      dropout = nn.Dropout(p=self.drop_out)
      lstm_input = dropout(lstm_input)
    # Compute new decoder state.
    decoder_state = self.lstm_cell(lstm_input, previous_state)
    # Compute classifier input (alternatively a pre-output can be computed
    # before with a NN layer).

    classifier_input = torch.cat(
      (previous_embedding, decoder_state[0], context_vector), dim=-1)
    predictions = self.classifier(classifier_input)

    # generation probability
    if self.use_pointer_generator:
      p_gen_input = torch.cat((context_vector, decoder_state[0], lstm_input), -1)
      p_gen = self.p_gen_linear(p_gen_input)
      p_gen = torch.sigmoid(p_gen).to(device=device)

      # create the extended vocabulary probabilities
      extended_vocab_probabilities = torch.zeros((batch_size, extended_vocab_size)).to(device=device)
      output_vocab_size = predictions.shape[1]
      extended_vocab_probabilities[:, :output_vocab_size] = p_gen * predictions
      extended_vocab_probabilities = extended_vocab_probabilities.scatter_add(1, indices, (1 - p_gen) * attention)

      return decoder_state, extended_vocab_probabilities
    else:
      return decoder_state, predictions


class Decoder(nn.Module):

  def __init__(self,
                output_vocab_size: int,
                config: CfgNode,
                device: str = "cpu",
                glove_embeddings: Dict=None):
    super(Decoder, self).__init__()
    self.device = device
    self.output_vocab_size = output_vocab_size
    self.decoder_step = DecoderStep(output_vocab_size, config, glove_embeddings)
    self.initial_state_layer_h = nn.Linear(
      config.HIDDEN_SIZE, config.HIDDEN_SIZE)
    self.initial_state_layer_c = nn.Linear(
      config.HIDDEN_SIZE, config.HIDDEN_SIZE)
    self.config = config

  def compute_initial_decoder_state(self,
                                    last_encoder_states: Tuple[torch.Tensor, torch.Tensor]):
    """Create the initial decoder state by passing the last encoder state though
    a linear layer. Since it's an LSTM, we'll be passing both the c and h vectors.

    Args:
      last_encoder_states: Encoder c and h for the last element in the sequence.
        Both c and h have the shape (num_enc_layers, batch size, hidden size).
    """
    # Only use the value from the last layer, since we simply use an LSTM cell
    # (only layer) in the decoder.
    enc_h = last_encoder_states[0][-1]
    enc_c = last_encoder_states[1][-1]
    initial_h = self.initial_state_layer_h(enc_h)
    initial_c = self.initial_state_layer_h(enc_c)
    return (initial_h, initial_c)


  def forward(self,
              encoder_output: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
              attention_mask: torch.Tensor,
              extended_vocab_size: int,
              indices: torch.Tensor,
              teacher_forcing_ratio: float = 0.0,
              decoder_inputs: torch.Tensor = None,
              max_out_length: int = None):
    """Bahdanau style decoder.

    Args:
      encoder_output: Encoder output consisting of the encoder states, a tensor
        with shape (input seq len, batch size, hidden size) and the last states
        of the encoder (h, c), where both tensors have shapes
        (num_enc_layers, batch size, hidden size).
      attention_mask: Attention mask (tensor of bools), shape
        (batch size, input seq len).
      indices: the index of each word in the input sequence corresponding to the
        extended vocab. This is needed in order to compute the probability for oov
        words that appeared in source sequence.
      decoder_inputs (torch.Tensor): Decoder input for each step, with shape
        (output seq len, batch size). These should be sent on the train flow,
        and consist of the gold output sequence. On the inference flow, they
        should not be used (None by default).
      max_out_length: Maximum output length (sent on the inference flow), for
        the sequences to have a fixed size in the batch. None by default.
    """
    if self.training:
      output_seq_len = decoder_inputs.shape[0]
    else:
      output_seq_len = max_out_length
    encoder_states = encoder_output[0]
    batch_size = encoder_states.shape[1]
    # Create the initial decoder state by passing the last encoder state
    # through a linear layer.
    decoder_state = self.compute_initial_decoder_state(encoder_output[1])
    # Create a batch of initial tokens.
    previous_token = torch.full((batch_size,), BOS_IDX).to(device=self.device)

    if self.config.USE_POINTER_GENERATOR:
      # When using pointer generator we also need the words in the input sequence
      # therefore we use the extended vocab
      all_logits_vocab_size = extended_vocab_size
    else:
      all_logits_vocab_size = self.output_vocab_size

    all_logits = torch.zeros(
      (output_seq_len, batch_size, all_logits_vocab_size)).to(device=self.device)
    all_predictions = torch.zeros((output_seq_len, batch_size))
    for i in range(output_seq_len):
      decoder_state, logits = self.decoder_step(
        previous_token, decoder_state, encoder_states, attention_mask, indices, extended_vocab_size, batch_size, self.device)
      # Get predicted token.
      step_predictions = torch.softmax(logits, dim=-1)
      predicted_token = torch.argmax(step_predictions, dim=-1)
      all_predictions[i] = predicted_token
      # Get the next decoder input, either from gold (if train & teacher forcing,
      # or use the predicted token).
      teacher_forcing = random.random() < teacher_forcing_ratio
      if self.training and teacher_forcing:
        previous_token = decoder_inputs[i]
      else:
        previous_token = predicted_token
      all_logits[i] = logits

    return all_logits, all_predictions

class Seq2seq(nn.Module):

  def __init__(self,
               input_vocab_size: int,
               output_vocab_size: int,
               # config CONCEPT_IDENTIFICATION.LSTM_BASED
               config: CfgNode,
               glove_embeddings: Dict = None,
               device="cpu"):
    super(Seq2seq, self).__init__()
    self.encoder = Encoder(input_vocab_size, config, glove_embeddings=glove_embeddings)
    self.decoder = Decoder(output_vocab_size, config, device=device, glove_embeddings=glove_embeddings)
    self.device = device
    if config.USE_POINTER_GENERATOR:
      self.encoder.embedding.weight = self.decoder.decoder_step.output_embedding.weight
    self.use_character_level_embeddings = config.CHAR_EMB_DIM != 0

  def create_mask(self, input_lengths: torch.Tensor, mask_seq_len: int):
    arr_range = torch.arange(mask_seq_len)
    mask = arr_range.unsqueeze(dim=0) < input_lengths.unsqueeze(dim=1)
    mask = mask.to(self.device)
    return mask

  def forward(self,
              input_sequence: torch.Tensor,
              input_lengths: torch.Tensor,
              extended_vocab_size: int= None,
              indices: torch.Tensor= None,
              teacher_forcing_ratio: float = 0.0,
              gold_output_sequence: torch.Tensor = None,
              max_out_length: int = None,
              character_inputs: torch.Tensor=None,
              character_inputs_lengths: torch.Tensor=None):
    """Forward seq2seq.

    Args:
        input_sequence (torch.Tensor): Input sequence of shape
          (input seq len, batch size).
        input_lengths: Lengths of the sequences in the batch (batch size).
        extended_vocab_size: the size of the extended vocab.
        indices: indices in the extended vocab.
        gold_output_sequence: Output sequence (output seq len, batch size).
        max_out_length: Maximum output length (sent on the inference flow), for
        the sequences to have a fixed size in the batch. None by default.
    Returns:
      predictions of shape (out seq len, batch size, output no of classes).
    """
    if self.use_character_level_embeddings:
      encoder_output = self.encoder(input_sequence, input_lengths, character_inputs, character_inputs_lengths)
    else:
      encoder_output = self.encoder(input_sequence, input_lengths)
    input_seq_len = input_sequence.shape[0]
    attention_mask = self.create_mask(input_lengths, input_seq_len)
    indices = indices.to(device=self.device) if indices is not None else None
    if self.training:
      logits, predictions = self.decoder(
        encoder_output, attention_mask, extended_vocab_size, indices,
                teacher_forcing_ratio, gold_output_sequence)
    else:
      logits, predictions = self.decoder(
        encoder_output, attention_mask, extended_vocab_size, indices, max_out_length = max_out_length)
    return logits, predictions

class DenseMLP(nn.Module):
  """
  MLP from Dependency parsing as Head Selection.
  """

  def __init__(self,
               node_repr_size:int,
               config: CfgNode):
    super(DenseMLP, self).__init__()
    self.Ua = nn.Linear(node_repr_size, config.DENSE_MLP_HIDDEN_SIZE, bias=False)
    self.Wa = nn.Linear(node_repr_size, config.DENSE_MLP_HIDDEN_SIZE, bias=False)
    self.va = nn.Linear(config.DENSE_MLP_HIDDEN_SIZE, 1, bias=False)


  def forward(self, parent: torch.Tensor, child: torch.Tensor):
    """
    Args:
        parent: Parent node representation (batch size, node repr size).
        child: Child node representation (batch size, node repr size).
    Returns
      edge_score: (batch size).
    """
    edge_score = self.va(torch.tanh(self.Ua(parent) + self.Wa(child)))
    # Return score of (batch size), not (batch size, 1).
    return edge_score[:,0]

class EdgeScoring(nn.Module):

  def __init__(self, config: CfgNode):
    super(EdgeScoring, self).__init__()
    self.dense_mlp = DenseMLP(2 * config.HIDDEN_SIZE, config)

  def forward(self, concepts: torch.tensor):
    """
    Args:
        concepts: Sequence of ordered concepts, shape
          (batch size, seq len, concept size).
        The first concept is the fake root concept.
    Returns:
      scores between each pair of concepts, shape
      (batch size, no of concepts, no of concepts).
    """
    batch_size, no_of_concepts, _ = concepts.shape
    scores = torch.zeros((batch_size, no_of_concepts, no_of_concepts))
    no_concepts = concepts.shape[1]
    #TODO: think if this can be done without loops.
    for i in range(no_concepts):
      for j in range(no_concepts):
        parent = concepts[:,i]
        child = concepts[:,j]
        score = self.dense_mlp(parent, child)
        scores[:,i,j] = score
    return scores

class HeadsSelection(nn.Module):
  """
  Module for training heads selection separately.
  The input is a list of numericalized concepts & the output is a matrix of
  edge scores.
  """

  def __init__(self, concept_vocab_size,
               # config HEAD_SELECTION
               config: CfgNode,
               glove_embeddings: Dict=None):
    super(HeadsSelection, self).__init__()
    self.encoder = Encoder(concept_vocab_size, config,
                           use_bilstm=True, glove_embeddings=glove_embeddings)
    self.edge_scoring = EdgeScoring(config)
    self.config = config

  @staticmethod
  def get_predictions(score_mat: torch.tensor, threshold: float = 0.5):
    """
    Args:
      score_mat (torch.tensor): obtained scores
      threshold: value to separate edge from non edge

    Returns:
      Prediction if is edge or not based on score matrix (binary matrix)
    """
    sigmoid_mat = torch.sigmoid(score_mat)
    prediction_mat = torch.where(sigmoid_mat >= threshold, 1, 0)
    return prediction_mat

  def forward(self, concepts: torch.tensor, concepts_lengths: torch.tensor,
              character_inputs: torch.Tensor=None,
              character_inputs_lengths: torch.Tensor=None):
    """
    Args:
      concepts (torch.tensor): Concepts (seq len, batch size).
      concepts_lengths (torch.tensor): Concept sequences lengths (batch size).

    Returns:
      Edge scores (batch size, seq len, seq len).
    """
    encoded_concepts, _ = self.encoder(concepts, concepts_lengths, character_inputs, character_inputs_lengths)
    encoded_concepts = encoded_concepts.transpose(0,1)
    scores = self.edge_scoring(encoded_concepts)
    predictions = self.get_predictions(scores, self.config.EDGE_THRESHOLD)
    return scores, predictions
