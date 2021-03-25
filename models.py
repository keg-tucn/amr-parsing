from typing import Tuple
import random
import torch
import torch.nn as nn
from yacs.config import CfgNode

DENSE_MLP_HIDDEN_SIZE = 30
SAMPLING_RATIO = 2

EDGE_THRESHOLD = 0.5

#TODO: move this.
BOS_IDX = 1

class Encoder(nn.Module):


  def __init__(self, input_vocab_size, config: CfgNode, use_bilstm=False):
    super(Encoder, self).__init__()
    self.embedding = nn.Embedding(input_vocab_size, config.EMB_DIM)
    self.lstm = nn.LSTM(
      config.EMB_DIM, config.HIDDEN_SIZE, config.NUM_LAYERS,
      bidirectional=use_bilstm)

  def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
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
    embedded_inputs = self.embedding(inputs)
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
    return context_vector

class DecoderClassifier(nn.Module):

  def __init__(self, output_vocab_size:int, config: CfgNode):
    super(DecoderClassifier, self).__init__()
    # Classifier input is a concatenation of previous embedding - EMB_DIM,
    # decoder state - HIDDEN_SIZE and context vector - HIDDEN_SIZE.
    self.linear_layer = nn.Linear(
      config.EMB_DIM + 2 * config.HIDDEN_SIZE, output_vocab_size)

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

  def __init__(self, output_vocab_size: int, config: CfgNode):
    super(DecoderStep, self).__init__()
    # Lstm input has size EMB_DIM + HIDDEN_SIZE (will take as input the
    # previous embedding - EMB_DIM and the context vector - HIDDEN_SIZE).
    self.lstm_cell = nn.LSTMCell(
      config.EMB_DIM + config.HIDDEN_SIZE, config.HIDDEN_SIZE)
    self.additive_attention = AdditiveAttention(config.HIDDEN_SIZE)
    self.output_embedding = nn.Embedding(output_vocab_size, config.EMB_DIM)
    # Classifier input: previous embedding, decoder state, context vector.
    self.classifier = DecoderClassifier(output_vocab_size, config)

  def forward(self,
              input: torch.Tensor,
              previous_state: Tuple[torch.Tensor, torch.Tensor],
              encoder_states: torch.Tensor,
              attention_mask: torch.Tensor):
    """
    Args:
      input: Decoder input (previous prediction or gold label), with shape
        (batch size).
      previous_state: LSTM previous state (c,h) where each tuple element has
        shape (batch size, hidden size).
      encoder_states: Encoder outputs with shape
        (input seq len, batch size, hidden size).
      attention_mask: Attention mask (batch size, input seq len).
    Returns:
       A tuple of:
         decoder state with shape (batch size, hidden size)
         predictions: probability distribution over output classes, with shape
           (batch size, number of output classes).
    """
    # Embed input.
    previous_embedding = self.output_embedding(input)
    # Compute context vector.
    h = previous_state[0]
    context_vector = self.additive_attention(h, encoder_states, attention_mask)
    # Compute lstm input.
    lstm_input =  torch.cat((previous_embedding, context_vector), dim=-1)
    # Compute new decoder state.
    decoder_state = self.lstm_cell(lstm_input, previous_state)
    # Compute classifier input (alternatively a pre-output can be computed
    # before with a NN layer).
    classifier_input = torch.cat(
      (previous_embedding, decoder_state[0], context_vector), dim=-1)
    predictions = self.classifier(classifier_input)
    return decoder_state, predictions


class Decoder(nn.Module):

  def __init__(self,
                output_vocab_size: int,
                config: CfgNode,
                teacher_forcing_ratio: float = 0.5,
                device: str = "cpu"):
    super(Decoder, self).__init__()
    self.device = device
    self.output_vocab_size = output_vocab_size
    self.teacher_forcing_ratio = teacher_forcing_ratio
    self.decoder_step = DecoderStep(output_vocab_size, config)
    self.initial_state_layer_h = nn.Linear(
      config.HIDDEN_SIZE, config.HIDDEN_SIZE)
    self.initial_state_layer_c = nn.Linear(
      config.HIDDEN_SIZE, config.HIDDEN_SIZE)

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
      decoder_inputs (torch.Tensor): Decoder input for each step, with shape
        (output seq len, batch size). These should be sent on the train flow,
        and consist of the gold output sequence. On the inference flow, they
        should not be used (None by default).
      max_output_length: Maximum output length (sent on the inference flow), for
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


    all_logits = torch.zeros(
      (output_seq_len, batch_size, self.output_vocab_size)).to(device=self.device)
    all_predictions = torch.zeros((output_seq_len, batch_size))
    for i in range(output_seq_len):
      decoder_state, logits = self.decoder_step(
        previous_token, decoder_state, encoder_states, attention_mask)
      # Get predicted token.
      step_predictions = torch.softmax(logits, dim=-1)
      predicted_token = torch.argmax(step_predictions, dim=-1)
      all_predictions[i] = predicted_token
      # Get the next decoder input, either from gold (if train & teacher forcing,
      # or use the predicted token).
      teacher_forcing = random.random() < self.teacher_forcing_ratio
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
               device="cpu"):
    super(Seq2seq, self).__init__()
    hidden_size = config.HIDDEN_SIZE
    self.encoder = Encoder(input_vocab_size, config)
    self.decoder = Decoder(output_vocab_size, config, device=device)
    self.device = device

  def create_mask(self, input_lengths: torch.Tensor, mask_seq_len: int):
    arr_range = torch.arange(mask_seq_len)
    mask = arr_range.unsqueeze(dim=0) < input_lengths.unsqueeze(dim=1)
    mask = mask.to(self.device)
    return mask

  def forward(self,
              input_sequence: torch.Tensor,
              input_lengths: torch.Tensor,
              gold_output_sequence: torch.Tensor = None,
              max_out_length: int = None):
    """Forward seq2seq.

    Args:
        input_sequence (torch.Tensor): Input sequence of shape
          (input seq len, batch size).
        input_lengths: Lengths of the sequences in the batch (batch size).
        gold_output_sequence: Output sequence (output seq len, batch size).
    Returns:
      predictions of shape (out seq len, batch size, output no of classes).
    """
    encoder_output = self.encoder(input_sequence, input_lengths)
    input_seq_len = input_sequence.shape[0]
    attention_mask = self.create_mask(input_lengths, input_seq_len)
    if self.training:
      logits, predictions = self.decoder(
        encoder_output, attention_mask, gold_output_sequence)
    else:
      logits, predictions = self.decoder(
        encoder_output, attention_mask, max_out_length = max_out_length)
    return logits, predictions

class DenseMLP(nn.Module):
  """
  MLP from Dependency parsing as Head Selection.
  """

  def __init__(self,
               node_repr_size:int):
    super(DenseMLP, self).__init__()
    self.Ua = nn.Linear(node_repr_size, DENSE_MLP_HIDDEN_SIZE, bias=False)
    self.Wa = nn.Linear(node_repr_size, DENSE_MLP_HIDDEN_SIZE, bias=False)
    self.va = nn.Linear(DENSE_MLP_HIDDEN_SIZE, 1, bias=False)


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

  def __init__(self, hidden_size: int):
    super(EdgeScoring, self).__init__()
    self.dense_mlp = DenseMLP(2 * hidden_size)

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

  def __init__(self, concept_vocab_size, hidden_size: int):
    super(HeadsSelection, self).__init__()
    self.encoder = Encoder(concept_vocab_size, hidden_size, use_bilstm=True)
    self.edge_scoring = EdgeScoring(hidden_size)

  @staticmethod
  def create_padding_mask(concepts_lengths: torch.tensor, seq_len: int):
    """
    Args:
      concepts_lengths: Batch of concept sequence lengths (batch size).
      seq_len: Maximum seq length.

    Returns:
      Batch of padding masks, with shape (batch size, max len, max len), where
      each padding mask has False where the padding is and True otherwise, that
      is the submatrix of size (concept len, concept len) is all True.
    """
    arr_range = torch.arange(seq_len)
    # Create sequence mask.
    mask_1d = arr_range.unsqueeze(dim=0) < concepts_lengths.unsqueeze(dim=1)
    # Create 2d mask for each element in the batch by multiplying a repeated
    # vector with its transpose.
    x = mask_1d.unsqueeze(1).repeat(1, seq_len, 1)
    y = x.transpose(1, 2)
    mask = x * y
    return mask

  @staticmethod
  def create_sampling_mask(gold_adj_mat: torch.tensor,
                           sampling_ratio: int = SAMPLING_RATIO):
    """Create sampling mask (for balancing negative and positive classes).
    Args:
      gold_adj_mat: Gold adjacency matrix (batch size, seq len, seq len).
      sampling_ratio: How many negative edges should be sampled for a positive
        edge. We sample this at concept level (if a concept has n heads, we
        sample sampling_ration * n non-heads).
    Returns:
      Mask of boolean values with shape (batch size, seq len, seq len).
    """
    #TODO: implement this!!!!
    # Dummy implementation for now.
    mask = torch.full(gold_adj_mat.shape, True)
    return mask

  @staticmethod
  def create_fake_root_mask(batch_size, seq_len, root_idx = 0):
    """
    Args:
      batch_size Batch size.
      seq_len: Seq len.
      root_idx: Index of the root in the scores matrix.

    Returns:
      Boolean mask ensuring the root cannot be a child of another concept,
      which means the root column should be False.
    """
    mask = torch.full((batch_size, seq_len, seq_len), True)
    mask[:,:,root_idx] = False
    return mask

  @staticmethod
  def create_mask(seq_len: int,
                  concepts_lengths: torch.tensor,
                  training: bool,
                  gold_adj_mat = None):
    """
    Creates a mask for masking scores (the scores will be masked with -inf to
    obtain 0 when passing through the sigmoid). This mask will be a mask of
    boolean values:
      False: masking out
      True: masking in
    Need to mask several things:
      padding -> the sequences of concepts are padded.
      sampling -> we don't want to use all non-existing arcs, we will "sample"
        from them by masking out the ones we don't want to use.
      fake root -> the fake root should not have any parent.

    Args:
      seq_len: max length of concepts sequence.
      concepts_lengths: Batch of concept sequence lengths (batch size).
      training: bool for training flow.
      gold_adj_mat: Gold adj mat (matrix of relations), only sent on training
        of shape (batch size, seq len, seq len).

    Returns mask of shape (batch size, seq len, seq len).
    """
    mat_sent_on_training = training and gold_adj_mat is not None
    mat_not_sent_on_inference = not training and gold_adj_mat is None
    assert mat_sent_on_training or mat_not_sent_on_inference
    batch_size = concepts_lengths.shape[0]
    mask = HeadsSelection.create_padding_mask(concepts_lengths, seq_len)
    if training:
      sampling_mask = HeadsSelection.create_sampling_mask(gold_adj_mat)
      mask = mask * sampling_mask
    fake_root_mask = HeadsSelection.create_fake_root_mask(
      batch_size, seq_len)
    mask = mask * fake_root_mask
    return mask

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

  def forward(self,
              concepts: torch.tensor,
              concepts_lengths: torch.tensor,
              gold_adj_mat = None):
    """
    Args:
      concepts (torch.tensor): Concepts (seq len, batch size).
      concepts_lengths (torch.tensor): Concept sequences lengths (batch size).
      gold_adj_mat: Gold adj mat (matrix of relations), only sent on training
        of shape (batch size, seq len, seq len).

    Returns:
      Edge scores (batch size, seq len, seq len).
    """
    seq_len = concepts.shape[0]
    encoded_concepts, _ = self.encoder(concepts, concepts_lengths)
    encoded_concepts = encoded_concepts.transpose(0,1)
    scores = self.edge_scoring(encoded_concepts)
    mask = self.create_mask(seq_len, concepts_lengths, self.training, gold_adj_mat)
    # TODO: would it make sense to instead weight the loss?
    # scores = scores.masked_fill(mask == 0, -float('inf'))
    predictions = self.get_predictions(scores, EDGE_THRESHOLD)
    return scores, predictions
