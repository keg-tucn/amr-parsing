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
EOS_IDX = 1

class CharacterLevelEmbedding(nn.Module):


  def __init__(self, config: CfgNode):
    super(CharacterLevelEmbedding, self).__init__()
    self.embedding = nn.Embedding(NUMBER_OF_ASCII_CHARACTERS, config.CHAR_EMB_DIM)
    #TODO: biGRU
    self.gru = nn.GRU(config.CHAR_EMB_DIM, config.CHAR_HIDDEN_SIZE)
    self.char_hidden_size = config.CHAR_HIDDEN_SIZE

  def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
    """
    Compute the learned embeddings at character level
       Args:
           inputs: (torch.Tensor): Inputs (word len, input seq len, batch size).
           input_lengths: (torch.Tensor): Inputs Length (input seq len, batch size)
       Returns:
           output: the embedded input sequence
    """
    # word_len x input_seq_len x batch_size x char_emb_dim
    embedded_inputs = self.embedding(inputs)
    word_len, input_seq_len, batch_size, char_emb_size = embedded_inputs.shape
    # view will return a tensor with the same data but with a different shape
    # the reshape and view are used in order to feed the embedded_inputs and inputs_length to the pack_padded_sequence
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
    self.drop_out = config.ENCODER_DROPOUT_RATE
    if self.use_glove:
      self.glove = nn.Embedding(len(glove_embeddings.keys()), config.GLOVE_EMB_DIM)
      weight_matrix = get_weight_matrix(glove_embeddings, config.GLOVE_EMB_DIM)
      self.glove.load_state_dict({'weight': weight_matrix})
      self.glove.weight.requires_grad = False
    emb_dim = 0
    if self.use_glove & self.use_character_level_embeddings:
      emb_dim += config.GLOVE_EMB_DIM + config.CHAR_HIDDEN_SIZE
    elif self.use_glove:
      emb_dim += config.GLOVE_EMB_DIM
    elif self.use_character_level_embeddings:
      emb_dim += config.CHAR_HIDDEN_SIZE

    if self.use_trainable_embeddings:
      emb_dim += config.EMB_DIM
    self.lstm = nn.LSTM(
      emb_dim, config.HIDDEN_SIZE, config.NUM_LAYERS,
      bidirectional=use_bilstm, dropout=config.ENCODER_DROPOUT_RATE)
    if self.use_character_level_embeddings:
      self.character_level_embeddings = CharacterLevelEmbedding(config)

  def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor,
              character_inputs: torch.Tensor=None, character_inputs_lengths: torch.Tensor=None):
    """
    Args:
        inputs (torch.Tensor): Inputs (input seq len, batch size).
        input_lengths (torch.Tensor): (batch size).
        character_inputs: (torch.Tensor): Inputs split per charactes
         (word len, input seq len, batch size).
        character_inputs_lengths: (torch.Tensor): Characters Inputs Length
        (input seq len, batch size)
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
      # concatenate embedded_inputs with character level embeddings on the last dimention
      embedded_inputs = torch.cat((embedded_inputs, character_embedded_inputs), dim=-1)
    if self.training:
      # Add dropout to lstm input.
      dropout = nn.Dropout(p=self.drop_out)
      embedded_inputs = dropout(embedded_inputs)
    #TODO: see if enforce_sorted would help to be True (eg efficiency).
    packed_embedded = nn.utils.rnn.pack_padded_sequence(
      embedded_inputs, input_lengths, enforce_sorted = False)
    packed_lstm_states, final_states = self.lstm(packed_embedded)
    lstm_states, _ = nn.utils.rnn.pad_packed_sequence(
      packed_lstm_states)
    if self.training:
      # Add dropout to lstm output.
      dropout = nn.Dropout(p=self.drop_out)
      lstm_states = dropout(lstm_states)
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
    self.drop_out = config.DECODER_DROPOUT_RATE
    self.use_pointer_generator = config.USE_POINTER_GENERATOR
    if self.use_pointer_generator:
        self.p_gen_linear = nn.Linear(config.HIDDEN_SIZE * 3 + config.EMB_DIM, 1, bias=True)
    self.output_vocab_size = output_vocab_size
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
    previous_embedding = self.output_embedding(input)
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
    #TODO: rename to logits
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
                device: str = "cpu"):
    super(Decoder, self).__init__()
    self.device = device
    self.output_vocab_size = output_vocab_size
    self.decoder_step = DecoderStep(output_vocab_size, config)
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
    #TODO: investigate if BOS is actually in the vocabulary
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


    #(batch size, hidden size, beam_width)
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




class BeamDecoder(nn.Module):

  def __init__(self,
                output_vocab_size: int,
                config: CfgNode,
                device: str = "cpu"):
    super(BeamDecoder, self).__init__()
    self.device = device
    self.output_vocab_size = output_vocab_size
    self.decoder_step = DecoderStep(output_vocab_size, config)
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
    #assert self.training == False
    
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

    #the beam dimension
    beam_width = 3
    hidden_size = encoder_output[0].shape[2]

    #create the 1D tensor that tells us if we reached eos for each sentence
    eos_reach_check_prev = torch.full((batch_size * beam_width,), False).to(device=self.device)

    previous_token = torch.full((batch_size,), BOS_IDX).to(device=self.device)


    #create the 1D tensor that keeps the predicted index
    previous_token_index = torch.zeros((batch_size * beam_width,)).to(device=self.device)

    #create the 1D tensor that keeps the value of the probability for the prediction of the index
    previous_token_values = torch.zeros((batch_size * beam_width,)).to(device=self.device)

    if self.config.USE_POINTER_GENERATOR:
      # When using pointer generator we also need the words in the input sequence
      # therefore we use the extended vocab
      all_logits_vocab_size = extended_vocab_size
    else:
      all_logits_vocab_size = self.output_vocab_size

    #keeps the indices picked by beam-seach (the first beam-with indices) 
    all_predictions_beam = torch.zeros((output_seq_len, batch_size * beam_width), dtype = torch.int64).to(device=self.device)

    #keeps the final result
    all_predictions = torch.zeros((output_seq_len, batch_size), dtype = torch.int64).to(device=self.device)

    #the first decoding step to generate the first predictions
    #the first step is different because the input size is independent of beam-width
    decoder_state, logits = self.decoder_step(
        previous_token, decoder_state, encoder_states, attention_mask, indices, extended_vocab_size, batch_size, self.device)

    #compute the probabilities
    scores = logits.log_softmax(dim=-1)

    #select top beam_width values and positions from the probability distribution
    previous_token_values, previous_token_index = scores.topk(beam_width , dim=-1)

    #change the shape of the predictions to be compatible with the decoder input
    previous_token_index = previous_token_index.view((batch_size * beam_width))
    previous_token_values = previous_token_values.view((batch_size * beam_width))

    #check if eos is reached for each sentence
    #we could skip this assuming all sentences have at least one word
    eos_reach_check_prev = (previous_token_index == EOS_IDX)
    
    #save the indecies of the first prediction
    all_predictions_beam[0] = previous_token_index.view(batch_size * beam_width)
    
    #repeat each decoder_state beam_with times
    #([batch_size, decoder_hidden]) -> ([batch_size * beam_width, decoder_hidden])
    decoder_state = (
                      decoder_state[0].repeat_interleave(beam_width, dim = 0), 
                      decoder_state[1].repeat_interleave(beam_width, dim = 0)
    )

    #repeat each encoder_state beam_with times
    #([out_seq_len, batch_size, encoder_hidden]) -> ([out_seq_len, batch_size * beam_width, encoder_hidden])
    encoder_states = encoder_states.repeat_interleave(beam_width, dim = 1)
    
    #repeat the indices and the attention_mask, same as above
    indices = indices.repeat_interleave(beam_width, dim = 0)
    attention_mask = attention_mask.repeat_interleave(beam_width, dim = 0)


    for i in range(1, output_seq_len):
      #generate the prediction and the new decoding state
      decoder_state, logits = self.decoder_step(
        previous_token_index, decoder_state, encoder_states, attention_mask, indices, extended_vocab_size, batch_size * beam_width, self.device)
      # Get predicted token.
      
      #compute the probabilities for the current prediction
      scores = logits.log_softmax(dim=-1)
      
      #repeat each probability from the previous step beam_width times
      previous_values_repeated = previous_token_values.unsqueeze(dim = -1).repeat_interleave(extended_vocab_size, dim = -1)

  
      # make sure if reached eos we don't add anything to the score
      scores = torch.where(
                            eos_reach_check_prev.unsqueeze(-1),
                            torch.zeros(scores.shape), 
                            scores 
      )


      #compute the score from current probabilities and the previous score
      scores += previous_values_repeated


      #copy the score before the normalization step
      scores_without_normalization = scores.clone()

      #normalize the score

      normalization_constant = (i + 1) ** 0.7

      normalization_matrix = torch.where(
                                          eos_reach_check_prev.unsqueeze(-1),
                                          torch.ones(scores.shape), 
                                          torch.full(scores.shape, normalization_constant)                                      
      )

      scores /= normalization_matrix
      

      #select the first beam_width probabilities and their indices from the normalized scores
      #the view function puts all the probabilities that are related to a sentence in the same tensor so we can pick the first beam_width indices and values
      predicted_token_values, predicted_token_index = scores.view(batch_size, beam_width * extended_vocab_size).topk(beam_width, dim = -1)
    
       

      # modify the shape of the unnormalized score to corespound with the shape of the normalized score 
      # ([batch_size * beam_width, extended_vocab_size]) -> ([batch_size, beam_width * extended_vocab_size])
      scores_without_normalization = scores_without_normalization.view(batch_size, beam_width * extended_vocab_size)



      #aleg doar acele probabilitati care corespund celor mai mari topk scoruri
      #predicted_token_index are size ul (batch_size, beam_width)
      #astfel in ultima dimensiune din tensorul final raman doar beam_width probabilitati nenormalizate

      #select the probabilities that corespound to the topk scores for each prediction
      #predicted_token_index has size (batch_size, beam_width)
      #([batch_size, beam_width * extended_vocab_size]) -> ([batch_size, beam_width])
      scores_without_normalization = scores_without_normalization.gather(dim = -1, index = predicted_token_index)

      #modify the dimensions ([batch_size, beam_width]) -> ([batch_size * beam_width])
      scores_without_normalization = scores_without_normalization.view(batch_size * beam_width)

      predicted_token_values = predicted_token_values.view(batch_size * beam_width)

      

      #select the value that will be added to the new scores the normalized one or the unnormalized one
      previous_token_values = torch.where(
                                            eos_reach_check_prev,
                                            predicted_token_values,
                                            scores_without_normalization                                
      )




      #save the predicted indices
      all_predictions_beam[i] = predicted_token_index.view(batch_size * beam_width)


      #get the positions for every index that was used to make a new prediction for a sentence
      positions_in_previous = previous_token_index.div(extended_vocab_size, rounding_mode='floor')

      #normalize the predictions to be in the (0, extended_vocab_size) range, before was (0, beam_width * extended_vocab_size)
      previous_token_index %= extended_vocab_size

      #get the positions in the batch-wise index tensor
      decoder_index = positions_in_previous + torch.arange(batch_size * beam_width).div(beam_width, rounding_mode='floor') * beam_width



      #select the decoder state for the new decoding step

      decoder_state= (
                        decoder_state[0].index_select(dim = 0, index = decoder_index),
                        decoder_state[1].index_select(dim = 0, index = decoder_index)
      )

      #select the previous state of the eos reached
      eos_reach_check_prev = eos_reach_check_prev.index_select(dim = 0, index = decoder_index)


      #calculate eos_check_prev = eos_check_present or eos_check_past
      #update eos_reach_check_prev
      eos_reach_check_prev = eos_reach_check_prev.logical_or(previous_token_index == EOS_IDX)
    
    
    

    #get the last predictions
    #size ([batch_size * beam_width])
    final_predictions = predicted_token_values

    #separate the values for each sentence into one dimension, get argmax on it
    #([batch_size * beam_width]) -> ([batch_size, beam_width]) -> ([batch_size])

    max_predictions_index = final_predictions.view(batch_size, beam_width).argmax(dim = -1)

    
    
    #get the vector for adding the positions in batch-wize tensor of previous predictions
    possition_for_adding = torch.arange(batch_size) * beam_width

    #pornind de la predictia finala, argmax
    #selectez mergand spre pozitia anterioara in vectorul de predictii beam valorile care au generat predictia curenta

    #starting from the final prediction
    #go backwards in the predictions vector(that keeps a beam prediction for each sentence)
    #keep only the predictions that led us to the final prediction
    for i in range(output_seq_len - 1, -1, -1): # from output_seq_len - 1 to 0
      
      max_predictions_index += possition_for_adding

      index_values_in_beam_vocab = all_predictions_beam[i].index_select(dim = -1, index = max_predictions_index)

      all_predictions[i] = index_values_in_beam_vocab % extended_vocab_size

      max_predictions_index = index_values_in_beam_vocab.div(extended_vocab_size, rounding_mode='floor')

    return None, all_predictions




class Seq2seq(nn.Module):

  def __init__(self,
               input_vocab_size: int,
               output_vocab_size: int,
               beam : bool,# pt true o sa faca beam deconding
               # config CONCEPT_IDENTIFICATION.LSTM_BASED
               config: CfgNode,
               glove_embeddings: Dict = None,
               device="cpu"):
    super(Seq2seq, self).__init__()
    self.encoder = Encoder(input_vocab_size, config, glove_embeddings=glove_embeddings)


    if beam == True: #and not self.traning: Am scos self.training fiindca nu il vede
      # aici am adaugat beam deconding in seq2seq
      self.decoder = BeamDecoder(output_vocab_size, config, device=device) 
    else: 
      self.decoder = Decoder(output_vocab_size, config, device=device)


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

  def __init__(self, config: CfgNode):
    super(DenseMLP, self).__init__()
    self.va = nn.Linear(config.DENSE_MLP_HIDDEN_SIZE, 1, bias=False)


  def forward(self, classifier_input):
    """
    Args:
        classifier_input: arc representation of Dense MLP hidden size for each
          concept pair; shape (batch_size, seq_len, seq_len, dense_mlp_hidden_size)
    Returns
      edge_score: (batch size, seq_len, seq_len).
    """
    edge_score = self.va(classifier_input)
    # Drop last dimension (we don't want vectors of 1 element).
    return edge_score.squeeze(-1)

class EdgeScoring(nn.Module):

  def __init__(self, config: CfgNode):
    super(EdgeScoring, self).__init__()
    self.dense_mlp = DenseMLP(config)
    self.Ua = nn.Linear(2 * config.HIDDEN_SIZE, config.DENSE_MLP_HIDDEN_SIZE, bias=False)
    self.Wa = nn.Linear(2 * config.HIDDEN_SIZE, config.DENSE_MLP_HIDDEN_SIZE, bias=False)

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
    parent_representations = self.Ua(concepts)
    child_representations = self.Wa(concepts)

    parent_mat = torch.repeat_interleave(parent_representations.unsqueeze(2), no_of_concepts, dim=-2)
    child_mat = torch.repeat_interleave(child_representations.unsqueeze(2), no_of_concepts, dim=-2)
    child_mat = torch.transpose(child_mat, -2, -3)

    # shape (batch_size, seq_len, seq_len, dense_mlp_hidden_size)
    classifier_input = torch.tanh(parent_mat + child_mat)
    scores = self.dense_mlp(classifier_input)
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
