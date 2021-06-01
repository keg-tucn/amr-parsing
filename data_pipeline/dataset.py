from typing import List
import copy

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import pad as torch_pad

import penman
from penman.models import noop
from penman.surface import AlignmentMarker

from data_pipeline.training_entry import TrainingEntry
from data_pipeline.data_reading import extract_triples, get_paths
from data_pipeline.vocab import Vocabs
from data_pipeline.glove_embeddings import GloVeEmbeddings

PAD = '<pad>'
UNK = '<unk>'
EOS = '<eos>'
PAD_IDX = 0

AMR_ID_KEY = 'amr_id'
SENTENCE_KEY = 'sentence'
CHAR_SENTENCE_KEY = 'char_sentence'
SENTENCE_STR_KEY = 'initial_sentence'
SENTENCE_LEN_KEY = 'sentence_lengts'
CHAR_SENTENCE_LEN_KEY = "char_sentence_length"
CONCEPTS_KEY = 'concepts'
CHAR_CONCEPTS_KEY = 'char_concepts'
CONCEPTS_STR_KEY = 'concepts_string'
GLOVE_CONCEPTS_KEY = 'glove_concepts'
CONCEPTS_LEN_KEY = 'concepts_lengths'
CHAR_CONCEPTS_LEN_KEY = "char_concepts_length"
ADJ_MAT_KEY = 'adj_mat'
AMR_STR_KEY = 'amr_str'

def pad_string_sequnece(batch_sequnece_strings, max_seq_len):
  """
     Adds padding to a batch of string sequences

     Args:
        batch_sequnece_strings: the set of sentances that need padding
        max_seq_len: the length of the longest sequence in the batch

     Returns:
        padded_sequences: the set of padded string sequences
  """
  padded_sequences = []
  for sequence in batch_sequnece_strings:
    padded_sequence = copy.deepcopy(sequence)
    for i in range(max_seq_len - len(sequence)):
      padded_sequence.append(PAD)
    padded_sequences.append(padded_sequence)
  return padded_sequences

def pad_char_sequence(batch_sequnece_char, max_seq_len, max_char_seq_len):
  """
      Adds padding to a batch of string sequences separated per characters

      Args:
         batch_sequnece_strings: the set of sentences, separated
         into characters for each word that need padding
         max_seq_len: the length of the longest sequence in the batch
        max_char_seq_len: the length of the longest sequence of characters in the batch
      Returns:
         padded_sequences: the set of padded string sequences separated into characters.
         It pads both the length of the characters per word and the sentences in the batch.
   """
  padded_batch = []
  padded_word = []
  for sentence in batch_sequnece_char:
    for word in sentence:
      padded_char = copy.deepcopy(word)
      for i in range(max_char_seq_len - len(word)):
        padded_char.append(0)
      padded_word.append(padded_char)
    for i in range(max_seq_len - len(sentence)):
      padded_word.append([0] * max_char_seq_len)
    padded_batch.append(padded_word)
    padded_word = []

  return padded_batch

def compute_char_length(batch_char_seg, max_seq_len):
  """
  Computes the length for the batch containing the
  sentences split into character
  Args:
    batch_char_seg: the set of sentences, separated
         into characters
    max_seq_len: the length of the longest sentence
  Returns:
    char_seq_len: the length of the sequence Tensor (input_seq_size, batch_size)
  """
  # add empty list to have the same length
  for sentance in batch_char_seg:
    while len(sentance) < max_seq_len:
      sentance.append([0])

  char_seq_len = [[len(word) for word in sentance] for sentance in batch_char_seg]
  return char_seq_len

def add_eos(training_entry: TrainingEntry, eos_token: str):
  training_entry.sentence.append(eos_token)
  training_entry.concepts.append(eos_token)

# TODO: create fuction to numericalize after the ASCII code
def numericalize_char(sentence):
  """
    Processes a sentence that is split into words into lists of integers
    that can be easily converted into tensors.
    The conversion is done for each character after it's ASCII code.
    Args:
      sentence: a sentence that is split into words
    Returns a tuple of:
      char_sentance_numericalized: List of list of character indices.
      Each word will be represented by a list of ASCII code for each
      character it contains.
    """
  # split the sentence into characters and get the character ASCII code
  # the ord() functions return the ASCII code for each character
  char_sentance_numericalized = [[ord(char) for char in word]for word in sentence]
  return char_sentance_numericalized

def numericalize(training_entry: TrainingEntry,
                 vocabs: Vocabs,
                 use_shared: bool,
                 glove_embeddings: GloVeEmbeddings):
  """
  Processes the train entry into lists of integers that can be easily converted
  into tensors. For the adjacency matrix 0 will be used in case the relation
  does not exist (is None).
  Args:
    vocabs: Vocabs object with the 3 vocabs (tokens, concepts, relations).
    use_shared: a flag telling us whether to use the shared vocabulary or
    the concept vocabulary and token vocabulary separately.
    This depends if the pointer generator is used or not.
    glove_embeddings: pertained Glove embeddings
  Returns a tuple of:
    sentence: List of token indices.
    concepts: List of concept indices.
    glove_embeddings: List of pertained Glove embeddings indices.
    adj_mat: Adjacency matrix which contains arc labels indices in the vocab.
  """
  # Process sentence.
  processed_sentence = [vocabs.get_token_idx(t, use_shared) for t in training_entry.sentence]
  # Process concepts.
  processed_concepts = [vocabs.get_concept_idx(c, use_shared) for c in training_entry.concepts]
  # Process glove concepts.
  processed_glove_concepts = [glove_embeddings.get_glove_concept_idx(c) for c in training_entry.concepts] \
    if glove_embeddings != None else []
  # Process adjacency matrix.
  processed_adj_mat = []
  for row in training_entry.adjacency_mat:
    processed_row = [0 if r is None else vocabs.get_relation_idx(r) for r in row]
    processed_adj_mat.append(processed_row)
  return processed_sentence, processed_concepts, processed_glove_concepts, processed_adj_mat

def remove_alignments(g: penman.Graph):
  """Creates a new penman graph AMR with no alignments. This is used if
  the amr string is needed with no alignments (eg. for smatch).

  Args:
      g (penman.Graph): Input penman graph (that might contain alignments).

  Returns:
      Output penman graph (same as the input one, but with no alignments).
  """
  # Create a deep copy of the input to not modify that object.
  new_g = copy.deepcopy(g)
  for epidata in g.epidata.items():
    triple, triple_data = epidata
    new_data = [d for d in triple_data if not isinstance(d, AlignmentMarker)]
    new_g.epidata[triple] = new_data
  return new_g

class AMRDataset(Dataset):
  """
  Dataset of sentence - amr entries, where the amrs are represented as a list
  of concepts and adjacency matrix.
  Arguments:
    paths: data paths.
    vocabs: the 3 vocabs (tokens, concepts, relations).
    device: cpu or cuda.
    seq2seq_setting: If true only the data for the seq2seq setting is returned
      (sequence of tokens with their lengths and concepts).
    ordered: if True the entries are ordered (decreasingly) by sentence length.
  """

  def __init__(self, paths: List[str], vocabs: Vocabs,
               device: str, seq2seq_setting: bool,
               ordered: bool,
               use_shared: bool = False,
               max_sen_len: int = None,
               glove: GloVeEmbeddings = None):
    super(AMRDataset, self).__init__()
    self.device = device
    self.seq2seq_setting = seq2seq_setting
    self.ids = []
    # Dictionary of amr id -> fields, where fields is a dict with the info
    # for each example.
    self.fields_by_id = {}
    for path in paths:
      triples = extract_triples(path)
      for triple in triples:
        id, sentence, amr_str = triple
        amr_penman_graph = penman.decode(amr_str, model=noop.model)
        # Get the amr string (with no alignment info).
        amr_penman_graph_no_align = remove_alignments(amr_penman_graph)
        amr_str_no_align = penman.encode(amr_penman_graph, model=noop.model)
        # Store the amr string with no alignment in a dictionary.
        training_entry = TrainingEntry(
          sentence=sentence.split(),
          g=amr_penman_graph,
          unalignment_tolerance=1)
        # Process the training entry (add EOS for sentence and concepts).
        if self.seq2seq_setting:
          add_eos(training_entry, EOS)
        # Numericalize the training entry (str -> vocab ids).
        sentence, concepts, glove_concepts, adj_mat = numericalize(training_entry, vocabs, use_shared, glove)
        # Numericalize the training entry sentence after each character ASCII code.
        char_sentence = numericalize_char(training_entry.sentence)
        # Numericalize the training entry concepts after each character ASCII code.
        char_concepts = numericalize_char(training_entry.concepts)
        # Collect the data.
        self.ids.append(id)
        field = {
          SENTENCE_KEY: torch.tensor(sentence, dtype=torch.long),
          CHAR_SENTENCE_KEY: char_sentence,
          SENTENCE_STR_KEY: training_entry.sentence,
          CONCEPTS_KEY: torch.tensor(concepts, dtype=torch.long),
          CHAR_CONCEPTS_KEY: char_concepts,
          CONCEPTS_STR_KEY: training_entry.concepts,
          GLOVE_CONCEPTS_KEY: torch.tensor(glove_concepts, dtype=torch.long),
          ADJ_MAT_KEY: torch.tensor(adj_mat, dtype=torch.long),
          AMR_STR_KEY: amr_str_no_align
        }
        self.fields_by_id[id] = field
    # Order them by sentence length.
    if ordered:
      # Sort dictionary items by sentence length.
      sorted_dict = sorted(self.fields_by_id.items(),
                           key=lambda item: len(item[1][SENTENCE_KEY]))
      # Retrieve the sorted amr ids.
      self.ids = [item[0] for item in sorted_dict]
    # Filter them out by sentence length.
    if max_sen_len is not None:
      self.ids = [id for id in self.ids \
                    if (len(self.fields_by_id[id][SENTENCE_KEY]) <= max_sen_len)]
    # Get max no of concepts.
    concept_lengths = [len(self.fields_by_id[id][CONCEPTS_KEY]) for id in self.ids]
    self.max_concepts_length = max(concept_lengths)

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, item):
    """Returns: id, sentence, sentence_str, concepts, concepts_str, adj_mat, amr_str."""
    id = self.ids[item]
    sentence = self.fields_by_id[id][SENTENCE_KEY]
    char_sentence = self.fields_by_id[id][CHAR_SENTENCE_KEY]
    sentence_str = self.fields_by_id[id][SENTENCE_STR_KEY]
    concepts = self.fields_by_id[id][CONCEPTS_KEY]
    char_concepts = self.fields_by_id[id][CHAR_CONCEPTS_KEY]
    concepts_str = self.fields_by_id[id][CONCEPTS_STR_KEY]
    glove_concepts = self.fields_by_id[id][GLOVE_CONCEPTS_KEY]
    adj_mat = self.fields_by_id[id][ADJ_MAT_KEY]
    amr_str = self.fields_by_id[id][AMR_STR_KEY]
    return id, sentence, char_sentence, sentence_str, concepts,\
           char_concepts, concepts_str, glove_concepts, adj_mat, amr_str

  def collate_fn(self, batch):
    amr_ids = []
    batch_sentences = []
    batch_char_sentence = []
    batch_sentences_strings = []
    batch_concepts = []
    batch_char_concepts = []
    batch_concepts_strings = []
    batch_glove_concepts = []
    batch_adj_mats = []
    amr_strings = []
    sentence_lengths = []
    concepts_lengths = []
    for entry in batch:
      amr_id, sentence, char_sentence, sentence_str, concepts, \
        char_concepts, concepts_str, glove_concepts, adj_mat, amr_str = entry
      amr_ids.append(amr_id)
      batch_sentences.append(sentence)
      batch_char_sentence.append(char_sentence)
      batch_sentences_strings.append(sentence_str)
      batch_concepts.append(concepts)
      batch_char_concepts.append(char_concepts)
      batch_concepts_strings.append(concepts_str)
      batch_glove_concepts.append(glove_concepts)
      batch_adj_mats.append(adj_mat)
      amr_strings.append(amr_str)
      sentence_lengths.append(len(sentence))
      concepts_lengths.append(len(concepts))
    # Get max lengths for padding.
    max_sen_len = max([len(s) for s in batch_sentences])
    max_char_sen_len = max([len(word) for s in batch_char_sentence for word in s])
    max_char_concepts_len = max([len(word) for s in batch_char_concepts for word in s])
    max_sen_str_len = max([len(s) for s in batch_sentences_strings])
    max_concepts_len = max([len(s) for s in batch_concepts])
    max_concepts_str_len = max([len(s) for s in batch_concepts_strings])
    max_glove_concepts_len = max([len(s) for s in batch_glove_concepts])
    max_adj_mat_size = max([len(s) for s in batch_adj_mats])
    # Compute the input_length for the sentence split into characters
    char_sentence_lengths = compute_char_length(batch_char_sentence, max_sen_len)
    # Compute the input_length for the concepts split into characters
    char_concepts_lengths = compute_char_length(batch_char_concepts, max_concepts_len)
    # Pad sentences.
    padded_sentences = [
      torch_pad(s, (0, max_sen_len - len(s))) for s in batch_sentences]

    # Pad char sentences.
    padded_char_sentences = pad_char_sequence(batch_char_sentence, max_sen_len, max_char_sen_len)
    # Pad char concepts.
    padded_char_concepts = pad_char_sequence(batch_char_concepts, max_concepts_len, max_char_concepts_len)

    # Pad initial sentences.
    padded_initial_sentences = pad_string_sequnece(batch_sentences_strings, max_sen_str_len)
    # Pad concepts
    padded_concepts = [
      torch_pad(c, (0, max_concepts_len - len(c))) for c in batch_concepts]
    # Pad concepts string sequence.
    padded_concepts_string  = pad_string_sequnece(batch_concepts_strings, max_concepts_str_len)
    # Pad glove concepts
    padded_glove_concepts = [
        torch_pad(c, (0, max_glove_concepts_len - len(c))) for c in batch_glove_concepts]
    # Pad adj matrices (pad on both dimensions).
    padded_adj_mats = []
    for adj_mat in batch_adj_mats:
      # Since it's a square matrix, the padding is the same on both dimensions.
      pad_size = max_adj_mat_size - len(adj_mat[0])
      padded_adj_mats.append(torch_pad(adj_mat, (0, pad_size, 0, pad_size)))
    # TODO: maybe by default do not put (seq_len, batch size) but have some
    # processing method for doing so after loading the data.
    if self.seq2seq_setting:
      new_batch = {
        SENTENCE_KEY: torch.transpose(torch.stack(padded_sentences), 0, 1).to(self.device),
        CHAR_SENTENCE_KEY: torch.transpose(torch.tensor(padded_char_sentences), 0, 2).to(self.device),
        # This is left on the cpu for 'pack_padded_sequence'.
        SENTENCE_STR_KEY: padded_initial_sentences,
        SENTENCE_LEN_KEY: torch.tensor(sentence_lengths),
        CHAR_SENTENCE_LEN_KEY: torch.transpose(torch.tensor(char_sentence_lengths), 0, 1),
        CONCEPTS_KEY: torch.transpose(torch.stack(padded_concepts), 0, 1).to(self.device),
        CONCEPTS_STR_KEY: padded_concepts_string
      }
    else:
      new_batch = {
        AMR_ID_KEY: amr_ids,
        CONCEPTS_KEY: torch.transpose(torch.stack(padded_concepts), 0, 1),
        GLOVE_CONCEPTS_KEY: torch.transpose(torch.stack(padded_glove_concepts), 0, 1),
        CHAR_CONCEPTS_KEY: torch.transpose(torch.tensor(padded_char_concepts), 0, 2).to(self.device),
        # This is left on the cpu for 'pack_padded_sequence'.
        CHAR_CONCEPTS_LEN_KEY: torch.transpose(torch.tensor(char_concepts_lengths), 0, 1),
        CONCEPTS_LEN_KEY: torch.tensor(concepts_lengths),
        ADJ_MAT_KEY: torch.stack(padded_adj_mats),
        AMR_STR_KEY: amr_strings
      }
    return new_batch

#TODO: remove this and add tests.
if __name__ == "__main__":
  subsets = ['bolt', 'cctv', 'dfa', 'dfb', 'guidelines',
             'mt09sdl', 'proxy', 'wb', 'xinhua']
  paths = get_paths('training', subsets)

  #TODO: a special token like 'no-relation' instead of None.
  special_words = ([PAD, EOS, UNK], [PAD, EOS, UNK], [PAD, UNK, None])
  vocabs = Vocabs(paths, UNK, special_words, min_frequencies=(1, 1, 1))

  dataset = AMRDataset(paths, vocabs, device='cpu', seq2seq_setting=False, ordered=True)

  #TODO: see if thr bactching could somehow be done by size (one option
  # would be to order the elements in the dataset, have fixed batches and
  # somehow shuffle the batches instead of the elements themselves).
  dataloader = DataLoader(dataset, batch_size=3, collate_fn=dataset.collate_fn)

  i = 0
  for batch in dataloader:
    if i == 2:
      break
    i+=1
    print('Batch ',i)
    print('Amr ids')
    print(batch[AMR_ID_KEY])
    print('Amrs')
    print(batch[AMR_STR_KEY])
    print(batch[CONCEPTS_KEY].shape)