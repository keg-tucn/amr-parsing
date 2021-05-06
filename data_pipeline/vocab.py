from typing import List, Tuple, Dict
from collections import Counter
import os
import pickle

import penman
from penman.models import noop

from utils import definitions
from data_pipeline.data_reading import extract_triples
from data_pipeline.training_entry import TrainingEntry

VOCAB_PATH = 'temp/vocabs'
TOKENS_CACHE_FILE = 'tokens.pickle'
CONCEPTS_CACHE_FILE = 'concepts.pickle'
RELATIONS_CACHE_FILE = 'relations.pickle'

def build_vocab(words: List[str], special_words:List[str], min_frequency: int):
  words_counter = Counter(words)
  words_and_freq = sorted(
    words_counter.items(), key=lambda pair: pair[1], reverse=True)
  # Filter out words with freq < min frequency.
  filtered_words_and_freq = [
    pair for pair in words_and_freq if pair[1] >= min_frequency]
  filtered_words = [wf[0] for wf in filtered_words_and_freq]
  vocab_words = special_words + filtered_words
  vocab = {word: i for i, word in enumerate(vocab_words)}
  return vocab
  
def get_cache_paths():
  cache_dir = os.path.join(definitions.PROJECT_ROOT_DIR, VOCAB_PATH)
  cache_files = [TOKENS_CACHE_FILE, CONCEPTS_CACHE_FILE, RELATIONS_CACHE_FILE]
  cache_paths = [os.path.join(cache_dir, f) for f in cache_files]
  return cache_paths

def read_cached_vocabs():
  """
  Returns:
    The 3 cached vocabs (tokens, concepts, relations) or None if nothing cached.
  """
  cache_paths = get_cache_paths()
  vocabs = []
  for cache_path in cache_paths:
    if os.path.isfile(cache_path):
      with open(cache_path, 'rb') as vocab_file:
        vocab = pickle.load(vocab_file)
        vocabs.append(vocab)
    else:
      return None
  return vocabs[0], vocabs[1], vocabs[2]

def cache_vocabs(token_vocab: Dict[str, int],
                 concept_vocab: Dict[str, int],
                 relation_vocab: Dict[str, int]):
  cache_dir = os.path.join(definitions.PROJECT_ROOT_DIR, VOCAB_PATH)
  if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    return None
  cache_paths = get_cache_paths()
  vocabs = [token_vocab, concept_vocab, relation_vocab]
  for i in range(len(cache_paths)):
    with open(cache_paths[i], 'wb') as vocab_file:
      pickle.dump(vocabs[i], vocab_file)

def build_vocabs(paths: List[str],
                 special_words: Tuple[List[str], List[str], List[str]],
                 min_frequencies: Tuple[int, int, int]):
  """
  Builds the 3 vocabularies. The vocabularies are cached and constructed again
  only if the caching doesn't exist.
  """
  #TODO: control caching though param.
  vocabs = read_cached_vocabs()
  if vocabs is not None:
    return vocabs
  # Unpack tuples.
  special_tokens, special_concepts, special_relations = special_words
  min_freq_tokens, min_freq_concepts, min_freq_relations = min_frequencies
  # Extract all the words.
  all_tokens = []
  all_concepts = []
  all_relations = []
  for path in paths:
    triples = extract_triples(path)
    for triple in triples:
      id, sentence, amr_str = triple
      amr_penman_graph = penman.decode(amr_str, model=noop.model)
      training_entry = TrainingEntry(
        sentence=sentence.split(),
        g=amr_penman_graph,
        unalignment_tolerance=1)
      tokens, concepts, relations = training_entry.get_labels()
      all_tokens += tokens
      all_concepts += concepts
      all_relations += relations
  # Extract the vocabs.
  token_vocab = build_vocab(all_tokens, special_tokens, min_freq_tokens)
  concept_vocab = build_vocab(all_concepts, special_concepts, min_freq_concepts)
  relation_vocab = build_vocab(all_relations, special_relations, min_freq_relations)
  # Cache the vocabs.
  cache_vocabs(token_vocab, concept_vocab, relation_vocab)
  return token_vocab, concept_vocab, relation_vocab


class Vocabs():

  def __init__(self,
               paths: List[str],
               unknown_special_word: str,
               special_words: Tuple[List[str], List[str], List[str]],
               min_frequencies: Tuple[int, int, int]):
    """
    Args:
      paths: Amr data file paths.
      unknown_special_word: Unknown special word.
      special_words: Tuple of special words (for sentence tokens, for concepts
        and for relations). Each list contains special words like PAD or UNK
        (these are added at the beginning of the vocab, with PAD usually at
        index 0).
      min_frequencies: Tuple of min frequencies for tokens, concepts and
        relations. If the words have a frequency < than the given frequency,
        they are not added to the vocab.
    """
    self.unknown_special_word = unknown_special_word
    token_vocab, concept_vocab, relation_vocab = build_vocabs(
      paths, special_words, min_frequencies)
    self.token_vocab = token_vocab
    self.concept_vocab = concept_vocab
    self.relation_vocab = relation_vocab
    self.token_vocab_size = len(token_vocab.keys())
    self.concept_vocab_size = len(concept_vocab.keys())
    self.relation_vocab_size = len(relation_vocab.keys())

  def get_token_idx(self, token: str):
    if token in self.token_vocab.keys():
      return self.token_vocab[token]
    return self.token_vocab[self.unknown_special_word]

  def get_concept_idx(self, concept: str):
    if concept in self.concept_vocab.keys():
      return self.concept_vocab[concept]
    return self.concept_vocab[self.unknown_special_word]

  def get_relation_idx(self, relation: str):
    if relation in self.relation_vocab.keys():
      return self.relation_vocab[relation]
    return self.relation_vocab[self.unknown_special_word]