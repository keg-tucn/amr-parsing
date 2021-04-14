from typing import List, Tuple, Dict
from collections import Counter
import os
import pickle

import penman
from penman.models import noop

import definitions
from data_pipeline.data_reading import extract_triples, get_paths
from data_pipeline.dummy.dummy_training_entry import DummyTrainingEntry
from data_pipeline.vocab import get_cache_paths, read_cached_vocabs, cache_vocabs, build_vocab

VOCAB_PATH = 'temp/vocabs'
TOKENS_CACHE_FILE = 'letter_tokens.pickle'
CONCEPTS_CACHE_FILE = 'letter_concepts.pickle'
RELATIONS_CACHE_FILE = 'letter_relations.pickle'

def build_vocabs(
                 sentences: List[str],
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
  for sentence in sentences:
    training_entry = DummyTrainingEntry(sentence=sentence,
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


class DummyVocabs():

  def __init__(self,
               sentences: List[str],
               unkown_special_word: str,
               special_words: Tuple[List[str], List[str], List[str]],
               min_frequencies: Tuple[int, int, int]):
    """
    Args:
      sentences: Dummy sentences.
      unkown_special_word: Unknown special word.
      special_words: Tuple of special words (for sentence tokens, for concepts
        and for relations). Each list contains special words like PAD or UNK
        (these are addded at the begining of the vocab, with PAD usually at
        index 0).
      min_frequencies: Tuple of min frequencies for tokens, concepts and
        relations. If the words have a frequency < than the given frequency,
        they are not added to the vocab.
    """
    self.unkown_special_word = unkown_special_word
    token_vocab, concept_vocab, relation_vocab = build_vocabs(
      sentences, special_words, min_frequencies)
    self.token_vocab = token_vocab
    self.concept_vocab = concept_vocab
    self.relation_vocab = relation_vocab
    self.token_vocab_size = len(token_vocab.keys())
    self.concept_vocab_size = len(concept_vocab.keys())
    self.relation_vocab_size = len(relation_vocab.keys())

  def get_token_idx(self, token: str):
    if token in self.token_vocab.keys():
      return self.token_vocab[token]
    return self.token_vocab[self.unkown_special_word]

  def get_concept_idx(self, concept: str):
    if concept in self.concept_vocab.keys():
      return self.concept_vocab[concept]
    return self.concept_vocab[self.unkown_special_word]

  def get_relation_idx(self, relation: str):
    if relation in self.relation_vocab.keys():
      return self.relation_vocab[relation]
    return self.relation_vocab[self.unkown_special_word]