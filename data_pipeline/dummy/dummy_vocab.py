"""
   File for Dummy Vocabulary
"""

from typing import List, Tuple
from copy import deepcopy
from collections import Counter


def build_vocab(words: List[str],
                special_words: List[str],
                min_frequency: int):
  words_counter = Counter(words)
  words_and_freq = sorted(
    words_counter.items(), key=lambda pair: pair[1], reverse=True)
  filtered_words_and_freq = [
    pair for pair in words_and_freq if pair[1] >= min_frequency]
  filtered_words = [wf[0] for wf in filtered_words_and_freq]
  vocab_words = special_words + filtered_words
  vocab = {word: i for i, word in enumerate(vocab_words)}
  return vocab

def build_vocabs(sentences: List[str],
                 special_words: Tuple[List[str], List[str], List[str]],
                 min_frequencies: Tuple[int, int]):
  """
  Builds the 3 dummy vocabularies.
  """
  # Unpack tuples.
  special_tokens, special_concepts = special_words
  min_freq_tokens, min_freq_concepts = min_frequencies
  # Extract all the words.
  all_tokens = []
  all_concepts = []
  for sentence in sentences:
    concepts = [char for char in sentence[::-1]]
    tokens = [char for char in sentence]
    all_tokens += tokens
    all_concepts += concepts
  # Extract the vocabs.
  token_vocab = build_vocab(all_tokens, special_tokens, min_freq_tokens)
  concept_vocab = build_vocab(all_concepts, special_concepts, min_freq_concepts)
  return token_vocab, concept_vocab


class DummyVocabs():
  """
   Class for Dummy Vocabulary
  """
  def __init__(self,
               sentences: List[str],
               unkown_special_word: str,
               special_words: Tuple[List[str], List[str]],
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
    token_vocab, concept_vocab = build_vocabs(
        sentences, special_words, min_frequencies)
    self.token_vocab = token_vocab
    self.concept_vocab = concept_vocab
    self.shared_vocab = self.create_shared_vocab()
    self.shared_vocab_size = len(self.shared_vocab.keys())
    self.token_vocab_size = len(token_vocab.keys())
    self.concept_vocab_size = len(concept_vocab.keys())

  def create_shared_vocab(self):
   """
   Creates the shared vocabulary between input and output.
   """
   shared_vocab = deepcopy(self.token_vocab)
   new_index = len(shared_vocab.items())
   for concept in self.concept_vocab.keys():
        if concept not in shared_vocab.keys():
            shared_vocab[concept] = new_index
            new_index += 1
   return shared_vocab

  def get_token_idx(self, token: str, use_shared = False):
    """
    Gives token index in vocabulary
    """
    if token in self.token_vocab.keys():
      return self.token_vocab[token]
    return self.token_vocab[self.unkown_special_word]

  def get_concept_idx(self, concept: str, use_shared = False):
    """
    Gives concept index in vocabulary
    """
    if concept in self.concept_vocab.keys():
      return self.concept_vocab[concept]
    return self.concept_vocab[self.unkown_special_word]

