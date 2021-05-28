"""
   File for Dummy Vocabulary
"""

from typing import List, Tuple

from data_pipeline.dummy.dummy_training_entry import DummyTrainingEntry
from data_pipeline.vocab import build_vocab

def build_vocabs(sentences: List[str],
                 special_words: Tuple[List[str], List[str], List[str]],
                 min_frequencies: Tuple[int, int, int]):
  """
  Builds the 3 dummy vocabularies.
  """
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
  return token_vocab, concept_vocab, relation_vocab


class DummyVocabs():
  """
   Class for Dummy Vocabulary
  """
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

  def get_token_idx(self, token: str, use_shared=False):
    """
    Gives token index in vocabulary
    """
    if token in self.token_vocab.keys():
      return self.token_vocab[token]
    return self.token_vocab[self.unkown_special_word]

  def get_concept_idx(self, concept: str, use_shared=False):
    """
    Gives concept index in vocabulary
    """
    if concept in self.concept_vocab.keys():
      return self.concept_vocab[concept]
    return self.concept_vocab[self.unkown_special_word]

  def get_relation_idx(self, relation: str):
    if relation in self.relation_vocab.keys():
      return self.relation_vocab[relation]
    return self.relation_vocab[self.unkown_special_word]
