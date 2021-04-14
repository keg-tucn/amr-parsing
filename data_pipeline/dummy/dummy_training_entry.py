from typing import List, Dict
import penman
from data_pipeline.concepts import get_ordered_concepts_ids
from data_pipeline.adjacency_mat import get_adjacency_mat

ROOT_ID = -1
ROOT = 'root'

class DummyTrainingEntry():
  
  def __init__(self,
    sentence,
    unalignment_tolerance: float = 0):
    "Create training entry with sentence and AMR."

    self.sentence = [char for char in sentence]
    # Reverse string + split
    self.concepts = [char for char in sentence[::-1]]
    self.adjacency_mat = []

  def get_labels(self):
    """Returns the tokens, concepts and relations in a training entry.
    """
    tokens = self.sentence
    concepts = self.concepts
    relations = []
    return tokens, concepts, relations