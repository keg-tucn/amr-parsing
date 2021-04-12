from typing import List, Dict
import penman
from data_pipeline.concepts import get_ordered_concepts_ids
from data_pipeline.adjacency_mat import get_adjacency_mat

ROOT_ID = -1
ROOT = 'root'

class DummyTrainingEntry():
  
  def __init__(self,
    sentence,
    # g: penman.Graph,
    unalignment_tolerance: float = 0):
    "Create training entry with sentence and AMR."
    # concepts, adj_mat = TrainingEntry.construct_from_penman(g,
    #   unalignment_tolerance)
    # if concepts is None:
      # return None
    self.sentence = [char for char in sentence]
    # Reverse string + split
    self.concepts = [char for char in sentence[::-1]]
    print("training entry sentence", self.sentence)
    print("training entry concepts", self.concepts)
    self.adjacency_mat = []

  @staticmethod
  def construct_from_penman(g: penman.Graph, unalignment_tolerance: float):
    # Get ordered concepts.
    ordered_concept_ids = get_ordered_concepts_ids(g, unalignment_tolerance)
    if ordered_concept_ids is None:
      return None, None
    ordered_concepts = [g.triples[c_id][2] for c_id in ordered_concept_ids]
    # Add fake root.
    ordered_concept_ids.insert(0, ROOT_ID)
    ordered_concepts.insert(0, ROOT)
    # Get adjacency matrix.
    adj_mat = get_adjacency_mat(g, ordered_concept_ids, ROOT_ID)
    # Return training entry.
    return ordered_concepts, adj_mat

  def get_labels(self):
    """Returns the tokens, concepts and relations in a training entry.
    """
    tokens = self.sentence
    concepts = self.concepts
    relations = []
    for row in self.adjacency_mat:
      for relation in row:
        if relation != None:
          relations.append(relation)
    return tokens, concepts, relations