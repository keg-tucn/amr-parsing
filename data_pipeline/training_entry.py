from typing import List
import penman
from data_pipeline.concepts import get_ordered_concepts_ids
from data_pipeline.adjacency_mat import get_adjacency_mat

ROOT_ID = -1
ROOT = 'root'

class TrainingEntry():
  
  def __init__(self, concepts: List[str], adjacency_mat: List[List[str]]):
    self.concepts = concepts
    self.adjacency_mat = adjacency_mat

  @staticmethod
  def construct_from_penman(g: penman.Graph, unalignment_tolerance: float):
    # Get ordered concepts.
    ordered_concept_ids = get_ordered_concepts_ids(g, unalignment_tolerance)
    ordered_concepts = [g.triples[c_id][2] for c_id in ordered_concept_ids]
    # Add fake root.
    ordered_concept_ids.insert(0, ROOT_ID)
    ordered_concepts.insert(0, ROOT)
    # Get adjacency matrix.
    adj_mat = get_adjacency_mat(g, ordered_concept_ids, ROOT_ID)
    # Return training entry.
    return TrainingEntry(ordered_concepts, adj_mat)