from typing import List, Dict
import penman
from data_pipeline.concepts import get_ordered_concepts_ids
from data_pipeline.adjacency_mat import get_adjacency_mat

ROOT_ID = -1
ROOT = 'root'

class TrainingEntry():
  
  def __init__(self,
    sentence: List[str],
    g: penman.Graph,
    unalignment_tolerance: float = 0):
    "Create training entry with sentence and AMR."
    concepts, adj_mat = TrainingEntry.construct_from_penman(g,
      unalignment_tolerance)
    if concepts is None:
      return None
    self.sentence = sentence
    self.concepts = concepts
    self.adjacency_mat = adj_mat

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

  def update_vocabs(self,
                    token_vocab: Dict[str, int],
                    concept_vocab: Dict[str, int],
                    relation_vocab: Dict[str, int]):
    """
    Update the three vocabularies with elements from the training entry.
    Args:
      token_vocab: Vocabulary of sentence tokens.
      concept_vocab: Vocabulary of concepts.
      relation_vocab: Vocabulary of arc relations.
    Returns:
      A tuple of the updated vocabularies:
      (token_vocab, concept_vocab, relation_vocab).
    """
    # Update token vocab.
    for token in self.sentence:
      if token not in token_vocab.keys():
        token_vocab.update({token: len(token_vocab.keys())})
    # Update concept vocab.
    for concept in self.concepts:
      if concept not in concept_vocab.keys():
        concept_vocab.update({concept: len(concept_vocab.keys())})
    # Update relation vocab.
    for row in self.adjacency_mat:
      for relation in row:
        if relation != None and relation not in relation_vocab.keys():
          relation_vocab.update({relation: len(relation_vocab.keys())})
    return token_vocab, concept_vocab, relation_vocab

  def process(self,
              token_vocab: Dict[str, int],
              concept_vocab: Dict[str, int],
              relation_vocab: Dict[str, int]):
    """
    Processes the train entry into lists of integeres that can be easily converted
    into tensors. For the adjacency matrix 0 will be used in case the relation
    does not exist (is None).
    Args:
      token_vocab: Vocabulary of sentence tokens.
      concept_vocab: Vocabulary of concepts.
      relation_vocab: Vocabulary of arc relations.
    Returns a tuple of:
      sentece: List of token indices.
      concepts: List of concept indices.
      adj_mat: Adjacency matrix which contains arc labels indices in the vocab.
    """
    # Process sentence.
    processed_sentence = [token_vocab[t] for t in self.sentence]
    # Process concepts.
    processed_concepts = [concept_vocab[c] for c in self.concepts]
    # Process adjacency matrix.
    processed_adj_mat = []
    for row in self.adjacency_mat:
      processed_row = [0 if r is None else relation_vocab[r] for r in row]
      processed_adj_mat.append(processed_row)
    return processed_sentence, processed_concepts, processed_adj_mat