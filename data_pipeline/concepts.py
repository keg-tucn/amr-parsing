from random import randrange
from typing import List, Dict

import penman
from penman.graph import Triple
from penman.surface import alignments
# Extract from the AMR the nodes in inference order.
# The nodes are represented by the position in the triples list (from the penman
# amr representation).

# The amr triples contain in the target part [2] the concepts, but not only (
# they can contain variables). The triples are first filtered in order to only
# have triples associated with the concepts.


def get_concept_ids(g: penman.Graph) -> List[int]:
  """
  Constructs a list of concept ids.

  Args:
    g: amr graph (in penman library representation).
  Returns:
    A list of the triple ids (positions in the original triples list) for the
    triples associated with concepts (that have a concept on the 3rd position). 
  """
  triples = g.triples
  variables = [t[0] for t in triples]
  variables = list(set(variables))
  filtered_triples_ids = []
  for triple_id in range(len(triples)):
    t = triples[triple_id]
    src = t[0]
    rel = t[1]
    trgt = t[2]
    if trgt not in variables or rel == ':instance':
      filtered_triples_ids.append(triple_id)
  return filtered_triples_ids


def get_concepts_alignments(g: penman.Graph) -> Dict[int, List[int]]:
  """
  Construct a dictionary of concept id -> alignments for an AMR graph.

  Args:
    g: amr graph (in penman library representation).
  Returns:
    dictionary of concept_id -> alignments where the concept id is the position
    in the g.triples list and alignments is a list of ints (token ids).
  """
  concept_ids = get_concept_ids(g)
  node_alignments = alignments(g)
  concepts_alignments = {}
  for concept_id in concept_ids:
    triple = g.triples[concept_id]
    if triple in node_alignments.keys():
      concepts_alignments[concept_id] = node_alignments[triple].indices
    else:
      concepts_alignments[concept_id] = []
  return concepts_alignments


def get_ordered_concepts_ids(g: penman.Graph, unalignment_tolerance = 0):
  """
  Constructs a list of ordered concepts.

  Args:
    g: amr graph (in penman library representation).
    unalignment_tolerance: the percentage of concepts with no alignemnt info
      allowed. If the amr has a larger percantage than allowed trhough this
      parameter, the function returns None. If the amr has a non-zero percentage
      of unaligned concepts but smaller than the tolerance, the unaligned
      concepts are placed at random in the revariables = [t[0] for t in triples]
  variables = list(set(variables))turn list.
  Returns:
    List of concepts ordered by inference order or None (where each concept is
    represented by the triple id in the penman graph representation).
  """
  concepts_alignments = get_concepts_alignments(g)
  concept_ids = concepts_alignments.keys()
  no_concepts = len(concept_ids)
  aligned_concepts = [c for c in concept_ids if len(concepts_alignments[c])>0]
  unaligned_concepts = [c for c in concept_ids if c not in aligned_concepts]
  no_unaligned_concepts = len(unaligned_concepts)
  if no_unaligned_concepts/no_concepts > unalignment_tolerance:
    return None
  # Only use the first alignment for sorting (for now).
  result = sorted(aligned_concepts, key = lambda c: concepts_alignments[c][0])
  # Place unaligned concepts randomly.
  for c in unaligned_concepts:
    random_index = 0
    if len(result) != 0:
      random_index = randrange(len(result))
    result.insert(random_index, c)
  return result