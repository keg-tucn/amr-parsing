from typing import List
import penman

INSTANCE_ARC_LABEL = ":instance"
ROOT_LABEL = ':root'


def get_concepts_to_triple_ids(g: penman.Graph):
  """
  Extract a dictionary concept -> triple id (where the 
  concept is instanced).
  """
  variables = g.variables()
  concepts_to_ids = {}
  for triple_id in range(len(g.triples)):
    triple = g.triples[triple_id]
    src = triple[0]
    role_label = triple[1]
    trgt = triple[2]
    if role_label == INSTANCE_ARC_LABEL:
      concepts_to_ids[src] = triple_id
    if trgt not in variables:
      concepts_to_ids[trgt] = triple_id
  return concepts_to_ids


def get_edges(g: penman.Graph, ordered_concept_ids: List[int], fake_root_id: int):
  variables = g.variables()
  edges = {}
  # Add (src_var, trg_var) -> edge
  for triple in g.triples:
    src = triple[0]
    edge_label = triple[1]
    target = triple[2]
    # Need to check that src is different from trgt to not add the 
    # instance edge between the variable i and concept i.
    if src in variables and target in variables and src!=target:
      edges[(src, target)] = edge_label
  # Add (src_var, constant) -> edge
  for concept_pos in range(len(ordered_concept_ids)):
    penman_concept_id = ordered_concept_ids[concept_pos]
    if penman_concept_id != fake_root_id:
      triple = g.triples[penman_concept_id]
      src = triple[0]
      edge_label = triple[1]
      target = triple[2]
      if edge_label != INSTANCE_ARC_LABEL:
        # concept with no variable, src is parent.
        edges[(src, target)] = edge_label
  return edges


def get_adjacency_mat(g: penman.Graph,
                      ordered_concepts_ids: List[int],
                      fake_root_id: int):
  """
  Construct an adjacency matrix from the penman graph representation of the AMR
  and the list of ordered concepts (represented as penman triple ids).
  Note: The ordered list of concepts contains on the first position a fake root
  (will be added in the adjacency matrix so the amr top is encoded).

  Returns
    an adjacency matrix that contains
      None: no edge
      edge label (str): if edge
  """
  n = len(ordered_concepts_ids)
  # 1. Construct (src_var, trgt_var/constant) -> edge label
  edges = get_edges(g, ordered_concepts_ids, fake_root_id)
  # 2. Construct concept -> triple id dict.
  concepts_to_ids = get_concepts_to_triple_ids(g)
  # 3. Construct triple_id -> concept pos dict.
  triple_to_concept = {ordered_concepts_ids[c]: c for c in range(n)}
  # 4. Construct adjacency matrix.
  adj_mat = [[None for i in range(n)] for j in range(n)]
  for edge, edge_label in edges.items():
    src, trgt = edge
    # Go from var/constant -> triple id -> ordered concept position.
    src_triple_id = concepts_to_ids[src]
    trgt_triple_id = concepts_to_ids[trgt]
    src = triple_to_concept[src_triple_id]
    trgt = triple_to_concept[trgt_triple_id]
    # src = triple_to_concept[concepts_to_ids[src]]
    # trgt = triple_to_concept[concepts_to_ids[trgt]]
    adj_mat[src][trgt] = edge_label
  # Add root.
  top = triple_to_concept[concepts_to_ids[g.top]]
  adj_mat[0][top] = ROOT_LABEL
  return adj_mat
