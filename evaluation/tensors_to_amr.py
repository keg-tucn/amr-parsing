from typing import List

import torch
import random

NO_VAR = '_'

def tensors_to_lists(concepts: torch.tensor,
                     concepts_length: torch.tensor,
                     adj_mat: torch.tensor,
                     concept_vocab):
  """
  Args:
    concepts: Concept sequence (max seq len).
    concepts_length: Concept sequence length scalar.
    adj_mat (torch.tensor): Adj matrix; shape (max len, max len).
    vocabs: Vocabs object (the 3 vocabs).

  Returns:
    Python lists with no padding. The fake root is also removed and the root
    concept is returned besides the list of string concepts and adj matrix.
  """
  # Remove padding.
  concepts_no_padding = concepts[0:concepts_length]
  adj_mat_no_padding = adj_mat[0:concepts_length, 0:concepts_length]
  # Extract real root (1: to not take fake root).
  root_indexes = torch.nonzero(adj_mat_no_padding[0, 1:])
  root_indexes = root_indexes + 1 #shift them back because 1:
  root_idx = random.randrange(1, concepts_length)
  if(len(root_indexes.tolist()) == 1):
    root_idx = int(root_indexes[0])
  if(len(root_indexes.tolist()) > 1):
    # chosen_index = random.randrange(len(root_indexes.tolist()))
    root_idx = int(root_indexes[0])
  # Remove fake root
  concepts_no_fake_root = concepts_no_padding[1:]
  adj_mat_no_fake_root = adj_mat_no_padding[1:,1:]
  root_idx = root_idx - 1
  # Transform to lists.
  concepts_as_list = concepts_no_fake_root.tolist()
  adj_mat_as_list = adj_mat_no_fake_root.tolist()
  # Un-numericalize concepts.
  ids_to_concepts_list = list(concept_vocab.keys())
  concepts_as_list = [ids_to_concepts_list[id] for id in concepts_as_list]
  return root_idx, concepts_as_list, adj_mat_as_list

def is_constant_concept(concept: str):
  constants = ['-', '+', 'imperative', 'interrogative', 'expressive']
  # Check if it belongs to the list of constants.
  if concept in constants:
    return True
  # Check if it is enclosed by quotes.
  if concept[0]=='"'and concept[-1]=='"':
    return True
  # Check if it is a number.
  try:
    float(concept)
    return True
  except ValueError:
    return False

def generate_variables(concepts: List[str]):
  variables = []
  for concept in concepts:
    if is_constant_concept(concept):
      variables.append(NO_VAR)
    else:
      prefix = concept[0].lower()
      var = prefix
      digit = 2
      while var in variables:
        var = prefix + str(digit)
        digit+=1
      variables.append(var)
  return variables

def generate_amr_str_rec(root: int, seen_nodes: List[int], depth,
                         concepts: List[str], concepts_var: List[str], adj_mat: List[List[int]],
                         labels: List[str], unk_relation_label: str):

  amr_str = "( {} / {} ".format(concepts_var[root], concepts[root])
  no_concepts = len(concepts)
  has_children = False
  for i in range(no_concepts):
    relation_id = adj_mat[root][i]
    if relation_id != 0:
      relation_label = labels[relation_id] if unk_relation_label is None else unk_relation_label
      has_children = True
      # If there is an edge i is a child node.
      # Check if it's a constant or a node with variable.
      if concepts_var[i] == NO_VAR:
        child_representation = "{} {}".format(relation_label, concepts[i])
      else: # Node with variable.
        if i in seen_nodes:
          # If i was seen it will be represented as a reentrant node (only var).
          child_representation = "{} {}".format(relation_label, concepts_var[i])
        else:
          # The child i was not already visited. It's marked as seen and it
          # becomes the root in the recursive call.
          if depth < no_concepts:
            seen_nodes.append(i)
            rec_repr = generate_amr_str_rec(i, seen_nodes, depth+1,
                                            concepts, concepts_var, adj_mat,
                                            labels, unk_relation_label)
            child_representation = "{} {}".format(relation_label, rec_repr)
          else:
            child_representation = ''
            break
      amr_str += "\n".ljust(depth + 1, "\t") + child_representation
  amr_str += ")"
  return amr_str

def get_amr_str_from_tensors(concepts: torch.tensor,
                             concepts_length: torch.tensor,
                             adj_mat: torch.tensor,
                             concept_vocab,
                             relation_vocab,
                             unk_rel_label: str):
  """
  Args:
    concepts: Concept sequence (max seq len).
    concepts_length: Concept sequence length scalar.
    adj_mat (torch.tensor): Adj matrix; shape (max len, max len).
    unk_rel_label: label that will be put on edges
      (in the unlabelled setting).
  """
  # Post-processing (don't allow self edges)
  max_seq_len = adj_mat.shape[1]
  mask = torch.eye(max_seq_len, max_seq_len, dtype=bool).to(adj_mat.device)
  adj_mat.masked_fill_(mask, 0)

  root_idx, concepts_as_list, adj_mat_as_list = tensors_to_lists(
    concepts, concepts_length, adj_mat, concept_vocab)
  concepts_var = generate_variables(concepts_as_list)
  relation_vocab = relation_vocab
  labels = list(relation_vocab.keys())
  amr_str = generate_amr_str_rec(
      root_idx, seen_nodes=[root_idx], depth=1,
      concepts=concepts_as_list, concepts_var=concepts_var,
      adj_mat=adj_mat_as_list,
      labels=labels,
      unk_relation_label=unk_rel_label)
  return amr_str

def get_amr_strings_from_tensors(concepts: torch.tensor,
                                 concepts_lengths: torch.tensor,
                                 adj_mats: torch.tensor,
                                 concept_vocab,
                                 relation_vocab,
                                 unk_rel_label: str):
  """
  This method creates a labelled or unlabelled AMR, depending on the value
  given to the unk_rel_label variable:
    - None: labelled AMR.
    - otherwise: unlabelled, using the given value for all the
                 arc labels in the AMR.
  Args:
      concepts: Batch of concept sequences (max seq len, batch size).
      concepts_lengths: Batch of sequences lentgths (batch size).
      adj_mats (torch.tensor): Batch of adj matrices (batch size, max seq len, max seq len):
        - unlabelled AMR: binary adj mats.
        - labelled AMR: adj mats with label indices.
      unk_rel_label: label that will be put on edges for unlabelled AMR.

  Returns: batch of (un)labelled AMR strings
  """
  amrs = []
  batch_size = concepts.shape[1]
  for batch in range(batch_size):
    amr_string = get_amr_str_from_tensors(concepts[:,batch],
                                          concepts_lengths[batch],
                                          adj_mats[batch],
                                          concept_vocab,
                                          relation_vocab,
                                          unk_rel_label)
    amrs.append(amr_string)
  return amrs