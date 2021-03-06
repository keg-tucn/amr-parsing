from typing import List
from absl.testing import absltest
import penman
import torch

from evaluation.tensors_to_amr import tensors_to_lists, generate_variables, \
  get_unlabelled_amr_str_from_tensors, generate_amr_str_rec


"""
Tests for evaluation/tensors_to_amr.py
Run from project dir with 'python -m tests.evaluation.tensors_to_amr_test'
"""

class TensorsToAmrTest(absltest.TestCase):

  def test_tensors_to_lists(self):

    class MyVocabs:
      
      def __init__(self):
        self.concept_vocab = {'<pad>': 0, 'dog': 1, 'eat-01': 2, 'bone': 3}

    concepts = torch.tensor([10, 1, 2, 3, 0, 0])
    concept_length = 4
    adj_mat = [
      [0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 1, 0, 1, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0]
    ]
    adj_mat = torch.tensor(adj_mat)
    vocabs = MyVocabs()
    root_idx, concepts_as_list, adj_mat_as_list = tensors_to_lists(
      concepts, concept_length, vocabs, adj_mat)
    expected_concepts = ['dog', 'eat-01', 'bone']
    expected_adj_mat = [
      [0, 0, 0],
      [1, 0, 1],
      [0, 0, 0],
    ]
    self.assertEqual(root_idx, 1)
    self.assertEqual(concepts_as_list, expected_concepts)
    self.assertEqual(adj_mat_as_list, expected_adj_mat)

  def test_generate_variables(self):
    concepts = ['establish-01', 'model', 'industry', 'innovate-01']
    expected_variables = ['e', 'm', 'i', 'i2']
    variables = generate_variables(concepts)
    self.assertEqual(variables, expected_variables)

  def test_generate_amr_str_rec(self):
    """
    Test for bolt12_07_4800.1:
    (e / establish-01
      :ARG1 (m / model
              :mod (i / innovate-01
                      :ARG1 (i2 / industry))))
    """
    concepts = ['establish-01', 'model', 'industry', 'innovate-01']
    concepts_var = ['e', 'm', 'i', 'i2']
    adj_mat = [
      [0, 1, 0, 0],
      [0, 0, 0, 1],
      [0, 0, 0, 0],
      [0, 0, 1, 0]
    ]
    expected_amr_str = """
    ( e / establish-01 
        :unk-label ( m / model 
                :unk-label ( i2 / innovate-01 
                        :unk-label ( i / industry ))))"""
    root = 0
    amr_str = generate_amr_str_rec(
      root, seen_nodes=[], depth=1,
      concepts=concepts, concepts_var=concepts_var, adj_mat=adj_mat,
      relation_label=':unk-label')
    # Compare them with no extra spaces.
    amr_str = ' '.join(amr_str.split())
    expected_amr_str = ' '.join(expected_amr_str.split())
    self.assertEqual(amr_str.strip(), expected_amr_str.strip())

if __name__ == '__main__':
  absltest.main()