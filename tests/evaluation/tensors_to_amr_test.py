from absl.testing import absltest
import torch

from evaluation.tensors_to_amr import tensors_to_lists, generate_variables, \
  get_unlabelled_amr_str_from_tensors, generate_amr_str_rec, get_unlabelled_amr_strings_from_tensors


"""
Tests for evaluation/tensors_to_amr.py
Run from project dir with 'python -m tests.evaluation.tensors_to_amr_test'
"""

UNK_REL_LABEL = ':unk-label'

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
      concepts, concept_length, adj_mat, vocabs)
    expected_concepts = ['dog', 'eat-01', 'bone']
    expected_adj_mat = [
      [0, 0, 0],
      [1, 0, 1],
      [0, 0, 0],
    ]
    self.assertEqual(root_idx, 1)
    self.assertEqual(concepts_as_list, expected_concepts)
    self.assertEqual(adj_mat_as_list, expected_adj_mat)

  def test_tensors_to_lists_multiple_roots(self):
    class MyVocabs:
      def __init__(self):
        self.concept_vocab = {'<pad>': 0, 'dog': 1, 'eat-01': 2, 'bone': 3}

    concepts = torch.tensor([10, 1, 2, 3, 0, 0])
    concept_length = 4
    adj_mat = [
      [0, 0, 1, 1, 0],
      [0, 0, 0, 0, 0],
      [0, 1, 0, 1, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0]
    ]
    adj_mat = torch.tensor(adj_mat)
    vocabs = MyVocabs()
    root_idx, concepts_as_list, adj_mat_as_list = tensors_to_lists(
      concepts, concept_length, adj_mat, vocabs)
    expected_concepts = ['dog', 'eat-01', 'bone']
    expected_adj_mat = [
      [0, 0, 0],
      [1, 0, 1],
      [0, 0, 0],
    ]
    self.assertTrue(root_idx == 1 or root_idx == 2)
    self.assertEqual(concepts_as_list, expected_concepts)
    self.assertEqual(adj_mat_as_list, expected_adj_mat)

  def test_tensors_to_lists_no_roots(self):
    class MyVocabs:
      def __init__(self):
        self.concept_vocab = {'<pad>': 0, 'dog': 1, 'eat-01': 2, 'bone': 3}

    concepts = torch.tensor([10, 1, 2, 3, 0, 0])
    concept_length = 4
    adj_mat = [
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 1, 0, 1, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0]
    ]
    adj_mat = torch.tensor(adj_mat)
    vocabs = MyVocabs()
    root_idx, concepts_as_list, adj_mat_as_list = tensors_to_lists(
      concepts, concept_length, adj_mat, vocabs)
    expected_concepts = ['dog', 'eat-01', 'bone']
    expected_adj_mat = [
      [0, 0, 0],
      [1, 0, 1],
      [0, 0, 0],
    ]
    print(root_idx)
    self.assertTrue(root_idx == 0 or root_idx == 1 or root_idx == 2 or root_idx == 3)
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
      relation_label=UNK_REL_LABEL)
    # Compare them with no extra spaces.
    amr_str = ' '.join(amr_str.split())
    expected_amr_str = ' '.join(expected_amr_str.split())
    self.assertEqual(amr_str.strip(), expected_amr_str.strip())

  def test_generate_amr_str_rec_for_duplicate_root(self):
    """
    Test for case with duplicate node as root:
      ( m / multi-sentence
      :unk-label ( t / too
          :unk-label ( m / multi-sentence
              :unk-label t
              :unk-label m
              :unk-label ( m2 / many
                  :unk-label t
                  :unk-label m
                  :unk-label m2)))
      :unk-label m
      :unk-label m2)
    """

    root_idx = 5
    concepts_as_list = ['history', 'give-01', 'and', 'we', 'too', 'multi-sentence', 'many', 'lesson', 'interrogative',
                        '<unk>', '<unk>', '<unk>']
    adj_mat_as_list = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    expected_amr_str = """( m / multi-sentence 
                              :unk-label ( t / too 
                                  :unk-label m)
                              :unk-label m
                              :unk-label ( m2 / many 
                                  :unk-label t
                                  :unk-label m
                                  :unk-label m2))"""

    concepts_var = generate_variables(concepts_as_list)
    amr_str = generate_amr_str_rec(
      root_idx, seen_nodes=[root_idx], depth=1,
      concepts=concepts_as_list, concepts_var=concepts_var,
      adj_mat=adj_mat_as_list,
      relation_label=UNK_REL_LABEL)

    # Compare them with no extra spaces.
    amr_str = ' '.join(amr_str.split())
    expected_amr_str = ' '.join(expected_amr_str.split())
    self.assertEqual(amr_str.strip(), expected_amr_str.strip())

  def test_get_unlabelled_amr_str_from_tensors(self):
    class MyVocabs:
      def __init__(self):
        self.concept_vocab = {'<pad>': 0, 'establish-01': 1, 'model': 2, 'industry': 3, 'innovate-01': 4}

    concepts = torch.tensor([10, 1, 2, 3, 4, 0])
    concept_length = 5
    adj_mat = [
      [0, 1, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 0]
    ]
    adj_mat = torch.tensor(adj_mat)
    vocabs = MyVocabs()
    expected_amr_str = """
        ( e / establish-01 
            :unk-label ( m / model 
                    :unk-label ( i2 / innovate-01 
                            :unk-label ( i / industry ))))"""

    amr_str = get_unlabelled_amr_str_from_tensors(concepts, concept_length, adj_mat, vocabs, unk_rel_label=':unk-label')

    # Compare them with no extra spaces.
    amr_str = ' '.join(amr_str.split())
    expected_amr_str = ' '.join(expected_amr_str.split())
    self.assertEqual(amr_str.strip(), expected_amr_str.strip())

  def test_get_unlabelled_amr_strings_from_tensors(self):
    class MyVocabs:
      def __init__(self):
        self.concept_vocab = {'<pad>': 0, 'establish-01': 1, 'model': 2, 'industry': 3, 'innovate-01': 4}

    concepts = torch.tensor([[10, 1, 2, 3, 4, 0]])
    # Concepts should have shape (seq len, batch size).
    concepts = torch.transpose(concepts, 0, 1)
    concept_length = torch.tensor([5])
    adj_mat = [[
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0]]]
    adj_mat = torch.tensor(adj_mat)
    vocabs = MyVocabs()
    expected_amr_str = """
        ( e / establish-01 
            :unk-label ( m / model 
                    :unk-label ( i2 / innovate-01 
                            :unk-label ( i / industry ))))"""

    amr_strings = get_unlabelled_amr_strings_from_tensors(
      concepts, concept_length, adj_mat, vocabs, unk_rel_label=UNK_REL_LABEL)

    # Compare them with no extra spaces.
    amr_str = ' '.join(amr_strings[0].split())
    expected_amr_str = ' '.join(expected_amr_str.split())
    self.assertEqual(amr_str.strip(), expected_amr_str.strip())

if __name__ == '__main__':
  absltest.main()