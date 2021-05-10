import torch
from absl.testing import absltest

from train_concept_identification import compute_fScore


class TrainConceptIdentificationTest(absltest.TestCase):

  def test_compute_fScore(self):
    class MyVocabs:

          def __init__(self):
              self.concept_vocab = {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'root': 3, 'eat-01': 4,  'dog':  5, 'car': 6}

    vocabs = MyVocabs()
    gold_outputs = torch.tensor([
        [3,   3,    3,   3,    3],
        [6,   4,    5,   6,    4],
        [1,   1,    1,   1,    1],
        [0,   1,    1,   4,    4]])

    predicted_outputs =torch.tensor([
        [3,   3,    3,   3,    3],
        [1,   1,    5,   6,    1],
        [1,   1,    1,   6,    1],
        [0,   1,    1,   1,    0]])

    f_score = compute_fScore(gold_outputs, predicted_outputs, vocabs)

    print("f-score", f_score)
    self.assertEqual(f_score, 0.4)

if __name__ == '__main__':
  absltest.main()