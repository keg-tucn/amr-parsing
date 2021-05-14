from absl.testing import absltest
import torch

from evaluation.smatch_score import to_pred_mat, compute_accuracy, compute_f_score


class TensorsToAmrTest(absltest.TestCase):

    def test_to_pred_mat(self):
        mat = torch.tensor([
            [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        sentence_len = 3
        expected_rez = torch.tensor([False, False, True, False, False, False, False, True, False])

        res_edges = to_pred_mat(mat, sentence_len)
        self.assertEqual(expected_rez.tolist(), res_edges.tolist())

    def test_compute_f_score(self):
        gold = torch.tensor([[
            [0, 0, 2, 0],
            [3, 0, 1, 0],
            [4, 0, 0, 0],
            [0, 0, 0, 0]
        ]])

        pred = torch.tensor([[
            [0, 2, 0, 0],
            [0, 0, 1, 0],
            [4, 0, 2, 0],
            [0, 0, 0, 0]
        ]])

        f_score = compute_f_score(gold, pred)
        self.assertEqual(f_score.__round__(2), 0.67)

    def test_compute_accuracy(self):
        gold = torch.tensor([[
            [0, 0, 2, 0],
            [3, 0, 1, 0],
            [4, 0, 0, 0],
            [0, 0, 0, 0]
        ]])

        pred = torch.tensor([[
            [0, 2, 0, 0],
            [0, 0, 1, 0],
            [4, 0, 2, 0],
            [0, 0, 0, 0]
        ]])

        accuracy = compute_accuracy(gold, pred)
        self.assertEqual(accuracy, 0.75)
