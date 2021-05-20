from absl.testing import absltest
import torch

from evaluation.arcs_evaluation_metrics import unpad_mat_to_list, \
    compute_accuracy, compute_f_score, compute_smatch, SmatchScore, compute_multiclass_f_score


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

        res_edges = unpad_mat_to_list(mat, sentence_len)
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

        f_score, precision, recall = compute_f_score(gold, pred)
        self.assertEqual(0.5, f_score)
        self.assertEqual(0.5, precision)
        self.assertEqual(0.5, recall)

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

    def test_compute_smatch(self):
        '''
            Precision = positive predicted values.
                All existent triplets were predicted correctly. => Precision should be 1
            Recall = relevant predicted values.
                2 of of 3 predicted correctly => Recall = 2/3
            F_score = 2 * (precision * recall) / (precision + recall)
                => F_score should be 0.8
        '''
        amr = '''(a2 / attract-01~e.1 
                    :degree (v / very~e.0))'''

        amr2 = '''(a2 / attract-01~e.1 
                    :unk-label (m / much~e.0)
                    :degree (v / very~e.0))'''

        smatch = compute_smatch([amr], [amr2])

        self.assertEqual(1.0, smatch[SmatchScore.PRECISION])
        self.assertEqual(0.67, smatch[SmatchScore.RECALL].__round__(2))
        self.assertEqual(0.8, smatch[SmatchScore.F_SCORE])

    def test_compute_multiclass_f_score(self):
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

        rel_dict = {
            'pad': 0,
            'ARG0': 1,
            'ARG1': 2,
            'neg': 3,
            'no-rel': 4,
        }

        f_score, precision, recall = compute_multiclass_f_score(gold, pred, rel_dict, len(rel_dict.keys()))
        self.assertEqual(0.57, f_score.__round__(2))
        self.assertEqual(0.57, precision.__round__(2))
        self.assertEqual(0.57, recall.__round__(2))
