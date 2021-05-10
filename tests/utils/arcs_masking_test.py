from absl.testing import absltest
import torch

from utils.config import get_default_config
from utils.arcs_masking import create_mask, create_sampling_mask, \
    create_fake_root_mask, create_padding_mask


class MaskingTest(absltest.TestCase):
    def test_create_padding_mask(self):
        max_seq_len = 5
        concept_lengths = torch.tensor([2, 3])
        expected_padding_mask = [
            [
                [True, True, False, False, False],
                [True, True, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False]
            ],
            [
                [True, True, True, False, False],
                [True, True, True, False, False],
                [True, True, True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False]
            ]
        ]
        expected_padding_mask = torch.tensor(expected_padding_mask)
        padding_mask = create_padding_mask(concept_lengths, max_seq_len)
        self.assertTrue(torch.equal(padding_mask, expected_padding_mask))

    def test_create_fake_root_mask(self):
        batch_size = 2
        seq_len = 5
        expected_fake_root_mask = [
            [
                [False, True, True, True, True],
                [False, True, True, True, True],
                [False, True, True, True, True],
                [False, True, True, True, True],
                [False, True, True, True, True]
            ],
            [
                [False, True, True, True, True],
                [False, True, True, True, True],
                [False, True, True, True, True],
                [False, True, True, True, True],
                [False, True, True, True, True]
            ]
        ]
        expected_fake_root_mask = torch.tensor(expected_fake_root_mask)
        fake_root_mask = create_fake_root_mask(batch_size, seq_len)
        self.assertTrue(torch.equal(fake_root_mask, expected_fake_root_mask))

    def test_sampling_mask(self):
        sampling_ratio = 2
        mat = [
            [
                [0, 0, 4, 0, 0, 0],
                [0, 0, 0, 3, 0, 0],
                [0, 8, 0, 2, 0, 0],
                [0, 0, 0, 0, 6, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 9, 0, 0],
                [0, 3, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [0, 4, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ]
        ]
        mat = torch.Tensor(mat)

        mask = [
            [
                [False, True, True, True, True, False],
                [False, True, True, True, True, False],
                [False, True, True, True, True, False],
                [False, True, True, True, True, False],
                [False, True, True, True, True, False],
                [False, False, False, False, False, False]
            ],
            [
                [False, True, True, True, False, False],
                [False, True, True, True, False, False],
                [False, True, True, True, False, False],
                [False, True, True, True, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False]
            ]
        ]
        mask = torch.Tensor(mask)

        final_mask = create_sampling_mask(mat, mask, sampling_ratio)
        entries_before_sampling = int(torch.count_nonzero(mask))
        entries_after_sampling = int(torch.count_nonzero(final_mask))
        self.assertTrue(entries_before_sampling > entries_after_sampling)

    def test_create_mask(self):
        gold_adj_mat = [
            [
                [0, 0, 4, 0, 0, 0],
                [0, 0, 0, 3, 0, 0],
                [0, 8, 0, 2, 0, 0],
                [0, 0, 0, 0, 6, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 9, 0, 0],
                [0, 3, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [0, 4, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ]
        ]
        gold_adj_mat = torch.Tensor(gold_adj_mat)
        concept_lengths = torch.tensor([2, 3])
        expected_mask = [
            [
                [False, True, False, False, False, False],
                [False, True, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False]
            ],
            [
                [False, True, True, False, False, False],
                [False, False, True, False, False, False],
                [False, True, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False]
            ]
        ]
        expected_mask = torch.tensor(expected_mask)
        cfg = get_default_config()
        mask = create_mask(gold_adj_mat, concept_lengths, cfg.HEAD_SELECTION)
        self.assertTrue(torch.equal(mask, expected_mask))


if __name__ == '__main__':
    absltest.main()
