import torch

from yacs.config import CfgNode


def create_padding_mask(concepts_lengths: torch.tensor, seq_len: int):
    """
    Args:
      concepts_lengths: Batch of concept sequence lengths (batch size).
      seq_len: Maximum seq length.

    Returns:
      Batch of padding masks, with shape (batch size, max len, max len), where
      each padding mask has False where the padding is and True otherwise, that
      is the submatrix of size (concept len, concept len) is all True.
    """
    arr_range = torch.arange(seq_len)
    # Create sequence mask.
    mask_1d = arr_range.unsqueeze(dim=0) < concepts_lengths.unsqueeze(dim=1)
    # Create 2d mask for each element in the batch by multiplying a repeated
    # vector with its transpose.
    x = mask_1d.unsqueeze(1).repeat(1, seq_len, 1)
    y = x.transpose(1, 2)
    mask = x * y
    return mask


def create_sampling_mask(gold_adj_mat: torch.tensor,
                         partial_mask: torch.tensor,
                         sampling_ratio: int = 0.5):
    """Create sampling mask (for balancing negative and positive classes).
    Args:
      gold_adj_mat: Gold adjacency matrix (batch size, seq len, seq len).
      partial_mask: Mask obtained by masking fake root and padding, so that
        the sampling process will not consider them again.
      sampling_ratio: How many negative edges should be sampled for a positive
        edge. We sample this at batch level.
    Returns:
      Mask of boolean values with shape (batch size, seq len, seq len).
    """
    no_positives = int(torch.count_nonzero(gold_adj_mat))
    no_samples = sampling_ratio * no_positives

    bin_adj_mat = torch.full(gold_adj_mat.shape, True)
    bin_adj_mat[gold_adj_mat == 0] = False

    sampling_candidates = torch.logical_and(torch.logical_not(bin_adj_mat), partial_mask)
    no_potential_candidates = int(torch.count_nonzero(sampling_candidates))

    prob_value = no_samples / no_potential_candidates if no_potential_candidates != 0 else 0.
    prob_value = min(1., prob_value)

    probabilities = torch.where(sampling_candidates, prob_value, 0.)
    selection = torch.bernoulli(probabilities)

    mask = (selection == 1.)
    return mask


def create_fake_root_mask(batch_size, seq_len, root_idx=0):
    """
    Args:
      batch_size Batch size.
      seq_len: Seq len.
      root_idx: Index of the root in the scores matrix.

    Returns:
      Boolean mask ensuring the root cannot be a child of another concept,
      which means the root column should be False.
    """
    mask = torch.full((batch_size, seq_len, seq_len), True)
    mask[:, :, root_idx] = False
    return mask


def create_mask(gold_adj_mat, concepts_lengths: torch.tensor, config: CfgNode):
    """
    Creates a mask for weighting the loss.
    This mask will be a mask of boolean values:
      False: masking out
      True: masking in
    Need to mask several things:
      padding -> the sequences of concepts are padded.
      sampling -> we don't want to use all non-existing arcs, we will "sample"
        from them by masking out the ones we don't want to use.
      fake root -> the fake root should not have any parent.

    Args:
      gold_adj_mat: Gold adj mat (matrix of relations), only sent on training
        of shape (batch size, seq len, seq len).
      concepts_lengths: Batch of concept sequence lengths (batch size).
      config: configuration file

    Returns mask of shape (batch size, seq len, seq len).
    """
    batch_size = concepts_lengths.shape[0]
    seq_len = gold_adj_mat.shape[1]
    mask = create_padding_mask(concepts_lengths, seq_len)
    fake_root_mask = create_fake_root_mask(batch_size, seq_len)
    mask = mask * fake_root_mask
    mask = create_sampling_mask(gold_adj_mat, mask, config.SAMPLING_RATIO)
    return mask
