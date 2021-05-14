import torch
from enum import Enum


class SmatchScore(Enum):
    PRECISION = 1,
    RECALL = 2,
    F_SCORE = 3


def initialize_smatch():
    smatch = {
        SmatchScore.PRECISION: 0.0,
        SmatchScore.RECALL: 0.0,
        SmatchScore.F_SCORE: 0.0
    }
    return smatch


def to_pred_mat(mat: torch.tensor, sentence_len: int):
    unpadded_mat = mat[:sentence_len, :sentence_len]
    edges_mat = (unpadded_mat != 0).view(-1)
    return edges_mat


def compute_f_score(golds, preds):
    true_pos = (torch.eq(golds, preds).logical_and(golds)).sum().item()
    false_pos = (torch.eq(torch.logical_not(golds), preds).logical_and(golds)).sum().item()
    false_neg = (torch.eq(torch.logical_not(preds), golds).logical_and(golds)).sum().item()
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)

    f_score = 0
    if precision + recall != 0:
        f_score = 2 * (precision * recall) / (precision + recall)
    return f_score


def compute_accuracy(golds, preds):
    good_preds = torch.eq(golds, preds)
    correct_preds = good_preds.sum().item()
    no_entries = torch.numel(golds)
    return correct_preds / no_entries


def calc_edges_scores(gold_mat: torch.tensor, predictions: torch.tensor, inputs_lengths: torch.tensor):
    index = 0
    f_score = 0
    accuracy = 0
    for example in range(inputs_lengths.shape[0]):
        gold_edges = to_pred_mat(gold_mat[index], inputs_lengths[index].item())
        pred_edges = to_pred_mat(predictions[index], inputs_lengths[index].item())
        f_score += compute_f_score(gold_edges, pred_edges)
        accuracy += compute_accuracy(gold_edges, pred_edges)
        index += 1
    f_score = f_score / index
    accuracy = accuracy / index
    return f_score, accuracy
