import torch
import io

from enum import Enum
from smatch import score_amr_pairs
from data_pipeline.vocab import Vocabs


class SmatchScore(Enum):
    '''
    This enum is used for readability purposes when computing the smatch score.
    It comprises of 3 metrics:
        - precision
        - recall
        - f_score
    '''
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


def compute_smatch(gold_outputs, predictions):
  """
  Args:
    gold_outputs: list of gold amrs as strings
    predictions: list of predictions of amrs as strings

  Returns:
    smatch: the smatch score is composed of 3 metrics:
      - precision
      - recall
      - best f-score
  """
  gold_outputs = ' \n\n '.join(gold_outputs)
  predictions = ' \n\n '.join(predictions)

  gold_file = io.StringIO(gold_outputs)
  pred_file = io.StringIO(predictions)

  smatch_score = initialize_smatch()
  try:
    smatch_score[SmatchScore.PRECISION], smatch_score[SmatchScore.RECALL], smatch_score[SmatchScore.F_SCORE] = \
      next(score_amr_pairs(gold_file, pred_file))
  except (AttributeError, ValueError) as e:
    print('Something went wrong went calculating smatch.')

  return smatch_score


def unpad_mat_to_list(mat: torch.tensor, sentence_len: int):
    '''
    This functions transforms the given adjacency matrix in a list of booleans
    which determine the existent edges, by removing the padding.
    Args:
        mat: adjacency matrix (seq len, seq len) with padding
        sentence_len: length of the concepts (to determine where to remove padding)
    Returns:
        A list of booleans of the relations between edges.
    '''
    unpadded_mat = mat[:sentence_len, :sentence_len]
    edges_mat = (unpadded_mat != 0).view(-1)
    return edges_mat


def compute_f_score(golds, preds):
    '''
    Params:
        golds: gold adjacency matrix
        preds: boolean adjacency matrix of the predictions

    Returns:
        f_score, precision and recall between the expected relations (gold)
        and the actual predictions (preds)
    '''
    true_pos = (preds.logical_and(golds)).sum().item()
    false_pos = (preds.logical_and(torch.logical_not(golds))).sum().item()
    false_neg = (torch.logical_not(preds).logical_and(golds)).sum().item()
    no_pos_preds = true_pos + false_pos
    precision = true_pos / no_pos_preds if no_pos_preds != 0 else 0.0
    no_pos_golds = true_pos + false_neg
    recall = true_pos / no_pos_golds if no_pos_golds != 0 else 0.0

    f_score = 0
    if precision + recall != 0:
        f_score = 2 * (precision * recall) / (precision + recall)
    return f_score, precision, recall


def compute_accuracy(golds, preds):
    '''
    Params:
        golds: gold adjacency matrix
        preds: boolean adjacency matrix of the predictions

    Returns:
        The accuracy of the edge predictions, which is computed by calculating
        the number of correctly predicted edges over the total number of edges.
    '''
    good_preds = torch.eq(golds, preds)
    correct_preds = good_preds.sum().item()
    no_entries = torch.numel(golds)
    return correct_preds / no_entries


def calc_edges_scores(gold_mat: torch.tensor, predictions: torch.tensor, inputs_lengths: torch.tensor):
    '''
    Params:
        gold_mat: gold adjacency matrix
        predictions: boolean adjacency matrix of the predictions
        inputs_lengths: length of the sequence for each example

    Returns:
        F_score, precision, racall and accuracy
        between unpadded gold data and actual predictions
    '''
    index = 0
    f_score = 0
    accuracy = 0
    precision = 0
    recall = 0
    for example in range(inputs_lengths.shape[0]):
        gold_edges = unpad_mat_to_list(gold_mat[index], inputs_lengths[index].item())
        pred_edges = unpad_mat_to_list(predictions[index], inputs_lengths[index].item())
        aux_f_score, aux_precision, aux_recall = compute_f_score(gold_edges, pred_edges)
        accuracy += compute_accuracy(gold_edges, pred_edges)
        f_score += aux_f_score
        precision += aux_precision
        recall += aux_recall
        index += 1
    f_score = f_score / index
    recall = recall / index
    precision = precision / index
    accuracy = accuracy / index
    return f_score, precision, recall, accuracy


def compute_multiclass_f_score(golds, preds, rel_dictionary, no_rels):
    '''
    Params:
        golds: gold adjacency matrix
        preds: boolean adjacency matrix of the predictions
    Returns:
        f_score, precision and recall between the expected relations (gold)
        and the actual predictions (preds)
    '''
    f_score = 0
    precision = 0
    recall = 0
    for relation in rel_dictionary.values():
        rel_gold_mat = (golds == relation).view(-1)
        rel_pred_mat = (preds == relation).view(-1)
        rel_f_score, rel_precision, rel_recall = compute_f_score(rel_gold_mat, rel_pred_mat)
        f_score += rel_f_score
        precision += rel_precision
        recall += rel_recall

    f_score = f_score / no_rels
    precision = precision / no_rels
    recall = recall / no_rels

    return f_score, precision, recall

def calc_labels_scores(gold_mat: torch.Tensor, predictions: torch.Tensor, inputs_lengths: torch.Tensor, vocabs: Vocabs,
                       device):
    '''
    Params:
        gold_mat: gold adjacency matrix with indexes for labels
        predictions: adjacency matrix of the predictions
        inputs_lengths: length of the sequence for each example
        vocabs: for dictionary to decode labels from indexes to strings
    Returns:
        F_score, precision and recall
        between unpadded gold data and actual label predictions
    '''
    rel_dictionary = vocabs.relation_vocab
    no_rels = vocabs.relation_vocab_size
    index = 0
    f_score = 0
    precision = 0
    recall = 0
    for example in range(inputs_lengths.shape[0]):
        sentence_len = inputs_lengths[index].item()
        gold_edges = gold_mat[index][:sentence_len, :sentence_len].to(device)
        pred_edges = predictions[index][:sentence_len, :sentence_len].to(device)
        aux_f_score, aux_precision, aux_recall = compute_multiclass_f_score(
            gold_edges, pred_edges, rel_dictionary, no_rels)
        f_score += aux_f_score
        precision += aux_precision
        recall += aux_recall
        index += 1
    f_score = f_score / index
    recall = recall / index
    precision = precision / index
    return f_score, precision, recall
