from typing import List
import torch
from torch.utils.data import Dataset

from torch.nn.functional import pad as torch_pad
from data_pipeline.dataset import EOS, BOS, numericalize
from data_pipeline.dummy.dummy_training_entry import DummyTrainingEntry
from data_pipeline.dummy.dummy_vocab import DummyVocabs



def add_eos(training_entry: DummyTrainingEntry, eos_token: str):
  """
    Add EOS token to DummyTrainingEntry
  """
  training_entry.sentence.append(eos_token)
  training_entry.concepts.append(eos_token)

def add_bos(training_entry: DummyTrainingEntry, bos_token: str):
  """
    Add BOS token to DummyTrainingEntry
  """
  training_entry.sentence.insert(0, bos_token)
  training_entry.concepts.insert(0, bos_token)

class DummySeq2SeqDataset(Dataset):
  """
  Dataset of sentence - amr entries, where the amrs are represented as a list
  of concepts and adjacency matrix.
  Arguments:
    sentences: dummy random sentences.
    vocabs: the 3 dummy vocabs (tokens, concepts, relations).
    device: cpu or cuda.
    seq2seq_setting: If true only the data for the seq2seq setting is returned
      (sequence of tokens with their lengths and concepts).
    ordered: if True the entries are ordered (decreasingly) by sentence length.
  """

  def __init__(self, sentences: List[str],
               vocabs: DummyVocabs,
               device: str,
               seq2seq_setting: bool = True,
               ordered: bool = True,
               max_sen_len: bool = None):
    super(DummySeq2SeqDataset, self).__init__()
    self.device = device
    self.seq2seq_setting = seq2seq_setting
    self.sentences_list = []
    self.concepts_list = []
    self.adj_mat_list = []
    self.ids = []
    self.amr_strings_by_id = {}
    i = 0
    for sentence in sentences:
      print("dummy sentence: ", sentence)
      self.amr_strings_by_id[i] = []
      i = i + 1
      training_entry = DummyTrainingEntry(
          sentence=sentence,
          unalignment_tolerance=1)
      # Process the training entry (add EOS for sentence and concepts).
      if self.seq2seq_setting:
        add_bos(training_entry, BOS)
        add_eos(training_entry, EOS)
      # Numericalize the training entry (str -> vocab ids).
      sentence, concepts, adj_mat = numericalize(training_entry, vocabs)
      print('sentence numericalized', sentence)
      print('concepts numericalized', concepts)
      # Convert to pytorch tensors.
      sentence = torch.tensor(sentence, dtype=torch.long)
      concepts = torch.tensor(concepts, dtype=torch.long)
      adj_mat = torch.tensor(adj_mat, dtype=torch.long)
      # Collect the data.
      self.ids.append(i)
      self.sentences_list.append(sentence)
      self.concepts_list.append(concepts)
      self.adj_mat_list.append(adj_mat)
    # Order them by sentence length.
    if ordered:
      zipped = list(zip(self.sentences_list, self.concepts_list, self.adj_mat_list))
      zipped.sort(key=lambda elem: len(elem[0]), reverse=True)
      ordered_lists = zip(*zipped)
      self.sentences_list, self.concepts_list, self.adj_mat_list = ordered_lists
    # Filter them out by sentence length.
    if max_sen_len is not None:
      lengths = [len(s) for s in self.sentences_list]
      self.sentences_list = [
          self.sentences_list[i] for i in range(len(lengths)) if lengths[i] <= max_sen_len]
      self.concepts_list = [
          self.concepts_list[i] for i in range(len(lengths)) if lengths[i] <= max_sen_len]
      self.adj_mat_list = [
          self.adj_mat_list[i] for i in range(len(lengths)) if lengths[i] <= max_sen_len]
    # Get max no of concepts.
    concept_lengths = [len(c) for c in self.concepts_list]
    self.max_concepts_length = max(concept_lengths)

  def __len__(self):
    return len(self.sentences_list)

  def __getitem__(self, item):
    return self.ids[item], self.sentences_list[item], self.concepts_list[item],  self.adj_mat_list[item]

  def collate_fn(self, batch):
    batch_sentences = []
    batch_concepts = []
    batch_adj_mats = []
    sentence_lengths = []
    concepts_lengths = []
    for entry in batch:
      amr_id, sentence, concepts, adj_mat = entry
      batch_sentences.append(sentence)
      batch_concepts.append(concepts)
      if adj_mat.nelement():
        batch_adj_mats.append(adj_mat)
      sentence_lengths.append(len(sentence))
      concepts_lengths.append(len(concepts))
    # Get max lengths for padding.
    max_sen_len = max([len(s) for s in batch_sentences])
    max_concepts_len = max([len(s) for s in batch_concepts])
    if batch_adj_mats:
      max_adj_mat_size = max([len(s) for s in batch_adj_mats])
    # Pad sentences.
    padded_sentences = [
        torch_pad(s, (0, max_sen_len - len(s))) for s in batch_sentences]
    # Pad concepts
    padded_concepts = [
        torch_pad(c, (0, max_concepts_len - len(c))) for c in batch_concepts]
    # Pad adj matrices (pad on both dimensions).
    padded_adj_mats = []
    for adj_mat in batch_adj_mats:
      if adj_mat.nelement():
        # Since it's a square matrix, the padding is the same on both dimensions.
        pad_size = max_adj_mat_size - len(adj_mat[0])
        padded_adj_mats.append(torch_pad(adj_mat, (0, pad_size, 0, pad_size)))
    if self.seq2seq_setting:
      new_batch = {
          'sentence': torch.transpose(torch.stack(padded_sentences), 0,1).to(self.device),
          # This is left on the cpu for 'pack_padded_sequence'.
          'sentence_lengts': torch.tensor(sentence_lengths),
          'concepts': torch.transpose(torch.stack(padded_concepts), 0,1).to(self.device)
      }
    else:
      new_batch = {
          'amr_id': amr_id,
          'concepts': torch.transpose(torch.stack(padded_concepts), 0,1).to(self.device),
          # This is left on the cpu for 'pack_padded_sequence'.
          'concepts_lengths': torch.tensor(concepts_lengths),
          'adj_mat': torch.stack(padded_adj_mats).to(self.device)
      }
    return new_batch
