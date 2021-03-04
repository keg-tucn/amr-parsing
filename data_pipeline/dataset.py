from typing import List
import torch
from torch.utils.data import Dataset, DataLoader

from torch.nn.functional import pad as torch_pad
import penman
from penman.models import noop
from data_pipeline.training_entry import TrainingEntry
from data_pipeline.data_reading import extract_triples, get_paths
from data_pipeline.vocab import Vocabs

PAD = '<pad>' 
UNK = '<unk>'
EOS = '<eos>'
PAD_IDX = 0

def add_eos(training_entry: TrainingEntry, eos_token: str):
  training_entry.sentence.append(eos_token)
  training_entry.concepts.append(eos_token)

def numericalize(training_entry: TrainingEntry,
                 vocabs: Vocabs):
  """
  Processes the train entry into lists of integeres that can be easily converted
  into tensors. For the adjacency matrix 0 will be used in case the relation
  does not exist (is None).
  Args:
    vocabs: Vocabs object with the 3 vocabs (tokens, concepts, relations).
  Returns a tuple of:
    sentece: List of token indices.
    concepts: List of concept indices.
    adj_mat: Adjacency matrix which contains arc labels indices in the vocab.
  """
  # Process sentence.
  processed_sentence = [vocabs.get_token_idx(t) for t in training_entry.sentence]
  # Process concepts.
  processed_concepts = [vocabs.get_concept_idx(c) for c in training_entry.concepts]
  # Process adjacency matrix.
  processed_adj_mat = []
  for row in training_entry.adjacency_mat:
    processed_row = [0 if r is None else vocabs.get_relation_idx(r) for r in row]
    processed_adj_mat.append(processed_row)
  return processed_sentence, processed_concepts, processed_adj_mat

class AMRDataset(Dataset):
  """
  Dataset of sentence - amr entries, where the amrs are represented as a list
  of concepts and adjacency matrix.
  Arguments:
    paths: data paths.
    vocabs: the 3 vocabs (tokens, concepts, relations).
    device: cpu or cuda.
    seq2seq_setting: If true only the data for the seq2seq setting is returned
      (sequence of tokens with their lengths and concepts).
    ordered: if True the entries are ordered (decreasingly) by sentence length.
  """

  def __init__(self, paths: List[str], vocabs: Vocabs,
               device: str, seq2seq_setting: bool,
               ordered: bool,
               max_sen_len: bool = None):
    super(AMRDataset, self).__init__()
    self.device = device
    self.seq2seq_setting = seq2seq_setting
    self.sentences_list= []
    self.concepts_list = []
    self.adj_mat_list = []
    for path in paths:
      triples = extract_triples(path)
      for triple in triples:
        id, sentence, amr_str = triple
        amr_penman_graph = penman.decode(amr_str, model=noop.model)
        training_entry = TrainingEntry(
          sentence=sentence.split(),
          g=amr_penman_graph,
          unalignment_tolerance=1)
        # Process the training entry (add EOS for sentence and concepts).
        add_eos(training_entry, EOS)
        # Numericalize the training entry (str -> vocab ids).
        sentence, concepts, adj_mat = numericalize(training_entry, vocabs)
        # Convert to pytorch tensors.
        #TODO: should I use pytorch or numpy tensors?
        sentence = torch.tensor(sentence, dtype=torch.long)
        concepts = torch.tensor(concepts, dtype=torch.long)
        adj_mat = torch.tensor(adj_mat, dtype=torch.long)
        # Collect the data.
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
    return self.sentences_list[item], self.concepts_list[item],  self.adj_mat_list[item]

  def collate_fn(self, batch):
    batch_sentences = []
    batch_concepts = []
    batch_adj_mats = []
    sentence_lengths = []
    for entry in batch:
      sentence, concepts, adj_mat = entry
      batch_sentences.append(sentence)
      batch_concepts.append(concepts)
      batch_adj_mats.append(adj_mat)
      sentence_lengths.append(len(sentence))
    # Get max lengths for padding.
    max_sen_len = max([len(s) for s in batch_sentences])
    max_concepts_len = max([len(s) for s in batch_concepts])
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
      # Since it's a square matrix, the padding is the same on both dimensions.
      pad_size = max_adj_mat_size - len(adj_mat[0])
      padded_adj_mats.append(torch_pad(adj_mat, (0, pad_size, 0, pad_size)))
    #TODO: maybe by default do not put (seq_len, batch size) but have some
    # processing method for doing so after loading the data.
    if self.seq2seq_setting:
      new_batch = {
        'sentence': torch.transpose(torch.stack(padded_sentences),0,1).to(self.device),
        # This is left on the cpu for 'pack_padded_sequence'.
        'sentence_lengts': torch.tensor(sentence_lengths),
        'concepts': torch.transpose(torch.stack(padded_concepts),0,1).to(self.device)
      }
    else:
      new_batch = {
        'sentence': torch.transpose(torch.stack(padded_sentences),0,1).to(self.device),
        # This is left on the cpu for 'pack_padded_sequence'.
        'sentence_lengts': torch.tensor(sentence_lengths),
        'concepts': torch.transpose(torch.stack(padded_concepts),0,1).to(self.device),
        'adj_mat': torch.transpose(torch.stack(padded_adj_mats),0,1).to(self.device)
      }
    return new_batch

#TODO: remove this and add tests.
if __name__ == "__main__":
  subsets = ['bolt', 'cctv', 'dfa', 'dfb', 'guidelines',
             'mt09sdl', 'proxy', 'wb', 'xinhua']
  paths = get_paths('training', subsets)

  #TODO: a special token like 'no-relation' instead of None.
  special_words = ([PAD, UNK], [PAD, UNK], [PAD, UNK, None])
  vocabs = Vocabs(paths, UNK, special_words, min_frequencies=(1, 1, 1))

  dataset = AMRDataset(paths, vocabs)
  
  #TODO: see if thr bactching could somehow be done by size (one option
  # would be to order the elements in the dataset, have fixed batches and
  # somehow shuffle the batches instead of the elements themselves).
  dataloader = DataLoader(dataset, batch_size=3, collate_fn=dataset.collate_fn)

  i = 0
  for batch in dataloader:
    if i == 2:
      break
    i+=1
    print('Batch ',i)
    print(batch['sentence'].shape)