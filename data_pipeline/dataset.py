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

class AMRDataset(Dataset):
  """
  Dataset of sentence - amr entries, where the amrs are represented as a list
  of concepts and adjacency matrix.
  """

  def __init__(self, paths: List[str], vocabs: Vocabs):
    super(AMRDataset, self).__init__()
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
        # Process the training entry (str -> vocab ids).
        sentence, concepts, adj_mat = training_entry.numericalize(
          vocabs.token_vocab, vocabs.concept_vocab, vocabs.relation_vocab)
        # Convert to pytorch tensors.
        #TODO: should I use pytorch or numpy tensors?
        sentence = torch.tensor(sentence, dtype=torch.int)
        concepts = torch.tensor(concepts, dtype=torch.int)
        adj_mat = torch.tensor(adj_mat, dtype=torch.int)
        # Collect the data.
        self.sentences_list.append(sentence)
        self.concepts_list.append(concepts)
        self.adj_mat_list.append(adj_mat)

  def __len__(self):
    return len(self.sentences_list)

  def __getitem__(self, item):
    return self.sentences_list[item], self.concepts_list[item],  self.adj_mat_list[item]

def collate_fn(batch):
  batch_sentences = []
  batch_concepts = []
  batch_adj_mats = []
  for entry in batch:
    sentence, concepts, adj_mat = entry
    batch_sentences.append(sentence)
    batch_concepts.append(concepts)
    batch_adj_mats.append(adj_mat)
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
  new_batch = {
    'sentence': torch.transpose(torch.stack(padded_sentences),0,1),
    'concepts': torch.transpose(torch.stack(padded_concepts),0,1),
    'adj_mat': torch.transpose(torch.stack(padded_adj_mats),0,1)
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
  
  dataloader = DataLoader(dataset, batch_size=3, collate_fn=collate_fn)

  i = 0
  for batch in dataloader:
    if i == 2:
      break
    i+=1
    print('Batch ',i)
    print(batch['sentence'].shape)