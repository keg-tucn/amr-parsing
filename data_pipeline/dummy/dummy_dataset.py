from typing import List
import torch
import string
import random
from torch.utils.data import Dataset

from torch.nn.functional import pad as torch_pad
from data_pipeline.dataset import PAD, EOS, BOS, UNK
from data_pipeline.dummy.dummy_vocab import DummyVocabs


def generate_sentences(len: int, number_of_sentences: int):
    """
      Generates random sentences of length len for dummy dataset.
      Args:
        len (int): length of sentences
        number_of_sentences (int): number of sentences to generate
      Returns:
        Train Sentences, Test Sentences, Dummy Vocabs
    """
    all_sentences = []
    for i in range(number_of_sentences):
        letters = string.ascii_lowercase
        sentence = ''.join(random.choice(letters) for i in range(len))
        all_sentences.append(sentence)
    print("all training sentences", all_sentences)

    return all_sentences


def add_eos(sentence, concepts, eos_token: str):
    """ Add EOS token to DummyTrainingEntry """
    sentence.append(eos_token)
    concepts.append(eos_token)


def add_bos(concepts, bos_token: str):
    """Add BOS token to DummyTrainingEntry """
    concepts.insert(0, bos_token)


def numericalize(sentence: List[str],
                 concepts: List[str],
                 vocabs: DummyVocabs):
    """
    Processes sentence and concepts lists of integers to be converted as tensors.
    Args:
      sentence (List[str]): Sentence to be numericalized
      concepts (List[str]): Corresponding concepts to be numericalized
      vocabs (DummyVocabs): Vocabs object with the 3 vocabs (tokens, concepts, relations).
    Returns a tuple of:
      sentece (List[int]): List of token indices.
      concepts (List[int]): List of concept indices.
    """
    processed_sentence = [vocabs.get_token_idx(t) for t in sentence]
    processed_concepts = [vocabs.get_concept_idx(c) for c in concepts]
    return processed_sentence, processed_concepts


def build_dummy_vocab():
    """
        Function that builds Vocabulary for Dummy Dataset 
        Returns:
          DummyVocabs object
    """
    special_words = ([PAD, BOS, EOS, UNK], [PAD, BOS, EOS, UNK])
    letters = string.ascii_lowercase
    vocabs = DummyVocabs(letters, UNK, special_words,
                         min_frequencies=(1, 1))
    return vocabs


class DummySeq2SeqDataset(Dataset):
    """
    Dataset of sentence - amr entries, where the amrs are represented as a list
    of concepts and adjacency matrix.
    Arguments:
      vocabs (DummyVocabs): alphabet vocabulary
      dataset_size (int): number of random sentences
      sentence_len (int): length of randomly generated sentences
      max_sen_len (int): maximum sentence length
      ordered (bool): if True the entries are ordered (decreasingly) by sentence length.
    """

    def __init__(self,
                 vocabs: DummyVocabs,
                 dataset_size: int,
                 sentence_length: int,
                 max_sen_len: int = None,
                 ordered: bool = True):
        super(DummySeq2SeqDataset, self).__init__()
        self.sentences_list = []
        self.concepts_list = []
        self.ids = []
        i = 0
        sentences = generate_sentences(sentence_length, dataset_size)
        self.vocabs = vocabs
        for sentence in sentences:
            print("dummy sentence: ", sentence)
            i = i + 1
            concepts = [char for char in sentence[::-1]]
            tokens = [char for char in sentence]
            # Process the training entry (add EOS for sentence and concepts).
            add_eos(tokens, concepts, EOS)
            add_bos(tokens, BOS)
            sentence, concepts = numericalize(tokens, concepts, vocabs)
            # Convert to pytorch tensors.
            sentence = torch.tensor(sentence, dtype=torch.long)
            concepts = torch.tensor(concepts, dtype=torch.long)
            self.ids.append(i)
            self.sentences_list.append(sentence)
            self.concepts_list.append(concepts)
        # Order them by sentence length.
        if ordered:
            zipped = list(
                zip(self.sentences_list, self.concepts_list))
            zipped.sort(key=lambda elem: len(elem[0]), reverse=True)
            ordered_lists = zip(*zipped)
            self.sentences_list, self.concepts_list = ordered_lists
        # Filter them out by sentence length.
        if max_sen_len is not None:
            lengths = [len(s) for s in self.sentences_list]
            self.sentences_list = [
                self.sentences_list[i] for i in range(len(lengths)) if lengths[i] <= max_sen_len]
            self.concepts_list = [
                self.concepts_list[i] for i in range(len(lengths)) if lengths[i] <= max_sen_len]
        concept_lengths = [len(c) for c in self.concepts_list]
        self.max_concepts_length = max(concept_lengths)

    def __len__(self):
        return len(self.sentences_list)

    def __getitem__(self, item):
        """Returns: id, sentence, concepts"""
        return self.ids[item], self.sentences_list[item], self.concepts_list[item]

    def collate_fn(self, batch):
        """Splits sentences into batches"""
        batch_sentences = []
        batch_concepts = []
        sentence_lengths = []
        concepts_lengths = []
        for entry in batch:
            amr_id, sentence, concepts = entry
            batch_sentences.append(sentence)
            batch_concepts.append(concepts)
            sentence_lengths.append(len(sentence))
            concepts_lengths.append(len(concepts))
        # Get max lengths for padding.
        max_sen_len = max([len(s) for s in batch_sentences])
        max_concepts_len = max([len(s) for s in batch_concepts])
        padded_sentences = [
            torch_pad(s, (0, max_sen_len - len(s))) for s in batch_sentences]
        padded_concepts = [
            torch_pad(c, (0, max_concepts_len - len(c))) for c in batch_concepts]

        new_batch = {
            'sentence': torch.transpose(torch.stack(padded_sentences), 0, 1),
            'sentence_lengts': torch.tensor(sentence_lengths),
            'concepts': torch.transpose(torch.stack(padded_concepts), 0, 1)
        }

        return new_batch
