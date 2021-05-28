import torch
from torch.nn.functional import pad as torch_pad
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2

from collections import Counter
from data_pipeline.copy_sequence.copy_sequence_vocab import CopySequenceVocab
from data_pipeline.dataset import EOS, UNK, BOS
from constants import SENTENCE_KEY, SENTENCE_LEN_KEY, CONCEPTS_KEY


def add_bos(sentence):
    """
        Add '<bos>' at the beginning of sentence
    """
    sentence.insert(0, BOS)


def add_eos(sentence):
    """
        Add '<eos>' at the end of sentence
    """
    sentence.append(EOS)


def numericalize(raw_text_iter, vocab: CopySequenceVocab, tokenizer):
    """
      Convert dataset sentences into numericalized tensors
      Args:
        raw_text_iter: text iterator
        vocab (CopySequenceVocab): vocab to tokenize
        tokenizer: tokenizer object
      Return:
        sentences and concepts as iterable objects
    """
    data = []
    concepts = []
    word_data = []
    for item in raw_text_iter:
        # Remove headers
        if "= " not in item:
            word_sentence = []
            sentence = []
            decode_sentence = []
            for token in tokenizer(item):
                # Remove punctuation
                if token not in ",.'!?-()@=†\\s":
                    word_sentence.append(token)
            add_eos(word_sentence)
            sentence = torch.tensor([vocab.get_token_idx(token) for token in word_sentence],
                                    dtype=torch.long)
            add_bos(word_sentence)
            decode_sentence = torch.tensor([vocab.get_token_idx(token) for token in word_sentence],
                                           dtype=torch.long)
            data.append(sentence)
            concepts.append(decode_sentence)
            word_data.append(word_sentence)
    return filter(lambda t: t.numel() > 2, data), filter(lambda t: t.numel() > 2, concepts)


def denumericalize(sentences, vocab: CopySequenceVocab):
    """
      Converts numericalized sentences into word sentences
    """
    word_data = []
    for sentence in sentences:
        words = []
        for token in sentence:
            words.append(vocab.get_word_by_token_idx(token.numpy()))
        word_data.append(words)
    return word_data


def build_copy_vocab(dataset, special_words):
    """
        Function that builds Vocabulary for Copy Sequence Task
        Args:
          dataset: iterator object containing dataset paragraphs
          special_words: Special words in vocabulary
        Returns:
          CopySequenceVocab object
    """
    tokenizer = get_tokenizer('basic_english')
    # Hash Dictionary of words
    counter = Counter()
    # Line is a paragraph
    for line in dataset:
        # Tokenizer(line) does a split for English language
        counter.update(tokenizer(line))

    vocab = CopySequenceVocab(counter, UNK, special_words, 1)
    return vocab


class CopySequenceDataset(Dataset):
    """
      Dataset for Copy Sequence task, using Wikitext2
    """

    def __init__(self, vocab: CopySequenceVocab, tokenizer, max_sen_len: int = None):
        self.vocab = vocab
        self.tokenizer = get_tokenizer('basic_english')
        self.fields_by_id = {}
        self.ids = []
        id = 0
        train_iter, val_iter, test_iter = WikiText2()

        sentences, concepts = numericalize(
            train_iter, self.vocab, self.tokenizer)

        for sentence in sentences:
            concept = torch.cat(
                (torch.tensor([self.vocab.get_token_idx("<bos>")]), sentence))
            self.ids.append(id)
            field = {
                SENTENCE_KEY: torch.tensor(sentence, dtype=torch.long),
                CONCEPTS_KEY: torch.tensor(concept, dtype=torch.long),
            }
            self.fields_by_id[id] = field
            id = id + 1

        sorted_dict = sorted(self.fields_by_id.items(),
                             key=lambda item: len(item[1][SENTENCE_KEY]))
        # Filter them out by sentence length.
        # Retrieve the sorted amr ids.
        self.ids = [item[0] for item in sorted_dict]
        if max_sen_len is not None:
            self.ids = [id for id in self.ids
                        if (len(self.fields_by_id[id][SENTENCE_KEY]) <= max_sen_len)]
        concept_lengths = [len(self.fields_by_id[id][CONCEPTS_KEY])
                           for id in self.ids]
        self.max_concepts_length = max(concept_lengths)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        """Returns: id, sentence, concepts"""
        id = self.ids[item]
        sentence = self.fields_by_id[id][SENTENCE_KEY]
        concepts = self.fields_by_id[id][CONCEPTS_KEY]
        return id, sentence, concepts

    def copy_sequence_collate_fn(self, batch):
        """Splits sentences into batches"""
        batch_sentences = []
        sentence_lengths = []
        batch_concepts = []
        concepts_lengths = []
        for entry in batch:
            id, sentence, concepts = entry
            batch_sentences.append(sentence)
            batch_concepts.append(concepts)
            sentence_lengths.append(len(sentence))
            concepts_lengths.append(len(concepts))

        max_sen_len = max([len(s) for s in batch_sentences])
        max_concepts_len = max([len(s) for s in batch_concepts])
        padded_sentences = [
            torch_pad(s, (0, max_sen_len - len(s))) for s in batch_sentences]
        padded_concepts = [
            torch_pad(c, (0, max_concepts_len - len(c))) for c in batch_concepts]

        new_batch = {
            SENTENCE_KEY: torch.transpose(torch.stack(padded_sentences), 0, 1),
            SENTENCE_LEN_KEY: torch.tensor(sentence_lengths),
            CONCEPTS_KEY: torch.transpose(torch.stack(padded_concepts), 0, 1),
            "word_sentence": denumericalize(padded_sentences, self.vocab),
            "word_concepts": denumericalize(padded_concepts, self.vocab),
        }
        return new_batch


if __name__ == "__main__":

    train_iter, val_iter, test_iter = WikiText2()
    special_words = [BOS, EOS, "<extra_pad>"]

    vocab = build_copy_vocab(train_iter, special_words)
    dataset = CopySequenceDataset(vocab, train_iter)
    dataloader = DataLoader(dataset, batch_size=3,
                            collate_fn=dataset.copy_sequence_collate_fn)

    for batch in dataloader:
        print(batch)
