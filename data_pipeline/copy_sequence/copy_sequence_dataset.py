import torch
from torch.nn.functional import pad as torch_pad
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2

from torch import Tensor
from collections import Counter
from data_pipeline.copy_sequence.copy_sequence_vocab import CopySequenceVocab
from data_pipeline.dataset import EOS, PAD, UNK
from data_pipeline.training_entry import ROOT
from constants import SENTENCE_KEY, SENTENCE_LEN_KEY, WORD_SENTENCE


def add_bos_eos(sentence):
    sentence.insert(0, ROOT)
    sentence.append(EOS)


def numericalize(raw_text_iter, vocab: CopySequenceVocab, tokenizer):
    data = []
    word_data = []
    for item in raw_text_iter:
        word_sentence = []
        sentence = []
        for token in tokenizer(item):
            word_sentence.append(token)
        add_bos_eos(word_sentence)
        sentence = torch.tensor([vocab.get_token_idx(token) for token in word_sentence],
                                    dtype=torch.long)
        data.append(sentence)
        word_data.append(word_sentence)
    return filter(lambda t: t.numel() > 2, data), filter(lambda t: t.numel() > 2, word_data)

def denumericalize(sentences, vocab):
    word_data = []
    for sentence in sentences:
        words = []
        for token in sentence:
            words.append(vocab.get_word_by_token_idx(token.numpy()))
        word_data.append(words)
    return word_data

def build_copy_vocab(dataset, tokenizer, special_words):
    # Hash Dictionary of words
    counter = Counter()
    # Line is a paragraph
    for line in dataset:
        # Tokenizer(line) does a split for English language
        counter.update(tokenizer(line))
    
    vocab = CopySequenceVocab(counter, UNK, special_words, 1)
    return vocab


class CopySequenceDataset(Dataset):
    def __init__(self, vocab: CopySequenceVocab, tokenizer, max_sen_len: bool = None):
        self.vocab = vocab
        self.tokenizer = get_tokenizer('basic_english')
        self.fields_by_id = {}
        self.ids = []
        id = 0
        train_iter, val_iter, test_iter = WikiText2()

        sentences, word_sentences = numericalize(train_iter, vocab, self.tokenizer)
        for sentence in sentences:
            self.ids.append(id)
            field = {SENTENCE_KEY: torch.tensor(sentence, dtype=torch.long)}
            self.fields_by_id[id] = field
            id = id + 1

        sorted_dict = sorted(self.fields_by_id.items(),
                             key=lambda item: len(item[1][SENTENCE_KEY]))
        # Retrieve the sorted amr ids.
        self.ids = [item[0] for item in sorted_dict]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        """Returns: id, sentence"""
        id = self.ids[item]
        sentence = self.fields_by_id[id][SENTENCE_KEY]
        return id, sentence

    def copy_sequence_collate_fn(self, batch):
        batch_sentences = []
        sentence_lengths = []

        for entry in batch:
            id, sentence = entry
            batch_sentences.append(sentence)
            sentence_lengths.append(len(sentence))

        max_sen_len = max([len(s) for s in batch_sentences])
        padded_sentences = [
            torch_pad(s, (0, max_sen_len - len(s))) for s in batch_sentences]

        new_batch = {
            SENTENCE_KEY: torch.transpose(torch.stack(padded_sentences), 0, 1),
            WORD_SENTENCE: denumericalize(padded_sentences, self.vocab),
            SENTENCE_LEN_KEY: torch.tensor(sentence_lengths)
        }
        return new_batch


if __name__ == "__main__":

    train_iter, val_iter, test_iter = WikiText2()
    tokenizer = get_tokenizer('basic_english')
    special_words = [PAD, EOS, UNK]

    vocab = build_copy_vocab(train_iter, tokenizer, special_words)
    dataset = CopySequenceDataset(vocab, train_iter)
    dataloader = DataLoader(dataset, batch_size=3,
                            collate_fn=dataset.copy_sequence_collate_fn)

    for batch in dataloader:
        print(batch)
