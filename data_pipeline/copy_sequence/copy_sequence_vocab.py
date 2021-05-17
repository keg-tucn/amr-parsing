from typing import List
from collections import Counter


def build_vocab(words_counter: Counter,
                special_words: List[str],
                min_frequency: int):
    words_and_freq = sorted(
        words_counter.items(), key=lambda pair: pair[1], reverse=True)
    filtered_words_and_freq = [
        pair for pair in words_and_freq if pair[1] >= min_frequency]
    filtered_words = [wf[0] for wf in filtered_words_and_freq]
    vocab_words = filtered_words + special_words
    vocab = {word: i for i, word in enumerate(vocab_words)}
    return vocab


class CopySequenceVocab():

    def __init__(self,
                 concepts: Counter,
                 unkown_special_word: str,
                 special_words: List[str],
                 min_frequency: int):
        """
        Args:
        """
        token_vocab = build_vocab(concepts, special_words, min_frequency)
        self.unkown_special_word = unkown_special_word
        self.token_vocab = token_vocab
        self.token_vocab_size = len(token_vocab.keys())

    def get_token_idx(self, token: str):
        if token in self.token_vocab.keys():
            return self.token_vocab[token]
        return self.token_vocab[self.unkown_special_word]
