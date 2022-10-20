import os
import sys
import torch
from typing import List, Dict

import definitions
import numpy as np
import pickle

GLOVE_PATH = 'temp/glove'
GLOVE_CACHE_FILE = 'glove.pickle'
GLOVE_WORD_2_IDX_CACHE_FILE = 'word2idx.pickle'
SENSE_IDENTIFICATOR = '-0'
ERROR_MESSAGE = 'Incorrect GloVe embeddings dimension {}. Choose between 50, 100, 200 or 300.'

glove_dimensions = {
    50: 'glove.6B.50d.txt',
    100: 'glove.6B.100d.txt',
    200: 'glove.6B.200d.txt',
    300: 'glove.6B.300d.txt'
}

cached_files = {
    50: 'glove50',
    100: 'glove100',
    200: 'glove200',
    300: 'glove300'
}

def get_data_file(emb_dim: int):
    return glove_dimensions.get(emb_dim, ERROR_MESSAGE)

def get_cache_paths(dim: int):
    dim_file_path = cached_files.get(dim, ERROR_MESSAGE)
    if dim_file_path == ERROR_MESSAGE:
        return list([])
    cache_dir = os.path.join(definitions.PROJECT_ROOT_DIR, GLOVE_PATH, dim_file_path)
    cache_files = [GLOVE_CACHE_FILE, GLOVE_WORD_2_IDX_CACHE_FILE]
    cache_paths = [os.path.join(cache_dir, f) for f in cache_files]
    return cache_paths


def check_cached_data(dim: int):
    cache_paths = get_cache_paths(dim)
    if len(cache_paths) == 0:
        return None
    data = []
    for cache_path in cache_paths:
        if os.path.isfile(cache_path):
            with open(cache_path, 'rb') as data_file:
                data_extracted = pickle.load(data_file)
                data.append(data_extracted)
        else:
            return None
    return data[0], data[1]


def cache_data(glove_embs: Dict, word2idx: Dict, dim: int):
    dim_file_path = cached_files.get(dim)
    cache_dir = os.path.join(definitions.PROJECT_ROOT_DIR, GLOVE_PATH, dim_file_path)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_paths = get_cache_paths(dim)
    data = [glove_embs, word2idx]
    for i in range(len(cache_paths)):
        with open(cache_paths[i], 'wb') as vocab_file:
            pickle.dump(data[i], vocab_file)


def get_weight_matrix(glove_embeddings: Dict, dim: int):
    no_glove_embs = len(glove_embeddings.keys())
    weight_matrix = torch.zeros((no_glove_embs, dim))
    for (key, value) in glove_embeddings.items():
        weight_matrix[key] = torch.tensor(value)
    return weight_matrix


class GloVeEmbeddings:

    def __init__(self, emb_dim: int, unk_word: str, special_words: List[str]):
        self.unknown_special_word = unk_word
        self.embeddings_vocab, self.word2idx = \
            self.extract_embeddings(emb_dim, unk_word, special_words)

    @staticmethod
    def extract_embeddings(emb_dim: int, unknown_special_word: str, special_words: List[str]):
        data = check_cached_data(emb_dim)
        if data is not None:
            print("Found cached data for GloVe Embeddings dim {}.".format(emb_dim))
            return data

        path = os.path.join(definitions.PROJECT_ROOT_DIR, definitions.GLOVE_PATH)
        glove_data_file = get_data_file(emb_dim)
        if glove_data_file == ERROR_MESSAGE:
            sys.exit(ERROR_MESSAGE.format(emb_dim))

        words = []
        idx = 0
        word2idx = {}
        glove_embeddings = {}

        # Add special words first
        for word in special_words:
            words.append(word)
            word2idx[word] = idx
            idx += 1
            embedding = np.zeros(emb_dim)
            glove_embeddings[word] = embedding

        # Then add words from the file with their corresponding embeddings
        with open(f'{path}/{glove_data_file}', 'rb') as file:
            for read_data in file:
                line = read_data.decode().split()
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                embedding = np.array(line[1:]).astype(np.float)
                glove_embeddings[word] = embedding

        # Change embeddings value for unknown special word to the mean of all embeddings
        glove_embeddings[unknown_special_word] = np.mean(np.array(list(glove_embeddings.values())), axis=0)

        # Obtain the GloVe dictionary with indexes
        glove = {word2idx[w]: glove_embeddings[w] for w in words}

        cache_data(glove, word2idx, emb_dim)
        print('Created and cached data for GloVe Embeddings dim {}.'.format(emb_dim))
        return glove, word2idx

    def get_glove_concept_idx(self, concept: str):
        if SENSE_IDENTIFICATOR in concept:
            a = concept.find(SENSE_IDENTIFICATOR)
            concept = concept[:a]
        if concept in self.word2idx.keys():
            return self.word2idx[concept]
        return self.word2idx[self.unknown_special_word]
