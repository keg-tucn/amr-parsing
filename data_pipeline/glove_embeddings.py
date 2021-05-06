import os
import torch
from typing import List, Dict

from utils import definitions
import numpy as np
import bcolz
import pickle

GLOVE_DATA_FILE = 'glove.6B.50d.txt'
GLOVE_VECTOR_FILE = '6B.50.dat'
GLOVE_PATH = 'temp/glove'
GLOVE_CACHE_FILE = 'glove.pickle'
GLOVE_WORD_2_IDX_CACHE_FILE = 'word2idx.pickle'
SENSE_IDENTIFICATOR = '-0'


def get_cache_paths():
    cache_dir = os.path.join(definitions.PROJECT_ROOT_DIR, GLOVE_PATH)
    cache_files = [GLOVE_CACHE_FILE, GLOVE_WORD_2_IDX_CACHE_FILE]
    cache_paths = [os.path.join(cache_dir, f) for f in cache_files]
    return cache_paths


def check_cached_data():
    cache_paths = get_cache_paths()
    data = []
    for cache_path in cache_paths:
        if os.path.isfile(cache_path):
            with open(cache_path, 'rb') as data_file:
                data_extracted = pickle.load(data_file)
                data.append(data_extracted)
        else:
            return None
    return data[0], data[1]


def cache_data(glove_embs: Dict, word2idx: Dict):
    cache_dir = os.path.join(definitions.PROJECT_ROOT_DIR, GLOVE_PATH)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_paths = get_cache_paths()
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
        data = check_cached_data()
        if data is not None:
            print("Found cached data for GloVe Embeddings.")
            return data

        path = os.path.join(definitions.PROJECT_ROOT_DIR, definitions.GLOVE_PATH)
        embeddings_dim = emb_dim
        words = []
        idx = 0
        word2idx = {}
        vectors = bcolz.carray(np.zeros(1), rootdir=f'{path}/{GLOVE_VECTOR_FILE}', mode='w')

        # Add special words first
        for word in special_words:
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.zeros(embeddings_dim)
            vectors.append(vect)

        # Then add words from the file with their corresponding embeddings
        with open(f'{path}/{GLOVE_DATA_FILE}', 'rb') as file:
            for read_data in file:
                line = read_data.decode().split()
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float)
                vectors.append(vect)

        vectors = bcolz.carray(vectors[1:].reshape((idx, embeddings_dim)),
                    rootdir=f'{path}/{GLOVE_VECTOR_FILE}', mode='w')
        vectors.flush()

        # Change embeddings value for unknown special word to the mean of all embeddings
        vectors[word2idx[unknown_special_word]] = np.mean(vectors, axis=0)

        # Obtain the GloVe dictionary
        glove = {word2idx[w]: vectors[word2idx[w]] for w in words}

        cache_data(glove, word2idx)
        print('Created and cached data for GloVe Embeddings.')
        return glove, word2idx

    def get_glove_concept_idx(self, concept: str):
        if SENSE_IDENTIFICATOR in concept:
            a = concept.find(SENSE_IDENTIFICATOR)
            concept = concept[:a]
        if concept in self.word2idx.keys():
            return self.word2idx[concept]
        return self.word2idx[self.unknown_special_word]
