from config import get_default_config
from flask import Flask
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
from typing import List
import os
import os.path
import torch

from data_pipeline.vocab import read_cached_vocabs
from data_pipeline.glove_embeddings import GloVeEmbeddings
from models import RelationIdentification

from evaluation.tensors_to_amr import get_amr_str_from_tensors

from data_pipeline.dataset import PAD, EOS, UNK

APP = Flask(__name__)
CORS(APP)
API = Api(APP)

PRETRAINED_MODEL_PATH = 'temp/relation_identification.pth'
INPUT_ARG = 'input'
WITH_LABEL_ARG = 'with_labels'
OUTPUT_ARG = 'Prediction'
API_PATH = '/predict/relations'

def get_concept_idx(concept: str, vocab):
    if concept in vocab.keys():
        return vocab[concept]
    return vocab[UNK]

def numericalize(concepts: List[str], vocabs):
    processed_concepts = [get_concept_idx(c, vocabs) for c in concepts]
    return processed_concepts

class Predict(Resource):

    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument(INPUT_ARG)
        parser.add_argument(WITH_LABEL_ARG)

        token_vocab, concept_vocab, relation_vocab = read_cached_vocabs()
        cfg = get_default_config()
        glove_embeddings = GloVeEmbeddings(cfg.RELATION_IDENTIFICATION.GLOVE_EMB_DIM, UNK, [PAD, EOS, UNK])

        rel_identif = RelationIdentification(len(concept_vocab), len(relation_vocab), cfg.RELATION_IDENTIFICATION,
                                             glove_embeddings.embeddings_vocab)
        if (os.path.exists(PRETRAINED_MODEL_PATH)):
            MODEL = torch.load(PRETRAINED_MODEL_PATH, map_location=torch.device('cpu'))
            rel_identif.load_state_dict(MODEL)
        rel_identif.eval()
        args = parser.parse_args()

        with_labels = True if args[WITH_LABEL_ARG] == 'true' else False
        concepts = args[INPUT_ARG].split(',')
        concepts_len = torch.tensor([len(concepts)])
        concepts = numericalize(concepts, concept_vocab)
        concepts_list = torch.tensor(concepts, dtype=torch.long)
        concepts = concepts_list.unsqueeze(0)
        concepts = torch.transpose(concepts, 0, 1)

        try:
            _, prediction, _, labelled_prediction = rel_identif(concepts, concepts_len)
            amr_str = get_amr_str_from_tensors(
                concepts_list, concepts_len, prediction[0], concept_vocab, relation_vocab, ':unk-label')
            labelled_amr_str = get_amr_str_from_tensors(
                concepts_list, concepts_len, labelled_prediction[0], concept_vocab, relation_vocab, None)
            print(amr_str)
            print(labelled_amr_str)
            resulted_amr = labelled_amr_str if with_labels else amr_str
            out = {OUTPUT_ARG: resulted_amr}
            code = 200
        except:
            out = {'ERROR_CODE': 'FUCTIONAL',
                'ERROR_REASON': 'The given list of concepts cannot be converted into an AMR graph.'}
            code = 400
        return out, code


API.add_resource(Predict, API_PATH)

if __name__ == '__main__':
    APP.run(debug=True, port=1080)