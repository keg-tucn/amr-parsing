from config import get_default_config
from flask import Flask
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
import os
import os.path
import torch

from data_pipeline.dummy.dummy_dataset import numericalize, build_dummy_vocab, denumericalize
from model.transformer import TransformerSeq2Seq
from train_concept_identification import build_dummy_vocab

APP = Flask(__name__)
CORS(APP)
API = Api(APP)

load_path = 'temp/pretrained_model.pth'
vocabs = build_dummy_vocab()
cfg = get_default_config()

transf = TransformerSeq2Seq(vocabs.token_vocab_size, 
                            vocabs.concept_vocab_size, 
                            vocabs.bos_idx, 
                            cfg.CONCEPT_IDENTIFICATION.TRANSF_BASED)
if(os.path.exists(load_path)):
  MODEL = torch.load(load_path, map_location=torch.device('cpu'))
  transf.load_state_dict(MODEL)
transf.eval()   

class Predict(Resource):

    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument('input')

        args = parser.parse_args()  # creates dict

        vocabs = build_dummy_vocab()

        input = [char for char in args['input']]
        max_out_len = len(input)
        gold_output = [char for char in input[::-1]]

        input, gold_output = numericalize(input, gold_output, vocabs )
        print(input)
        input = torch.tensor(input, dtype=torch.long)
        print(input)
        input = input.unsqueeze(0)
        input = torch.transpose(input, 0, 1)
        print(input)
        print(input.shape)
        gold_output = torch.tensor(gold_output, dtype=torch.long)

        # convert input to array
        # try: 
        _, prediction = transf(input_sequence=input, max_out_length=max_out_len)
        print(prediction.squeeze().tolist())
        # print(prediction.flatten(start_dim=0, end_dim=1).shape)
        prediction = denumericalize(prediction.tolist(), vocabs)
        print(prediction)
        result = ""
        for letter in prediction:
          result += letter[0]
        out = {'Prediction': result}
        return out, 200

        # except: 
        #   print("An error occured")
        #   return "ERROR", 500


API.add_resource(Predict, '/predict/synthetic')

if __name__ == '__main__':
    APP.run(debug=True, port='1080')