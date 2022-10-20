from yacs.config import CfgNode as CN

_C = CN()

# Configs for Concept Identification.
_C.CONCEPT_IDENTIFICATION = CN()
# Save Path
_C.CONCEPT_IDENTIFICATION.SAVE_PATH = ''
# Load Path
_C.CONCEPT_IDENTIFICATION.LOAD_PATH = ''
# Persisted Component
_C.CONCEPT_IDENTIFICATION.PERSISTED_COMPONENT = ''
# input sequence to output sequence or input sequence to input sequence
_C.CONCEPT_IDENTIFICATION.COPY_SEQUENCE = False
# Steps for pretraining
_C.CONCEPT_IDENTIFICATION.PRETRAIN_STEPS = 15
# Configs for LSTM-based Concept Identification.
_C.CONCEPT_IDENTIFICATION.LSTM_BASED = CN()
# Embeddings dimension.
_C.CONCEPT_IDENTIFICATION.LSTM_BASED.EMB_DIM = 512
# Glove embeddings dimension.
_C.CONCEPT_IDENTIFICATION.LSTM_BASED.GLOVE_EMB_DIM = 300
# Character-Level embeddings dimension.
_C.CONCEPT_IDENTIFICATION.LSTM_BASED.CHAR_EMB_DIM = 5
# Hidden size.
_C.CONCEPT_IDENTIFICATION.LSTM_BASED.HIDDEN_SIZE = 512
# Character-Level hidden size.
_C.CONCEPT_IDENTIFICATION.LSTM_BASED.CHAR_HIDDEN_SIZE = 124
# Lemma embeddings hidden size
_C.CONCEPT_IDENTIFICATION.LSTM_BASED.LEMMA_EMB_DIM = 5
# Number of layers.
_C.CONCEPT_IDENTIFICATION.LSTM_BASED.NUM_LAYERS = 1
# Drop-out encoder.
_C.CONCEPT_IDENTIFICATION.LSTM_BASED.ENCODER_DROPOUT_RATE = 0.1
# Drop-out decoder.
_C.CONCEPT_IDENTIFICATION.LSTM_BASED.DECODER_DROPOUT_RATE = 0.4
# Pointer generator
_C.CONCEPT_IDENTIFICATION.LSTM_BASED.USE_POINTER_GENERATOR = False
# Configs for Transformer-based Concept Identification.
_C.CONCEPT_IDENTIFICATION.TRANSF_BASED = CN()
# Configs for Transformer-based Concept Identification.
# Embeddings dimension.
_C.CONCEPT_IDENTIFICATION.TRANSF_BASED.EMB_DIM = 512
# Hidden size
_C.CONCEPT_IDENTIFICATION.TRANSF_BASED.HIDDEN_SIZE = 2048
# Number of layers.
_C.CONCEPT_IDENTIFICATION.TRANSF_BASED.NUM_LAYERS = 6
# Number of heads.
_C.CONCEPT_IDENTIFICATION.TRANSF_BASED.NUM_HEADS = 8
# Drop-out.
_C.CONCEPT_IDENTIFICATION.TRANSF_BASED.DROPOUT_RATE = 0.1
# Maximum Positional Encodings Lenght
_C.CONCEPT_IDENTIFICATION.TRANSF_BASED.MAX_POS_ENC_LEN = 500

_C.RELATION_IDENTIFICATION = CN()
# Embeddings dimension.
_C.RELATION_IDENTIFICATION.EMB_DIM = 150
# Glove embeddings dimension.
_C.RELATION_IDENTIFICATION.GLOVE_EMB_DIM = 50
# Character-Level embeddings dimension.
_C.RELATION_IDENTIFICATION.CHAR_EMB_DIM = 0
# Hidden size.
_C.RELATION_IDENTIFICATION.HIDDEN_SIZE = 512
# Character-Level hidden size.
_C.RELATION_IDENTIFICATION.CHAR_HIDDEN_SIZE = 0
# Lemma embeddings hidden size
_C.RELATION_IDENTIFICATION.LEMMA_EMB_DIM = 0
# Number of layers.
_C.RELATION_IDENTIFICATION.NUM_LAYERS = 3
# MLP Hidden size.
_C.RELATION_IDENTIFICATION.DENSE_MLP_HIDDEN_SIZE = 512
# Drop-out.
_C.RELATION_IDENTIFICATION.ENCODER_DROPOUT_RATE = 0.2
# Edge threshold.
_C.RELATION_IDENTIFICATION.EDGE_THRESHOLD = 0.6
# Sampling ratio.
_C.RELATION_IDENTIFICATION.SAMPLING_RATIO = 3
# Positive class weight
_C.RELATION_IDENTIFICATION.POSITIVE_CLASS_WEIGHT = 7.0
# Negative class weight
_C.RELATION_IDENTIFICATION.NEGATIVE_CLASS_WEIGHT = 10.0
# Logging start epoch for training flow.
_C.RELATION_IDENTIFICATION.LOGGING_START_EPOCH_TRAIN = 30
# Logging start epoch for evaluation flow.
_C.RELATION_IDENTIFICATION.LOGGING_START_EPOCH_DEV = 1
# Config for which module to train
_C.RELATION_IDENTIFICATION.HEADS_SELECTION = True
_C.RELATION_IDENTIFICATION.ARCS_LABELLING = True


def get_default_config():
  return _C.clone()