from yacs.config import CfgNode as CN

_C = CN()

# Configs for Concept Identification.
_C.CONCEPT_IDENTIFICATION = CN()
# Configs for LSTM-based Concept Identification.
_C.CONCEPT_IDENTIFICATION.LSTM_BASED = CN()
# Embeddings dimension.
_C.CONCEPT_IDENTIFICATION.LSTM_BASED.EMB_DIM = 50
# Glove embeddings dimension.
_C.CONCEPT_IDENTIFICATION.LSTM_BASED.GLOVE_EMB_DIM = 50
# Hidden size.
_C.CONCEPT_IDENTIFICATION.LSTM_BASED.HIDDEN_SIZE = 40
# Number of layers.
_C.CONCEPT_IDENTIFICATION.LSTM_BASED.NUM_LAYERS = 1
# Configs for Transformer-based Concept Identification.
_C.CONCEPT_IDENTIFICATION.TRANSF_BASED = CN()
#TODO: add transformer based configs.

_C.RELATION_IDENTIFICATION = CN()
# Embeddings dimension.
_C.RELATION_IDENTIFICATION.EMB_DIM = 50
# Glove embeddings dimension.
_C.RELATION_IDENTIFICATION.GLOVE_EMB_DIM = 50
# Hidden size.
_C.RELATION_IDENTIFICATION.HIDDEN_SIZE = 40
# Number of layers.
_C.RELATION_IDENTIFICATION.NUM_LAYERS = 1
# MLP Hidden size.
_C.RELATION_IDENTIFICATION.DENSE_MLP_HIDDEN_SIZE = 30
# Edge threshold.
_C.RELATION_IDENTIFICATION.EDGE_THRESHOLD = 0.6
# Sampling ratio.
_C.RELATION_IDENTIFICATION.SAMPLING_RATIO = 3
# Positive class weight
_C.RELATION_IDENTIFICATION.POSITIVE_CLASS_WEIGHT = 0.7
# Negative class weight
_C.RELATION_IDENTIFICATION.NEGATIVE_CLASS_WEIGHT = 1.0
# Logging start epoch for training flow.
_C.RELATION_IDENTIFICATION.LOGGING_START_EPOCH_TRAIN = 30
# Logging start epoch for evaluation flow.
_C.RELATION_IDENTIFICATION.LOGGING_START_EPOCH_DEV = 20
# Config for which module to train
_C.RELATION_IDENTIFICATION.HEADS_SELECTION = True
_C.RELATION_IDENTIFICATION.ARCS_LABELLING = True


def get_default_config():
  return _C.clone()