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

_C.HEAD_SELECTION = CN()
# Embeddings dimension.
_C.HEAD_SELECTION.EMB_DIM = 50
# Glove embeddings dimension.
_C.HEAD_SELECTION.GLOVE_EMB_DIM = 50
# Hidden size.
_C.HEAD_SELECTION.HIDDEN_SIZE = 40
# Number of layers.
_C.HEAD_SELECTION.NUM_LAYERS = 1
# MLP Hidden size.
_C.HEAD_SELECTION.DENSE_MLP_HIDDEN_SIZE = 30
# Edge threshold.
_C.HEAD_SELECTION.EDGE_THRESHOLD = 0.5
# Sampling ratio.
_C.HEAD_SELECTION.SAMPLING_RATIO = 4
# Positive class weight
_C.HEAD_SELECTION.POSITIVE_CLASS_WEIGHT = 0.7
# Negative class weight
_C.HEAD_SELECTION.NEGATIVE_CLASS_WEIGHT = 1.0
# Logging start epoch for training flow.
_C.HEAD_SELECTION.LOGGING_START_EPOCH_TRAIN = 30
# Logging start epoch for evaluation flow.
_C.HEAD_SELECTION.LOGGING_START_EPOCH_DEV = 20


def get_default_config():
  return _C.clone()