from enum import Enum


class SmatchScore(Enum):
    PRECISION = 1,
    RECALL = 2,
    F_SCORE = 3


def initialize_smatch():
    smatch = {
        SmatchScore.PRECISION: 0.0,
        SmatchScore.RECALL: 0.0,
        SmatchScore.F_SCORE: 0.0
    }
    return smatch