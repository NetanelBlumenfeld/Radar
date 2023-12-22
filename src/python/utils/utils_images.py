from enum import Enum, auto

import numpy as np

EPSILON = 1e-8


class Normalization(Enum):
    NONE = auto()
    Range_0_1 = auto()
    Range_neg_1_1 = auto()


def normalize_img(img: np.ndarray, type: Normalization) -> np.ndarray:
    if type == Normalization.NONE:
        return img
    elif type == Normalization.Range_0_1:
        return (img - np.min(img)) / (np.max(img) - np.min(img))
    elif type == Normalization.Range_neg_1_1:
        return (img - np.min(img)) / (np.max(img) - np.min(img) + EPSILON) * 2 - 1
    else:
        raise ValueError("Unknown normalization type: " + str(type))
