from enum import Enum, auto

import cv2
import numpy as np

EPSILON = 1e-8


class Normalization(Enum):
    NONE = auto()
    Range_0_1 = auto()
    Range_neg_1_1 = auto()


def normalize_img(img: np.ndarray, pix_norm: Normalization) -> np.ndarray:
    if pix_norm == Normalization.NONE:
        return img
    elif pix_norm == Normalization.Range_0_1:
        return (img - np.min(img)) / (np.max(img) - np.min(img) + EPSILON)
    elif pix_norm == Normalization.Range_neg_1_1:
        return (img - np.min(img)) / (np.max(img) - np.min(img) + EPSILON) * 2 - 1
    else:
        raise ValueError("Unknown normalization type: " + str(pix_norm))


def down_sample_img(
    x: np.ndarray, row_factor: int, col_factor: int, original_dim: bool = False
) -> np.ndarray:
    def _down_sample(img: np.ndarray, row_factor: int, col_factor: int) -> np.ndarray:
        return img[::row_factor, ::col_factor]

    def _up_scale(img: np.ndarray, dim_up: tuple[int, int]) -> np.ndarray:
        real_img = np.real(img)
        imag_img = np.imag(img)
        data_real_up = cv2.resize(real_img, dim_up, interpolation=cv2.INTER_CUBIC)
        data_imag_up = cv2.resize(imag_img, dim_up, interpolation=cv2.INTER_CUBIC)
        return data_real_up + 1j * data_imag_up

    assert x.ndim == 2
    org_dim = (x.shape[1], x.shape[0])
    img = _down_sample(x, row_factor, col_factor)
    if original_dim:
        img = _up_scale(img, org_dim)

    return img
