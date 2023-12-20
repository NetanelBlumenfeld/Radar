from collections import namedtuple
from enum import Enum, auto
from functools import partial

import cv2
import numpy as np
import torch
from scipy.fftpack import fft, fftshift
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from utils.utils_paths import ensure_path_exists


class Normalization(Enum):
    NONE = auto()
    Range_0_1 = auto()
    Range_neg_1_1 = auto()


GESTURES = [
    "PinchIndex",
    "PinchPinky",
    "FingerSlider",
    "FingerRub",
    "SlowSwipeRL",
    "FastSwipeRL",
    "Push",
    "Pull",
    "PalmTilt",
    "Circle",
    "PalmHold",
    "NoHand",
]
numberOfInstanceWindows = 3
lengthOfSubWindow = 32


def tiny_radar_for_classifier(data_dir: str, batch_size: int, pix_norm: Normalization):
    pass


def tiny_radar_for_sr_classifier():
    pass
