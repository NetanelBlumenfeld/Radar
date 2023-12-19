from collections import namedtuple
from functools import partial

import cv2
import numpy as np
import torch
from data_loader.tiny_radar_loader import loadFeatures
from scipy.fftpack import fft, fftshift
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from utils.utils_paths import ensure_path_exists


def down_sample_selective(
    data,
    row_factor,
    col_factor,
):
    return data[::row_factor, ::col_factor]


class GestureDataset(Dataset):
    def __init__(self, dataX, dataY):
        hight_res = np.concatenate([np.array(d) for d in dataX])
        s0, s1, s2, s3, s4 = hight_res.shape
        low_res = np.empty((s0, s1, s2 // 4, s3 // 4, s4))
        for i in range(hight_res.shape[0]):
            for j in range(hight_res.shape[1]):
                for k in range(hight_res.shape[4]):
                    low_res[i, j, :, :, k] = down_sample_selective(
                        hight_res[i, j, :, :, k], 4, 4
                    )

        self.x_train = np.transpose(low_res, (0, 1, 4, 2, 3))
        self.hight_res = np.transpose(hight_res, (0, 1, 4, 2, 3))
        self.tempy = np.concatenate([np.array(dy) for dy in dataY])
        self.label = np.empty((self.tempy.shape[0], self.tempy.shape[1]))

        for idx in range(self.tempy.shape[0]):
            for j in range(self.tempy.shape[1]):
                for i in range(self.tempy.shape[2]):
                    if self.tempy[idx][j][i] == 1:
                        self.label[idx][j] = i
        del self.tempy

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x_train[idx], [
            self.hight_res[idx],
            torch.LongTensor(self.label[idx]),
        ]


def setupDataset(
    dataX, dataY, test_size=0.2, random_state=42
) -> tuple[Dataset, Dataset]:
    """
    Split the dataset into training and validation sets.

    Parameters:
    - dataX: List of data samples.
    - dataY: Corresponding list of labels.
    - test_size: Fraction of the dataset to be used as validation data.
    - random_state: Seed for the random number generator for reproducibility.

    Returns:
    - traindataset: Training dataset.
    - valdataset: Validation dataset.
    """
    # Flatten the list of arrays
    flat_dataX = np.concatenate(dataX)
    flat_dataY = np.concatenate(dataY)

    # Split the dataset
    X_train, X_val, Y_train, Y_val = train_test_split(
        flat_dataX, flat_dataY, test_size=test_size, random_state=random_state
    )

    # Generate datasets
    traindataset = GestureDataset([X_train], [Y_train])
    valdataset = GestureDataset([X_val], [Y_val])
    return traindataset, valdataset


def get_drnl_data_loader(
    pathToDataset: str,
    listPeople: list[int],
    listGestures: list[str],
    batch_size: int = 128,
    scale: bool = False,
) -> tuple[DataLoader, DataLoader]:
    # Dataset parameters
    numberOfInstanceWindows = 3
    lengthOfSubWindow = 32

    featureList = loadFeatures(
        pathToDataset,
        listPeople,
        listGestures,
        numberOfInstanceWindows,
        lengthOfSubWindow,
        scale,
    )
    dataX = list(map(lambda x: x[0], featureList))
    dataY = list(map(lambda x: x[1], featureList))

    traindataset, valdataset = setupDataset(dataX, dataY)

    training_generator = DataLoader(
        traindataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_generator = DataLoader(
        valdataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return training_generator, val_generator
