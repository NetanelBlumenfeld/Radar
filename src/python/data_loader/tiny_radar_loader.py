import os
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
import torch
from data_loader.utils_tiny import (
    doppler_maps,
    doppler_maps_mps,
    down_sample_and_save,
    down_sample_data,
    load_tiny_data,
    normalize_tiny_data,
    normalize_tiny_data_mps,
    npy_feat_reshape,
)
from scipy.fftpack import fft, fftshift, ifft, ifftshift
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from utils.utils_images import Normalization


class ReconstractDataset(Dataset):
    def __init__(self, imgs):
        imgs_high_res = np.concatenate([np.array(d) for d in imgs])
        imgs_high_res = doppler_maps(imgs_high_res)
        imgs_high_res = normalize_tiny_data(imgs_high_res, Normalization.Range_0_1)
        x = down_sample_data(imgs_high_res, 4, 4, False)
        x = torch.tensor(x, dtype=torch.float32)
        x_len, time_steps, rows, cols, channels = imgs_high_res.shape
        self.x = x.permute(0, 1, 4, 2, 3).reshape(
            x_len * time_steps * channels, 1, rows // 4, cols // 4
        )
        self.y = (
            torch.tensor(imgs_high_res, dtype=torch.float32)
            .permute(0, 1, 4, 2, 3)
            .reshape(x_len * time_steps * channels, 1, rows, cols)
        )

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx], self.y[idx]


class SRClassifierDataset(Dataset):
    def __init__(self, low_res, hight_resY, labelsY):
        _x_train = np.concatenate([np.array(d) for d in low_res])
        _hight_res_y = np.concatenate([np.array(d) for d in hight_resY])
        self.x_train = np.transpose(_x_train, (0, 1, 4, 2, 3))
        self.hight_res_y = np.transpose(_hight_res_y, (0, 1, 4, 2, 3))

        self.tempy = np.concatenate([np.array(dy) for dy in labelsY])

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
            self.hight_res_y[idx],
            torch.LongTensor(self.label[idx]),
        ]


class ClassifierDataset(Dataset):
    def __init__(self, dataX, dataY):
        _x_train = np.concatenate([np.array(d) for d in dataX])
        self.x_train = np.transpose(_x_train, (0, 1, 4, 2, 3))
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

        return self.x_train[idx], torch.LongTensor(self.label[idx])


class SRDataset(Dataset):
    def __init__(self, low_res, hight_res):
        _x_train = np.concatenate([np.array(d) for d in low_res])
        _hight_res_y = np.concatenate([np.array(d) for d in hight_res])
        # self.x_train = np.transpose(_x_train, (0, 1, 4, 2, 3))
        # self.hight_res_y = np.transpose(_hight_res_y, (0, 1, 4, 2, 3))
        self.x_train = _x_train
        self.hight_res_y = _hight_res_y

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x_train[idx], self.hight_res_y[idx]


def setup_dataset_2t(dataX, test_size, random_state=42) -> tuple[Dataset, Dataset]:
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

    # Split the dataset
    X_train, X_val = train_test_split(
        dataX, test_size=test_size, random_state=random_state
    )

    # Generate datasets
    traindataset = ReconstractDataset([X_train])
    valdataset = ReconstractDataset([X_val])

    return traindataset, valdataset


def setup_dataset_2(
    dataX, dataY, test_size, random_state=42
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

    # Split the dataset
    X_train, X_val, Y_train, Y_val = train_test_split(
        dataX, dataY, test_size=test_size, random_state=random_state
    )

    # Generate datasets
    traindataset = SRDataset([X_train], [Y_train])
    valdataset = SRDataset([X_val], [Y_val])

    return traindataset, valdataset


def setup_dataset_3(
    low_res_imgs, hight_res_imgs, labels, test_size=0.1, random_state=42
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

    # Split the data into training and testing sets
    (
        train_low_res,
        test_low_res,
        train_high_res,
        test_high_res,
        train_labels,
        test_labels,
    ) = train_test_split(
        low_res_imgs,
        hight_res_imgs,
        labels,
        test_size=test_size,
        random_state=random_state,
    )

    # Generate datasets
    traindataset = SRClassifierDataset(
        [train_low_res], [train_high_res], [train_labels]
    )
    valdataset = SRClassifierDataset([test_low_res], [test_high_res], [test_labels])

    return traindataset, valdataset


def tiny_radar_for_classifier(
    data_dir: str,
    people: list[int],
    gestures: list[str],
    batch_size: int,
    pix_norm: Normalization,
    test_size: float = 0.1,
) -> tuple[DataLoader, DataLoader, str]:
    dataX, dataY = load_tiny_data(data_dir, people, gestures, "doppler")
    if pix_norm != Normalization.NONE:
        dataX = normalize_tiny_data(dataX, pix_norm)
    for i in range(dataX.shape[0]):
        for j in range(dataX.shape[1]):
            for k in range(dataX.shape[4]):
                sig = dataX[i, j, :, :, k]
                ds_sig = sig[::1, ::64]
                # low_pass_sig = np.zeros_like(sig)
                # low_pass_sig[12:20, :] = sig[12:20, :]
                # sig_time = ifft(ifftshift(low_pass_sig, axes=0), axis=0)
                # ds_sig = sig_time[::4, ::4]
                # low_pass_sig = abs(fftshift(fft(ds_sig, axis=0), axes=0)).astype(
                #     np.float32
                # )
                up_sig = cv2.resize(ds_sig, (492, ds_sig.shape[0] * 1))
                dataX[i, j, :, :, k] = up_sig
    traindataset, valdataset = setup_dataset_2(dataX, dataY, test_size)
    trainloader = DataLoader(
        traindataset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    valloader = DataLoader(
        valdataset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    data_set_name = data_dir.split("/")[-2] + "_" + str(pix_norm).lower()
    return trainloader, valloader, data_set_name


def tiny_radar_for_sr_classifier_on_disk(
    high_res_dir: str,
    low_res_dir: str,
    people: list[int],
    gestures: list[str],
    batch_size: int,
    pix_norm: Normalization,
    test_size: float = 0.1,
) -> tuple[DataLoader, DataLoader, str]:
    hight_res, labels = load_tiny_data(high_res_dir, people, gestures, "doppler")
    low_res, _ = load_tiny_data(low_res_dir, people, gestures, "doppler")
    if pix_norm != Normalization.NONE:
        low_res = normalize_tiny_data(low_res, pix_norm)
        hight_res = normalize_tiny_data(hight_res, pix_norm)

    traindataset, valdataset = setup_dataset_3(low_res, hight_res, labels, test_size)
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)
    data_set_name = low_res_dir.split("/")[-2] + "_" + str(pix_norm).lower()

    return trainloader, valloader, data_set_name


def tiny_radar_of_disk(
    data_dir: str,
    row_factor: int,
    col_factor: int,
    original_dim: bool,
    people: list[int],
    gestures: list[str],
    with_labels: bool,
    batch_size: int,
    pix_norm: Normalization,
    test_size: float = 0.1,
    mps: bool = True,
) -> tuple[DataLoader, DataLoader, str]:
    high_res_raw, labels = load_tiny_data(
        data_dir, people, gestures, "npy"
    )  # loading npy data (raw)
    high_res_raw = npy_feat_reshape(high_res_raw)  # reshape to doppler map shape
    low_res_raw = down_sample_data(
        high_res_raw, row_factor, col_factor, original_dim
    )  # down sample data

    print("Getting Doppler map ")
    if mps:
        num_workers = os.cpu_count()
        print(f"down sampling data with {num_workers//2} cpus")
        # high_res_raw = npy_feat_reshape(high_res_raw, classifier_shape=False)
        # low_res_raw = npy_feat_reshape(low_res_raw, classifier_shape=False)
        print(low_res_raw.shape)
        with Pool(num_workers) as p:
            high_res = p.map(doppler_maps_mps, high_res_raw)
            low_res = p.map(doppler_maps_mps, low_res_raw)
        # with Pool(num_workers) as p:
        #     norm_func = partial(normalize_tiny_data_mps, pix_norm=pix_norm)
        #     high_res_norm = p.map(norm_func, high_res)
        #     low_res_norm = p.map(norm_func, low_res)

        high_res = np.array(high_res, dtype=np.float32)
        low_res = np.array(low_res, dtype=np.float32)

    else:
        high_res = doppler_maps(high_res_raw)
        low_res = doppler_maps(low_res_raw)

    if pix_norm != Normalization.NONE:
        low_res = normalize_tiny_data(low_res, pix_norm)
        high_res = normalize_tiny_data(high_res, pix_norm)
    print(low_res.shape, high_res.shape)
    del high_res_raw, low_res_raw

    print("splliting data")

    traindataset, valdataset = setup_dataset_2(low_res, high_res, test_size)
    del low_res, high_res
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)
    dataset_name = (
        f"row_{row_factor}_col_{col_factor}_original_{original_dim}_pix_{pix_norm}"
    )
    return trainloader, valloader, dataset_name


def tiny_radar_for_sr(
    high_res_dir: str,
    low_res_dir: str,
    people: list[int],
    gestures: list[str],
    batch_size: int,
    pix_norm: Normalization,
    test_size: float = 0.1,
) -> tuple[DataLoader, DataLoader, str]:
    hight_res, _ = load_tiny_data(high_res_dir, pix_norm, people, gestures, "doppler")
    low_res, _ = load_tiny_data(low_res_dir, pix_norm, people, gestures, "doppler")
    traindataset, valdataset = setup_dataset_2(low_res, hight_res, test_size)
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)
    data_set_name = low_res_dir.split("/")[-2] + "_" + str(pix_norm).lower()

    return trainloader, valloader, data_set_name


def tiny_tt(
    data_dir: str,
    people: list[int],
    gestures: list[str],
    batch_size: int,
    pix_norm: Normalization,
    test_size: float = 0.1,
) -> tuple[DataLoader, DataLoader, str]:
    dataX, _ = load_tiny_data(data_dir, people, gestures, "npy")
    dataX = npy_feat_reshape(dataX)
    traindataset, valdataset = setup_dataset_2t(dataX, test_size)
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)
    data_set_name = data_dir.split("/")[-2] + "_" + str(pix_norm).lower()
    return trainloader, valloader, data_set_name
