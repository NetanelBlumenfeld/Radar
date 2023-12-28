import cv2
import numpy as np
import torch
from data_loader.tiny_loader import load_tiny_data, load_tiny_data_sr
from data_loader.utils_tiny import normalize_tiny_data
from scipy.fftpack import fft, fftshift
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from utils.utils_images import Normalization, down_sample_img, normalize_img


class DopplerDataset(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs

    def process_data(self, x: np.ndarray) -> np.ndarray:
        x = np.abs(fftshift(fft(x, axis=0), axes=0))
        x = normalize_img(x, Normalization.Range_0_1)
        x = np.expand_dims(x, axis=0)
        return x

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        high_res_time = self.imgs[idx]
        high_res = self.process_data(high_res_time)
        low_res_time = down_sample_img(high_res_time, 4, 4)
        low_res = self.process_data(low_res_time)
        return low_res, high_res


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
        self.x = np.expand_dims(low_res, axis=1)
        self.y = np.expand_dims(hight_res, axis=1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx], self.y[idx]


def setup_sr_data_loader(
    X: np.ndarray, y: np.ndarray, test_size: float, batch_size: int, random_state=42
) -> tuple[DataLoader, DataLoader]:
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
        X, y, test_size=test_size, random_state=random_state
    )

    # Generate datasets
    traindataset = SRDataset(X_train, Y_train)
    del X_train, Y_train
    valdataset = SRDataset(X_val, Y_val)
    del X_val, Y_val
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)

    return trainloader, valloader


def setup_data_loader(
    y: np.ndarray, test_size: float, batch_size: int, random_state=42
) -> tuple[DataLoader, DataLoader]:
    """
    Split the dataset into training and validation sets.

    Parameters:
    - y: high res images.
    - test_size: Fraction of the dataset to be used as validation data.
    - random_state: Seed for the random number generator for reproducibility.

    Returns:
    - traindataset: Training dataset.
    - valdataset: Validation dataset.
    """

    # Split the dataset
    Y_train, Y_val = train_test_split(y, test_size=test_size, random_state=random_state)

    # Generate datasets
    traindataset = DopplerDataset(Y_train)
    del Y_train
    valdataset = DopplerDataset(Y_val)
    del Y_val
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)

    return trainloader, valloader


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


def dataset_tiny_sr_off_disk(
    data_dir: str,
    people: int,
    gestures: list[str],
    batch_size: int,
    pix_norm: Normalization,
    test_size: float = 0.05,
) -> tuple[DataLoader, DataLoader, str]:
    print(f"People: {people}, Gestures: {gestures}, Batch Size: {batch_size}")
    high_res, low_res = load_tiny_data_sr(
        data_dir=data_dir,
        people=people,
        gestures=gestures,
        data_type="npy",
        pix_norm=pix_norm,
    )
    trainloader, valloader = setup_sr_data_loader(
        low_res, high_res, test_size, batch_size
    )

    data_set_name = data_dir.split("/")[-2] + "_" + str(pix_norm).lower()
    return trainloader, valloader, data_set_name


def dataset_tiny(
    data_dir: str,
    people: int,
    gestures: list[str],
    batch_size: int,
    pix_norm: Normalization,
    test_size: float = 0.05,
    data_type: str = "npy",
) -> tuple[DataLoader, DataLoader, str]:
    print(f"People: {people}, Gestures: {gestures}, Batch Size: {batch_size}")
    high_res = load_tiny_data(data_dir, people, gestures, data_type)
    trainloader, valloader = setup_data_loader(high_res, test_size, batch_size)

    data_set_name = data_dir.split("/")[-2] + "_" + str(pix_norm).lower()
    return trainloader, valloader, data_set_name
