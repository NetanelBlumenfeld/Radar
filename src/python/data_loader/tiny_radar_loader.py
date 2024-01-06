from typing import Callable

import numpy as np
import torch
from config import DataCfg
from data_loader.tiny_loader import (
    load_tiny_data,
    load_tiny_data_sr_4090,
    load_tiny_data_sr_classifier_4090,
)
from scipy.fftpack import fft, fftshift
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from utils.utils_images import Normalization, down_sample_img, normalize_img


class SrDataSet_3080(Dataset):
    def __init__(self, imgs: np.ndarray, transform: Callable) -> None:
        self.imgs = imgs
        self.transform = transform
        assert self.transform is not None

    def process_data(self, x: np.ndarray) -> np.ndarray:
        # TODO: adding this as preprocess pipeline
        x = np.abs(fftshift(fft(x, axis=0), axes=0))
        x = normalize_img(x, Normalization.Range_0_1)
        x = np.expand_dims(x, axis=0)
        return x

    def __len__(self) -> int:
        return self.imgs.shape[0]

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        high_res_time = self.imgs[idx]
        high_res = self.transform(high_res_time)
        low_res_time = down_sample_img(high_res_time, 4, 4)
        low_res = self.transform(low_res_time)

        # high_res = self.process_data(high_res_time)
        # low_res_time = down_sample_img(high_res_time, 4, 4)
        # low_res = self.process_data(low_res_time)
        return low_res, high_res


class ClassifierDataset(Dataset):
    def __init__(self, dataX, dataY):
        # self.x_train = np.transpose(dataX, (0, 1, 4, 2, 3))  # (N, 5, 2 ,32,492)
        self.x_train = dataX
        self.tempy = dataY
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


class SrClassifierDataset_3080(Dataset):
    def __init__(
        self,
        hight_res: np.ndarray,
        labels: np.ndarray,
        transform: Callable,
    ) -> None:
        self.hight_res = np.transpose(hight_res, (0, 1, 4, 2, 3))  # (N, 5, 2 ,32,492)
        self.transform = transform
        self.tempy = labels
        self.label = np.empty((self.tempy.shape[0], self.tempy.shape[1]))
        print(self.label.shape)
        for idx in range(self.tempy.shape[0]):
            for j in range(self.tempy.shape[1]):
                for i in range(self.tempy.shape[2]):
                    if self.tempy[idx][j][i] == 1:
                        self.label[idx][j] = i
        del self.tempy

    def __len__(self) -> int:
        return self.hight_res.shape[0]

    def __getitem__(self, idx) -> tuple[np.ndarray, list]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        high_res_time = self.hight_res[idx]
        low_res, high_res = self.transform(high_res_time)
        # low_res_time = down_sample_img(high_res_time, 4, 4)
        # low_res = self.transform(low_res_time)
        return low_res, [high_res, torch.LongTensor(self.label[idx])]


class SrDataSet_4090(Dataset):
    def __init__(self, low_res, hight_res):
        """ "
        take input with shape (N,H,W) and add channel dim (N,1,H,W)
        """
        self.x = np.expand_dims(low_res, axis=1)
        self.y = np.expand_dims(hight_res, axis=1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx], self.y[idx]


class SrClassifier_4090(Dataset):
    def __init__(self, low_res: np.ndarray, high_res: np.ndarray, labels: np.ndarray):
        self.low_res = low_res
        self.high_res = high_res
        self.labels = labels

    def __len__(self):
        return self.low_res.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.low_res[idx], [self.high_res[idx], self.labels[idx]]


def loader_sr_classifier_3080(
    high_res: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    batch_size: int,
    transform_pipeline: Callable,
    random_state=42,
) -> tuple[DataLoader, DataLoader]:
    high_res_train, high_res_val, labels_train, labels_val = train_test_split(
        high_res, labels, test_size=test_size, random_state=random_state
    )
    traindataset = SrClassifierDataset_3080(
        high_res_train, labels_train, transform_pipeline
    )
    del high_res_train, labels_train
    valdataset = SrClassifierDataset_3080(high_res_val, labels_val, transform_pipeline)
    del high_res_val, labels_val
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=False)
    return trainloader, valloader


def loader_classifier(
    X: np.ndarray, y: np.ndarray, test_size: float, batch_size: int, random_state=42
) -> tuple[DataLoader, DataLoader]:
    """
    Split the dataset into training and validation sets.

    Parameters:
    - X: low res images.
    - y: labels.
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
    print("X_train" + str(X_train.shape))
    print("Y_train" + str(Y_train.shape))

    # Generate datasets
    traindataset = ClassifierDataset(X_train, Y_train)
    del X_train, Y_train
    valdataset = ClassifierDataset(X_val, Y_val)
    del X_val, Y_val
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)

    return trainloader, valloader


def loader_sr_3080(
    y: np.ndarray,
    test_size: float,
    batch_size: int,
    transform_pipeline: Callable,
    random_state=42,
) -> tuple[DataLoader, DataLoader]:
    """
    Split the dataset into training and validation sets.

    Parameters:
    - y: high res images.
    - test_size: Fraction of the dataset to be used as validation data.
    - random_state: Seed for the random number generator for reproducibility.

    """

    # Split the dataset
    Y_train, Y_val = train_test_split(y, test_size=test_size, random_state=random_state)

    # Generate datasets
    traindataset = SrDataSet_3080(Y_train, transform_pipeline)
    del Y_train
    valdataset = SrDataSet_3080(Y_val, transform_pipeline)
    del Y_val
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader


def loader_sr_4090(
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
    traindataset = SrDataSet_4090(X_train, Y_train)
    del X_train, Y_train
    valdataset = SrDataSet_4090(X_val, Y_val)
    del X_val, Y_val
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)

    return trainloader, valloader


def loader_sr_classifier_4090(
    low_res: np.ndarray,
    high_res: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    batch_size: int,
    random_state=42,
) -> tuple[DataLoader, DataLoader]:
    print(
        f"shapes low_res - {low_res.shape}, high_res - {high_res.shape}, labels - {labels.shape}"
    )
    (
        train_low_res,
        test_low_res,
        train_high_res,
        test_high_res,
        train_labels,
        test_labels,
    ) = train_test_split(
        low_res,
        high_res,
        labels,
        test_size=test_size,
        random_state=random_state,
    )
    # Generate datasets
    traindataset = SrClassifier_4090(train_low_res, train_high_res, train_labels)
    valdataset = SrClassifier_4090(test_low_res, test_high_res, test_labels)
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)

    return trainloader, valloader


def tiny_sr_4090(
    data_dir: str,
    data_cfg: DataCfg,
    batch_size: int,
    test_size: float = 0.05,
) -> tuple[DataLoader, DataLoader]:
    assert data_cfg.task == "sr"
    assert data_cfg.transform_pipeline is not None
    high_res, low_res = load_tiny_data_sr_4090(
        data_dir=data_dir,
        people=data_cfg.people,
        gestures=data_cfg.gestures,
        data_type="npy",
        load_data_func=data_cfg.transform_pipeline,
    )
    trainloader, valloader = loader_sr_4090(low_res, high_res, test_size, batch_size)

    return trainloader, valloader


def tiny_classifier(
    data_dir: str,
    data_cfg: DataCfg,
    test_size: float = 0.1,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader]:
    X, y = load_tiny_data(
        data_dir,
        data_cfg.people,
        data_cfg.gestures,
        data_cfg.data_type,
        task="classifier",
    )
    assert X.ndim == 5
    assert data_cfg.transform_pipeline is not None

    X = data_cfg.transform_pipeline(X)
    trainloader, valloader = loader_classifier(X, y, test_size, batch_size)

    return trainloader, valloader


def tiny_sr_3080(
    data_dir: str,
    data_cfg: DataCfg,
    batch_size: int,
    test_size: float = 0.05,
) -> tuple[DataLoader, DataLoader]:
    high_res, _ = load_tiny_data(
        data_dir,
        data_cfg.people,
        data_cfg.gestures,
        data_cfg.data_type,
        task="sr",
    )
    assert data_cfg.transform_pipeline is not None
    trainloader, valloader = loader_sr_3080(
        high_res, test_size, batch_size, data_cfg.transform_pipeline
    )

    return trainloader, valloader


def tiny_sr_classifier_3080(
    data_dir: str,
    data_cfg: DataCfg,
    test_size: float = 0.1,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader]:
    high_res, labels = load_tiny_data(
        data_dir,
        data_cfg.people,
        data_cfg.gestures,
        data_cfg.data_type,
        task="sr_classifier",
    )
    assert data_cfg.transform_pipeline is not None
    trainloader, valloader = loader_sr_classifier_3080(
        high_res, labels, test_size, batch_size, data_cfg.transform_pipeline
    )
    return trainloader, valloader


def tiny_sr_classifier_4090(
    data_dir: str,
    data_cfg: DataCfg,
    test_size: float = 0.1,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader]:
    assert data_cfg.transform_pipeline is not None

    low_res, high_res, labels = load_tiny_data_sr_classifier_4090(
        data_dir,
        data_cfg.people,
        data_cfg.gestures,
        data_cfg.data_type,
        load_data_func=data_cfg.transform_pipeline,
    )
    trainloader, valloader = loader_sr_classifier_4090(
        low_res=low_res,
        high_res=high_res,
        labels=labels,
        test_size=test_size,
        batch_size=batch_size,
    )
    return trainloader, valloader


def get_tiny_dataloader(
    data_cfg: DataCfg,
    data_dir: str,
    pc: str,
    test_size: float = 0.1,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader]:
    print(
        f"People: {data_cfg.people}, Gestures: {data_cfg.gestures}, Batch Size: {batch_size}"
    )

    if data_cfg.task == "classifier":
        trainloader, valloader = tiny_classifier(
            data_dir=data_dir,
            data_cfg=data_cfg,
            test_size=test_size,
            batch_size=batch_size,
        )
    elif data_cfg.task == "sr" and pc == "4090":
        trainloader, valloader = tiny_sr_4090(
            data_dir=data_dir, data_cfg=data_cfg, batch_size=batch_size
        )

    elif data_cfg.task == "sr" and pc == "3080":
        trainloader, valloader = tiny_sr_3080(
            data_dir=data_dir, data_cfg=data_cfg, batch_size=batch_size
        )
    elif data_cfg.task == "sr_classifier" and pc == "3080":
        trainloader, valloader = tiny_sr_classifier_3080(
            data_dir=data_dir,
            data_cfg=data_cfg,
            test_size=test_size,
            batch_size=batch_size,
        )
    elif data_cfg.task == "sr_classifier" and pc == "4090":
        trainloader, valloader = tiny_sr_classifier_4090(
            data_dir=data_dir,
            data_cfg=data_cfg,
            test_size=test_size,
            batch_size=batch_size,
        )

    else:
        raise ValueError("task must be sr, classifier or sr_classifier")
    return trainloader, valloader
