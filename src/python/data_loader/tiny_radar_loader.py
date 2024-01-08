from typing import Callable

import numpy as np
import torch
from config import DataCfg
from data_loader.tiny_loader import (
    load_tiny_data,
    load_tiny_data_sr_4090,
    load_tiny_data_sr_classifier_4090,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from utils.utils_images import down_sample_img

RANDOM_STATE = 42


class SrDataSet_3080(Dataset):
    def __init__(self, imgs: np.ndarray, transform: Callable) -> None:
        self.imgs = imgs
        self.transform = transform
        assert self.transform is not None

    def __len__(self) -> int:
        return self.imgs.shape[0]

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # high_res_time = self.imgs[idx]
        # high_res = self.transform(high_res_time)
        # low_res_time = down_sample_img(high_res_time, 4, 4)
        # low_res = self.transform(low_res_time)
        # return low_res, high_res
        high_res_time = self.transform(self.imgs[idx])
        low_res_time = np.zeros_like(high_res_time)
        low_res_time[0] = down_sample_img(high_res_time[0], 4, 4)
        low_res_time[1] = down_sample_img(high_res_time[1], 4, 4)
        return low_res_time, high_res_time


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
        if low_res.ndim == 3:
            low_res = np.expand_dims(low_res, axis=1)
        if hight_res.ndim == 3:
            hight_res = np.expand_dims(hight_res, axis=1)
        self.x = low_res
        self.y = hight_res

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
        self.tempy = labels
        self.label = np.empty((self.tempy.shape[0], self.tempy.shape[1]))
        for idx in range(self.tempy.shape[0]):
            for j in range(self.tempy.shape[1]):
                for i in range(self.tempy.shape[2]):
                    if self.tempy[idx][j][i] == 1:
                        self.label[idx][j] = i
        del self.tempy

    def __len__(self):
        return self.low_res.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.low_res[idx], [
            self.high_res[idx],
            torch.LongTensor(self.label[idx]),
        ]


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
    X_train, X_val, Y_train, Y_val = train_test_split(
        low_res, high_res, test_size=test_size, random_state=RANDOM_STATE
    )

    # Generate datasets
    traindataset = SrDataSet_4090(X_train, Y_train)
    del X_train, Y_train
    valdataset = SrDataSet_4090(X_val, Y_val)
    del X_val, Y_val
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)

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
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    traindataset = ClassifierDataset(X_train, Y_train)
    del X_train, Y_train

    valdataset = ClassifierDataset(X_val, Y_val)
    del X_val, Y_val
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)

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
    Y_train, Y_val = train_test_split(
        high_res, test_size=test_size, random_state=RANDOM_STATE
    )
    traindataset = SrDataSet_3080(Y_train, data_cfg.transform_pipeline)
    del Y_train
    valdataset = SrDataSet_3080(Y_val, data_cfg.transform_pipeline)
    del Y_val
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=False)

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
    high_res_train, high_res_val, labels_train, labels_val = train_test_split(
        high_res, labels, test_size=test_size, random_state=RANDOM_STATE
    )
    traindataset = SrClassifierDataset_3080(
        high_res_train, labels_train, data_cfg.transform_pipeline
    )
    del high_res_train, labels_train
    valdataset = SrClassifierDataset_3080(
        high_res_val, labels_val, data_cfg.transform_pipeline
    )
    del high_res_val, labels_val
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=False)

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
        random_state=RANDOM_STATE,
    )
    # Generate datasets
    traindataset = SrClassifier_4090(train_low_res, train_high_res, train_labels)
    valdataset = SrClassifier_4090(test_low_res, test_high_res, test_labels)
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)

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
