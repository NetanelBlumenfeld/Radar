import numpy as np
import torch
from data_loader.utils_tiny import down_sample_doppler_maps, load_tiny_doppler_maps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from utils.utils_images import Normalization


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


def setup_classifier_dataset(
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
    # Flatten the list of arrays
    flat_dataX = np.concatenate(dataX)
    flat_dataY = np.concatenate(dataY)

    # Split the dataset
    X_train, X_val, Y_train, Y_val = train_test_split(
        flat_dataX, flat_dataY, test_size=test_size, random_state=random_state
    )

    # Generate datasets
    traindataset = ClassifierDataset([X_train], [Y_train])
    valdataset = ClassifierDataset([X_val], [Y_val])

    return traindataset, valdataset


def setup_sr_classifier_dataset(
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
    flat_low_res_imgs = np.concatenate(low_res_imgs)
    flat_high_res_imgs = np.concatenate(hight_res_imgs)
    flat_labels = np.concatenate(labels)

    # Split the data into training and testing sets
    (
        train_low_res,
        test_low_res,
        train_high_res,
        test_high_res,
        train_labels,
        test_labels,
    ) = train_test_split(
        flat_low_res_imgs,
        flat_high_res_imgs,
        flat_labels,
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
    dataX, dataY = load_tiny_doppler_maps(data_dir, pix_norm, people, gestures)
    traindataset, valdataset = setup_classifier_dataset(dataX, dataY, test_size)
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)
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
    hight_res, labels = load_tiny_doppler_maps(high_res_dir, pix_norm, people, gestures)
    low_res, _ = load_tiny_doppler_maps(low_res_dir, pix_norm, people, gestures)
    traindataset, valdataset = setup_sr_classifier_dataset(
        low_res, hight_res, labels, test_size
    )
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)
    data_set_name = low_res_dir.split("/")[-2] + "_" + str(pix_norm).lower()

    return trainloader, valloader, data_set_name


def tiny_radar_for_sr_classifier_of_disk(
    high_res_dir: str,
    row_scale: int,
    col_scale: int,
    original_dim: bool,
    batch_size: int,
    pix_norm: Normalization,
    test_size: int = 0.1,
) -> tuple[DataLoader, DataLoader]:
    hight_res, labels = load_tiny_doppler_maps(high_res_dir, pix_norm)
    low_res = down_sample_doppler_maps(hight_res, row_scale, col_scale, original_dim)
    traindataset, valdataset = setup_sr_classifier_dataset(
        low_res, hight_res, labels, test_size
    )
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)
    return trainloader, valloader
