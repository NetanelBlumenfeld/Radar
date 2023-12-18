from collections import namedtuple
from functools import partial

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class SRGestureDataset(Dataset):
    def __init__(self, dataX, hight_resY, labelsY):
        _x_train = np.concatenate([np.array(d) for d in dataX])
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


def loadPerson(paramList, scale: bool = False):
    SubjectData_hight_res = list()
    SubjectData_low_res = list()
    SubjectLabel = list()
    print(f"Doing {paramList.personIdx}")
    for gestureIdx, gestureName in enumerate(paramList.listOfGestures):
        # Create filename
        filename = (
            "p"
            + str(paramList.personIdx)
            + "/"
            + gestureName
            + "_1s_wl"
            + str(paramList.lengthOfWindow)
            + "_"
            + "doppl.npy"
        )

        # Load data gesture data from disk
        try:
            GestureData_los_res = np.load(paramList.pathToLowRes + filename)
            GestureData_hight_res = np.load(paramList.pathToLowRes + filename)
            if scale:
                for i in range(GestureData_los_res.shape[0]):
                    for j in range(GestureData_los_res.shape[1]):
                        for k in range(GestureData_los_res.shape[4]):
                            max_val_los_res = (
                                GestureData_los_res[i, j, :, :, k].max() + 0.0001
                            )
                            GestureData_los_res[i, j, :, :, k] = (
                                GestureData_los_res[i, j, :, :, k] / max_val_los_res
                            )
                            max_val_hight_res = (
                                GestureData_hight_res[i, j, :, :, k].max() + 0.0001
                            )
                            GestureData_hight_res[i, j, :, :, k] = (
                                GestureData_hight_res[i, j, :, :, k] / max_val_hight_res
                            )
        except IOError:
            print("Could not open file: " + filename)
            continue
        else:
            if 0 >= GestureData_los_res.shape[0] or 0 >= GestureData_hight_res.shape[0]:
                print("Skip datafile (no data): " + filename)
                continue

            SubjectData_low_res.append(GestureData_los_res)
            SubjectData_hight_res.append(GestureData_hight_res)

            for idx in range(0, GestureData_los_res.shape[0]):
                GestureLabel = list()
                for jdx in range(0, GestureData_los_res.shape[1]):
                    GestureLabel.append(
                        np.identity(len(paramList.listOfGestures))[gestureIdx]
                    )
                SubjectLabel.append(np.asarray(GestureLabel))

            # Check if there is some data for this person
            if (0 >= len(SubjectData_low_res)) or (0 >= len(SubjectLabel)):
                print("No entries found for person with index 'p" + str(idx) + "'")
                return

    return (
        np.concatenate(SubjectData_low_res, axis=0),
        np.concatenate(SubjectData_hight_res, axis=0),
        np.asarray(SubjectLabel),
    )


def loadFeatures(
    los_res_path,
    hight_res_path,
    listOfPeople,
    listofGestures,
    numberofInstanceCopies,
    lengthOfWindow,
    scale: bool,
):
    ParamList = namedtuple(
        "ParamList",
        "personIdx, pathToLowRes,pathToHightRes ,listOfGestures, numberOfInstanceCopies, lengthOfWindow",
    )
    personList = []
    for i in listOfPeople:
        personList.append(
            ParamList(
                i,
                los_res_path,
                hight_res_path,
                listofGestures,
                numberofInstanceCopies,
                lengthOfWindow,
            )
        )
    loadPerson_scale = partial(loadPerson, scale=scale)

    featureList = list(map(loadPerson_scale, personList))
    return featureList


def setupLOOCV(
    low_res_imgs, hight_res_imgs, labels, validationPerson: int
) -> tuple[Dataset, Dataset]:
    # Split people into train and validation set
    low_res_train = [
        *low_res_imgs[0:validationPerson],
        *low_res_imgs[validationPerson + 1 :],
    ]
    hight_res_train = [
        *hight_res_imgs[0:validationPerson],
        *hight_res_imgs[validationPerson + 1 :],
    ]
    labels_train = [*labels[0:validationPerson], *labels[validationPerson + 1 :]]

    low_res_val = [low_res_imgs[validationPerson]]
    hight_res_val = [hight_res_imgs[validationPerson]]
    labels_val = [labels[validationPerson]]

    # Generate dataset from lists
    traindataset = SRGestureDataset(low_res_train, hight_res_train, labels_train)
    valdataset = SRGestureDataset(low_res_val, hight_res_val, labels_val)

    return traindataset, valdataset


def setupDataset(
    low_res_imgs, hight_res_imgs, labels, test_size=0.2, random_state=42
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
        random_state=test_size,
    )

    # Generate datasets
    traindataset = SRGestureDataset([train_low_res], [train_high_res], [train_labels])
    valdataset = SRGestureDataset([test_low_res], [test_high_res], [test_labels])

    return traindataset, valdataset


def get_sr_tiny_radar_data_loader(
    los_res_path: str,
    hight_res_path: str,
    listPeople: list[int],
    listGestures: list[str],
    batch_size: int = 128,
    scale: bool = False,
) -> tuple[DataLoader, DataLoader]:
    # Dataset parameters
    numberOfInstanceWindows = 3
    lengthOfSubWindow = 32

    # Set the validation Person used for Leave-one-out cross-validation
    validationPerson = 10

    featureList = loadFeatures(
        los_res_path,
        hight_res_path,
        listPeople,
        listGestures,
        numberOfInstanceWindows,
        lengthOfSubWindow,
        scale,
    )
    low_res_imgs = list(map(lambda x: x[0], featureList))
    hight_res_imgs = list(map(lambda x: x[1], featureList))
    labels = list(map(lambda x: x[2], featureList))

    traindataset, valdataset = setupLOOCV(
        low_res_imgs, hight_res_imgs, labels, validationPerson
    )

    training_generator = DataLoader(
        traindataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_generator = DataLoader(
        valdataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return training_generator, val_generator
