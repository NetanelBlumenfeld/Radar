from collections import namedtuple
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
import torch
from scipy.fftpack import fft, fftshift
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from utils.utils_paths import ensure_path_exists


class GestureDataset(Dataset):
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


def down_sample_data(data, row_factor, col_factor, interpolation):
    """
    Down samples a matrix by the given row and column factors and then convert it back to the original shape.
    :param data: The matrix to be down sampled.
    :param row_factor: The factor to down sample the rows by.
    :param col_factor: The factor to down sample the columns by.
    :return: The down sampled matrix.
    """

    def down_sample_selective(
        data,
        row_factor,
        col_factor,
    ):
        return data[::row_factor, ::col_factor]

    width_down = int(data.shape[1] / col_factor)
    height_down = int(data.shape[0] / row_factor)
    dim_down = (width_down, height_down)
    dim_up = (data.shape[1], data.shape[0])
    if data.dtype == np.complex64:
        data_real = data.real
        data_imag = data.imag
        if interpolation is None:
            data_real_down = down_sample_selective(data_real, row_factor, col_factor)
            data_imag_down = down_sample_selective(data_imag, row_factor, col_factor)
        else:
            data_real_down = cv2.resize(
                data_real, dim_down, interpolation=cv2.INTER_LINEAR
            )
            data_imag_down = cv2.resize(
                data_imag, dim_down, interpolation=cv2.INTER_LINEAR
            )
        data_real_up = cv2.resize(data_real_down, dim_up, interpolation=cv2.INTER_CUBIC)
        data_imag_up = cv2.resize(data_imag_down, dim_up, interpolation=cv2.INTER_CUBIC)
        data_down = data_real_up + 1j * data_imag_up
    return data_down


def doppler_maps(x, take_abs=True, do_shift=True):
    x_len = x.shape[0]
    num_windows_per_instance = x.shape[1]
    time_wind = x.shape[2]
    num_range_points = x.shape[3]
    num_sensors = x.shape[4]

    if take_abs:
        doppler = np.zeros(
            (x_len, num_windows_per_instance, time_wind, num_range_points, num_sensors),
            dtype=np.float32,
        )  # take the absolute value, thus not complex data type
        for i_x in range(0, x_len):
            for i_instance in range(0, num_windows_per_instance):
                for i_range in range(0, num_range_points):
                    for i_sensor in range(0, num_sensors):
                        if do_shift:
                            doppler[i_x, i_instance, :, i_range, i_sensor] = abs(
                                fftshift(fft(x[i_x, i_instance, :, i_range, i_sensor]))
                            )
                        else:
                            doppler[i_x, i_instance, :, i_range, i_sensor] = abs(
                                fft(x[i_x, i_instance, :, i_range, i_sensor])
                            )

    else:
        doppler = np.zeros(
            (x_len, num_windows_per_instance, time_wind, num_range_points, num_sensors),
            dtype=np.complex64,
        )  # complex value
        for i_x in range(0, x_len):
            for i_instance in range(0, num_windows_per_instance):
                for i_range in range(0, num_range_points):
                    for i_sensor in range(0, num_sensors):
                        if do_shift:
                            doppler[i_x, i_instance, :, i_range, i_sensor] = fftshift(
                                fft(x[i_x, i_instance, :, i_range, i_sensor])
                            )
                        else:
                            doppler[i_x, i_instance, :, i_range, i_sensor] = fft(
                                x[i_x, i_instance, :, i_range, i_sensor]
                            )
    return doppler


def load_data_doppler_map(datasetPath):
    persons = 25
    people = list(range(1, persons, 1))
    rows = (persons - 1) * 11 * 105
    X = np.empty((rows, 5, 32, 492, 2), dtype=np.float32)
    y = np.empty((rows, 5, 1), int)
    i = 0
    temp = 0
    for gdx, gestureName in enumerate(gestures):
        print(gestureName, gdx)
        for pdx, person in enumerate(people):
            path = (
                datasetPath
                + "/"
                + "p"
                + str(person)
                + "/"
                + gestureName
                + "_"
                + "1s_"
                + "wl32_doppl.npy"
            )
            # print(path)
            x = np.load(path)
            x_len = x.shape[0]
            temp = x_len
            X[i : i + x_len] = x
            y[i : i + x_len] = np.zeros((5, 1)) + gdx
            i = i + x_len

    return X[: i - temp], y[: i - temp]


def down_sample_and_save(folder_path, row_factor, col_factor, interpolation=None):
    """
    Down samples all the .npy files in a data_npy folder by the given row and column factors and saves the down sampled
    doppler-range map in a new folder with the same name as the original folder, but with the suffix
    "_down_sample_row_{row_factor}_col_{col_factor}".

    Parameters:
    folder_path (str): The path to the folder containing the .npy files to be down sampled.
    row_factor (int): The down sampling factor for rows.
    col_factor (int): The down sampling factor for columns.

    Returns:
    None
    """
    windowLength = 32
    npy_folder_path = folder_path + "/data_npy"
    new_folder_path = (
        folder_path + f"/data_feat_ds_row_{row_factor}_col_{col_factor}_d_none_u_cubic"
    )
    ensure_path_exists(new_folder_path)
    for gdx, gestureName in enumerate(gestures):
        for pdx, person in enumerate(people):
            path = f"{npy_folder_path}/p{person}/{gestureName}_1s.npy"
            print(path)
            person_folder_path = new_folder_path + f"/p{person}"
            file_path = person_folder_path + f"/{gestureName}_1s_wl32_doppl.npy"

            ensure_path_exists(person_folder_path)

            x = np.load(path)
            numberOfWindows = x.shape[0]
            numberOfSweeps = x.shape[1]
            numberOfRangePoints = x.shape[2]
            numberOfSensors = x.shape[3]

            numberOfSubWindows = int(numberOfSweeps / windowLength)

            x = x.reshape(
                (
                    numberOfWindows,
                    numberOfSubWindows,
                    windowLength,
                    numberOfRangePoints,
                    numberOfSensors,
                )
            )
            for win in range(numberOfWindows):
                for sub_win in range(numberOfSubWindows):
                    for sen in range(numberOfSensors):
                        x[win, sub_win, :, :, sen] = down_sample_data(
                            x[win, sub_win, :, :, sen],
                            row_factor,
                            col_factor,
                            interpolation,
                        )
            x = doppler_maps(x, take_abs=True)
            np.save(file_path, x)


def loadPerson(paramList, scale: bool):
    SubjectData = list()
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
            GestureData = np.load(paramList.pathToFeatures + filename)
            if scale:
                for i in range(GestureData.shape[0]):
                    for j in range(GestureData.shape[1]):
                        for k in range(GestureData.shape[4]):
                            GestureData[i, j, :, :, k] = (
                                GestureData[i, j, :, :, k]
                                / GestureData[i, j, :, :, k].max()
                            )
        except IOError:
            print("Could not open file: " + filename)
            continue
        else:
            if 0 >= GestureData.shape[0]:
                print("Skip datafile (no data): " + filename)
                continue

            SubjectData.append(GestureData)

            for idx in range(0, GestureData.shape[0]):
                GestureLabel = list()
                for jdx in range(0, GestureData.shape[1]):
                    GestureLabel.append(
                        np.identity(len(paramList.listOfGestures))[gestureIdx]
                    )
                SubjectLabel.append(np.asarray(GestureLabel))

            # Check if there is some data for this person
            if (0 >= len(SubjectData)) or (0 >= len(SubjectLabel)):
                print("No entries found for person with index 'p" + str(idx) + "'")
                return

    return np.concatenate(SubjectData, axis=0), np.asarray(SubjectLabel)


def loadFeatures(
    pathtoDataset,
    listOfPeople,
    listofGestures,
    numberofInstanceCopies,
    lengthOfWindow,
    scale,
):
    ParamList = namedtuple(
        "ParamList",
        "personIdx, pathToFeatures, listOfGestures, numberOfInstanceCopies, lengthOfWindow",
    )
    personList = []
    for i in listOfPeople:
        personList.append(
            ParamList(
                i,
                pathtoDataset,
                listofGestures,
                numberofInstanceCopies,
                lengthOfWindow,
            )
        )
    # with Pool(8) as p:
    loadPerson_scale = partial(loadPerson, scale=scale)
    featureList = list(map(loadPerson_scale, personList))
    return featureList


def setupLOOCV(dataX, dataY) -> tuple[Dataset, Dataset]:
    # Split people into train and validation set
    dataX_train = [*dataX[0:validationPerson], *dataX[validationPerson + 1 :]]
    dataY_train = [*dataY[0:validationPerson], *dataY[validationPerson + 1 :]]

    # Set the validation Person used for Leave-one-out cross-validation
    validationPerson = 10
    dataX_val = [dataX[validationPerson]]
    dataY_val = [dataY[validationPerson]]

    # Generate dataset from lists
    traindataset = GestureDataset(dataX_train, dataY_train)
    valdataset = GestureDataset(dataX_val, dataY_val)

    return traindataset, valdataset


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


def get_tiny_radar_data_loader(
    pathToDataset: str,
    listPeople: list[int],
    listGestures: list[str],
    batch_size: int = 128,
    scale: bool = True,
) -> tuple[DataLoader, DataLoader]:
    # Dataset parameters
    numberOfTimeSteps = 5
    numberOfSensors = 2
    numberOfRangePointsPerSensor = 492
    numberOfInstanceWindows = 3
    lengthOfSubWindow = 32
    numberOfGestures = 12

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


if __name__ == "__main__":
    gestures = [
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
    persons = 26
    people = list(range(1, persons, 1))

    path = "/mnt/netaneldata/11G"
    down_sample_and_save(path, 8, 64, None)
