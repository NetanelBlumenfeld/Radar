from collections import namedtuple
from functools import partial
from typing import Optional

import cv2
import numpy as np
from scipy.fftpack import fft, fftshift
from utils.utils_images import Normalization, normalize_img
from utils.utils_paths import ensure_path_exists

numberOfInstanceWindows = 3
lengthOfSubWindow = 32


def npy_feat_reshape(x: np.ndarray, classifier_shape: bool = True) -> np.ndarray:
    numberOfWindows = x.shape[0]
    numberOfSweeps = x.shape[1]
    numberOfRangePoints = x.shape[2]
    numberOfSensors = x.shape[3]

    numberOfSubWindows = int(numberOfSweeps / lengthOfSubWindow)

    if classifier_shape:
        x = x.reshape(
            (
                numberOfWindows,
                numberOfSubWindows,
                lengthOfSubWindow,
                numberOfRangePoints,
                numberOfSensors,
            )
        )
    else:
        d0, d1, d2, d3, d4 = x.shape
        x = x.reshape(
            d0 * d1 * d4,
            1,
            d2,
            d3,
        )
        x = x.reshape(-1, d2, d3)
    return x


def normalize_tiny_data(dataX, pix_norm: Normalization):
    """normalize the doppler maps of tiny radar dataset"""
    for i in range(dataX.shape[0]):
        for j in range(dataX.shape[1]):
            for k in range(dataX.shape[4]):
                dataX[i, j, :, :, k] = normalize_img(dataX[i, j, :, :, k], pix_norm)
    return dataX


def normalize_tiny_data_mps(img, pix_norm: Normalization):
    EPSILON = 1e-8

    """normalize the doppler maps of tiny radar dataset"""
    if pix_norm == Normalization.NONE:
        return img
    elif pix_norm == Normalization.Range_0_1:
        return (img - np.min(img)) / (np.max(img) - np.min(img) + EPSILON)
    elif pix_norm == Normalization.Range_neg_1_1:
        return (img - np.min(img)) / (np.max(img) - np.min(img) + EPSILON) * 2 - 1
    else:
        raise ValueError("Unknown normalization type: " + str(type))
    # x = normalize_img(x, pix_norm)
    # return x


def loadPerson(paramList, data_type: str):
    SubjectData = list()
    SubjectLabel = list()
    for gestureIdx, gestureName in enumerate(paramList.listOfGestures):
        # Create filename
        if data_type == "doppler":
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
        elif data_type == "npy":
            filename = "p" + str(paramList.personIdx) + "/" + gestureName + "_1s.npy"

        try:
            GestureData = np.load(paramList.pathToFeatures + filename)
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
    data_type,
) -> list:
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
    loadperson = partial(loadPerson, data_type=data_type)
    featureList = list(map(loadperson, personList))
    return featureList


def doppler_maps_mps(x):
    res = np.zeros(x.shape)
    for i in range(x.shape[1]):
        res = abs(fftshift(fft(x[:, i])))
    return res


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


def load_tiny_data(
    data_dir: str,
    people: list[int],
    gestures: list[str],
    data_type: str = "doppler",
) -> tuple[np.ndarray, np.ndarray]:
    featureList = loadFeatures(
        data_dir,
        people,
        gestures,
        numberOfInstanceWindows,
        lengthOfSubWindow,
        data_type,
    )
    dataX = np.concatenate(list(map(lambda x: x[0], featureList)), axis=0)
    dataY = np.concatenate(list(map(lambda x: x[1], featureList)), axis=0)
    return dataX, dataY


def down_sample_data(
    data: np.ndarray, row_factor: int, col_factor: int, original_dim: bool = False
) -> np.ndarray:
    def _down_sample(img: np.ndarray, row_factor: int, col_factor: int) -> np.ndarray:
        return img[::row_factor, ::col_factor]

    def _up_scale(img: np.ndarray, dim_up: tuple[int, int]) -> np.ndarray:
        real_img = np.real(img)
        imag_img = np.imag(img)
        data_real_up = cv2.resize(real_img, dim_up, interpolation=cv2.INTER_CUBIC)
        data_imag_up = cv2.resize(imag_img, dim_up, interpolation=cv2.INTER_CUBIC)
        return data_real_up + 1j * data_imag_up

    if original_dim:
        res = np.empty_like(data)
        org_dim = (data.shape[3], data.shape[2])
    else:
        res = np.empty(
            (
                data.shape[0],
                data.shape[1],
                data.shape[2] // row_factor,
                data.shape[3] // col_factor,
                data.shape[4],
            ),
            dtype=np.complex64,
        )
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[4]):
                img = _down_sample(data[i, j, :, :, k], row_factor, col_factor)
                if original_dim:
                    img = _up_scale(img, org_dim)

                res[i, j, :, :, k] = img
    return res


def down_sample_and_save(
    folder_path,
    row_factor,
    col_factor,
    gestures,
    people,
    original_dim: bool,
    interpolation=None,
    save: bool = False,
) -> Optional[tuple[list, list]]:
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
    npy_folder_path = folder_path + "/data_npy"
    new_folder_path = (
        folder_path + f"/_row_{row_factor}_col_{col_factor}_d_orgdim_{original_dim}"
    )
    low_res, high_res = [], []
    ensure_path_exists(new_folder_path)
    for gdx, gestureName in enumerate(gestures):
        for pdx, person in enumerate(people):
            path = f"{npy_folder_path}/p{person}/{gestureName}_1s.npy"
            print(path)
            person_folder_path = new_folder_path + f"/p{person}"
            file_path = person_folder_path + f"/{gestureName}_1s_wl32_doppl.npy"

            ensure_path_exists(person_folder_path)

            x = np.load(path)
            x = npy_feat_reshape(x)
            res = down_sample_data(x, row_factor, col_factor, original_dim)
            res = doppler_maps(res, take_abs=True)
            if save:
                np.save(file_path, res)
            else:
                low_res.append(res)
                high_res.append(x)

    return high_res, low_res


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

    folder_path = "/mnt/netaneldata/11G/"
    down_sample_and_save(folder_path, 4, 4, gestures, people, True)
