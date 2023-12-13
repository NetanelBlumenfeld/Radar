import os
from typing import Optional

import cv2
import numpy as np
from scipy.fftpack import fft, fftfreq, fftshift


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
persons = 25
people = list(range(1, persons, 1))


def ensure_path_exists(path):
    """
    Checks if a given path exists, and if not, creates it.

    Parameters:
    path (str): The path to be checked and potentially created.

    Returns:
    None
    """
    if not os.path.exists(path):
        os.makedirs(path)


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
    ]
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
        folder_path
        + f"/data_feat_down_sample_row_{row_factor}_col_{col_factor}_data_feat"
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


def sr_tiny_radar_data(folder_path_original, folder_path_ds):
    x_original, y = load_data_doppler_map(folder_path_original)
    x_ds, _ = load_data_doppler_map(folder_path_ds)
    X = np.stack((x_original, x_ds), axis=1)
    return X, y


if __name__ == "__main__":
    path = "/mnt/netaneldata/11G"
    down_sample_and_save(path, 8, 64, None)
