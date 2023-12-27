import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy.fftpack import fft, fftshift
from utils.utils_images import Normalization, down_sample_img, normalize_img


def down_sample_data_sr(
    x: np.ndarray, row_factor: int, col_factor: int, original_dim: bool = False
) -> np.ndarray:
    assert x.ndim == 3
    if original_dim:
        res = np.empty_like(x)
    else:
        res = np.empty(
            (
                x.shape[0],
                x.shape[1] // row_factor,
                x.shape[2] // col_factor,
            ),
            dtype=np.complex64,
        )
    x_len = x.shape[0]
    for i in range(x_len):
        res[i] = down_sample_img(x[i], row_factor, col_factor, original_dim)
    return res


def data_paths(data_dir: str, people: int, gestures: list, data_type: str) -> list[str]:
    lengthOfSubWindow = 32
    if data_type == "doppler":
        folder_path = os.path.join(data_dir, "data_feat")
    elif data_type == "npy":
        folder_path = os.path.join(data_dir, "data_npy")
    else:
        raise ValueError("data type must be npy or doppler")
    data_paths = []
    for person in range(1, people):
        for gesture in gestures:
            if data_type == "npy":
                file_name = f"p{str(person)}/{gesture}_1s.npy"
            elif data_type == "doppler":
                file_name = (
                    f"p{str(person)}/{gesture}_1s_wl{lengthOfSubWindow}_doppl.npy"
                )
            else:
                raise ValueError("data type must be npy or doppler")
            data_paths.append(os.path.join(folder_path, file_name))
    return data_paths


def load_data(data_paths: str, with_labels: bool = False) -> np.ndarray:
    # TODO: add labels
    try:
        data = np.load(data_paths)
    except Exception as e:
        raise ValueError(f"failed to load data from {data_paths}: {str(e)}")
    if with_labels:
        pass
    return data


def npy_feat_reshape(x: np.ndarray) -> np.ndarray:
    numberOfWindows = x.shape[0]
    numberOfSweeps = x.shape[1]
    numberOfRangePoints = x.shape[2]
    numberOfSensors = x.shape[3]
    lengthOfSubWindow = 32

    numberOfSubWindows = int(numberOfSweeps / lengthOfSubWindow)

    x = x.reshape(
        (
            numberOfWindows,
            numberOfSubWindows,
            lengthOfSubWindow,
            numberOfRangePoints,
            numberOfSensors,
        )
    )
    return x


def feat_sr_reshape(x: np.ndarray) -> np.ndarray:
    """
    reshape data to fit the SR model input
    output shape is (samples*sub_windows*sensors,range_points, doppler_points)
    """
    assert x.ndim == 5, "data must be 5D"
    x_reshape = x.transpose(0, 1, 4, 2, 3)
    samples, sub_windows, sensors, range_points, doppler_points = x_reshape.shape
    x_reshape = x_reshape.reshape(
        samples * sub_windows * sensors, range_points, doppler_points
    )
    return x_reshape


def doppler_map(x: np.ndarray, ax: int = 1) -> np.ndarray:
    """input shape is (N,doppler_points,range_points)"""
    assert x.ndim == 3, "data must be 3D"
    return np.abs(fftshift(fft(x, axis=ax), axes=ax))


def normalize_sr_data(
    x: np.ndarray, norm: Normalization = Normalization.Range_0_1
) -> np.ndarray:
    """input shape is (N,doppler_points,range_points)"""
    assert x.ndim == 3, "data must be 3D"
    x_len = x.shape[0]
    for i in range(x_len):
        x[i] = normalize_img(x[i], norm)
    return x


def load_tiny_data_sr_pipeline(
    path: str, norm_func, down_sample_func
) -> tuple[np.ndarray, np.ndarray]:
    print(f"loading data from -- {path}")
    high_res_time = load_data(path)
    high_res_time = feat_sr_reshape(npy_feat_reshape(high_res_time))
    low_res_time = down_sample_func(high_res_time)
    high_res = doppler_map(high_res_time)
    high_res = norm_func(high_res)
    low_res = doppler_map(low_res_time)
    low_res = norm_func(low_res)
    high_res = high_res[~np.all(high_res == 0, axis=(1, 2))]
    low_res = low_res[~np.all(low_res == 0, axis=(1, 2))]
    return low_res, high_res


def load_tiny_data_sr(
    data_dir: str, people: int, gestures: list, data_type: str, pix_norm: Normalization
) -> tuple[np.ndarray, np.ndarray]:
    res = data_paths(data_dir, people, gestures, data_type)
    ds_func = partial(
        down_sample_data_sr, row_factor=4, col_factor=4, original_dim=False
    )
    norm_func = partial(normalize_img, pix_norm=pix_norm)
    num_workers = os.cpu_count()
    load_data_func = partial(
        load_tiny_data_sr_pipeline, norm_func=norm_func, down_sample_func=ds_func
    )
    print(f"down sampling data with {num_workers} cpu cores")
    with Pool(num_workers) as p:
        data = p.map(load_data_func, res)
    print("concatenating data")
    high_res = np.concatenate(list(map(lambda x: x[1], data)))
    low_res = np.concatenate(list(map(lambda x: x[0], data)))

    assert high_res.shape[0] == low_res.shape[0]
    return high_res, low_res


def load_tiny_data(
    data_dir: str, people: int, gestures: list, data_type: str
) -> np.ndarray:
    res = data_paths(data_dir, people, gestures, data_type)
    num_workers = os.cpu_count()
    load_data_func = partial(load_data, with_labels=False)
    print(f"loading data with {num_workers} cpu cores")
    with Pool(num_workers) as p:
        data = p.map(load_data_func, res)
    print("concatenating data")
    data = np.concatenate(data)
    data = feat_sr_reshape(npy_feat_reshape(data))

    return data
