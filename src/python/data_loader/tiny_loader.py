import os
from functools import partial
from multiprocessing import Pool
from typing import Callable, Optional

import numpy as np
from data_loader.utils_tiny1 import feat_sr_reshape, npy_feat_reshape
from utils.utils_images import down_sample_img


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


def load_data(
    data_path: str,
    gestures: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    def _get_gesture_index(data_dir: str, gestures: list[str]) -> int:
        file_name = data_dir.split("/")[-1]
        gesture = file_name.split("_")[0]
        return gestures.index(gesture)

    try:
        data = np.load(data_path)
        data = npy_feat_reshape(data)
    except Exception as e:
        raise ValueError(f"failed to load data from {data_path}: {str(e)}")
    else:
        SubjectLabel = []
        if gestures:
            gestureIdx = _get_gesture_index(data_path, gestures)
            gestures_num = len(gestures)
            for idx in range(0, data.shape[0]):
                GestureLabel = []
                for jdx in range(0, data.shape[1]):
                    GestureLabel.append(np.identity(gestures_num)[gestureIdx])
                SubjectLabel.append(np.asarray(GestureLabel))
    return data, np.array(SubjectLabel)


def load_tiny_data(
    data_dir: str, people: int, gestures: list, data_type: str, task: str
):
    res = data_paths(data_dir, people, gestures, data_type)
    num_workers = os.cpu_count()
    load_data_func = partial(load_data, gestures=gestures)
    print(f"loading data with {num_workers} cpu cores")
    with Pool(num_workers) as p:
        data = p.map(load_data_func, res)
    print(f"y - {data[0][1].shape}, len - {len(data)}")
    y = np.concatenate(list(map(lambda x: x[1], data)))
    X = np.concatenate(list(map(lambda x: x[0], data)))
    if task == "sr":
        X = feat_sr_reshape(X)
    print(f"X,y shapes - {X.shape}, {y.shape}")
    return X, y


def load_tiny_data_sr_4090(
    data_dir: str, people: int, gestures: list, data_type: str, load_data_func: Callable
) -> tuple[np.ndarray, np.ndarray]:
    "this function used for loading low res images before training the sr model"
    res = data_paths(data_dir, people, gestures, data_type)
    num_workers = os.cpu_count()
    print(f"down sampling data with {num_workers} cpu cores")
    with Pool(num_workers) as p:
        data = p.map(load_data_func, res)
    high_res = np.concatenate(list(map(lambda x: x[1], data)))
    low_res = np.concatenate(list(map(lambda x: x[0], data)))

    assert high_res.shape[0] == low_res.shape[0]
    return high_res, low_res


def load_tiny_data_sr_classifier_4090(
    data_dir: str, people: int, gestures: list, data_type: str, load_data_func: Callable
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    "this function used for loading low res images before training the sr model"
    res = data_paths(data_dir, people, gestures, data_type)
    num_workers = os.cpu_count()
    print(f"down sampling data with {num_workers} cpu cores")
    with Pool(num_workers) as p:
        data = p.map(load_data_func, res)
    high_res = np.concatenate(list(map(lambda x: x[1], data)))
    low_res = np.concatenate(list(map(lambda x: x[0], data)))
    labels = np.concatenate(list(map(lambda x: x[2], data)))

    assert high_res.shape[0] == low_res.shape[0]
    return low_res, high_res, labels
