import numpy as np
from data_loader.tiny_loader import load_data
from data_loader.utils_tiny1 import doppler_map, feat_sr_reshape


def sr_4090_pipeline(
    path: str, norm_func, down_sample_func
) -> tuple[np.ndarray, np.ndarray]:
    high_res_time = load_data(path)[0]
    high_res_time = feat_sr_reshape(high_res_time)
    low_res_time = down_sample_func(high_res_time)
    high_res = doppler_map(high_res_time)
    high_res = norm_func(high_res)
    low_res = doppler_map(low_res_time)
    low_res = norm_func(low_res)
    high_res = high_res[~np.all(high_res == 0, axis=(1, 2))]
    low_res = low_res[~np.all(low_res == 0, axis=(1, 2))]
    return low_res, high_res


def sr_time_4090_pipeline(
    path: str, norm_func, down_sample_func
) -> tuple[np.ndarray, np.ndarray]:
    high_res_time = load_data(path)[0]
    high_res_time = feat_sr_reshape(high_res_time)
    low_res_time = down_sample_func(high_res_time)
    high_res = norm_func(high_res_time)
    low_res = norm_func(low_res_time)
    high_res = high_res[~np.all(high_res == 0, axis=(1, 2))]
    low_res = low_res[~np.all(low_res == 0, axis=(1, 2))]

    high_res = np.stack([np.real(high_res), np.imag(high_res)], axis=1)
    low_res = np.stack([np.real(low_res), np.imag(low_res)], axis=1)

    return low_res, high_res


def sr_classifier_4090_pipeline(
    path: str, gestures: list[str], down_sample_func, norm_func
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def _set_low_res(img: np.ndarray, down_sp_func) -> tuple[int, int]:
        low_res = down_sp_func(img)
        return low_res.shape[1:]

    high_res_time, labels = load_data(path, gestures)
    high_res_time = high_res_time.transpose(0, 1, 4, 2, 3)  # (N,5,2,32,492)
    result_high_res = np.zeros(high_res_time.shape, dtype=np.float32)
    low_res_example = _set_low_res(high_res_time[0, 0], down_sample_func)
    low_res_dim = [
        high_res_time.shape[0],
        high_res_time.shape[1],
        high_res_time.shape[2],
        low_res_example[0],
        low_res_example[1],
    ]
    result_low_res = np.zeros(low_res_dim, dtype=np.float32)
    for sample in range(high_res_time.shape[0]):
        for time_step in range(high_res_time.shape[1]):
            high_res = doppler_map(high_res_time[sample, time_step])
            result_high_res[sample, time_step] = norm_func(high_res)

            low_res_time = down_sample_func(high_res_time[sample, time_step])
            low_res = doppler_map(low_res_time)
            result_low_res[sample, time_step] = norm_func(low_res)

    return result_low_res, result_high_res, labels


# pipeline for sr_classifier - 3080
def sr_classifier_3080_pipeline(
    high_res_time: np.ndarray, norm_func, down_sample_func
) -> tuple[np.ndarray, np.ndarray]:
    low_res_list, high_res_list = [], []
    for time_step in range(high_res_time.shape[0]):
        high_res = doppler_map(high_res_time[time_step])
        high_res = norm_func(high_res)
        low_res_time = down_sample_func(high_res_time[time_step])
        low_res = doppler_map(low_res_time)
        low_res = norm_func(low_res)
        low_res_list.append(low_res)
        high_res_list.append(high_res)
    return (
        np.array(low_res_list),
        np.array(high_res_list),
    )


def classifier_pipeline(X: np.ndarray, norm_func) -> np.ndarray:
    X = X.transpose(0, 1, 4, 2, 3)
    res = np.zeros(X.shape, dtype=np.float32)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                data = X[i, j, k]
                data = norm_func(doppler_map(data, ax=0))
                res[i, j, k] = data
    return res
