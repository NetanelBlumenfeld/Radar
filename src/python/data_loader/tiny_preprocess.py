import cv2
import numpy as np
from scipy.fftpack import fft, fftshift
from utils.utils_images import Normalization, normalize_img

lengthOfSubWindow = 32


def down_sample_data(
    data: np.ndarray, row_factor: int, col_factor: int, up_sample: bool = False
) -> np.ndarray:
    def _down_sample(img: np.ndarray, row_factor: int, col_factor: int) -> np.ndarray:
        return img[::row_factor, ::col_factor]

    def _up_scale(img: np.ndarray, dim_up: tuple[int, int]) -> np.ndarray:
        real_img = np.real(img)
        imag_img = np.imag(img)
        data_real_up = cv2.resize(real_img, dim_up, interpolation=cv2.INTER_CUBIC)
        data_imag_up = cv2.resize(imag_img, dim_up, interpolation=cv2.INTER_CUBIC)
        return data_real_up + 1j * data_imag_up

    img = _down_sample(data, row_factor, col_factor)
    if up_sample:
        org_dim = (data.shape[1], data.shape[0])
        img = _up_scale(img, org_dim)
    return img


def npy_feat_reshape(x: np.ndarray) -> np.ndarray:
    numberOfWindows = x.shape[0]
    numberOfSweeps = x.shape[1]
    numberOfRangePoints = x.shape[2]
    numberOfSensors = x.shape[3]

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


def doppler_map(x: np.ndarray) -> np.ndarray:
    """Compute the doppler map for velocity of a single instance. The input is a 2D array of shape (freq_ax, range_ax)"""
    assert x.ndim == 2
    freq_ax, range_ax = x.shape
    res = np.zeros((freq_ax, range_ax), dtype=np.float32)
    for i in range(range_ax):
        res[:, i] = abs(fftshift(fft(x[:, i])))
    return res


def pipeline_sr(
    x: np.ndarray,
    row_factor: int,
    col_factor: int,
    up_sample: bool,
    pix_norm: Normalization,
) -> tuple[np.ndarray, np.ndarray]:
    """
    pipeline for super resolution
    params:
    x: input data, the raw data (data_npy)
    """
    assert x.ndim == 4
    x = npy_feat_reshape(x)
    samples, sub_windows, freq_ax, range_ax, sensors = x.shape
    low_res_freq_ax = freq_ax if up_sample else freq_ax // row_factor
    low_res_range_ax = range_ax if up_sample else range_ax // col_factor
    low_res = np.zeros(
        (samples, sub_windows, low_res_freq_ax, low_res_range_ax, sensors),
        dtype=np.float32,
    )
    high_res = np.zeros(
        (samples, sub_windows, freq_ax, range_ax, sensors), dtype=np.float32
    )
    for sample in range(samples):
        for sub_window in range(sub_windows):
            for sensor in range(sensors):
                img = x[sample, sub_window, :, :, sensor]
                low_res_img = down_sample_data(img, row_factor, col_factor, up_sample)
                low_res_doppler = doppler_map(low_res_img)
                high_res_doppler = doppler_map(img)
                low_res_doppler = normalize_img(low_res_doppler, pix_norm)
                high_res_doppler = normalize_img(high_res_doppler, pix_norm)
                low_res[sample, sub_window, :, :, sensor] = low_res_doppler
                high_res[sample, sub_window, :, :, sensor] = high_res_doppler
    return high_res, low_res
