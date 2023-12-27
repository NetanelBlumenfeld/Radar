import numpy as np
from scipy.fftpack import fft, fftshift
from utils.utils_images import Normalization, normalize_img


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
