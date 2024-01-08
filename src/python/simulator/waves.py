import numpy as np
from pydantic import BaseModel


class SignalParams(BaseModel):
    fc: float
    fs: float
    prf: float
    phase: float
    pulse_width: float

    @property
    def time_axis(self):
        return np.arange(0, 1 / self.prf, 1 / self.fs)


def pulse_wave(
    prf: int, pulse_width: float, fs: float, time_scale: float = 1
) -> np.ndarray:
    """
    Generates a pulse wave with a given max range, prf and sampling frequency.
    """
    # Generating the pulse waveform
    pulse = np.arange(0, 1 / prf, 1 / fs) * 0
    pulse_width_samples = round(
        pulse_width * fs / time_scale
    )  # Number of samples that the pulse is 'on'
    pulse[
        :pulse_width_samples
    ] = 1  # Set the pulse 'on' for the duration of the pulse width
    return pulse.reshape(-1, 1)


def get_carrier(fc: int, fs: int, prf: int, init_phase: float) -> np.ndarray:
    time = np.arange(0, 1 / prf, 1 / fs)
    return np.sin(2 * np.pi * fc * time + init_phase)
