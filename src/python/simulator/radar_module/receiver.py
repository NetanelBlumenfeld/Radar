import numpy as np

from python.simulator.waves import SignalParams


class Receiver:
    def __init__(self, sig_carrier: np.ndarray, signal_params: SignalParams):
        self.sig_carrier = sig_carrier
        self.signal_params = signal_params

    @property
    def sig_reference(self):
        return 1 / self.sig_carrier

    def receive(self, sig: np.ndarray) -> np.ndarray:
        """receive the signal band pass signal and convert him to base band"""
        # return signal * self.sig_reference
        i = (
            2
            * sig
            * np.cos(
                2 * np.pi * self.signal_params.fc * self.signal_params.time_axis[:-2]
            )
        )
        q = (
            2
            * sig
            * np.sin(
                2 * np.pi * self.signal_params.fc * self.signal_params.time_axis[:-2]
            )
        )
        return i + 1j * q
