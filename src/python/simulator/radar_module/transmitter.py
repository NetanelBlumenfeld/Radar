import numpy as np


class Transmitter:
    # read about anntena gain
    def __init__(
        self,
        Peek_power: float,
        gain: float,
        fc: int,
        signal_amplitude: np.ndarray,
        sig_carrier: np.ndarray,
    ):
        self.Peek_power = Peek_power
        self.gain = gain
        self.c = 299792458
        self.fc = fc
        self.signal_amplitude = signal_amplitude
        self.sig_carrier = sig_carrier
        assert signal_amplitude.shape == sig_carrier.shape

    @property
    def transmit(self):
        """transmit the band pass signal"""
        return self.signal_amplitude * self.sig_carrier

    def power_density_point(self, R):
        """the power density at a point in space"""
        return (self.gain * self.Peek_power) / (4 * np.pi * R**2)  # W/m^2

    def power_density(self, rsc, R):
        """
        the power density of scatter in space
        params:
        - rsc:  radar cross section
        - R:  distance from the radar
        """
        return self.power_density_point(R) * rsc
