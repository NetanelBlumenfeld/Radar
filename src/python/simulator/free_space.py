import math

import numpy as np
from scipy import constants as consts


class FreeSpace:
    """signal propagation in free space."""

    def __init__(
        self,
        propagation_speed=consts.c,
        operating_frequency=3e8,
        sample_rate=1e6,
        two_way_propagation=True,
    ):
        """
        :param number propagation_speed: speed of signal propagation in this space, default is speed of light
        :param number operating_frequency:
        :param number sample_rate:
        :param bool two_way_propagation: return signal transmitted either from origin to dest (if set to False)
            or in both directinos
        """
        self.propagation_speed = propagation_speed
        self.operating_frequency = operating_frequency
        self.sample_rate = sample_rate
        self._propagation_ratio = 2.0 if two_way_propagation else 1.0

    def step(self, signal, relative_pos, relative_vel):
        """
        Performs signal propagation in space configured.
        :param array-like signal: (maybe complex) amplitude of transmitted signal
        :param array-like relative_pos: relative position between the radar and the target
        :param array-like relative_vel: relative velocity between the radar and the target
        :returns array-like: received signal
        """
        # allocate space for result
        received = np.empty_like(signal, dtype=np.complex_)

        # tau is time shift between signal emission and reception
        distance = np.squeeze(np.linalg.norm(relative_pos))
        tau = self._propagation_ratio * distance / self.propagation_speed
        loss = self._loss(tau)
        phase_shift = self._phase_shift(tau)
        doppler_shift = self._doppler_shift(
            tau, relative_pos, relative_vel, signal.size
        )
        delay_frac, delay_int = math.modf(tau * self.sample_rate)
        delay_int = int(delay_int)

        aux_signal = np.empty(signal.size + 1, dtype=np.complex_)
        aux_signal[0] = 0
        aux_signal[1:] = loss * phase_shift * doppler_shift * signal
        received[:delay_int] = 0
        received[delay_int:] = aux_signal[: -delay_int - 1] * delay_frac + aux_signal[
            1 : aux_signal.size - delay_int
        ] * (1 - delay_frac)

        return received

    def _loss(self, tau):
        loss = np.power(4 * consts.pi * tau * self.operating_frequency, -1)
        return np.squeeze(loss)

    def _phase_shift(self, tau):
        shift = np.exp(-1j * 2 * consts.pi * self.operating_frequency * tau)
        return np.squeeze(shift)

    def _doppler_shift(self, tau, relative_pos, relative_vel, length):
        """
        Returns ndarray of doppler shifts
        """
        vel_projection = -np.dot(relative_vel, relative_pos) / np.linalg.norm(
            relative_pos
        )
        coef = (
            1j
            * 2
            * consts.pi
            * self._propagation_ratio
            * vel_projection
            * self.operating_frequency
            / self.propagation_speed
        )
        shift = np.exp(coef * (tau + np.arange(length) / self.sample_rate))
        return shift
