import numpy as np

from python.simulator.waves import get_carrier, pulse_wave

prf = 1
fs = 100
pulse_width = 0.1
pulse = pulse_wave(prf, pulse_width, fs)


def test_pule_wave_shape():
    assert pulse.shape == (100, 1)


def test_pulse_wave_width():
    assert np.all(pulse[:9] == 1)
    assert np.all(pulse[10:] == 0)


def test_carrier_phase():
    fc = 5
    fs = 100
    prf = 1
    init_phase = 0
    carrier = get_carrier(fc, fs, prf, init_phase)
    assert carrier[0] == 0
    init_phase = 90
    carrier = get_carrier(fc, fs, prf, init_phase)
    assert carrier[0] > carrier[1]
