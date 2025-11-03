"""Basic tests for FK solvers."""

import numpy as np
from fk_1d import fk_solve_1d
from utils import wave_speed, doubling_time, dice_coefficient, cfl_check


def test_growth():
    r = fk_solve_1d(T=50)
    assert r['volumes'][-1] > r['volumes'][0]


def test_wave():
    assert abs(wave_speed(0.012, 0.1) - 2 * np.sqrt(0.012 * 0.1)) < 1e-10


def test_dice():
    m = np.ones((10, 10))
    assert abs(dice_coefficient(m, m) - 1.0) < 1e-6


def test_cfl():
    assert cfl_check(0.1, 1.0, 0.1)['stable'] is True


if __name__ == '__main__':
    test_growth()
    test_wave()
    test_dice()
    test_cfl()
    print("All tests passed.")
