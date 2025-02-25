import numpy as np
from ebsc import signal_generators
import pytest


def test_generate_vtc23_signal():
    W_mean = 250
    t = np.arange(0, 1, 0.01)
    variable_bandwidth_signal = signal_generators.generate_vtc23_signal(W_mean)
    variable_bandwidth_signal_at_t = variable_bandwidth_signal(t)
    W_mean_calc = variable_bandwidth_signal.local_bandwidth(t).mean()
    np.testing.assert_approx_equal(W_mean_calc, W_mean, significant=3)


def test_generate_bounded_vtc23_signal():
    W_mean = 250
    t = np.arange(0, 1, 0.01)
    lower_bound = -4
    upper_bound = 4
    variable_bandwidth_signal = signal_generators.generate_bounded_vtc23_signal(
        W_mean, lower_bound=-4, upper_bound=4
    )
    variable_bandwidth_signal_at_t = variable_bandwidth_signal(t)
    W_mean_calc = variable_bandwidth_signal.local_bandwidth(t).mean()
    vbw_max = variable_bandwidth_signal_at_t.max()
    vbw_min = variable_bandwidth_signal_at_t.min()
    np.testing.assert_approx_equal(W_mean_calc, W_mean, significant=3)
    assert vbw_max < upper_bound
    assert vbw_min > lower_bound
