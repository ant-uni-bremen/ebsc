"""
Created on Fri Nov  4 15:59:39 2022

@author: christopher
"""

import numpy as np
from .function_types import LocalBandwidth, VariableBandwidthFunction


def generate_vtc23_signal(W_mean, sigma_gauss=0.1):
    # Generate local bandwidth
    t_W = np.arange(0, 1.001, 0.001)
    W = 4 / 3 * (W_mean - 250.0) + (4000.0 / 3 - 4 / 3 * W_mean) * np.exp(
        -((t_W - 0.5) ** 2) / (2 * sigma_gauss**2)
    )
    local_bandwidth = LocalBandwidth(t_W, W)

    # Generate actual signal
    variable_bandwidth_signal = VariableBandwidthFunction(local_bandwidth)
    variable_bandwidth_signal.set_sample_amplitudes(
        np.random.randn(variable_bandwidth_signal.degrees_of_freedom)
    )

    return variable_bandwidth_signal


def generate_bounded_vtc23_signal(
    W_mean, sigma_gauss=0.1, lower_bound=-4, upper_bound=4
):
    while True:
        signal = generate_vtc23_signal(W_mean, sigma_gauss=sigma_gauss)
        analog_signal = signal.to_analog_function()
        if (
            analog_signal.minimum() > lower_bound
            and analog_signal.maximum() < upper_bound
        ):
            break
    return signal


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t_eval = np.arange(0, 1, 0.0001)

    test_signal = generate_bounded_vtc23_signal(100, 1000)

    plt.plot(t_eval, test_signal(t_eval))
