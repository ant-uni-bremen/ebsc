import ebsc
from ebsc import projection, error_calculation, signal_generators
import matplotlib.pyplot as plt
import numpy as np


def test_sgram_bandwidth_estimator():
    para = projection.BandwidthEstimatorParameters(
        nperseg=256,
        nfft=512,
        window="hann",
        q=0.99,
        f_min=0.1,
    )
    W_mean = 250
    signal = signal_generators.generate_vtc23_signal(W_mean)
    analog_signal = signal.to_analog_function()
    fs = 1 / (analog_signal.t[1] - analog_signal.t[0])
    local_bandwidth = projection.sgram_bandwidth_estimator(analog_signal.a, fs, para)
    fs_B = 1 / (local_bandwidth.t[1] - local_bandwidth.t[0])
    nmse = error_calculation.calc_NMSE(
        signal.local_bandwidth,
        local_bandwidth,
        fs_B,
        local_bandwidth.x[0],
        local_bandwidth.x[-1],
    )
    plt.plot(local_bandwidth.t, signal.local_bandwidth(local_bandwidth.t))
    plt.plot(local_bandwidth.t, local_bandwidth(local_bandwidth.t))
    assert nmse < 1e-2


def test_project_analog_signal_to_vbw():
    para = projection.BandwidthEstimatorParameters(
        nperseg=256,
        nfft=512,
        window="hann",
        q=0.99,
        f_min=0.1,
    )
    W_mean = 350
    signal = signal_generators.generate_vtc23_signal(W_mean)
    analog_signal = signal.to_analog_function()
    fs = 1 / (analog_signal.t[1] - analog_signal.t[0])
    vbw = projection.project_analog_signal_to_vbw(analog_signal, fs, para)
    vbw_analog = vbw.to_analog_function()
    fs_B = 1 / (vbw.local_bandwidth.t[1] - vbw.local_bandwidth.t[0])
    nmse_local_bandwidth = error_calculation.calc_NMSE(
        signal.local_bandwidth,
        vbw.local_bandwidth,
        fs_B,
        vbw.local_bandwidth.x[0],
        vbw.local_bandwidth.x[-1],
    )
    nmse_signal = error_calculation.calc_NMSE(
        signal.to_analog_function(),
        vbw_analog,
        fs,
        vbw_analog.x[0],
        vbw_analog.x[-1],
    )

    plt.plot(vbw.local_bandwidth.t, signal.local_bandwidth(vbw.local_bandwidth.t))
    plt.plot(vbw.local_bandwidth.t, vbw.local_bandwidth(vbw.local_bandwidth.t))
    plt.figure()
    plt.plot(vbw_analog.t, analog_signal(vbw_analog.t))
    plt.plot(vbw_analog.t, vbw_analog(vbw_analog.t))
    assert nmse_local_bandwidth < 1e-2
    assert nmse_signal < 1e-2
