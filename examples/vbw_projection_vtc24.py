"""
Instantaneous bandwidth estimation and VBW projection example for algorithm described in:
Willuweit, Christopher, et al. "Instantaneous Bandwidth Estimation for Efficient Sampling of Electrocardiograms." 2024 IEEE 99th Vehicular Technology Conference (VTC2024-Spring). IEEE, 2024.
"""

import ebsc
import numpy as np
import matplotlib.pyplot as plt

para = ebsc.BandwidthEstimatorParameters(
    nperseg=256,
    nfft=512,
    window="hann",
    q=0.99,
    f_min=0.1,
)
W_mean = 350
signal = ebsc.generate_vtc23_signal(W_mean)
analog_signal = signal.to_analog_function()
fs = 1 / (analog_signal.t[1] - analog_signal.t[0])
vbw = ebsc.project_analog_signal_to_vbw(analog_signal, fs, para)
vbw_analog = vbw.to_analog_function()
fs_B = 1 / (vbw.local_bandwidth.t[1] - vbw.local_bandwidth.t[0])
nmse_local_bandwidth = ebsc.calc_NMSE(
    signal.local_bandwidth,
    vbw.local_bandwidth,
    fs_B,
    vbw.local_bandwidth.x[0],
    vbw.local_bandwidth.x[-1],
)
nmse_signal = ebsc.calc_NMSE(
    signal.to_analog_function(),
    vbw_analog,
    fs,
    vbw_analog.x[0],
    vbw_analog.x[-1],
)

plt.plot(
    vbw.local_bandwidth.t, signal.local_bandwidth(vbw.local_bandwidth.t), label="$B(t)$"
)
plt.plot(
    vbw.local_bandwidth.t,
    vbw.local_bandwidth(vbw.local_bandwidth.t),
    label="$\hat{B}(t)$",
)
plt.legend()
plt.figure()
plt.plot(vbw_analog.t, analog_signal(vbw_analog.t), label="$x(t)$")
plt.plot(vbw_analog.t, vbw_analog(vbw_analog.t), label="$\hat{x}(t)$")
plt.scatter(
    vbw.sample_times,
    vbw(vbw.sample_times),
    label="nonuniform samples $t_n$",
)
plt.legend()
plt.show()
