"""
VBW test signal generation example for:
Willuweit, Christopher, Carsten Bockelmann, and Armin Dekorsy.
"Energy and bandwidth efficiency of event-based communication." 
2023 IEEE 97th Vehicular Technology Conference (VTC2023-Spring). IEEE, 2023.
"""
import ebsc
import numpy as np
import matplotlib.pyplot as plt

sigma = 0.1
W_mean_list = [325.0, 475.0, 625.0, 775.0, 925.0]
s_max = 4

N_L_list = np.arange(10, 105, 5)

t = np.arange(0, 1.0, 0.005)
t_W = np.arange(0, 1.001, 0.001)


# Test signal generation
vbw_signals = []
for W_mean in W_mean_list:
    vbw = ebsc.generate_bounded_vtc23_signal(W_mean, lower_bound=-s_max, upper_bound=s_max)
    vbw_signals.append(vbw)

plt.figure()
for vbw in vbw_signals:
    plt.plot(t_W, vbw.local_bandwidth(t_W), label="$\\bar{W}$=" + str(W_mean))
plt.legend()

plt.figure()
for vbw, W_mean in zip(vbw_signals, W_mean_list):
    plt.plot(t, vbw(t), label="$\\bar{W}$=" + str(W_mean))
plt.hlines([-4, 4], t.min(), t.max(), color="black", label="bounds")
plt.legend()
plt.show()