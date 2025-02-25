"""
SoD sampling example for:
Willuweit, Christopher, Carsten Bockelmann, and Armin Dekorsy.
"Energy and bandwidth efficiency of event-based communication." 
2023 IEEE 97th Vehicular Technology Conference (VTC2023-Spring). IEEE, 2023.
"""
import ebsc
import numpy as np
import matplotlib.pyplot as plt

sigma = 0.1
W_mean = 325.0
s_max = 4

N_L_list = np.arange(10, 35, 5)

t = np.arange(0, 1.0, 0.0001)

# Test signal generation
vbw = ebsc.generate_bounded_vtc23_signal(W_mean, lower_bound=-s_max, upper_bound=s_max)
analog = vbw.to_analog_function()

# Test signal sampling
sample_list = []
for N_L in N_L_list:
    print(N_L)
    ref_levels = np.linspace(-4, 4, N_L)
    detector = ebsc.SendOnDeltaDetector(ref_levels, 0)
    events = detector.get_events_list(analog)
    samples = ebsc.lc_events_to_samples(events, ref_levels)
    sample_list.append(samples)


for N_L, samples in zip(N_L_list, sample_list):
    plt.figure()
    plt.title("N_L=" + str(N_L))
    plt.plot(t, analog(t), label="x(t)")
    plt.scatter(samples.get_t_as_array(), samples.get_a_as_array(), marker="x", label="SoD")
    plt.legend()
