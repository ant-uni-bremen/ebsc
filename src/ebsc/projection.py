#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20  15:59:39 2022

Algorithm described in 
Willuweit, Christopher, et al. "Instantaneous Bandwidth Estimation for Efficient Sampling of Electrocardiograms." 2024 IEEE 99th Vehicular Technology Conference (VTC2024-Spring). IEEE, 2024.
@author: christopher
"""

import numpy as np
from scipy.signal import spectrogram
from .function_types import LocalBandwidth, VariableBandwidthFunction, AnalogSignal
from dataclasses import dataclass
from typing import Optional


@dataclass(kw_only=True)
class SpectrogramParameters:
    nperseg: int
    nfft: int
    window: str
    noverlap: Optional[int] = None

    def __post_init__(self):
        if self.noverlap is None:
            self.noverlap = self.nperseg - 1


@dataclass(kw_only=True)
class BandwidthEstimatorParameters(SpectrogramParameters):
    q: float
    f_min: float


def sgram_bandwidth_estimator(
    x, f_s, estimation_para: BandwidthEstimatorParameters
) -> LocalBandwidth:
    ff, tt, Sxx = spectrogram(
        x,
        f_s,
        noverlap=estimation_para.noverlap,
        nperseg=estimation_para.nperseg,
        nfft=estimation_para.nfft,
        window=estimation_para.window,
        scaling="spectrum",
    )
    cum_Sxx = np.cumsum(Sxx, axis=0)

    mean_seg_energy = np.mean(cum_Sxx[-1, :])
    thresh = cum_Sxx[-1, :] - mean_seg_energy * (1 - estimation_para.q)

    est_B = np.zeros(len(tt))
    delta_f = ff[1] - ff[0]
    for seg_idx in range(len(tt)):
        w = np.where(cum_Sxx[:, seg_idx] < thresh[seg_idx])

        if len(w[0]) == 0:
            est_B[seg_idx] = estimation_para.f_min
        else:
            idx = np.max(w)
            B_tmp = ff[idx] + delta_f * (thresh[seg_idx] - cum_Sxx[idx, seg_idx]) / (
                cum_Sxx[idx + 1, seg_idx] - cum_Sxx[idx, seg_idx]
            )
            est_B[seg_idx] = max(B_tmp, estimation_para.f_min)

    return LocalBandwidth(tt, np.asarray(est_B))


def project_analog_signal_to_vbw(
    analog_signal: AnalogSignal,
    fs: float,
    estimation_para: BandwidthEstimatorParameters,
) -> VariableBandwidthFunction:
    local_bandwidth = sgram_bandwidth_estimator(
        analog_signal.a,
        fs,
        estimation_para,
    )
    vbw = VariableBandwidthFunction(local_bandwidth)
    vbw.set_sample_amplitudes(analog_signal(vbw.sample_times))
    return vbw
