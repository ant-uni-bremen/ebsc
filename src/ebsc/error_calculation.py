#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 11:59:17 2022

@author: christopher
"""
import numpy as np
from .function_types import AnalogSignal, BandlimitedFunction
from numpy.typing import NDArray
from typing import Union


def calc_squared_error(
    vbw1: Union[BandlimitedFunction, AnalogSignal],
    vbw2: Union[BandlimitedFunction, AnalogSignal],
    t: NDArray,
) -> NDArray:
    return (vbw1(t) - vbw2(t)) ** 2


def calc_absolute_error(
    vbw1: Union[BandlimitedFunction, AnalogSignal],
    vbw2: Union[BandlimitedFunction, AnalogSignal],
    t: NDArray,
) -> NDArray:
    return np.abs(vbw1(t) - vbw2(t))


def calc_sample_time_range(signal_1: AnalogSignal, signal_2: AnalogSignal):
    t_min = max(signal_1.x[0], signal_2.x[0])
    t_max = min(signal_1.x[-1], signal_2.x[-1])
    return t_min, t_max


def generate_regular_eval_times(t_min: float, t_max: float, f_s: float) -> NDArray:
    first_sample_time = t_min
    number_samples = int(np.floor((t_max - t_min) * (f_s)))  # war +1
    last_sample_time = t_min + (number_samples - 1) / (f_s)
    return np.linspace(first_sample_time, last_sample_time, number_samples)


def calc_signal_energy(
    signal: AnalogSignal,
    f_s: float,
    t_min: Union[None, float] = None,
    t_max: Union[None, float] = None,
) -> float:
    if t_min == None:
        t_min = signal.x[0]
    if t_max == None:
        t_max = signal.x[-1]

    t_eval = generate_regular_eval_times(t_min, t_max, f_s)
    return np.mean(signal(t_eval) ** 2)


def calc_MSE(
    signal_1: AnalogSignal,
    signal_2: AnalogSignal,
    f_s: float,
    t_min: float,
    t_max: float,
) -> float:
    """
    Calculates the MSE based on a regular sampling of signal_1 and signal_2.
    The sampling times will be at distance 1/f_s between on the valid time-
    range of both signals. The valid time range is detemined from breakpoints
    of signal_1 and signal_2
    """

    t_eval = generate_regular_eval_times(t_min, t_max, f_s)

    squared_error = calc_squared_error(signal_1, signal_2, t_eval)

    return np.mean(squared_error)


def calc_NMSE(
    true_signal: AnalogSignal,
    test_signal: AnalogSignal,
    f_s: float,
    t_min: float,
    t_max: float,
) -> float:
    """
    Calculates the MSE normalized to the Energy of true_signal
    """
    return calc_MSE(true_signal, test_signal, f_s, t_min, t_max) / calc_signal_energy(
        true_signal, f_s, t_min, t_max
    )


def get_vbw_time_range(t_n: NDArray) -> float:
    """
    Args:
        t_n (NDArray): Nonuniform sampling times of a VariableBandwidthFunction.

    Returns:
        float: Time between first and last sample.
    """
    return np.max(t_n) - np.min(t_n)


def calc_ASR(t_n: NDArray) -> float:
    """
    Args:
        t_n (NDArray): Nonuniform sampling times of a VariableBandwidthFunction.

    Returns:
        float: Average sampling rate.
    """
    section_time = get_vbw_time_range(t_n)
    asr = len(t_n) / section_time
    return asr
