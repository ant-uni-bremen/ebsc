import numpy as np
from ebsc import function_types
import pytest
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate


def test_PiecewiseLinearFunction_input():
    x = np.arange(0, 5, 1)
    c1 = [2, 1, 7, -1]
    teval = np.arange(0, x.max(), 0.01)
    with pytest.raises(NotImplementedError) as e_info:
        plf1 = function_types.PiecewiseLinearFunction(c1, x)


def test_PiecewiseLinearFunction():
    a = np.array([2, 1, 6, 2, 1])
    t = np.arange(0, 5, 1)
    teval = np.arange(0, 5, 0.01)
    coeffs = np.asarray([np.diff(a) / np.diff(t), a[:-1]])

    plf = function_types.PiecewiseLinearFunction(coeffs, t)

    np.testing.assert_allclose(plf(t), a)


def test_PiecewiseLinearFunction__neg__():
    a = np.array([2, 1, 6, 2, 1])
    t = np.arange(0, 5, 1)
    teval = np.arange(0, 5, 0.01)
    coeffs = np.asarray([np.diff(a) / np.diff(t), a[:-1]])

    plf = function_types.PiecewiseLinearFunction(coeffs, t)
    np.testing.assert_allclose(-plf(t), -a)


def test_PiecewiseLinearFunction__add__():
    a1 = np.array([2, 1, 6, 2, 1])
    a2 = np.array([-5, 1, 2, 2, -7])
    t = np.arange(0, 5, 1)
    teval = np.arange(0, 5, 0.01)
    c1 = np.asarray([np.diff(a1) / np.diff(t), a1[:-1]])
    c2 = np.asarray([np.diff(a2) / np.diff(t), a2[:-1]])

    plf1 = function_types.PiecewiseLinearFunction(c1, t)
    plf2 = function_types.PiecewiseLinearFunction(c2, t)
    plf3 = plf1 + plf2
    np.testing.assert_allclose(plf3(t), a1 + a2)


def test_PiecewiseLinearFunction__sub__():
    a1 = np.array([2, 1, 6, 2, 1])
    a2 = np.array([-5, 1, 2, 2, -7])
    t = np.arange(0, 5, 1)
    teval = np.arange(0, 5, 0.01)
    c1 = np.asarray([np.diff(a1) / np.diff(t), a1[:-1]])
    c2 = np.asarray([np.diff(a2) / np.diff(t), a2[:-1]])

    plf1 = function_types.PiecewiseLinearFunction(c1, t)
    plf2 = function_types.PiecewiseLinearFunction(c2, t)
    plf3 = plf1 - plf2
    np.testing.assert_allclose(plf3(t), a1 - a2)


def test_PiecewiseLinearFunction__mul__():
    a = np.array([2, 1, 6, 2, 1])
    t = np.arange(0, 5, 1)
    teval = np.arange(0, 5, 0.01)
    c = np.asarray([np.diff(a) / np.diff(t), a[:-1]])

    plf1 = function_types.PiecewiseLinearFunction(c, t)
    plf2 = plf1 * -102.5
    np.testing.assert_allclose(plf2(t), a * -102.5)


def test_PiecewiseLinearFunction__truediv__():
    a = np.array([2, 1, 6, 2, 1])
    t = np.arange(0, 5, 1)
    teval = np.arange(0, 5, 0.01)
    c = np.asarray([np.diff(a) / np.diff(t), a[:-1]])

    plf1 = function_types.PiecewiseLinearFunction(c, t)
    plf2 = plf1 / 12.55
    np.testing.assert_allclose(plf2(t), a / 12.55)


def test_PiecewiseLinearFunction_add_breakpoints():
    a = np.array([2, 1, 6, 2, 1])
    t = np.arange(0, 5, 1)
    t_new = 0.15
    teval = np.arange(0, 5, 0.01)
    c = np.asarray([np.diff(a) / np.diff(t), a[:-1]])

    plf1 = function_types.PiecewiseLinearFunction(c, t)
    plf2 = plf1._add_breakpoints([t_new])
    np.testing.assert_allclose(plf2(teval), plf1(teval))


def test_PiecewiseLinearFunction_breakpoints_before_coeffs_before():
    a = np.array([2, 1, 6, 2, 1])
    t = np.array([0, 0.1, 0.6, 2, 2.5])
    c = np.asarray([np.diff(a) / np.diff(t), a[:-1]])

    plf1 = function_types.PiecewiseLinearFunction(c, t)
    breakpoints_before = plf1.breakpoints_before(2)
    coeffs_before = plf1.coeffs_before(2)
    np.testing.assert_allclose(breakpoints_before, t[np.where(t < 2)])
    np.testing.assert_allclose(coeffs_before, c[:, np.where(t[:-1] < 2)[0]])


def test_PiecewiseLinearFunction_breakpoints_between_coeffs_between():
    a = np.array([2, 1, 6, 2, 1, 5])
    t = np.array([0, 0.1, 0.6, 2, 2.5, 3])
    c = np.asarray([np.diff(a) / np.diff(t), a[:-1]])

    plf1 = function_types.PiecewiseLinearFunction(c, t)
    breakpoints_between = plf1.breakpoints_between(0.1, 2.5)
    coeffs_between = plf1.coeffs_between(0.1, 2.5)
    np.testing.assert_allclose(breakpoints_between, t[2:4])
    np.testing.assert_allclose(coeffs_between, c[:, 2:4])


def test_PiecewiseLinearFunction_breakpoints_between_including_coeffs_between_including():
    a = np.array([2, 1, 6, 2, 1, 5])
    t = np.array([0, 0.1, 0.6, 2, 2.5, 3])
    c = np.asarray([np.diff(a) / np.diff(t), a[:-1]])

    plf1 = function_types.PiecewiseLinearFunction(c, t)
    breakpoints_between = plf1.breakpoints_between_including(0.1, 2.5)
    coeffs_between = plf1.coeffs_between_including(0.1, 2.5)
    np.testing.assert_allclose(breakpoints_between, t[1:5])
    np.testing.assert_allclose(coeffs_between, c[:, 1:4])

    breakpoints_between = plf1.breakpoints_between_including(0.1, 0.1)
    coeffs_between = plf1.coeffs_between_including(0.1, 0.1)
    np.testing.assert_allclose(breakpoints_between, t[1])
    np.testing.assert_allclose(coeffs_between, [[], []])


def test_PiecewiseLinearFunction_breakpoints_after_coeffs_after():
    a = np.array([2, 1, 6, 2, 1, 5])
    t = np.array([0, 0.1, 0.6, 2, 2.5, 3])
    c = np.asarray([np.diff(a) / np.diff(t), a[:-1]])

    plf1 = function_types.PiecewiseLinearFunction(c, t)
    breakpoints_between = plf1.breakpoints_after(2.2)
    coeffs_between = plf1.coeffs_after(2.2)
    np.testing.assert_allclose(breakpoints_between, t[4:])
    np.testing.assert_allclose(coeffs_between, c[:, 4:])

    breakpoints_between = plf1.breakpoints_after(1.9)
    coeffs_between = plf1.coeffs_after(1.9)
    np.testing.assert_allclose(breakpoints_between, t[3:])
    np.testing.assert_allclose(coeffs_between, c[:, 3:])


def test_PiecewiseLinearFunction_solve():
    import ebsc

    np.random.seed(1)
    len_amp = 5
    t = np.arange(0, len_amp, 1)
    teval = np.arange(0, len_amp, 0.01)
    a = np.random.randint(0, 100, len_amp)
    coeffs = np.asarray([np.diff(a) / np.diff(t), a[:-1]])
    signal = ebsc.PiecewiseLinearFunction(coeffs, t)
    a = np.random.randint(0, 100, len_amp)
    coeffs2 = np.asarray([np.diff(a) / np.diff(t), a[:-1]])
    bound = ebsc.PiecewiseLinearFunction(coeffs2, t)
    test_truth = signal - bound

    signal = function_types.PiecewiseLinearFunction(coeffs, t)
    bound = function_types.PiecewiseLinearFunction(coeffs2, t)
    test = signal - bound

    intersect_truth = test_truth.solve()
    intersect = test.solve()
    intersect = np.squeeze(intersect)
    plt.plot(t, signal(t), label="1")
    plt.plot(t, bound(t), label="2")
    plt.plot(t, test_truth(t), label="diff")
    plt.scatter(intersect_truth, np.zeros(len(intersect_truth)), label="intersect")
    plt.scatter(intersect, np.zeros(len(intersect)), marker="x", label="intersect")
    plt.legend()
    np.testing.assert_allclose(intersect_truth, intersect)


def test_Gamma_inverse():
    pass


def test_AnalogSignal():
    a = np.array([2, 1, 6, 2, 1, 5])
    t = np.array([0, 0.1, 2.5, 0.6, 2, 3])

    with pytest.raises(ValueError) as e_info:
        analog = function_types.AnalogSignal(t, a)
    with pytest.raises(ValueError) as e_info:
        analog = function_types.AnalogSignal(t[1:], a)

    t = np.sort(t)
    analog = function_types.AnalogSignal(t, a)
    np.testing.assert_allclose(analog(t), a)


def test_AnalogSignal_maximum():
    a = np.array([2, 1, 6, 2, 1, 5])
    t = np.array([0, 0.1, 0.6, 2, 2.5, 3])
    analog = function_types.AnalogSignal(t, a)
    analog_max = analog.maximum()
    np.testing.assert_allclose(analog_max, np.max(a))


def test_AnalogSignal_minimum():
    a = np.array([2, 1, 6, 2, 1, 5])
    t = np.array([0, 0.1, 0.6, 2, 2.5, 3])
    analog = function_types.AnalogSignal(t, a)
    analog_min = analog.minimum()
    np.testing.assert_allclose(analog_min, np.min(a))


def test_LocalBandwidth():
    B = np.array([2, 1, 6, 2, 1, 5])
    t = np.array([0, 0.1, 0.6, 2, 2.5, 3])
    with pytest.raises(ValueError) as e_info:
        lbw = function_types.LocalBandwidth(t, -B)
    lbw = function_types.LocalBandwidth(t, B)
    np.testing.assert_allclose(lbw(t), B)


def test_LocalBandwidth_degrees_of_freedom():
    B = np.array([2, 1, 6, 2, 1, 5])
    t = np.array([0, 0.1, 0.6, 2, 2.5, 3])
    lbw = function_types.LocalBandwidth(t, B)
    np.testing.assert_allclose(lbw.degrees_of_freedom, 20)


def test_LocalBandwidth_get_gamma():
    B = np.array([2, 1, 6, 2, 1, 5])
    t = np.array([0, 0.1, 0.6, 2, 2.5, 3])

    lbw = function_types.LocalBandwidth(t, B)
    gamma = lbw.get_gamma()
    temp = lbw.antiderivative()
    gamma_temp = interpolate.PPoly(2 * temp.c, temp.x)
    np.testing.assert_allclose(gamma(t), gamma_temp(t))


def test_calc_sample_times():
    B = np.array([2, 1, 6, 2, 1, 5])
    t = np.array([0, 0.1, 0.6, 2, 2.5, 3])
    st = np.array(
        [
            0.0,
            0.28284271,
            0.42426407,
            0.52915026,
            0.61673333,
            0.70250156,
            0.7921216,
            0.88616429,
            0.98535718,
            1.09065231,
            1.20333705,
            1.32522729,
            1.45903264,
            1.60912879,
            1.78348486,
            2.0,
            2.29289322,
            2.6545085,
            2.82569391,
            2.94782196,
        ]
    )
    lbw = function_types.LocalBandwidth(t, B)
    gamma = lbw.get_gamma()
    sample_times = function_types.calc_sample_times(gamma, lbw.degrees_of_freedom)
    np.testing.assert_allclose(sample_times, st)


def test_VariableBandwidthFunction():
    np.random.seed(1)
    B = np.array([2, 1, 6, 2, 1, 5])
    t = np.array([0, 0.1, 0.6, 2, 2.5, 3])
    vbw_truth = np.array(
        [0.417022, 0.68652836, 0.18395624, 0.67046751, 0.36681728, 0.24147736]
    )
    lbw = function_types.LocalBandwidth(t, B)
    amplitudes = np.random.random(lbw.degrees_of_freedom)
    vbw = function_types.VariableBandwidthFunction(lbw)
    vbw.set_sample_amplitudes(amplitudes)
    np.testing.assert_allclose(vbw.local_bandwidth(t), lbw(t))
    np.testing.assert_allclose(vbw(t), vbw_truth)


def test_VariableBandwidthFunction():
    np.random.seed(1)
    B = np.array([2, 1, 6, 2, 1, 5])
    t = np.array([0, 0.1, 0.6, 2, 2.5, 3])
    vbw_truth = np.array(
        [0.417022, 0.68652836, 0.18395624, 0.67046751, 0.36681728, 0.24147736]
    )
    lbw = function_types.LocalBandwidth(t, B)
    amplitudes = np.random.random(lbw.degrees_of_freedom)
    vbw = function_types.VariableBandwidthFunction(lbw)
    vbw.set_sample_amplitudes(amplitudes)
    np.testing.assert_allclose(vbw.local_bandwidth(t), lbw(t))
    np.testing.assert_allclose(vbw(t), vbw_truth)


def test_VariableBandwidthFunction_to_analog_function():
    np.random.seed(1)
    B = np.array([2, 1, 6, 2, 1, 5])
    t = np.array([0, 0.1, 0.6, 2, 2.5, 3])
    vbw_truth = np.array(
        [0.417022, 0.68652836, 0.18395624, 0.67046751, 0.36681728, 0.24147736]
    )
    lbw = function_types.LocalBandwidth(t, B)
    amplitudes = np.random.random(lbw.degrees_of_freedom)
    vbw = function_types.VariableBandwidthFunction(lbw)
    vbw.set_sample_amplitudes(amplitudes)
    analog = vbw.to_analog_function()
    np.testing.assert_allclose(analog(t[:-1]), vbw_truth[:-1], rtol=1e-02)


def test_BandlimitedFunction():
    a = np.array([2, 1, 6, 2, 1, 5])
    teval = np.arange(0, 6, 0.5)
    truth = np.array(
        [
            2.0,
            1.92844519,
            1.81544356,
            1.66963651,
            1.50201323,
            1.3254285,
            1.15399965,
            1.00241052,
            0.88515551,
            0.81576172,
            0.80602912,
            0.86532911,
        ]
    )
    bandlimited = function_types.BandlimitedFunction(a, 1 / len(a))

    np.testing.assert_allclose(bandlimited(teval), truth)


def test_BandlimitedFunction_t():
    a = np.array([2, 1, 6, 2, 1, 5])
    teval = np.array([0.0, 6.0, 12.0, 18.0, 24.0, 30.0])
    truth = np.array(
        [
            2.0,
            1.92844519,
            1.81544356,
            1.66963651,
            1.50201323,
            1.3254285,
            1.15399965,
            1.00241052,
            0.88515551,
            0.81576172,
            0.80602912,
            0.86532911,
        ]
    )
    bandlimited = function_types.BandlimitedFunction(a, 1 / len(a))
    np.testing.assert_allclose(bandlimited.t(), teval)
