"""
Created on Fri Nov  4 15:51:52 2022

@author: christopher
"""

import numpy as np
from scipy.interpolate import PPoly
import bisect
import warnings
from numpy.typing import NDArray


class PiecewiseLinearFunction(PPoly):
    """
    Provides additional methods for adding/subtracting PPolys from each other
    and multiplying/dividing PPolys with/by a constant factor
    """

    def __init__(self, c: NDArray, x: NDArray, extrapolate: bool | None = None) -> None:
        c = np.asarray(c)
        if c.shape[0] == 1:
            c = np.vstack([np.zeros(c.shape[1]), c])
        elif c.shape[0] > 2:
            raise NotImplementedError("Currently only piecewise linear is supported.")
        super().__init__(c, x, extrapolate=extrapolate)

    def __neg__(self) -> "PiecewiseLinearFunction":
        return PiecewiseLinearFunction(-self.c, self.x)

    def __add__(self, other: "PiecewiseLinearFunction") -> "PiecewiseLinearFunction":
        # Find and add additional breakpoints for self
        extended_self = self._add_breakpoints(other.x)
        extended_other = other._add_breakpoints(self.x)

        sum_breakpoints = extended_self.x
        sum_coefficients = extended_self.c + extended_other.c

        return PiecewiseLinearFunction(sum_coefficients, sum_breakpoints)

    def __sub__(self, other: "PiecewiseLinearFunction") -> "PiecewiseLinearFunction":
        return self + (-other)

    def __mul__(self, other: float) -> "PiecewiseLinearFunction":
        return PiecewiseLinearFunction(other * self.c, self.x)

    def __truediv__(self, other: float) -> "PiecewiseLinearFunction":
        return PiecewiseLinearFunction(self.c / other, self.x)

    def _calc_new_breakpoint_coefficients(self, x_new: NDArray) -> NDArray:
        """
        Calculate the new coefficients for single breakpoint (new or already
        existing)

        TODO: might be better to call self.__call__() instead of calculating
        the offset at x_new manually? Easier in every case...
        """

        if x_new in self.x:
            index = np.argwhere(self.x == x_new)
            if index == len(self.x) - 1:
                return np.asarray(
                    [
                        [self.c[0, -1]],
                        [self.c[1, -1] - self.c[0, -1] * (self.x[-2] - x_new)],
                    ]
                )
            else:
                return self.c[:, index[0]]
        else:
            if x_new < self.x[0]:
                c = np.asarray(
                    [
                        [self.c[0, 0]],
                        [self.c[1, 0] - self.c[0, 0] * (self.x[0] - x_new)],
                    ]
                )
            elif x_new > self.x[-2]:
                c = np.asarray(
                    [
                        [self.c[0, -1]],
                        [self.c[1, -1] - self.c[0, -1] * (self.x[-2] - x_new)],
                    ]
                )
            else:
                index = bisect.bisect_left(self.x, x_new) - 1
                c = np.asarray(
                    [
                        [self.c[0, index]],
                        [self.c[1, index] - self.c[0, index] * (self.x[index] - x_new)],
                    ]
                )
        return c

    def _add_breakpoints(
        self, additional_breakpoints: NDArray
    ) -> "PiecewiseLinearFunction":
        """
        Create a new PiecewiseLinearFunction-object with additional breakpoints,
        but representing the same mathematical function
        """

        total_breakpoints = np.unique(
            np.concatenate((self.x, np.asarray(additional_breakpoints)))
        )

        total_coefficients = []

        for bp in total_breakpoints[:-1]:
            total_coefficients.append(self._calc_new_breakpoint_coefficients(bp))

        total_coefficients = np.asarray(total_coefficients).squeeze().transpose()

        return PiecewiseLinearFunction(total_coefficients, total_breakpoints)

    def breakpoints_before(self, xmax: float) -> NDArray[np.floating]:
        indices = np.where(self.x < xmax)
        return self.x[indices]

    def coeffs_before(self, xmax: float) -> NDArray[np.floating]:
        indices = np.nonzero(self.x < xmax)
        return self.c[:, indices[0]]

    def breakpoints_between(self, xmin: float, xmax: float) -> NDArray[np.floating]:
        indices = np.where(np.logical_and(self.x > xmin, self.x < xmax))
        return self.x[indices]

    def coeffs_between(self, xmin: float, xmax: float) -> NDArray[np.floating]:
        indices = np.nonzero(np.logical_and(self.x > xmin, self.x < xmax))
        return self.c[:, indices[0]]

    def breakpoints_between_including(
        self, xmin: float, xmax: float
    ) -> NDArray[np.floating]:
        indices = np.where(np.logical_and(self.x >= xmin, self.x <= xmax))
        return self.x[indices]

    def coeffs_between_including(
        self, xmin: float, xmax: float
    ) -> NDArray[np.floating]:
        indices = np.nonzero(np.logical_and(self.x >= xmin, self.x <= xmax))
        return self.c[:, indices[0][:-1]]

    def breakpoints_after(self, xmin: float) -> NDArray[np.floating]:
        indices = np.where(self.x > xmin)
        return self.x[indices]

    def coeffs_after(self, xmin: float) -> NDArray[np.floating]:
        indices = np.nonzero(self.x > xmin)
        if len(indices[0]) < 2:
            return np.asarray([[], []])
        else:
            return self.c[:, indices[0][:-1]]


class Gamma(PiecewiseLinearFunction):
    """
    DO NOT INSTANTIATE DIRECTLY. Only from LocalBandwidth.get_gamma()

    A continuous and strict monotonically increasing piecewise linear function
    Both is ensured by generation from LocalBandwidth
    Thus it can be inverted.
    """

    def __init__(self, c: NDArray, x: NDArray) -> None:
        super().__init__(c, x, extrapolate=True)

    def inverse(self) -> "Gamma":
        x_inverted = self(self.x)
        c_inverted = None

        return Gamma(c_inverted, x_inverted)


class AnalogSignal(PiecewiseLinearFunction):
    """
    Basically a continuous linear-PPoly without extrapolation
    Due to continuity, the constructor interface can be simpler than PPoly

    Since it relies on linear interpolation, a bandlimited analog signal should
    be properly upsampled (sinc-interpolated) before conversion to AnalogSignal
    """

    def __init__(self, t: NDArray, a: NDArray, extrapolate: bool = False) -> None:
        if any(np.diff(t) <= 0):
            raise ValueError("t not unique and/or sorted")
        if len(t) != len(a):
            raise ValueError("t and a have to be of same length")

        self.t = t
        self.a = a

        coeffs = np.asarray([np.diff(self.a) / np.diff(self.t), self.a[:-1]])
        super().__init__(coeffs, self.t, extrapolate=extrapolate)

    def maximum(self) -> float:
        return max(self.a)

    def minimum(self) -> float:
        return min(self.a)

    def __call__(self, t: NDArray) -> NDArray[np.floating]:
        result = super().__call__(t)
        if np.isnan(result).any():
            warnings.warn("Analog signal was evaluated outside " "specified time range")
        return result


class LocalBandwidth(AnalogSignal):
    """
    This class represents the local bandwidth function B(t) giving local
    bandwidth in Hz at time t in seconds.
    It is an extension of AnalogSignal, which only allows for positive
    amplitudes
    """

    def __init__(self, t: NDArray, B: NDArray) -> None:
        super().__init__(t, B)
        if not all(B > 0):
            raise ValueError("Bandwidth has to be positive at all times")
        temp = super().antiderivative()
        self.gamma = PPoly(2 * temp.c, temp.x, extrapolate=False)
        self.degrees_of_freedom = int(np.floor(self.gamma(self.gamma.x[-1])) + 1)

    def get_gamma(self) -> PPoly:
        """
        FIXME: use Gamma instead of PPoly
        """
        return self.gamma


def calc_sample_times(gamma: PPoly, degrees_of_freedom: int) -> NDArray[np.floating]:
    sample_times = np.zeros(degrees_of_freedom)
    for n in range(degrees_of_freedom):
        sol = gamma.solve(n)
        # Solution lies on breakpoint -> two equal solutions exist
        if np.all(np.isclose(sol, sol[0])):
            sol = sol[0]
        sample_times[n] = sol
    return sample_times


class VariableBandwidthFunction:
    """
    This class describes a function that has variable bandwidth according to
    Time-Warping according to Clark, Palmer and Lawrence . To initialize
    one has to specify local bandwidth. After that, DoF for the function
    in specified time range of local bandwidth can be read.
    Knowing the DoF one can now set the sample amplitudes
    gamma(tmin) is considered to be 0 and to create a first sample at tmin.

    BEWARE: Even if different local_bandwidth are limited to the same time
            time interval, the resulting vbw-function will be valid in a
            varying time intervals (between first and last sample depending
            on the shape of local_bandwidth, not its duration)
    """

    def __init__(self, local_bandwidth: LocalBandwidth) -> None:
        self.local_bandwidth = local_bandwidth
        self.gamma = self.local_bandwidth.gamma
        self.degrees_of_freedom = self.local_bandwidth.degrees_of_freedom
        self.sample_times = calc_sample_times(self.gamma, self.degrees_of_freedom)

    def set_sample_amplitudes(self, amplitudes: NDArray) -> None:
        if len(amplitudes) != self.degrees_of_freedom:
            raise ValueError("amplitudes have to be of length degrees of freedom")
        self.amplitudes = amplitudes

    def to_analog_function(
        self,
        oversampling: int = 8,
        adaptive: bool = False,
        t_start: float | None = None,
        t_stop: float | None = None,
    ) -> AnalogSignal:
        """
        Convert to analog signal / piecewise linear approximation
        normal mode, adaptive=False:
            Provides uniform sampling based on
            oversampling_factor*2*max(local_bandwidth)
        adaptive mode, not yet implemented:
            Provides nonuniform sampling, rate of
            oversampling_factor*2*local_bandwidth(t)
        """
        if adaptive == True:
            raise NotImplementedError("Adaptive mode not yet implemented")
        else:
            if t_start == None:
                t_start = self.sample_times[0]
            if t_stop == None:
                t_stop = self.sample_times[-1]
            t_step = 1 / (oversampling * 2 * self.local_bandwidth.maximum())
            t = np.arange(t_start, t_stop + t_step, t_step)
        return AnalogSignal(t, self.__call__(t))

    def __call__(self, t: NDArray) -> NDArray[np.floating]:
        values = np.zeros(len(t))
        gamma_at_t = self.gamma(t)
        for n in range(0, self.degrees_of_freedom):
            values += self.amplitudes[n] * np.sinc(gamma_at_t - n)
        return values


class BandlimitedFunction:
    """
    A classically bandlimited function
    """

    def __init__(self, amplitudes: NDArray, f_s: float, t_start: float = 0) -> None:
        self.amplitudes = amplitudes
        self.f_s = f_s
        self.t_start = t_start
        self.N = len(amplitudes)

    def t(self) -> NDArray[np.floating]:
        return np.linspace(
            self.t_start, self.t_start + (self.N - 1) * 1 / self.f_s, self.N
        )

    def to_analog_function(
        self,
        oversampling: int = 8,
        t_start: float | None = None,
        t_stop: float | None = None,
    ) -> AnalogSignal:
        """
        Convert to analog signal / piecewise linear approximation
        normal mode:    Provides uniform sampling based on
                        oversampling_factor*f_s
        """
        if t_start == None:
            t_start = self.t_start
        if t_stop == None:
            t_stop = self.N / self.f_s
        t_step = 1 / (oversampling * self.f_s)
        t = np.arange(t_start, t_stop + t_step, t_step)
        return AnalogSignal(t, self.__call__(t))

    def __call__(self, t: NDArray) -> NDArray[np.floating]:
        values = np.zeros(len(t))
        for n in range(0, self.N):
            values += self.amplitudes[n] * np.sinc(self.f_s * (t - self.t_start) - n)
        return values
