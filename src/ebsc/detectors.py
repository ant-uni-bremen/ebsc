"""
Created on Fri Nov  4 15:13:01 2022

@author: christopher
"""

import bisect
import numpy as np
from .data_types import Event, EventType, EventSequence
from typing import Tuple, List
from .function_types import AnalogSignal
from numpy.typing import NDArray


class Comparator:
    """
    Simple model of a comparator
    TBD!

    """

    def __init__(
        self,
        ref_level: List,
        output_lc_types: bool = True,
        block_processing: bool = False,
    ) -> None:
        self.ref_level = ref_level
        self.internal_state = None  # Can be true, false or None
        self.block_processing = block_processing

        if output_lc_types == True:
            self.fall_event_type = EventType.LC_FALL
            self.rise_event_type = EventType.LC_RISE
        else:
            self.fall_event_type = EventType.CMP_FALL
            self.rise_event_type = EventType.CMP_RISE

    def __find_previous_breakpoint_amplitude(
        self, analog_signal: AnalogSignal, t: NDArray
    ) -> NDArray | None:
        # previous in strict sense, breakpointtime < t
        # return None if no previous breakpoint
        breakpoint_idx = bisect.bisect_left(analog_signal.x, t) - 1
        if breakpoint_idx < 0:
            return None
        else:
            return analog_signal(analog_signal.x[breakpoint_idx])

    def __find_next_breakpoint_amplitude(
        self, analog_signal: AnalogSignal, t: NDArray
    ) -> NDArray | None:
        # next in strict sense, breakpointtime > t
        # return None if no next breakpoint
        breakpoint_idx = bisect.bisect_right(analog_signal.x, t)
        if breakpoint_idx > (len(analog_signal.x) - 1):
            return None
        else:
            return analog_signal(analog_signal.x[breakpoint_idx])

    def __calc_solutions(self, analog_signal: AnalogSignal) -> NDArray:
        return analog_signal.solve(self.ref_level, extrapolate=False)

    def __remove_intervals(self, solutions: NDArray) -> NDArray:
        remove_idx = []
        for sidx in range(0, len(solutions)):
            if np.isnan(solutions[sidx]):
                remove_idx.append(sidx - 1)
                remove_idx.append(sidx)
        return np.delete(solutions, remove_idx)

    def __determine_state(
        self, analog_signal: AnalogSignal, solutions: NDArray
    ) -> NDArray | None:
        if self.internal_state is None:
            # search for first breakpoint that is not on level
            breakpoint_idx = 0
            while analog_signal(analog_signal.x[breakpoint_idx]) == self.ref_level:
                solutions = np.delete(solutions, [0])
                breakpoint_idx += 1
            if analog_signal(analog_signal.x[breakpoint_idx]) > self.ref_level:
                self.internal_state = True
            else:
                self.internal_state = False
            return solutions

    def __calc_events(
        self, analog_signal: AnalogSignal, solutions: NDArray
    ) -> List[Event]:
        event_list = []
        for solution_idx in range(0, len(solutions)):
            if (self.internal_state == True) and (
                self.__find_next_breakpoint_amplitude(
                    analog_signal, solutions[solution_idx]
                )
                < self.ref_level
            ):
                event_list.append(Event(solutions[solution_idx], self.fall_event_type))
                self.internal_state = False
            elif (self.internal_state == False) and (
                self.__find_next_breakpoint_amplitude(
                    analog_signal, solutions[solution_idx]
                )
                > self.ref_level
            ):
                event_list.append(Event(solutions[solution_idx], self.rise_event_type))
                self.internal_state = True
        return event_list

    def reset_state(self) -> None:
        self.internal_state = None

    def get_state(self) -> None | bool:
        return self.internal_state

    def get_events(self, analog_signal: AnalogSignal) -> EventSequence:
        sol = self.__calc_solutions(analog_signal)
        sol = self.__remove_intervals(sol)

        if self.internal_state is None:
            sol = self.__determine_state(analog_signal, sol)

        events = self.__calc_events(analog_signal, sol)
        if self.block_processing == False:
            self.reset_state()

        return EventSequence(events)


class SchmittTrigger:
    def __init__(
        self,
        ref_level: List,
        hysteresis: float | None,
        output_lc_types: bool = True,
        block_processing: bool = False,
        initial_state=None,
    ) -> None:
        self.ref_level = ref_level

        self.internal_state = None

        self.block_processing = block_processing

        if output_lc_types == True:
            self.fall_event_type = EventType.LC_FALL
            self.rise_event_type = EventType.LC_RISE
        else:
            self.fall_event_type = EventType.ST_FALL
            self.rise_event_type = EventType.ST_RISE

        if hysteresis == 0:
            raise ValueError("Schmitt-trigger has to have nonzero hysteresis")
        else:
            self.hysteresis = hysteresis
            self.upper_threshold = ref_level + hysteresis
            self.lower_threshold = ref_level - hysteresis

        self.upper_comparator = Comparator(
            self.upper_threshold, block_processing=block_processing
        )
        self.lower_comparator = Comparator(
            self.lower_threshold, block_processing=block_processing
        )

    def __filter_escape_events(
        self, upper_events: List[Event], lower_events: List[Event]
    ) -> List[Event]:
        escape_events = []
        # append upper
        for e in upper_events:
            if e.event_type == self.upper_comparator.rise_event_type:
                escape_events.append(e)
        # append lower
        for e in lower_events:
            if e.event_type == self.upper_comparator.fall_event_type:
                escape_events.append(e)
        # sort and return
        escape_events.sort(key=lambda e: e.t)
        return escape_events

    def __determine_initial_state(
        self, escape_events: List[Event], analog_signal: AnalogSignal
    ) -> None:
        # determine initial state
        if analog_signal(analog_signal.x[0]) > self.upper_threshold:
            self.internal_state = True
        elif analog_signal(analog_signal.x[0]) < self.lower_threshold:
            self.internal_state = False
        else:
            # determine first escape from hysteresis area
            if escape_events[0].event_type == self.upper_comparator.rise_event_type:
                self.internal_state = True
            else:
                self.internal_state = False

    def __calc_events(self, escape_events: List[Event]) -> List[Event]:
        schmitt_events = []
        for e in escape_events:
            if (self.internal_state == True) and (
                e.event_type == self.upper_comparator.fall_event_type
            ):
                schmitt_events.append(Event(e.t, self.fall_event_type))
                self.internal_state = False
            elif (self.internal_state == False) and (
                e.event_type == self.upper_comparator.rise_event_type
            ):
                schmitt_events.append(Event(e.t, self.rise_event_type))
                self.internal_state = True

        return schmitt_events

    def reset_state(self) -> None:
        # Reset internal state and states of comparators
        self.internal_state = None
        self.upper_comparator.reset_state()
        self.lower_comparator.reset_state()

    def get_events(self, analog_signal: AnalogSignal) -> EventSequence:

        upper_events = self.upper_comparator.get_events(analog_signal)
        lower_events = self.lower_comparator.get_events(analog_signal)

        escape_event_list = self.__filter_escape_events(upper_events, lower_events)

        if self.internal_state is None:
            self.__determine_initial_state(escape_event_list, analog_signal)
        event_list = self.__calc_events(escape_event_list)

        if self.block_processing == False:
            self.reset_state()

        return EventSequence(event_list)


class LevelCrossingDetector:
    def __init__(
        self,
        ref_level: NDArray,
        hysteresis: float | None,
        block_processing: bool = False,
    ) -> None:
        self.ref_level = ref_level
        if hysteresis == 0 or hysteresis is None:
            self.hysteresis = None
            self.block_processing = block_processing
            self.comp = Comparator(ref_level, block_processing=block_processing)
        else:
            self.block_processing = block_processing
            self.hysteresis = hysteresis
            self.st = SchmittTrigger(
                ref_level, hysteresis, block_processing=block_processing
            )

    def get_event_amplitude_mapping(self) -> dict[EventType, NDArray[np.floating]]:
        if self.hysteresis:
            mapping = {
                EventType.LC_FALL: self.ref_level - self.hysteresis,
                EventType.LC_RISE: self.ref_level + self.hysteresis,
            }
        else:
            mapping = {
                EventType.LC_FALL: self.ref_level,
                EventType.LC_RISE: self.ref_level,
            }
        return mapping

    def get_events(self, analog_signal: AnalogSignal) -> EventSequence:
        if self.hysteresis is None:
            return self.comp.get_events(analog_signal)
        else:
            return self.st.get_events(analog_signal)


class MultiLevelCrossingDetector:
    def __init__(
        self, ref_levels, hysteresis: float | None, block_processing: bool = False
    ) -> None:
        self.ref_levels = ref_levels
        self.Nlevels = len(ref_levels)
        # TODO: check if ref_levels \pm hys is unique!
        #       check if ref_levels and hystereses are of same size, or
        #       size of hystereses is one

        if np.isscalar(hysteresis):
            self.hystereses = [hysteresis]
        if len(self.hystereses) == 1:
            self.hystereses = np.full(self.Nlevels, self.hystereses)
        elif len(self.hysteresis) != self.Nlevels:
            raise ValueError("hystereses is of wrong length")

        self.block_processing = block_processing

        self.lcds = []
        for n in range(0, self.Nlevels):
            self.lcds.append(
                LevelCrossingDetector(
                    self.ref_levels[n],
                    self.hystereses[n],
                    block_processing=self.block_processing,
                )
            )

    def get_event_amplitude_mapping(self) -> dict[EventType, NDArray[np.floating]]:
        mapping = []
        for n in range(0, self.Nlevels):
            mapping.append(self.lcds[n].get_event_amplitude_mapping())
        return mapping

    def get_events_list(
        self, analog_signal: AnalogSignal, bit_output: bool = False
    ) -> List[Event] | EventSequence:
        if not bit_output:
            event_sequence_list = []
            for n in range(0, self.Nlevels):
                event_sequence_list.append(self.lcds[n].get_events(analog_signal))
            return event_sequence_list
        else:
            #
            bit_event_sequence = EventSequence()
            for n in range(0, self.Nlevels):
                event_seq = self.lcds[n].get_events(analog_signal)
                for e in event_seq:
                    # +3 is DIRTY!
                    bit_event_sequence.append(Event(e.t, e.event_type + 3, bit_idx=n))
            return bit_event_sequence


# Send-on-Delta version of MLCD
# Uses internal MLCD and filters the events according to SoD concept
# TODO: Implement True SOD output option. Now it is just a filtered version
# of MultiLevelCrossingDetector, i.e. it keeps the absolute level information
class SendOnDeltaDetector:
    def __init__(
        self,
        ref_levels: NDArray,
        hysteresis: float | None,
        block_processing: bool = False,
    ) -> None:
        self.mlcd = MultiLevelCrossingDetector(ref_levels, hysteresis, block_processing)
        self.ref_levels = ref_levels
        self.N_levels = len(ref_levels)

    def convert_to_mlcs_list(
        self, mlcd_events: List[Event]
    ) -> Tuple[NDArray[np.floating], NDArray[np.integer]]:
        # A special representation of Events from LCS. Stored as follows:
        # There are N Levels. Each level is numbered from 1 to N. Each can be crossed
        # rising or falling. We code rising crossings with positive level-no and falling
        # crossings with negative level-no. So e.g. Level-2-Rising is coded as 2 and
        # Level-4-Falling is coded as -4.
        mlcs_times_list = []
        mlcs_number_list = []
        for es_idx in range(len(mlcd_events)):
            for evt in mlcd_events[es_idx]:
                mlcs_times_list.append(evt.t)
                if evt.event_type == EventType.LC_RISE:
                    mlcs_number_list.append(es_idx + 1)
                elif evt.event_type == EventType.LC_FALL:
                    mlcs_number_list.append(-(es_idx + 1))
                else:
                    raise ValueError("Event types must be LC_RISE or LC_FALL")
        mlcs_times_list = np.asarray(mlcs_times_list)
        mlcs_number_list = np.asarray(mlcs_number_list)
        sort_idx = np.argsort(mlcs_times_list)
        return mlcs_times_list[sort_idx], mlcs_number_list[sort_idx]

    def sod_filter_events(self, mlcd_events: List[Event]) -> List[Event]:
        (mlcd_times, mlcd_numbers) = self.convert_to_mlcs_list(mlcd_events)
        # TODO: - convert LCS events to iterable sequence
        #       - Determine first state
        #       - Iterate through events and delete non-sod

        # Determine first state, states are according to level-no in the middle of the region
        # associated with the state.

        # Initialize SoD-Event-Sequence-List
        sod_events = []
        for n in range(0, self.N_levels):
            sod_events.append(EventSequence())

        # Determine first state
        state = np.abs(mlcd_numbers[0])
        if mlcd_numbers[0] > 0:
            sod_events[np.abs(mlcd_numbers[0]) - 1].append(
                Event(mlcd_times[0], EventType.DELTA_RISE)
            )
        else:
            sod_events[np.abs(mlcd_numbers[0]) - 1].append(
                Event(mlcd_times[0], EventType.DELTA_FALL)
            )

        for mlcd_idx in range(len(mlcd_numbers)):
            # Check wether Event will occur in SOD
            if np.abs(mlcd_numbers[mlcd_idx]) != state:
                state = np.abs(mlcd_numbers[mlcd_idx])
                if mlcd_numbers[mlcd_idx] > 0:
                    sod_events[np.abs(mlcd_numbers[mlcd_idx]) - 1].append(
                        Event(mlcd_times[mlcd_idx], EventType.DELTA_RISE)
                    )
                else:
                    sod_events[np.abs(mlcd_numbers[mlcd_idx]) - 1].append(
                        Event(mlcd_times[mlcd_idx], EventType.DELTA_FALL)
                    )

        return sod_events

    def get_event_amplitude_mapping(
        self,
    ) -> List[dict[EventType, NDArray[np.floating]]]:
        mapping = []
        for l in range(0, self.N_levels):
            mapping.append(
                {
                    EventType.DELTA_FALL: self.mlcd.ref_levels[l]
                    - self.mlcd.hystereses[l],
                    EventType.DELTA_RISE: self.mlcd.ref_levels[l]
                    + self.mlcd.hystereses[l],
                }
            )
        return mapping

    def get_events_list(self, analog_signal: AnalogSignal) -> List[Event]:
        mlcd_events = self.mlcd.get_events_list(analog_signal)
        return self.sod_filter_events(mlcd_events)


def level_delta_to_levels(
    analog_signal: AnalogSignal, level_delta: float
) -> NDArray[np.floating]:
    min_level = int(analog_signal.minimum() / level_delta) * level_delta
    max_level = int(analog_signal.maximum() / level_delta) * level_delta
    N_levels = int((max_level - min_level) / level_delta) + 1
    return np.linspace(min_level, max_level, N_levels)
