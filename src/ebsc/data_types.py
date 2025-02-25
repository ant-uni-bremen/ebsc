"""
Created on Fri Nov  4 10:44:08 2022

@author: christopher
"""

import numpy as np
import bisect
from enum import IntEnum, unique
from functools import total_ordering
from numpy.typing import NDArray
from typing import Union, Iterator, List

"""
    Sample and Sample sequence. Can be used for uniform and nonuniform sampling
    for uniform samples a representation via amplitude ndarray is definitely
    less memory-hungry
"""


@total_ordering
class Sample:
    def __init__(self, t: float, a: float) -> None:
        if np.isnan(t):
            raise ValueError("Sample time cannot be NaN")
        elif np.isnan(a):
            raise ValueError("Sample amplitude cannot be NaN")
        else:
            self.t = t
            self.a = a

    def __str__(self) -> str:
        return "Sample(t=" + str(self.t) + ", a=" + str(self.a) + ")"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: "Sample") -> bool:
        return self.t == other.t

    def __lt__(self, other: "Sample") -> bool:
        return self.t < other.t


@unique
class EventType(IntEnum):
    # Comparator Event-types
    CMP_FALL = 0
    CMP_RISE = 1

    # Schmitt-Trigger Event-types
    ST_FALL = 2
    ST_RISE = 3

    # General LC-Events
    LVL_CROSSING = 4

    # Bitwise Events
    BIT_FALL = 5
    BIT_RISE = 6

    # SoD Events
    DELTA_FALL = 7
    DELTA_RISE = 8

    # LC Events
    LC_FALL = 9
    LC_RISE = 10


@total_ordering
class Event:
    def __init__(
        self, t: float, event_type: EventType, bit_idx: bool | int = None
    ) -> None:
        self.t = t
        self.event_type = event_type

        self.bit_idx = bit_idx

    def __eq__(self, other: "Event") -> bool:
        return (
            self.t == other.t
            and self.event_type == other.event_type
            and self.bit_idx == other.bit_idx
        )

    def __lt__(self, other: "Event") -> bool:
        if self.t == other.t:
            if self.event_type == other.event_type:
                return self.bit_idx < other.bit_idx
            else:
                return self.event_type < other.event_type
        return self.t < other.t

    def __str__(self) -> str:
        output_string = str(self.t) + " " + str(self.event_type.name)
        return output_string

    def __repr__(self) -> str:
        return self.__str__()


class MultiLevelCrossingEvent(Event):
    def __init__(self, t: float, event_type: EventType, ref_level_idx: int) -> None:
        super().__init__(t, event_type)
        self.ref_level_idx = ref_level_idx


class Sequence:
    def __init__(self, sequence: NDArray) -> None:
        self.sequence = sequence

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, key: Union[int, NDArray[np.integer]]) -> Sample:
        return self.sequence[key]

    def __setitem__(self, key, value):
        raise ValueError("Never set directly. Use append() instead.")

    def __delitem__(self, key: int) -> None:
        self.sequence.pop(key)

    def __len__(self) -> int:
        return len(self.sequence)

    def __iter__(self) -> Iterator[Sample | Event]:
        return iter(self.sequence)

    def __str__(self) -> str:
        if len(self.sequence) == 0:
            output_string = "Empty" + type(self).__name__
        else:
            output_string = (
                type(self).__name__ + " of length " + str(len(self.sequence)) + ":\n"
            )
            for event in self.sequence:
                output_string += str(event) + "\n"
        return output_string


class SampleSequence(Sequence):
    def __init__(self, t: NDArray, a: NDArray) -> None:
        if any(np.diff(t) <= 0):
            raise ValueError("t not unique and/or sorted")
        if len(t) != len(a):
            raise ValueError("t and a have different size")

        self.sequence = [Sample(time, ampl) for time, ampl in zip(t, a)]

    def append(self, sample: Sample) -> None:
        if sample.t in self.get_t_as_array():
            raise ValueError("There is already a sample at t")
        else:
            insertion_idx = bisect.bisect(self.sequence, sample)
            self.sequence.insert(insertion_idx, sample)

    def remove_sample(self, t: float) -> None:
        remove_index = np.transpose(np.nonzero(self.get_t_as_array() == t)).tolist()
        self.sequence.pop(remove_index[0][0])

    def get_t_as_array(self) -> Union[float, NDArray[np.floating]]:
        return np.asarray([sample.t for sample in self.sequence])

    def get_a_as_array(self) -> Union[float, NDArray[np.floating]]:
        return np.asarray([sample.a for sample in self.sequence])


class EventSequence(Sequence):
    def __init__(self, event_list: List[Event] = []) -> None:
        # IT IS NOT CHECKED IF LIST IS IN ORDER!
        if not sorted(event_list) == event_list:
            raise ValueError("The input events should be sorted!")
        self.sequence = list(event_list)

    def append(self, event: Event) -> None:
        if event.t in self.get_all_event_times():
            raise ValueError("There is already an event at t")
        insertion_idx = bisect.bisect(self.sequence, event)
        self.sequence.insert(insertion_idx, event)

    # DAS IST NOCH NICHT GUT SO:
    def discard_crossing_slopes(self) -> None:
        for e in self.sequence:
            e.event_type = EventType.LVL_CROSSING

    def find_event_idx(self, t: NDArray) -> List[int]:
        return [i for i in range(0, len(self.sequence)) if self.sequence[i].t == t]

    def get_all_event_times(self) -> NDArray[np.floating]:
        return np.asarray([event.t for event in self.sequence])

    def get_all_event_types(self) -> NDArray[np.floating]:
        return np.asarray([event.event_type for event in self.sequence])

    def __eq__(self, other: "EventSequence") -> List[bool]:
        t_equal = all(self.get_all_event_times() == other.get_all_event_times())
        event_type_equal = all(
            self.get_all_event_types() == other.get_all_event_types()
        )
        return t_equal and event_type_equal


"""
Conversion functions
"""


def cut_events_list(
    events: Union[EventSequence, List], t_start: float, t_stop: float
) -> Union[EventSequence, List]:
    """
    Return events-list only including events with times between t_start and
    t_stop
    """
    if isinstance(events, list):
        new_list = []
        for es in events:
            new_sequence = EventSequence()
            for e in es:
                if e.t >= t_start and e.t <= t_stop:
                    new_sequence.append(e)
            new_list.append(new_sequence)
        return new_list
    elif isinstance(events, EventSequence):
        new_sequence = EventSequence()
        for e in events:
            if e.t >= t_start and e.t <= t_stop:
                new_sequence.append(e)
        return new_sequence


def events_list_to_mlcd_event_sequence(events: List) -> EventSequence:
    new_events = EventSequence()
    for event_sequence_idx in range(len(events)):
        for event in events[event_sequence_idx]:
            new_events.append(
                MultiLevelCrossingEvent(event.t, event.event_type, event_sequence_idx)
            )
    return new_events


def lc_events_to_samples(events: List, ref_levels: List) -> SampleSequence:
    sample_seq = SampleSequence([], [])
    for event_seq_index in range(len(events)):
        for event in events[event_seq_index]:
            sample_seq.append(Sample(event.t, ref_levels[event_seq_index]))
    return sample_seq


def mlc_events_to_states(
    events: List[EventSequence],
) -> tuple[NDArray[np.floating], List]:
    """
    Yields states and time-breakpoints for a given multi-level-crossing sequence.

    This function yields a time vector and a state vector containing integers.
    For N_L levels, N_L+1 states are possible, numbered from 0 to N_L.

    In state 0 the signal was smaller than the smallest reference voltate (level).
    In state 1 the signal was larger than the smallest but smaller than the second smallest ref voltage.

    When T is the number of event-times t, we gain T+1 states.
    The state before t[0], all the states between the event times t[1:-1] and
    the state after t[-1].
    """
    mlc_events = events_list_to_mlcd_event_sequence(events)
    t = mlc_events.get_all_event_times()
    states = []

    # Pre t[0] state
    first_event = mlc_events[0]
    if first_event.event_type == EventType.LC_FALL:
        states.append(first_event.ref_level_idx + 1)
    else:
        states.append(first_event.ref_level_idx)

    for e_idx in range(len(mlc_events)):
        e = mlc_events[e_idx]
        if e.event_type == EventType.LC_FALL:
            states.append(e.ref_level_idx)
        elif e.event_type == EventType.LC_RISE:
            states.append(e.ref_level_idx + 1)
        else:
            raise ValueError("Wrong Event-Types. Must be LC_FALL or LC_RISE")
    return t, states


def mlc_events_to_comparator_states(
    events: List, nr_of_ref_levels: int
) -> tuple[NDArray[np.floating], List]:
    """
    Yields comparator states and time-breakpoints for a given multi-level-crossing sequence.

    This function yields a time vector and an N dimensional state vector containing integers
    For N_L levels, N_L+1 states are possible, numbered from 0 to N_L.

    In state 0 the signal was smaller than the smallest reference voltate (level).
    In state 1 the signal was larger than the smallest but smaller than the second smallest ref voltage.

    When T is the number of event-times t, we gain T+1 states.
    The state before t[0], all the states between the event times t[1:-1] and
    the state after t[-1].
    """

    t, states = mlc_events_to_states(events)
    states_vector = np.zeros((len(states), nr_of_ref_levels), dtype=bool)
    for i, state in enumerate(states):
        states_vector[i, nr_of_ref_levels - state :] = True
    # states_vector = np.fliplr(np.bitwise_not(states_vector))
    return t, states_vector


def sod_events_to_states(events: List) -> tuple[NDArray[np.floating], List]:
    """
    At this time, this works with the "fake" SOD-Events, which still contain
    absolute ref_level. This is because i want to avoid the absolute amplited
    reconstruction by signal statistics...
    TODO: call these events not SOD, but something else...

    In SOD, for N_L levels, we have exactly N_L states.
    So we can just copy the ref_level_idx to states
    """
    mlc_sod_events = events_list_to_mlcd_event_sequence(events)
    t = mlc_sod_events.get_all_event_times()
    states = []

    # Pre t[0] state
    first_event = mlc_sod_events[0]
    if first_event.event_type == EventType.DELTA_FALL:
        states.append(first_event.ref_level_idx + 1)
    else:
        states.append(first_event.ref_level_idx - 1)

    for e_idx in range(len(mlc_sod_events)):
        e = mlc_sod_events[e_idx]
        if e.event_type == EventType.DELTA_FALL or e.event_type == EventType.DELTA_RISE:
            states.append(e.ref_level_idx)
        else:
            raise ValueError("Wrong Event-Types. Must be DELTA_FALL or DELTA_RISE")
    return t, states


def sod_events_to_bits(
    events: List, initial_state: bool = False
) -> tuple[NDArray[np.floating], list]:
    """
    The states of an SOD sampler can be described by 1.5 bits:
    The signal traveled one delta up, crossing the next higher level or
    the signal traveled one delta down, crossing the next lower level in comparison to the previous.
    State now describes the direction of travel and control if an event occured
    """
    mlc_sod_events = events_list_to_mlcd_event_sequence(events)
    t = mlc_sod_events.get_all_event_times()
    state = []
    control = []
    # Pre t[0] state
    state.append(initial_state)
    control.append(initial_state)
    for e_idx in range(len(mlc_sod_events)):
        e = mlc_sod_events[e_idx]
        control.append(np.logical_not(control[-1]))
        if e.event_type == EventType.DELTA_FALL:
            state.append(False)
        elif e.event_type == EventType.DELTA_RISE:
            state.append(True)
        else:
            raise ValueError("Wrong Event-Types. Must be DELTA_FALL or DELTA_RISE")
    return t, state, control
