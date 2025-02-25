import numpy as np
from ebsc import data_types
import pytest


def test_Sample():
    assert data_types.Sample(0.2, 1) < data_types.Sample(0.5, 1)
    assert data_types.Sample(0.2, 1) == data_types.Sample(0.2, 1)
    with pytest.raises(Exception) as e_info:
        data_types.Sample(0.2, np.nan)
    with pytest.raises(Exception) as e_info:
        data_types.Sample(np.nan, 2)


def test_SampleSequence():
    t = np.arange(0, 1, 1 / 10)
    a = np.random.randn(len(t))
    samples = data_types.SampleSequence(t, a)
    # Adding sample at existing time
    with pytest.raises(Exception) as e_info:
        samples.append(data_types.Sample(0, 2))
    np.testing.assert_array_equal(samples.get_t_as_array(), t)
    np.testing.assert_array_equal(samples.get_a_as_array(), a)
    # Adding new sample
    samples.append(data_types.Sample(-0.1, 2))
    np.testing.assert_array_equal(samples.get_t_as_array(), np.array([-0.1, *t]))
    np.testing.assert_array_equal(samples.get_a_as_array(), np.array([2, *a]))
    # Removing sample
    samples.remove_sample(-0.1)
    np.testing.assert_array_equal(samples.get_t_as_array(), t)
    np.testing.assert_array_equal(samples.get_a_as_array(), a)


def test_Event():
    event0 = data_types.Event(0, data_types.EventType.LVL_CROSSING)
    event1 = data_types.Event(1, data_types.EventType.DELTA_FALL)
    event2 = data_types.Event(1, data_types.EventType.DELTA_RISE)
    event3 = data_types.Event(1, data_types.EventType.DELTA_RISE)
    assert event0 < event1
    # Why is it defined like this?
    assert event1 < event2
    assert event2 == event3


def test_MultiLevelCrossingEvent():
    event0 = data_types.MultiLevelCrossingEvent(0, data_types.EventType.LC_FALL, 1)
    event1 = data_types.MultiLevelCrossingEvent(1, data_types.EventType.LC_FALL, 0)
    event2 = data_types.MultiLevelCrossingEvent(1, data_types.EventType.LC_RISE, 0)
    event3 = data_types.MultiLevelCrossingEvent(1, data_types.EventType.LC_RISE, 1)
    event4 = data_types.MultiLevelCrossingEvent(1, data_types.EventType.LC_FALL, 0)
    assert event0 < event1
    # Why is it defined like this?
    assert event1 < event2
    assert event2 == event3
    # FIXME: This does not work because bit_idx is not inherited
    # assert event1 < event4


def test_EventSequence():
    event0 = data_types.MultiLevelCrossingEvent(0.0, data_types.EventType.LC_FALL, 1)
    event1 = data_types.MultiLevelCrossingEvent(1.0, data_types.EventType.LC_FALL, 0)
    event2 = data_types.MultiLevelCrossingEvent(1.0, data_types.EventType.LC_RISE, 0)
    event_list = data_types.EventSequence([event0])
    event_list.append(event1)
    assert len(event_list) == 2
    np.testing.assert_array_equal(event_list.get_all_event_times(), [0, 1])
    assert event_list.find_event_idx(0.0)[0] == 0
    # This properly raises an exception
    with pytest.raises(Exception) as e_info:
        event_list.append(event1)
    # Should raise an exception because two events can't happen at the same time?
    with pytest.raises(Exception) as e_info:
        event_list.append(event2)
    # Should check order and duplicates?
    with pytest.raises(Exception) as e_info:
        event_list = data_types.EventSequence([event1, event0])
    with pytest.raises(Exception) as e_info:
        event_list = data_types.EventSequence([event0, event0])
    event_list0 = data_types.EventSequence([event0, event1])
    event_list1 = data_types.EventSequence([event0, event1])
    assert event_list0 == event_list1
    event_list1 = data_types.EventSequence([event0, event2])
    assert not event_list0 == event_list1


def test_cut_events_list():
    event0 = data_types.MultiLevelCrossingEvent(0.0, data_types.EventType.LC_FALL, 1)
    event1 = data_types.MultiLevelCrossingEvent(1.0, data_types.EventType.LC_FALL, 0)
    event2 = data_types.MultiLevelCrossingEvent(3.0, data_types.EventType.LC_RISE, 0)
    event3 = data_types.MultiLevelCrossingEvent(4.0, data_types.EventType.LC_FALL, 0)
    event4 = data_types.MultiLevelCrossingEvent(5.0, data_types.EventType.LC_RISE, 0)
    event_list = data_types.EventSequence([event0, event1, event2, event3, event4])
    np.testing.assert_array_equal(
        data_types.cut_events_list(event_list, 1.0, 4.0).get_all_event_times(),
        data_types.EventSequence([event1, event2, event3]).get_all_event_times(),
    )


def test_events_list_to_mlcd_event_sequence():
    level0_event_list = data_types.EventSequence(
        [
            data_types.Event(0.0, data_types.EventType.LC_RISE),
            data_types.Event(0.3, data_types.EventType.LC_FALL),
            data_types.Event(0.4, data_types.EventType.LC_RISE),
        ]
    )
    level1_event_list = data_types.EventSequence(
        [
            data_types.Event(0.1, data_types.EventType.LC_RISE),
            data_types.Event(0.2, data_types.EventType.LC_FALL),
            data_types.Event(0.5, data_types.EventType.LC_RISE),
        ]
    )
    mlcd_list = data_types.events_list_to_mlcd_event_sequence(
        [level0_event_list, level1_event_list]
    )
    np.testing.assert_array_equal(
        mlcd_list.get_all_event_times(), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )
    assert (
        mlcd_list[mlcd_list.find_event_idx(0.2)[0]].event_type
        == data_types.EventType.LC_FALL
        and mlcd_list[mlcd_list.find_event_idx(0.2)[0]].ref_level_idx == 1
    )


def test_lc_events_to_samples():
    ref_levels = [-1, 1]
    level0_event_list = data_types.EventSequence(
        [
            data_types.Event(0.0, data_types.EventType.LC_RISE),
            data_types.Event(0.3, data_types.EventType.LC_FALL),
            data_types.Event(0.4, data_types.EventType.LC_RISE),
        ]
    )
    level1_event_list = data_types.EventSequence(
        [
            data_types.Event(0.1, data_types.EventType.LC_RISE),
            data_types.Event(0.2, data_types.EventType.LC_FALL),
            data_types.Event(0.5, data_types.EventType.LC_RISE),
        ]
    )
    samples = data_types.lc_events_to_samples(
        [level0_event_list, level1_event_list], ref_levels
    )
    np.testing.assert_array_equal(
        samples.get_t_as_array(), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )
    np.testing.assert_array_equal(samples.get_a_as_array(), [-1, 1, 1, -1, -1, 1])
