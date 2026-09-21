import pytest

from pc.streamer import SourceFrameIds


def test_source_frame_ids_are_sequential_within_one_process():
    ids = SourceFrameIds(start_time_ns=1_000_000_000)

    assert [ids.next(), ids.next(), ids.next()] == [1_000_001, 1_000_002, 1_000_003]
    assert ids.frames_sent == 3


def test_source_frame_ids_advance_across_process_restarts():
    previous = SourceFrameIds(start_time_ns=1_000_000_000)
    previous_ids = [previous.next() for _ in range(240)]
    restarted = SourceFrameIds(start_time_ns=2_000_000_000)

    assert restarted.next() > previous_ids[-1]
    assert restarted.frames_sent == 1


def test_source_frame_ids_reject_negative_epoch():
    with pytest.raises(ValueError, match="non-negative"):
        SourceFrameIds(start_time_ns=-1)
