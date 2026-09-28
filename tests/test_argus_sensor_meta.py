from __future__ import annotations

import pytest

from jetson.deepstream.argus_sensor_meta import SensorFrame, SensorFrameIndex


def test_index_joins_by_pts_and_stays_bounded() -> None:
    index = SensorFrameIndex(capacity=2)
    for pts in (10, 20, 30):
        index.record(pts, SensorFrame(pts // 10, pts * 1000))
    index.record(40, None)
    assert index.pop(10) is None  # evicted
    assert index.pop(30) == SensorFrame(3, 30_000)
    assert index.pop(30) is None  # consumed once
    assert (index.recorded, index.missing) == (3, 1)


def test_reader_returns_none_for_a_buffer_without_argus_metadata() -> None:
    gi = pytest.importorskip("gi")
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    from jetson.deepstream.argus_sensor_meta import ArgusSensorMetaReader

    Gst.init(None)
    assert ArgusSensorMetaReader().read(Gst.Buffer.new_allocate(None, 16, None)) is None
