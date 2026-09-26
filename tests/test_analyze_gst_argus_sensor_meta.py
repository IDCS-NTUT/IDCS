import pytest

from tools.analyze_gst_argus_sensor_meta import analyze


def _row(index: int, *, sensor_ns: int, age_ns: int = 7_000_000):
    return {
        "frame_index": index,
        "sensor_frame_number": index,
        "sensor_start_ns": sensor_ns,
        "buffer_pts_ns": sensor_ns // 10,
        "pad_monotonic_ns": sensor_ns + age_ns,
    }


def test_analyzer_uses_sensor_metadata_not_buffer_pts():
    rows = [_row(1, sensor_ns=1_000_000_000), _row(2, sensor_ns=3_100_000_000), _row(3, sensor_ns=3_116_666_667)]
    report = analyze(rows)
    assert report["steady_frames_after_2s"] == 2
    assert report["sensor_start_to_source_pad_p50_ms"] == 7.0
    assert report["missing_sensor_frame_numbers"] == 0
    assert report["physical_exposure_timing_measured"] is False


def test_unordered_sensor_metadata_rejected():
    rows = [_row(1, sensor_ns=1_000_000_000), _row(2, sensor_ns=900_000_000)]
    with pytest.raises(ValueError, match="sensor timestamp"):
        analyze(rows)
