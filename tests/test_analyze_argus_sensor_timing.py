import pytest

from tools.analyze_argus_sensor_timing import analyze, read_samples


def test_incomplete_argus_run_keeps_valid_prefix_without_claiming_completion(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text(
        '{"frame_number":1,"sensor_start_ns":1000000000,"argus_frame_time_ns":1020000000,"acquire_monotonic_ns":1020100000,"exposure_duration_ns":5000000}\n'
        '{"frame_number":2,"sensor_start_ns":1033333333,"argus_frame_time_ns":1053333333,"acquire_monotonic_ns":1053433333,"exposure_duration_ns":5000000}\n'
        '{"frame_number":3,',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="line 3"):
        read_samples(path)
    samples, truncated = read_samples(path, allow_truncated_tail=True)
    assert truncated
    report = analyze(samples, incomplete=True)
    assert report["sensor_start_to_acquire_p50_ms"] == 20.1
    assert report["probe_completed_cleanly"] is False
    assert report["physical_exposure_start_measured"] is False
