from __future__ import annotations

import json

import pytest

from tools.analyze_clock_drift import OffsetSample, analyze, read_exchanges


def _linear_samples(*, ppm: float, width_ns: int = 2000, count: int = 11) -> list[OffsetSample]:
    baseline = 100_000_000
    return [
        OffsetSample(
            jetson_time_ns=tick * 1_000_000_000,
            offset_min_ns=baseline + round(ppm * tick * 1000) - width_ns // 2,
            offset_max_ns=baseline + round(ppm * tick * 1000) + width_ns // 2,
        )
        for tick in range(count)
    ]


def test_known_linear_drift_is_inside_interval_feasible_range() -> None:
    report = analyze(_linear_samples(ppm=5.0))
    assert report["observed_span_s"] == 10.0
    assert report["midpoint_regression_ppm"] == pytest.approx(5.0)
    assert report["narrow_half_midpoint_regression_ppm"] == pytest.approx(5.0)
    lower, upper = report["constant_slope_feasible_ppm"]
    assert lower < 5.0 < upper
    assert upper - lower == pytest.approx(0.4)
    assert report["future_drift_bound_established"] is False


def test_midpoint_bias_does_not_exclude_true_zero_slope() -> None:
    samples = [
        OffsetSample(tick * 1_000_000_000, 99_000_000, 101_000_000 + tick * 100_000)
        for tick in range(11)
    ]
    report = analyze(samples)
    assert report["midpoint_regression_ppm"] > 0
    lower, upper = report["constant_slope_feasible_ppm"]
    assert lower <= 0 <= upper


def test_subset_constraints_are_explicit_and_conservative() -> None:
    samples = _linear_samples(ppm=3.0, count=101)
    report = analyze(samples, max_slope_points=11)
    assert report["constant_slope_constraint_samples"] <= 11
    assert report["slope_constraints_use_subset"] is True
    lower, upper = report["constant_slope_feasible_ppm"]
    assert lower <= 3.0 <= upper


def test_jsonl_reader_recomputes_bounds_and_rejects_bad_order(tmp_path) -> None:
    path = tmp_path / "exchanges.jsonl"
    path.write_text(json.dumps({
        "jetson_send_ns": 1_000_000_000,
        "pc_receive_ns": 1_101_000_000,
        "pc_send_ns": 1_101_000_000,
        "jetson_receive_ns": 1_002_000_000,
    }) + "\n", encoding="utf-8")
    samples = read_exchanges(path)
    assert samples == [OffsetSample(1_001_000_000, 99_000_000, 101_000_000)]
    assert analyze(samples)["constant_slope_feasible_ppm"] is None
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps({
            "jetson_send_ns": 0,
            "pc_receive_ns": 0,
            "pc_send_ns": 0,
            "jetson_receive_ns": 1,
        }) + "\n")
    with pytest.raises(ValueError, match="strictly increasing"):
        analyze(read_exchanges(path))


def test_empty_or_invalid_intervals_fail_closed() -> None:
    with pytest.raises(ValueError, match="no valid"):
        analyze([])
    with pytest.raises(ValueError, match="invalid offset"):
        OffsetSample(1, 2, 1)
