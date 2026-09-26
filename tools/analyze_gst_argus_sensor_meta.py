"""Analyze 60 fps Argus sensor-start metadata joined to source-pad arrivals.

The sensor timestamp denotes first data arrival from the sensor, not optical
exposure start/midpoint. Both sensor and pad times must be same-clock plausible.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _quantile(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def read_samples(path: Path) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
                required = ("frame_index", "sensor_frame_number", "sensor_start_ns", "buffer_pts_ns", "pad_monotonic_ns")
                if not all(isinstance(row[key], int) for key in required):
                    raise ValueError("non-integer camera timestamp or frame number")
                rows.append(row)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise ValueError(f"invalid camera sample at line {number}: {exc}") from exc
    return rows


def analyze(rows: list[dict[str, int]]) -> dict[str, object]:
    if len(rows) < 2:
        raise ValueError("at least two camera frames required")
    for previous, current in zip(rows, rows[1:]):
        if current["sensor_frame_number"] <= previous["sensor_frame_number"]:
            raise ValueError("sensor frame number did not increase")
        if current["sensor_start_ns"] <= previous["sensor_start_ns"]:
            raise ValueError("sensor timestamp did not increase")
        if current["pad_monotonic_ns"] <= previous["pad_monotonic_ns"]:
            raise ValueError("source-pad arrival did not increase")
    ages_ns = [row["pad_monotonic_ns"] - row["sensor_start_ns"] for row in rows]
    if any(age < 0 or age >= 2_000_000_000 for age in ages_ns):
        raise ValueError("sensor and pad clocks are not plausibly comparable")
    first_pad = rows[0]["pad_monotonic_ns"]
    steady = [row for row in rows if row["pad_monotonic_ns"] - first_pad >= 2_000_000_000]
    if len(steady) < 2:
        raise ValueError("fewer than two steady frames after 2 s warmup")
    ages_ms = [(row["pad_monotonic_ns"] - row["sensor_start_ns"]) / 1e6 for row in steady]
    sensor_intervals_ms = [
        (current["sensor_start_ns"] - previous["sensor_start_ns"]) / 1e6
        for previous, current in zip(steady, steady[1:])
    ]
    pad_intervals_ms = [
        (current["pad_monotonic_ns"] - previous["pad_monotonic_ns"]) / 1e6
        for previous, current in zip(steady, steady[1:])
    ]
    span_s = (rows[-1]["sensor_start_ns"] - rows[0]["sensor_start_ns"]) / 1e9
    missing = sum(
        current["sensor_frame_number"] - previous["sensor_frame_number"] - 1
        for previous, current in zip(rows, rows[1:])
    )
    return {
        "frames": len(rows),
        "steady_frames_after_2s": len(steady),
        "sensor_span_s": span_s,
        "sensor_fps": (len(rows) - 1) / span_s,
        "missing_sensor_frame_numbers": missing,
        "sensor_start_to_source_pad_p50_ms": statistics.median(ages_ms),
        "sensor_start_to_source_pad_p95_ms": _quantile(ages_ms, 0.95),
        "sensor_start_to_source_pad_p99_ms": _quantile(ages_ms, 0.99),
        "sensor_start_to_source_pad_max_ms": max(ages_ms),
        "sensor_interval_p50_ms": statistics.median(sensor_intervals_ms),
        "sensor_interval_p99_ms": _quantile(sensor_intervals_ms, 0.99),
        "source_pad_interval_p99_ms": _quantile(pad_intervals_ms, 0.99),
        "sensor_timestamp_event": "first_data_arrival_not_exposure_start",
        "physical_exposure_timing_measured": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples_jsonl", type=Path)
    args = parser.parse_args()
    report = analyze(read_samples(args.samples_jsonl))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
