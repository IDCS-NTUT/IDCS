"""Summarize direct Argus sensor-start metadata from a camera-only probe.

Sensor start is first sensor data arrival, not beginning/centre of exposure.
An explicitly incomplete run may ignore one truncated final JSONL record.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _p(values: list[float], q: float) -> float | None:
    return sorted(values)[math.ceil(q * len(values)) - 1] if values else None


def read_samples(path: Path, *, allow_truncated_tail: bool = False) -> tuple[list[dict], bool]:
    lines = path.read_text(encoding="utf-8").splitlines()
    samples = []
    truncated = False
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
            for key in ("frame_number", "sensor_start_ns", "exposure_duration_ns", "argus_frame_time_ns", "acquire_monotonic_ns"):
                if not isinstance(row[key], int):
                    raise ValueError(f"{key} must be an integer")
            samples.append(row)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            if allow_truncated_tail and index == len(lines) - 1:
                truncated = True
                break
            raise ValueError(f"invalid camera sample on line {index + 1}: {exc}") from exc
    return samples, truncated


def analyze(samples: list[dict], *, incomplete: bool) -> dict[str, object]:
    if not samples:
        raise ValueError("no valid camera samples")
    start_ns = samples[0]["acquire_monotonic_ns"]
    steady = [row for row in samples if row["acquire_monotonic_ns"] - start_ns >= 2_000_000_000]
    if not steady:
        steady = samples
    plausible = [
        row for row in steady
        if 0 <= row["acquire_monotonic_ns"] - row["sensor_start_ns"] < 2_000_000_000
    ]
    ages_ms = [(row["acquire_monotonic_ns"] - row["sensor_start_ns"]) / 1e6 for row in plausible]
    sensor_to_argus_ms = [(row["argus_frame_time_ns"] - row["sensor_start_ns"]) / 1e6 for row in plausible]
    argus_to_acquire_ms = [(row["acquire_monotonic_ns"] - row["argus_frame_time_ns"]) / 1e6 for row in plausible]
    intervals_ms = [
        (right["sensor_start_ns"] - left["sensor_start_ns"]) / 1e6
        for left, right in zip(steady, steady[1:])
    ]
    numbers = [row["frame_number"] for row in samples]
    missing = sum(max(0, right - left - 1) for left, right in zip(numbers, numbers[1:]))
    span_s = (samples[-1]["sensor_start_ns"] - samples[0]["sensor_start_ns"]) / 1e9
    return {
        "frames": len(samples),
        "steady_frames_after_2s": len(steady),
        "observed_sensor_span_s": span_s,
        "observed_sensor_fps": (len(samples) - 1) / span_s if span_s > 0 else None,
        "missing_frame_numbers": missing,
        "same_clock_plausible_steady_frames": len(plausible),
        "sensor_start_to_acquire_p50_ms": statistics.median(ages_ms) if ages_ms else None,
        "sensor_start_to_acquire_p95_ms": _p(ages_ms, 0.95),
        "sensor_start_to_acquire_p99_ms": _p(ages_ms, 0.99),
        "sensor_start_to_acquire_max_ms": max(ages_ms) if ages_ms else None,
        "sensor_start_to_argus_frame_p50_ms": statistics.median(sensor_to_argus_ms) if sensor_to_argus_ms else None,
        "argus_frame_to_acquire_p50_ms": statistics.median(argus_to_acquire_ms) if argus_to_acquire_ms else None,
        "exposure_duration_p50_ms": statistics.median([row["exposure_duration_ns"] / 1e6 for row in steady]),
        "sensor_frame_interval_p50_ms": statistics.median(intervals_ms) if intervals_ms else None,
        "sensor_frame_interval_p99_ms": _p(intervals_ms, 0.99),
        "probe_completed_cleanly": not incomplete,
        "physical_exposure_start_measured": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples_jsonl", type=Path)
    parser.add_argument("--incomplete", action="store_true", help="record aborted run and allow one truncated tail record")
    args = parser.parse_args()
    samples, truncated = read_samples(args.samples_jsonl, allow_truncated_tail=args.incomplete)
    report = analyze(samples, incomplete=args.incomplete or truncated)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not args.incomplete and not truncated else 2


if __name__ == "__main__":
    raise SystemExit(main())
