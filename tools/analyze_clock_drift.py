"""Analyze measured PC-minus-Jetson offset intervals without certifying future drift.

Midpoint trends are descriptive estimates. The feasible slope range assumes
one constant relative clock frequency over the observed window; network
asymmetry and later clock changes can invalidate extrapolation. No result
from this tool grants a controller timing bound.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from jetson.control_v3.timing import ClockBounds


@dataclass(frozen=True)
class OffsetSample:
    jetson_time_ns: int
    offset_min_ns: int
    offset_max_ns: int

    def __post_init__(self) -> None:
        if self.offset_min_ns > self.offset_max_ns:
            raise ValueError("invalid offset interval")

    @property
    def width_ns(self) -> int:
        return self.offset_max_ns - self.offset_min_ns

    @property
    def midpoint_ns(self) -> float:
        return (self.offset_min_ns + self.offset_max_ns) / 2


def read_exchanges(path: Path) -> list[OffsetSample]:
    samples: list[OffsetSample] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                sent = int(row["jetson_send_ns"])
                received = int(row["jetson_receive_ns"])
                bounds = ClockBounds.from_exchange(
                    jetson_send_ns=sent,
                    pc_receive_ns=int(row["pc_receive_ns"]),
                    pc_send_ns=int(row["pc_send_ns"]),
                    jetson_receive_ns=received,
                )
                samples.append(OffsetSample(
                    jetson_time_ns=(sent + received) // 2,
                    offset_min_ns=bounds.offset_min_ns,
                    offset_max_ns=bounds.offset_max_ns,
                ))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid exchange on line {line_number}: {exc}") from exc
    return samples


def _regression_ppm(samples: list[OffsetSample]) -> float | None:
    if len(samples) < 2:
        return None
    t0 = samples[0].jetson_time_ns
    y0 = samples[0].midpoint_ns
    x = [(sample.jetson_time_ns - t0) / 1e9 for sample in samples]
    y = [sample.midpoint_ns - y0 for sample in samples]
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    denominator = math.fsum((value - mean_x) ** 2 for value in x)
    if denominator == 0:
        return None
    numerator = math.fsum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    return numerator / denominator / 1000  # ns/s -> ppm


def _narrow_subset(samples: list[OffsetSample], max_points: int) -> list[OffsetSample]:
    if len(samples) <= max_points:
        return samples
    span_ns = samples[-1].jetson_time_ns - samples[0].jetson_time_ns
    bucket_ns = max(1, math.ceil((span_ns + 1) / max_points))
    buckets: dict[int, OffsetSample] = {}
    for sample in samples:
        bucket = (sample.jetson_time_ns - samples[0].jetson_time_ns) // bucket_ns
        previous = buckets.get(bucket)
        if previous is None or sample.width_ns < previous.width_ns:
            buckets[bucket] = sample
    return [buckets[key] for key in sorted(buckets)]


def _feasible_constant_slope_ppm(samples: list[OffsetSample]) -> list[float] | None:
    if len(samples) < 2:
        return None
    lower_ppm = -math.inf
    upper_ppm = math.inf
    for index, earlier in enumerate(samples):
        for later in samples[index + 1:]:
            elapsed_ns = later.jetson_time_ns - earlier.jetson_time_ns
            if elapsed_ns <= 0:
                raise ValueError("exchange times must be strictly increasing")
            lower_ppm = max(
                lower_ppm,
                (later.offset_min_ns - earlier.offset_max_ns) / elapsed_ns * 1e6,
            )
            upper_ppm = min(
                upper_ppm,
                (later.offset_max_ns - earlier.offset_min_ns) / elapsed_ns * 1e6,
            )
    return [lower_ppm, upper_ppm]


def analyze(samples: list[OffsetSample], *, max_slope_points: int = 3000) -> dict[str, object]:
    if not samples:
        raise ValueError("no valid clock exchanges")
    if max_slope_points < 2:
        raise ValueError("max_slope_points must be at least two")
    if any(right.jetson_time_ns <= left.jetson_time_ns for left, right in zip(samples, samples[1:])):
        raise ValueError("exchange times must be strictly increasing")
    widths = [sample.width_ns for sample in samples]
    median_width = statistics.median(widths)
    narrow = [sample for sample in samples if sample.width_ns <= median_width]
    slope_samples = _narrow_subset(samples, max_slope_points)
    slope_range = _feasible_constant_slope_ppm(slope_samples)
    return {
        "samples": len(samples),
        "observed_span_s": (samples[-1].jetson_time_ns - samples[0].jetson_time_ns) / 1e9,
        "best_interval_width_ms": min(widths) / 1e6,
        "median_interval_width_ms": median_width / 1e6,
        "midpoint_regression_ppm": _regression_ppm(samples),
        "narrow_half_midpoint_regression_ppm": _regression_ppm(narrow),
        "constant_slope_feasible_ppm": slope_range,
        "constant_slope_model_feasible": slope_range is None or slope_range[0] <= slope_range[1],
        "constant_slope_constraint_samples": len(slope_samples),
        "slope_constraints_use_subset": len(slope_samples) < len(samples),
        "future_drift_bound_established": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("--max-slope-points", type=int, default=3000)
    args = parser.parse_args()
    report = analyze(read_exchanges(args.input_jsonl), max_slope_points=args.max_slope_points)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
