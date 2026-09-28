"""Compare two closed-loop controller traces after a common warm-up period."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = 0.95 * (len(ordered) - 1)
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summarize(path: Path, *, warmup_s: float) -> dict[str, float | int]:
    ticks = [
        item for line in path.read_text(encoding="utf-8").splitlines()
        if (item := json.loads(line)).get("type") == "tick"
    ]
    if not ticks:
        raise ValueError(f"no controller ticks in {path}")
    first_ns = ticks[0]["observation"]["created_monotonic_ns"]
    tracking = [
        item for item in ticks
        if item["intent"]["reason"] == "tracking"
        and (item["observation"]["created_monotonic_ns"] - first_ns) / 1e9 >= warmup_s
    ]
    if not tracking:
        raise ValueError(f"no post-warm-up tracking ticks in {path}")
    errors = [float(item["observation"]["target"]["pixel_error"][0]) for item in tracking]
    commands = [float(item["intent"]["yaw_rate_rad_s"]) for item in tracking]
    limits = [item["intent"]["limits"] for item in tracking]
    return {
        "controller_ticks": len(ticks),
        "tracking_ticks_after_warmup": len(tracking),
        "yaw_rms_px": math.sqrt(statistics.fmean(value * value for value in errors)),
        "yaw_p95_abs_px": _p95([abs(value) for value in errors]),
        "yaw_command_total_variation_rad_s": sum(
            abs(new - old) for old, new in zip(commands, commands[1:])
        ),
        "yaw_rate_limited_fraction": statistics.fmean(
            bool(item["yaw_rate_limited"]) for item in limits
        ),
        "acceleration_limited_fraction": statistics.fmean(
            bool(item["acceleration_limited"]) for item in limits
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--warmup-s", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = summarize(args.baseline, warmup_s=args.warmup_s)
    candidate = summarize(args.candidate, warmup_s=args.warmup_s)
    comparisons = {
        "rms_change_fraction": candidate["yaw_rms_px"] / baseline["yaw_rms_px"] - 1,
        "p95_change_fraction": candidate["yaw_p95_abs_px"] / baseline["yaw_p95_abs_px"] - 1,
        "command_variation_ratio": (
            candidate["yaw_command_total_variation_rad_s"]
            / baseline["yaw_command_total_variation_rad_s"]
        ),
        "rms_improvement_at_least_10pct": candidate["yaw_rms_px"] <= 0.9 * baseline["yaw_rms_px"],
        "p95_not_worse_by_10pct": candidate["yaw_p95_abs_px"] <= 1.1 * baseline["yaw_p95_abs_px"],
        "variation_within_1_25x": (
            candidate["yaw_command_total_variation_rad_s"]
            <= 1.25 * baseline["yaw_command_total_variation_rad_s"]
        ),
    }
    report = {
        "warmup_s": args.warmup_s,
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparisons,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
