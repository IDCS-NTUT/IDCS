"""Summarize controller timestamp mapping from a diagnostics JSONL trace."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tracking = [value for value in records if value.get("reason") == "tracking"]
    modes = Counter(
        str(value.get("timing", {}).get("estimator_time_source", "missing"))
        for value in tracking
    )
    mapped = [
        value["timing"] for value in tracking
        if value.get("timing", {}).get("estimator_time_source") == "mapped_pc_source"
    ]
    result = {
        "tracking_ticks": len(tracking),
        "estimator_time_sources": dict(sorted(modes.items())),
        "mapped_frame_age_ms": _summary([float(v["source_frame_age_ms"]) for v in mapped]),
        "clock_uncertainty_ms": _summary([float(v["source_clock_uncertainty_ms"]) for v in mapped]),
        "frame_gimbal_pose_age_ms": _summary([
            float(v["frame_gimbal_pose_age_ms"])
            for v in mapped if v.get("frame_gimbal_pose_age_ms") is not None
        ]),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
