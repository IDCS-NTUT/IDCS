"""Score bounded 30-second FF crossover trials with switch washout.

The two schedules (off/on/off and on/off/on) balance feedforward against
block position while keeping one hardware/video session continuous.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tools.analyze_video_hil import _jsonl, analyze_trial


SCHEDULES = {
    "off-on-off": (0.0, 0.5, 0.0),
    "on-off-on": (0.5, 0.0, 0.5),
}


def analyze_crossover(host_dir: Path, jetson_dir: Path) -> dict:
    safety = analyze_trial(host_dir, jetson_dir, allow_schedule=True)
    rows = _jsonl(jetson_dir / "trace.jsonl")
    meta = next(row for row in rows if row.get("type") == "meta")
    schedule = meta.get("feedforward_schedule")
    if schedule not in SCHEDULES or meta.get("duration_s") != 30:
        raise ValueError("not a bounded crossover trial")
    ticks = [row for row in rows if row.get("type") == "tick"]
    if len(ticks) < 1400:
        raise ValueError("crossover trial lacks most 50-Hz ticks")
    first_by_block: dict[int, int] = {}
    scored: dict[int, dict[int, dict]] = {0: {}, 1: {}, 2: {}}
    applied_by_block = [0, 0, 0]
    for row in ticks:
        block = row.get("schedule_block")
        if block not in (0, 1, 2):
            raise ValueError("crossover tick lacks block identity")
        scale = SCHEDULES[schedule][block]
        if row.get("feedforward_scale") != scale:
            raise ValueError("crossover tick has wrong feedforward scale")
        issued_ns = row["intent"]["issued_monotonic_ns"]
        first_by_block.setdefault(block, issued_ns)
        if row["pid_reason"] != "tracking":
            continue
        applied = any(abs(value) > 1e-5 for value in row.get("feedforward_rad_s", []))
        applied_by_block[block] += applied
        if scale == 0 and applied:
            raise ValueError("crossover off block applied feedforward")
        if issued_ns - first_by_block[block] < 2_000_000_000:
            continue
        frame_id = row.get("source_frame_id")
        error = row.get("raw_bearing_error_rad")
        if frame_id is not None and error is not None and all(math.isfinite(value) for value in error):
            scored[block].setdefault(frame_id, row)
    blocks = []
    for index, scale in enumerate(SCHEDULES[schedule]):
        frames = list(scored[index].values())
        if len(frames) < 150:
            raise ValueError(f"crossover block {index} lacks 150 scored source frames")
        if scale == 0.5 and applied_by_block[index] < 100:
            raise ValueError(f"crossover block {index} did not exercise feedforward")
        blocks.append({
            "index": index, "feedforward_scale": scale,
            "frames": len(frames), "feedforward_applied_ticks": applied_by_block[index],
            "yaw_sum_sq": sum(row["raw_bearing_error_rad"][0] ** 2 for row in frames),
            "pitch_sum_sq": sum(row["raw_bearing_error_rad"][1] ** 2 for row in frames),
        })
    return {"schedule": schedule, "safety": safety, "blocks": blocks}


def compare_crossovers(first: dict, second: dict) -> dict:
    if {first["schedule"], second["schedule"]} != set(SCHEDULES):
        raise ValueError("both reverse crossover schedules are required")
    for key in ("host_fixture_sha256", "jetson_config_digest", "yaw_kp", "pitch_kp"):
        if first["safety"][key] != second["safety"][key]:
            raise ValueError(f"crossover trials differ in {key}")
    pooled = {}
    for scale, name in ((0.0, "off"), (0.5, "on")):
        blocks = [block for trial in (first, second) for block in trial["blocks"]
                  if block["feedforward_scale"] == scale]
        frames = sum(block["frames"] for block in blocks)
        pooled[name] = {
            "blocks": len(blocks), "frames": frames,
            "yaw_rms_rad": math.sqrt(sum(block["yaw_sum_sq"] for block in blocks) / frames),
            "pitch_rms_rad": math.sqrt(sum(block["pitch_sum_sq"] for block in blocks) / frames),
        }
    return {
        "first": first, "second": second, "pooled": pooled,
        "yaw_rms_change_percent": 100 * (pooled["on"]["yaw_rms_rad"] / pooled["off"]["yaw_rms_rad"] - 1),
        "pitch_rms_change_percent": 100 * (pooled["on"]["pitch_rms_rad"] / pooled["off"]["pitch_rms_rad"] - 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first-host", "first-jetson", "second-host", "second-jetson"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = compare_crossovers(
        analyze_crossover(args.first_host, args.first_jetson),
        analyze_crossover(args.second_host, args.second_jetson),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
