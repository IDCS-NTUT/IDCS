#!/usr/bin/env python3
"""Compare hold-baseline and shadow-rate intents for a JSONL observation trace."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from common.schemas import ControlObservation
from jetson.control_replay import HoldPolicy, replay_observations
from jetson.shadow_rate_policy import ShadowRatePolicy, ShadowRatePolicyConfig


def _observations(path: Path) -> tuple[list[ControlObservation], int]:
    values: list[ControlObservation] = []
    invalid = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            if record.get("type") == "observation":
                values.append(ControlObservation.model_validate(record["observation"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            invalid += 1
    return values, invalid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--yaw-kp", type=float, default=1.0)
    parser.add_argument("--pitch-kp", type=float, default=1.0)
    parser.add_argument("--period-ms", type=float, default=20.0)
    args = parser.parse_args()
    observations, invalid = _observations(args.trace)
    hold, hold_result = replay_observations(observations, HoldPolicy())
    shadow, shadow_result = replay_observations(observations, ShadowRatePolicy(ShadowRatePolicyConfig(
        yaw_kp=args.yaw_kp, pitch_kp=args.pitch_kp, nominal_period_s=args.period_ms / 1000.0,
    )))
    assert len(hold) == len(shadow)
    deltas = [max(abs(a.yaw_rate_rad_s - b.yaw_rate_rad_s), abs(a.pitch_rate_rad_s - b.pitch_rate_rad_s))
              for a, b in zip(hold, shadow)]
    print(json.dumps({
        "physical_control_disabled": True,
        "invalid_records": invalid,
        "observations": len(observations),
        "hold": hold_result.__dict__, "shadow": shadow_result.__dict__,
        "shadow_reasons": dict(sorted(Counter(intent.reason for intent in shadow).items())),
        "saturation_count": sum(any(intent.limits.model_dump().values()) for intent in shadow),
        "max_shadow_rate_delta_rad_s": max(deltas, default=0.0),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
