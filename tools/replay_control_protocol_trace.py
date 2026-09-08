#!/usr/bin/env python3
"""Replay ControlObservation JSONL through a deterministic, non-actuating policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from common.schemas import ControlObservation
from jetson.control_replay import ControlPolicy, HoldPolicy, replay_observations
from jetson.shadow_rate_policy import ShadowRatePolicy, ShadowRatePolicyConfig


def _policy_from_args(args: argparse.Namespace) -> tuple[str, ControlPolicy]:
    """Construct an explicitly selected replay-only policy from CLI options."""

    if args.policy == "hold":
        return "shadow_hold", HoldPolicy(valid_for_ns=args.valid_for_ms * 1_000_000)
    return "shadow_rate", ShadowRatePolicy(
        ShadowRatePolicyConfig(
            yaw_kp=args.yaw_kp,
            pitch_kp=args.pitch_kp,
            yaw_kd=args.yaw_kd,
            pitch_kd=args.pitch_kd,
            yaw_rate_limit_rad_s=args.yaw_rate_limit,
            pitch_rate_limit_rad_s=args.pitch_rate_limit,
            yaw_accel_limit_rad_s2=args.yaw_accel_limit,
            pitch_accel_limit_rad_s2=args.pitch_accel_limit,
            nominal_period_s=args.period_ms / 1_000.0,
            valid_for_ns=args.valid_for_ms * 1_000_000,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--policy", choices=("hold", "shadow-rate"), default="hold")
    parser.add_argument("--yaw-kp", type=float, default=1.0)
    parser.add_argument("--pitch-kp", type=float, default=1.0)
    parser.add_argument("--yaw-kd", type=float, default=0.0)
    parser.add_argument("--pitch-kd", type=float, default=0.0)
    parser.add_argument("--yaw-rate-limit", type=float, default=1.0)
    parser.add_argument("--pitch-rate-limit", type=float, default=1.0)
    parser.add_argument("--yaw-accel-limit", type=float, default=2.0)
    parser.add_argument("--pitch-accel-limit", type=float, default=2.0)
    parser.add_argument("--period-ms", type=float, default=20.0)
    parser.add_argument("--valid-for-ms", type=int, default=50)
    args = parser.parse_args(argv)
    observations = []
    invalid = 0
    with args.trace.open(encoding="utf-8") as source:
        for line in source:
            try:
                record = json.loads(line)
                if record.get("type") == "observation":
                    observations.append(ControlObservation.model_validate(record["observation"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                invalid += 1
    policy_name, policy = _policy_from_args(args)
    intents, result = replay_observations(observations, policy)
    if args.output is not None:
        with args.output.open("w", encoding="utf-8") as output:
            for intent in intents:
                output.write(json.dumps({"type": "replayed_intent", "intent": intent.model_dump(mode="json")},
                                        separators=(",", ":"), sort_keys=True) + "\n")
    print(json.dumps({**result.__dict__, "invalid_records": invalid, "policy": policy_name,
                      "physical_control_disabled": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
