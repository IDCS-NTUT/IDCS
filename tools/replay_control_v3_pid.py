"""Deterministic, serial-free ControlObservation -> V3 PID shadow-intent replay."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from common.schemas import ControlObservation
from jetson.control_v3.pid import AxisPIDConfig, BasicPID
from jetson.control_v3.shadow_pid import ShadowPIDController
from jetson.control_v3.timing import ClockBounds


def _merge(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _round(value: float) -> float:
    return round(value, 9)


def replay(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    if fixture.get("schema_version") != 1:
        raise ValueError("unsupported V3 PID replay fixture version")
    config = fixture["controller"]
    pid = BasicPID(
        AxisPIDConfig(**config["yaw"]),
        AxisPIDConfig(**config["pitch"]),
    )
    controller = ShadowPIDController(
        pid,
        max_clock_sample_age_ns=config["max_clock_sample_age_ns"],
        max_capture_age_ns=config["max_capture_age_ns"],
        max_gimbal_age_ns=config["max_gimbal_age_ns"],
        max_safety_age_ns=config["max_safety_age_ns"],
    )
    clock = ClockBounds.from_exchange(**fixture["clock_exchange"])
    base = fixture["base_observation"]
    records: list[dict[str, Any]] = []
    for step in fixture["steps"]:
        observation = ControlObservation.model_validate(_merge(base, step["observation"]))
        result = controller.decide(observation, clock)
        records.append({
            "case": step["case"],
            "observation_sequence": observation.sequence,
            "reason": result.intent.reason,
            "timing_reason": result.timing.reason,
            "mode": result.intent.mode,
            "yaw_rate_rad_s": _round(result.intent.yaw_rate_rad_s),
            "pitch_rate_rad_s": _round(result.intent.pitch_rate_rad_s),
            "yaw_terms_rad_s": [
                _round(result.pid.yaw.proportional_rad_s),
                _round(result.pid.yaw.integral_rad_s),
                _round(result.pid.yaw.derivative_rad_s),
            ],
            "rate_limited": (
                result.intent.limits.yaw_rate_limited
                or result.intent.limits.pitch_rate_limited
            ),
            "acceleration_limited": result.intent.limits.acceleration_limited,
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--verify-golden", action="store_true")
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    records = replay(fixture)
    if args.verify_golden and records != fixture.get("golden"):
        print(json.dumps({"result": "golden_mismatch", "actual": records}, indent=2))
        return 1
    print(json.dumps({"result": "pass", "records": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
