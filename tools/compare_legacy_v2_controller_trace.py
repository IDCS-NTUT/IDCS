#!/usr/bin/env python3
"""Compare legacy ControlCmd and qualified V2 intent decisions offline.

Both policies consume the exact same immutable ControlObservation records. The
tool never opens sockets, serial devices, or hardware endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from common.config import load_config_bundle, resolve_active_video_profile, resolve_config_paths
from common.control import ControlConfig, LaserMountConfig
from common.schemas import CamState, ControlCmd, ControlIntent, ControlObservation
from jetson.controller import ControlLoop
from jetson.qualified_controller_profile import load_qualified_shadow_policy_config
from jetson.shadow_rate_policy import ShadowRatePolicy


class _CapturePublisher:
    def __init__(self) -> None:
        self.commands: list[ControlCmd] = []

    def send_string(self, payload: str, **_kwargs: Any) -> None:
        self.commands.append(ControlCmd.model_validate_json(payload))


def load_observations(path: Path) -> tuple[list[ControlObservation], int]:
    observations: list[ControlObservation] = []
    invalid = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            if record.get("type") == "observation":
                payload = record["observation"]
            elif record.get("type") == "tick":
                payload = record["observation"]
            else:
                continue
            observations.append(ControlObservation.model_validate(payload))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            invalid += 1
    return observations, invalid


def _legacy_commands(
    observations: Iterable[ControlObservation],
    config: ControlConfig,
    laser_mount: LaserMountConfig,
) -> list[ControlCmd]:
    publisher = _CapturePublisher()
    loop = ControlLoop(config, publisher, laser_mount=laser_mount)
    for observation in observations:
        now = observation.created_monotonic_ns / 1_000_000_000.0
        if observation.gimbal.valid:
            loop.update_cam_state(
                CamState(
                    frame_id=observation.sequence,
                    src_ts_ms=observation.created_monotonic_ns // 1_000_000,
                    pan=observation.gimbal.yaw_rad or 0.0,
                    tilt=observation.gimbal.pitch_rad or 0.0,
                    pan_rate=observation.gimbal.yaw_rate_rad_s,
                    tilt_rate=observation.gimbal.pitch_rate_rad_s,
                )
            )
        loop.update_control_observation(observation, received_at=now)
        before = len(publisher.commands)
        loop.tick(now=now)
        if len(publisher.commands) != before + 1:
            raise ValueError("legacy controller did not emit exactly one command per observation")
    return publisher.commands


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def compare_policies(
    observations: Sequence[ControlObservation],
    *,
    legacy_config: ControlConfig,
    laser_mount: LaserMountConfig,
    redesigned_policy: ShadowRatePolicy,
    max_safety_mismatches: int,
    max_rate_delta_rad_s: float,
) -> dict[str, Any]:
    legacy = _legacy_commands(observations, legacy_config, laser_mount)
    redesigned = [redesigned_policy.decide(value) for value in observations]
    deltas: list[float] = []
    safety_mismatches = 0
    records: list[dict[str, Any]] = []
    for observation, command, intent in zip(observations, legacy, redesigned):
        legacy_authorized = bool(command.target_ok)
        redesigned_authorized = intent.reason in {"tracking", "position_limit_hold"}
        mismatch = legacy_authorized != redesigned_authorized
        safety_mismatches += int(mismatch)
        delta = max(
            abs(float(command.pan_rate_cmd) - intent.yaw_rate_rad_s),
            abs(float(command.tilt_rate_cmd) - intent.pitch_rate_rad_s),
        )
        deltas.append(delta)
        records.append({
            "observation_sequence": observation.sequence,
            "legacy_target_ok": legacy_authorized,
            "redesigned_reason": intent.reason,
            "safety_decision_mismatch": mismatch,
            "rate_delta_rad_s": delta,
        })
    p95 = _percentile(deltas, 0.95)
    failures: list[str] = []
    if not observations:
        failures.append("no_observations")
    if safety_mismatches > max_safety_mismatches:
        failures.append("safety_decision_mismatches")
    if p95 > max_rate_delta_rad_s:
        failures.append("rate_delta_p95_above_limit")
    return {
        "format": "idcs.legacy_v2_controller_parity",
        "version": 1,
        "physical_control_disabled": True,
        "observations": len(observations),
        "safety_decision_mismatches": safety_mismatches,
        "max_rate_delta_rad_s": max(deltas, default=0.0),
        "p95_rate_delta_rad_s": p95,
        "thresholds": {
            "max_safety_mismatches": max_safety_mismatches,
            "max_rate_delta_rad_s": max_rate_delta_rad_s,
        },
        "qualified": not failures,
        "failures": failures,
        "records": records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--config", default="configs/network.yaml")
    parser.add_argument(
        "--config-extra",
        default="configs/perception.yaml,configs/control.yaml,configs/system.yaml",
    )
    parser.add_argument("--qualified-controller-report", required=True, type=Path)
    parser.add_argument("--max-safety-mismatches", type=int, default=0)
    parser.add_argument("--max-rate-delta-rad-s", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.max_safety_mismatches < 0 or args.max_rate_delta_rad_s < 0.0:
        parser.error("parity thresholds must be non-negative")
    bundle = load_config_bundle(resolve_config_paths(args.config, args.config_extra))
    config = bundle.mutable_copy()
    outer = config.get("control", {}).get("mpc", {}).get("outer_tuner", {})
    if isinstance(outer, dict):
        outer["enabled"] = False
        outer["load_on_start"] = False
        outer["save_on_update"] = False
    video, _profile = resolve_active_video_profile(config)
    legacy_config = ControlConfig.from_raw_config(
        config, (int(video["width"]), int(video["height"]))
    )
    laser_mount = LaserMountConfig.from_raw_config(config)
    report_path = args.qualified_controller_report.resolve()
    policy = ShadowRatePolicy(load_qualified_shadow_policy_config(report_path))
    observations, invalid = load_observations(args.trace)
    result = compare_policies(
        observations,
        legacy_config=legacy_config,
        laser_mount=laser_mount,
        redesigned_policy=policy,
        max_safety_mismatches=args.max_safety_mismatches,
        max_rate_delta_rad_s=args.max_rate_delta_rad_s,
    )
    result.update({
        "invalid_records": invalid,
        "trace": str(args.trace.resolve()),
        "qualified_controller_report": str(report_path),
        "qualified_controller_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        **bundle.provenance(),
    })
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["qualified"] and invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
