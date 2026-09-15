#!/usr/bin/env python3
"""Record versioned atomic ControlObservation snapshots from live metadata.

The recorder is passive: it subscribes to PerceptionSnapshotV2 and state
metadata, creates a shadow observation at a fixed rate, and writes JSONL.  It
has no ControlCmd publisher, serial import, or gimbal access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import zmq

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from common.config_sync import expand_config_paths, load_merged_config
from common.control import ControlConfig
from common.perception import perception_snapshot_from_json
from common.schemas import CamState, ManualControlState
from common.shutdown import install_signal_handlers
from jetson.control_observation import ControlObservationAssembler
from jetson.qualified_controller_profile import load_qualified_shadow_policy_config
from jetson.shadow_rate_policy import ShadowRatePolicy, ShadowRatePolicyConfig


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/network.yaml")
    parser.add_argument("--config-extra", default="configs/control.yaml,configs/system.yaml")
    parser.add_argument("--snapshot-sub", required=True)
    parser.add_argument("--camstate-sub", required=True)
    parser.add_argument("--manual-sub", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--loop-hz", type=float, default=50.0)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--shadow-rate", action="store_true",
                        help="also record non-actuating ShadowRatePolicy intents")
    parser.add_argument(
        "--qualified-controller-report", type=Path,
        help="passing offline report used to freeze PID/Kalman/feedforward values",
    )
    parser.add_argument("--yaw-kp", type=float, default=1.0)
    parser.add_argument("--pitch-kp", type=float, default=1.0)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy(args: argparse.Namespace) -> tuple[Optional[ShadowRatePolicy], dict[str, object]]:
    if not args.shadow_rate:
        return None, {"policy": "observation_only"}
    if args.qualified_controller_report is None:
        config = ShadowRatePolicyConfig(
            yaw_kp=args.yaw_kp, pitch_kp=args.pitch_kp,
            nominal_period_s=1.0 / args.loop_hz,
        )
        return ShadowRatePolicy(config), {"policy": "shadow_rate_explicit"}
    report = args.qualified_controller_report.resolve()
    config = load_qualified_shadow_policy_config(report)
    configured_hz = 1.0 / config.nominal_period_s
    if abs(configured_hz - args.loop_hz) > 1e-9:
        raise ValueError(
            f"qualified controller cadence is {configured_hz:g} Hz, not requested {args.loop_hz:g} Hz"
        )
    return ShadowRatePolicy(config), {
        "policy": "shadow_rate_qualified_los",
        "qualified_controller_report": str(report),
        "qualified_controller_report_sha256": _sha256(report),
    }


def _sub(ctx: zmq.Context, endpoint: str) -> zmq.Socket:
    socket = ctx.socket(zmq.SUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.connect(endpoint)
    return socket


def _latest(socket: zmq.Socket) -> Optional[bytes]:
    value = None
    while True:
        try:
            value = socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return value


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    if args.loop_hz <= 0 or (args.duration_s is not None and args.duration_s <= 0):
        raise SystemExit("--loop-hz and --duration-s must be positive")
    if args.qualified_controller_report is not None and not args.shadow_rate:
        raise SystemExit("--qualified-controller-report requires --shadow-rate")
    cfg = load_merged_config(expand_config_paths(args.config, args.config_extra))
    video = cfg.get("video", {}) if isinstance(cfg.get("video", {}), Mapping) else {}
    profiles = video.get("profiles", {}) if isinstance(video.get("profiles", {}), Mapping) else {}
    profile = profiles.get(str(video.get("active_profile", "720p")), {})
    frame_size = (int(profile.get("width", 1280)), int(profile.get("height", 720)))
    assembler = ControlObservationAssembler(ControlConfig.from_raw_config(cfg, frame_size))
    try:
        policy, policy_metadata = _policy(args)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid qualified controller report: {exc}") from exc
    ctx = zmq.Context()
    sockets = (_sub(ctx, args.snapshot_sub), _sub(ctx, args.camstate_sub), _sub(ctx, args.manual_sub))
    stop = install_signal_handlers()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    period = 1.0 / args.loop_hz
    next_tick = start
    count = errors = missed_periods = tracking_intents = hold_intents = limited_intents = 0
    deadline_lateness_sum_s = 0.0
    max_deadline_lateness_s = 0.0
    try:
        with args.output.open("w", encoding="utf-8", buffering=1) as output:
            output.write(json.dumps({"type": "meta", "format": "idcs.control_protocol_trace",
                                     "version": 2, "start_monotonic_ns": int(start * 1e9),
                                     "loop_hz": args.loop_hz, "physical_control_disabled": True,
                                     **policy_metadata}, sort_keys=True) + "\n")
            while not stop.is_set() and (args.duration_s is None or time.monotonic() - start < args.duration_s):
                now = time.monotonic()
                for socket, decoder, update in (
                    (sockets[0], perception_snapshot_from_json, assembler.update_perception_snapshot),
                    (sockets[1], lambda value: CamState(**json.loads(value)), assembler.update_cam_state),
                    (sockets[2], lambda value: ManualControlState(**json.loads(value)), assembler.update_manual_state),
                ):
                    payload = _latest(socket)
                    if payload is None:
                        continue
                    try:
                        update(decoder(payload), received_at=now)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        errors += 1
                if now >= next_tick:
                    deadline_lateness_s = max(0.0, now - next_tick)
                    missed_periods += int(deadline_lateness_s / period)
                    deadline_lateness_sum_s += deadline_lateness_s
                    max_deadline_lateness_s = max(
                        max_deadline_lateness_s, deadline_lateness_s
                    )
                    observation = assembler.build(now=now)
                    output.write(json.dumps({"type": "observation", "observation": observation.model_dump(mode="json")},
                                            separators=(",", ":"), sort_keys=True) + "\n")
                    if policy is not None:
                        intent = policy.decide(observation)
                        if intent.reason == "tracking":
                            tracking_intents += 1
                        else:
                            hold_intents += 1
                        if any(intent.limits.model_dump().values()):
                            limited_intents += 1
                        output.write(json.dumps({"type": "intent", "intent": intent.model_dump(mode="json")},
                                                separators=(",", ":"), sort_keys=True) + "\n")
                    count += 1
                    next_tick = now + period
                time.sleep(min(0.002, period / 4.0))
            summary = {
                'type': 'summary',
                'format': 'idcs.control_protocol_trace',
                'version': 2,
                'observations': count,
                'decode_errors': errors,
                'missed_periods': missed_periods,
                'mean_deadline_lateness_ms': (
                    1000.0 * deadline_lateness_sum_s / count if count else 0.0
                ),
                'max_deadline_lateness_ms': 1000.0 * max_deadline_lateness_s,
                'physical_control_disabled': True,
                'tracking_intents': tracking_intents,
                'hold_intents': hold_intents,
                'limited_intents': limited_intents,
                **policy_metadata,
            }
            output.write(
                json.dumps(summary, separators=(',', ':'), sort_keys=True) + '\n'
            )
    finally:
        for socket in sockets:
            socket.close(0)
        ctx.term()
    print(json.dumps({"format": "idcs.control_protocol_trace", "observations": count,
                      "decode_errors": errors, "tracking_intents": tracking_intents,
                      "hold_intents": hold_intents, "limited_intents": limited_intents,
                      "physical_control_disabled": True, **policy_metadata}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
