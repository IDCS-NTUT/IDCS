#!/usr/bin/env python3
"""Record versioned atomic ControlObservation snapshots from live metadata.

The recorder is passive: it subscribes to metadata only, creates a shadow
observation at a fixed rate, and writes JSONL.  It has no ControlCmd publisher,
serial import, or gimbal access.
"""

from __future__ import annotations

import argparse
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
from common.schemas import CamState, ManualControlState, detection_msg_from_json
from common.shutdown import install_signal_handlers
from jetson.control_observation import ControlObservationAssembler
from jetson.shadow_rate_policy import ShadowRatePolicy, ShadowRatePolicyConfig


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/network.yaml")
    parser.add_argument("--config-extra", default="configs/control.yaml,configs/system.yaml")
    parser.add_argument("--detection-sub", required=True)
    parser.add_argument("--camstate-sub", required=True)
    parser.add_argument("--manual-sub", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--loop-hz", type=float, default=50.0)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--shadow-rate", action="store_true",
                        help="also record non-actuating ShadowRatePolicy intents")
    parser.add_argument("--yaw-kp", type=float, default=1.0)
    parser.add_argument("--pitch-kp", type=float, default=1.0)
    return parser.parse_args(argv)


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
    cfg = load_merged_config(expand_config_paths(args.config, args.config_extra))
    video = cfg.get("video", {}) if isinstance(cfg.get("video", {}), Mapping) else {}
    profiles = video.get("profiles", {}) if isinstance(video.get("profiles", {}), Mapping) else {}
    profile = profiles.get(str(video.get("active_profile", "720p")), {})
    frame_size = (int(profile.get("width", 1280)), int(profile.get("height", 720)))
    assembler = ControlObservationAssembler(ControlConfig.from_raw_config(cfg, frame_size))
    policy = ShadowRatePolicy(ShadowRatePolicyConfig(
        yaw_kp=args.yaw_kp, pitch_kp=args.pitch_kp,
        nominal_period_s=1.0 / args.loop_hz,
    )) if args.shadow_rate else None
    ctx = zmq.Context()
    sockets = (_sub(ctx, args.detection_sub), _sub(ctx, args.camstate_sub), _sub(ctx, args.manual_sub))
    stop = install_signal_handlers()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    period = 1.0 / args.loop_hz
    next_tick = start
    count = errors = missed_periods = 0
    deadline_lateness_sum_s = 0.0
    max_deadline_lateness_s = 0.0
    try:
        with args.output.open("w", encoding="utf-8", buffering=1) as output:
            output.write(json.dumps({"type": "meta", "format": "idcs.control_protocol_trace",
                                     "version": 1, "start_monotonic_ns": int(start * 1e9),
                                     "loop_hz": args.loop_hz}, sort_keys=True) + "\n")
            while not stop.is_set() and (args.duration_s is None or time.monotonic() - start < args.duration_s):
                now = time.monotonic()
                for socket, decoder, update in (
                    (sockets[0], detection_msg_from_json, assembler.update_detection),
                    (sockets[1], lambda value: CamState(**json.loads(value)), assembler.update_cam_state),
                    (sockets[2], lambda value: ManualControlState(**json.loads(value)), assembler.update_manual_state),
                ):
                    payload = _latest(socket)
                    if payload is None:
                        continue
                    try:
                        update(decoder(payload.decode("utf-8")) if decoder is not detection_msg_from_json else decoder(payload), received_at=now)
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
                        output.write(json.dumps({"type": "intent", "intent": intent.model_dump(mode="json")},
                                                separators=(",", ":"), sort_keys=True) + "\n")
                    count += 1
                    next_tick = now + period
                time.sleep(min(0.002, period / 4.0))
            summary = {
                'type': 'summary',
                'format': 'idcs.control_protocol_trace',
                'version': 1,
                'observations': count,
                'decode_errors': errors,
                'missed_periods': missed_periods,
                'mean_deadline_lateness_ms': (
                    1000.0 * deadline_lateness_sum_s / count if count else 0.0
                ),
                'max_deadline_lateness_ms': 1000.0 * max_deadline_lateness_s,
                'physical_control_disabled': True,
            }
            output.write(
                json.dumps(summary, separators=(',', ':'), sort_keys=True) + '\n'
            )
    finally:
        for socket in sockets:
            socket.close(0)
        ctx.term()
    print(json.dumps({"format": "idcs.control_protocol_trace", "observations": count,
                      "decode_errors": errors, "physical_control_disabled": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
