#!/usr/bin/env python3
"""Run the controller at fixed cadence on a non-production shadow endpoint.

This compatibility entry point receives latest-only PerceptionSnapshotV2 and
CamState metadata, generates ControlCmd messages at the configured cadence,
and never opens serial hardware or the production ``net.zmq_control`` endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import zmq

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from common.config_sync import expand_config_paths, load_merged_config
from common.control import ControlConfig
from common.perception import perception_snapshot_from_json
from common.schemas import CamState
from common.shutdown import install_signal_handlers
from jetson.control_observation import ControlObservationAssembler
from jetson.controller import ControlLoop
from jetson.fixed_rate_controller import FixedRateController


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/network.yaml")
    parser.add_argument("--config-extra", default="configs/control.yaml,configs/system.yaml")
    parser.add_argument("--snapshot-sub", required=True)
    parser.add_argument("--camstate-sub", default=None)
    parser.add_argument("--shadow-control-bind", required=True)
    parser.add_argument("--loop-hz", type=float, default=None)
    parser.add_argument("--duration-s", type=float, default=None)
    return parser.parse_args(argv)


def _latest_recv(socket: zmq.Socket) -> bytes | None:
    latest: bytes | None = None
    while True:
        try:
            latest = socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return latest


def _same_tcp_port(first: str, second: str) -> bool:
    """Treat all TCP bind-address aliases for the production port as unsafe."""
    try:
        return urlparse(first).scheme == urlparse(second).scheme == "tcp" and urlparse(first).port == urlparse(second).port
    except ValueError:
        return False


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.duration_s is not None and args.duration_s <= 0.0:
        raise SystemExit("--duration-s must be > 0")
    cfg = load_merged_config(expand_config_paths(args.config, args.config_extra))
    raw_net = cfg.get("net", {})
    net: Mapping[str, Any] = raw_net if isinstance(raw_net, Mapping) else {}
    production = str(net.get("zmq_control", "")).strip()
    if _same_tcp_port(args.shadow_control_bind, production):
        raise SystemExit("--shadow-control-bind must not equal net.zmq_control")
    raw_video = cfg.get("video", {})
    video: Mapping[str, Any] = raw_video if isinstance(raw_video, Mapping) else {}
    profiles = video.get("profiles", {}) if isinstance(video.get("profiles", {}), Mapping) else {}
    active = str(video.get("active_profile", "720p"))
    profile = profiles.get(active, {}) if isinstance(profiles.get(active, {}), Mapping) else {}
    frame_size = (int(profile.get("width", 1280)), int(profile.get("height", 720)))
    control = ControlConfig.from_raw_config(cfg, frame_size)
    loop_hz = float(args.loop_hz if args.loop_hz is not None else (control.loop_hz or 50.0))

    ctx = zmq.Context()
    snapshot_sub = ctx.socket(zmq.SUB)
    snapshot_sub.setsockopt(zmq.CONFLATE, 1)
    snapshot_sub.setsockopt(zmq.LINGER, 0)
    snapshot_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    snapshot_sub.connect(args.snapshot_sub)
    state_sub = None
    if args.camstate_sub:
        state_sub = ctx.socket(zmq.SUB)
        state_sub.setsockopt(zmq.CONFLATE, 1)
        state_sub.setsockopt(zmq.LINGER, 0)
        state_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        state_sub.connect(args.camstate_sub)
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 1)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind(args.shadow_control_bind)
    runner = FixedRateController(ControlLoop(control, pub, cli_json_logs=True), loop_hz=loop_hz)
    assembler = ControlObservationAssembler(control)
    stop = install_signal_handlers()
    start = time.monotonic()
    try:
        while not stop.is_set() and (args.duration_s is None or time.monotonic() - start < args.duration_s):
            now = time.monotonic()
            payload = _latest_recv(snapshot_sub)
            if payload is not None:
                try:
                    snapshot = perception_snapshot_from_json(payload)
                    assembler.update_perception_snapshot(snapshot, received_at=now)
                    runner.update_control_observation(
                        assembler.build(now=now),
                        received_at=now,
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
            if state_sub is not None:
                payload = _latest_recv(state_sub)
                if payload is not None:
                    try:
                        raw = json.loads(payload.decode("utf-8"))
                        runner.update_cam_state(CamState(**raw))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
            runner.advance(now)
            time.sleep(min(0.002, runner.period_s / 4.0))
    finally:
        stats = runner.stats
        for socket in (snapshot_sub, state_sub, pub):
            if socket is not None:
                socket.close(0)
        ctx.term()
    print(json.dumps({"mode": "shadow_fixed_rate", "physical_control_disabled": True,
                      "loop_hz": loop_hz, **stats.__dict__}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
