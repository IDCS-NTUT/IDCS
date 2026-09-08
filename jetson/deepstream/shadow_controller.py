"""Simulation-only controller sidecar for DeepStream selected-target metadata.

This module deliberately does not modify the DeepStream verification pipeline
or bind the production control endpoint.  It consumes the control-free
DetectionMsg PUB stream and publishes commands only on a separately configured
simulation endpoint that an explicitly opted-in PC SimCamera may consume.
"""

from __future__ import annotations

import argparse
import json
import time
from urllib.parse import urlparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import zmq

from common.config_sync import merge_config_maps, parse_config_text
from common.control import ControlConfig
from common.schemas import detection_msg_from_json
from common.shutdown import install_signal_handlers
from jetson.controller import ControlLoop


def load_config(paths: Sequence[Path]) -> Mapping[str, Any]:
    return merge_config_maps(
        *(parse_config_text(path.read_text(encoding="utf-8"), str(path)) for path in paths)
    )


def _has_valid_preselection(message: Any) -> bool:
    if message.target_idx is None and message.target_track_id is None:
        return False
    if message.target_track_id is not None:
        return any(
            box.track_id is not None and int(box.track_id) == int(message.target_track_id)
            for box in message.boxes
        )
    return 0 <= int(message.target_idx) < len(message.boxes)


def _same_tcp_port(first: str, second: str) -> bool:
    try:
        return urlparse(first).scheme == urlparse(second).scheme == "tcp" and urlparse(first).port == urlparse(second).port
    except ValueError:
        return False


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idcs-config", action="append", type=Path, required=True,
                        help="IDCS YAML files in merge order; repeat for overrides")
    parser.add_argument("--detection-sub", required=True,
                        help="control-free DeepStream DetectionMsg PUB endpoint to connect")
    parser.add_argument("--sim-control-bind", required=True,
                        help="dedicated simulation-only ControlCmd PUB endpoint to bind")
    parser.add_argument("--duration-s", type=float, default=None,
                        help="optional bounded run duration for validation")
    args = parser.parse_args(argv)
    if args.duration_s is not None and args.duration_s <= 0:
        parser.error("--duration-s must be > 0")

    config = load_config(args.idcs_config)
    net = config.get("net", {}) if isinstance(config, Mapping) else {}
    production_control = str(net.get("zmq_control", "")) if isinstance(net, Mapping) else ""
    if _same_tcp_port(args.sim_control_bind, production_control):
        parser.error("--sim-control-bind must not use net.zmq_control (production endpoint)")

    video = config.get("video", {}) if isinstance(config, Mapping) else {}
    profiles = video.get("profiles", {}) if isinstance(video, Mapping) else {}
    active = str(video.get("active_profile", "720p")) if isinstance(video, Mapping) else "720p"
    profile = profiles.get(active, {}) if isinstance(profiles, Mapping) else {}
    frame_size = (int(profile.get("width", 1280)), int(profile.get("height", 720)))
    base_control = ControlConfig.from_raw_config(config, frame_size)
    # The sidecar consumes a selection produced upstream.  PID is the safe,
    # deterministic first closed-loop validation; it does not require encoder
    # feedback or write MPC tuner state.
    control = replace(base_control, controller="pid", target_selector="preselected", mpc=None)

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 1)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(args.detection_sub)
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 1)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind(args.sim_control_bind)
    controller = ControlLoop(control, pub, cli_json_logs=True)
    stop = install_signal_handlers()
    started = time.monotonic()
    received = selected = invalid = ticks = 0
    try:
        while not stop.is_set():
            now = time.monotonic()
            if args.duration_s is not None and now - started >= args.duration_s:
                break
            try:
                payload = sub.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                payload = None
            if payload is not None:
                try:
                    message = detection_msg_from_json(payload)
                except (TypeError, ValueError):
                    invalid += 1
                else:
                    received += 1
                    if _has_valid_preselection(message):
                        controller.update_detection(message)
                        selected += 1
            controller.tick(now)
            ticks += 1
            time.sleep(0.002)
    finally:
        sub.close()
        pub.close()
        ctx.term()
    print(json.dumps({
        "mode": "simulation_only_shadow_controller",
        "detection_messages": received,
        "selected_messages": selected,
        "invalid_messages": invalid,
        "ticks": ticks,
        "elapsed_s": round(time.monotonic() - started, 3),
        "controller": "pid",
        "control_endpoint": args.sim_control_bind,
        "physical_control_disabled": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
