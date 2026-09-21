"""Dedicated V2 fixed-rate controller runtime.

The runtime owns metadata sockets only. It never imports or opens serial
hardware; ``jetson.gimbal_bridge`` remains the sole actuator boundary.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import zmq

from common.config import (
    ConfigError,
    load_config_bundle,
    resolve_active_video_profile,
    resolve_config_paths,
)
from common.control import (
    ControlConfig,
    ControlConfigError,
    LaserConfigError,
    LaserMountConfig,
)
from common.perception import perception_snapshot_from_json
from common.schemas import CamState, ManualControlState, manual_control_state_from_json
from common.shutdown import install_signal_handlers
from jetson.control_observation import ControlObservationAssembler
from jetson.controller import ControlLoop
from jetson.fixed_rate_controller import FixedRateController


def _port(endpoint: str, name: str) -> int:
    parsed = urlsplit(endpoint)
    if parsed.scheme != "tcp" or not parsed.hostname:
        raise ValueError(f"{name} must be tcp://host:port")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{name} has an invalid port") from exc
    if port is None or not 1 <= port <= 65535:
        raise ValueError(f"{name} must include a valid port")
    return port


def _bind(endpoint: str, name: str) -> str:
    return f"tcp://0.0.0.0:{_port(endpoint, name)}"


def _sub(context: zmq.Context, endpoint: str) -> zmq.Socket:
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.connect(endpoint)
    return socket


def _latest(socket: zmq.Socket) -> bytes | None:
    value = None
    while True:
        try:
            value = socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return value


def _latest_manual(socket: zmq.Socket) -> bytes | None:
    value = None
    while True:
        try:
            value = socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return value


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    net = config.get("net")
    if not isinstance(net, Mapping):
        raise ValueError("configuration requires net mapping")
    video, profile = resolve_active_video_profile(config)
    frame_size = (int(video["width"]), int(video["height"]))
    control = ControlConfig.from_raw_config(config, frame_size)
    laser_mount = LaserMountConfig.from_raw_config(config)
    endpoints = {
        "snapshot_sub": str(net.get("zmq_perception_v2", "")),
        "gimbal_sub": str(net.get("zmq_gimbal_state", "")),
        "manual_bind": _bind(str(net.get("zmq_manual_state", "")), "net.zmq_manual_state"),
        "control_bind": _bind(str(net.get("zmq_control", "")), "net.zmq_control"),
    }
    _port(endpoints["snapshot_sub"], "net.zmq_perception_v2")
    _port(endpoints["gimbal_sub"], "net.zmq_gimbal_state")
    return {
        "frame_size": frame_size,
        "video_profile": profile,
        "control": control,
        "laser_mount": laser_mount,
        **endpoints,
    }


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/network.yaml")
    parser.add_argument(
        "--config-extra",
        default="configs/perception.yaml,configs/control.yaml,configs/system.yaml",
    )
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--enable-control-publish",
        action="store_true",
        help="required acknowledgement before binding production net.zmq_control",
    )
    args = parser.parse_args(argv)
    if args.duration_s is not None and args.duration_s <= 0.0:
        parser.error("--duration-s must be positive")
    paths = resolve_config_paths(args.config, args.config_extra)
    try:
        bundle = load_config_bundle(paths, required_sections=("net", "video", "control"))
        # ControlConfig still normalizes several nested legacy-shaped sections;
        # give it a detached mutable copy while the authoritative bundle stays
        # immutable and its provenance remains unchanged.
        settings = _settings(bundle.mutable_copy())
    except (ConfigError, ControlConfigError, LaserConfigError, KeyError, ValueError) as exc:
        parser.error(str(exc))

    control: ControlConfig = settings["control"]
    startup = {
        "mode": "v2_fixed_rate_controller",
        "frame_size": settings["frame_size"],
        "video_profile": settings["video_profile"],
        "controller": control.controller,
        "loop_hz": float(control.loop_hz or 50.0),
        "snapshot_sub": settings["snapshot_sub"],
        "gimbal_sub": settings["gimbal_sub"],
        "manual_bind": settings["manual_bind"],
        "control_bind": settings["control_bind"],
        "serial_access": False,
        **bundle.provenance(),
    }
    if args.check:
        print(json.dumps(startup, sort_keys=True, default=str))
        return 0
    if not args.enable_control_publish:
        raise SystemExit("--enable-control-publish is required")

    context = zmq.Context()
    snapshot_sub = _sub(context, settings["snapshot_sub"])
    gimbal_sub = _sub(context, settings["gimbal_sub"])
    manual_pull = context.socket(zmq.PULL)
    manual_pull.setsockopt(zmq.RCVHWM, 10)
    manual_pull.setsockopt(zmq.LINGER, 0)
    manual_pull.bind(settings["manual_bind"])
    command_pub = context.socket(zmq.PUB)
    command_pub.setsockopt(zmq.SNDHWM, 1)
    command_pub.setsockopt(zmq.LINGER, 100)
    command_pub.bind(settings["control_bind"])

    assembler = ControlObservationAssembler(
        control, laser_mount=settings["laser_mount"]
    )
    runner = FixedRateController(
        ControlLoop(
            control,
            command_pub,
            laser_mount=settings["laser_mount"],
            cli_json_logs=True,
        ),
        loop_hz=float(control.loop_hz or 50.0),
    )
    stop = install_signal_handlers()
    started = time.monotonic()
    snapshots = gimbal_states = manual_states = invalid = 0
    try:
        while not stop.is_set() and (
            args.duration_s is None or time.monotonic() - started < args.duration_s
        ):
            now = time.monotonic()
            payload = _latest(snapshot_sub)
            if payload is not None:
                try:
                    assembler.update_perception_snapshot(
                        perception_snapshot_from_json(payload), received_at=now
                    )
                except (TypeError, ValueError):
                    invalid += 1
                else:
                    snapshots += 1
            payload = _latest(gimbal_sub)
            if payload is not None:
                try:
                    state = CamState.model_validate_json(payload)
                except ValueError:
                    invalid += 1
                else:
                    assembler.update_cam_state(state, received_at=now)
                    runner.update_cam_state(state)
                    gimbal_states += 1
            payload = _latest_manual(manual_pull)
            if payload is not None:
                try:
                    manual = manual_control_state_from_json(payload)
                except (TypeError, ValueError, json.JSONDecodeError):
                    invalid += 1
                else:
                    assembler.update_manual_state(manual, received_at=now)
                    manual_states += 1
            observation = assembler.build(now=now)
            runner.update_control_observation(observation, received_at=now)
            runner.advance(now)
            time.sleep(min(0.002, runner.period_s / 4.0))
    finally:
        stats = runner.stats
        for socket in (snapshot_sub, gimbal_sub, manual_pull, command_pub):
            socket.close(0)
        context.term()
    print(
        json.dumps(
            {
                **startup,
                "duration_s": time.monotonic() - started,
                "snapshot_messages": snapshots,
                "gimbal_states": gimbal_states,
                "manual_states": manual_states,
                "invalid_messages": invalid,
                **stats.__dict__,
            },
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
