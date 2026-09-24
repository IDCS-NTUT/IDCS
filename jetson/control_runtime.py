"""Qualified V2 fixed-rate controller runtime.

Consumes native V2 perception, encoder CamState, and manual safety state and
emits short-lived ControlIntent messages. Serial access remains exclusively at
the gimbal bridge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import zmq

from common.config import ConfigError, load_config_bundle, resolve_active_video_profile, resolve_config_paths
from common.control import ControlConfig, ControlConfigError, LaserConfigError, LaserMountConfig
from common.perception import perception_snapshot_from_json
from common.schemas import CamState, ControlIntent, manual_control_state_from_json
from common.shutdown import install_signal_handlers
from jetson.control_observation import ControlObservationAssembler
from jetson.qualified_controller_profile import load_qualified_shadow_policy_config
from jetson.shadow_rate_policy import ShadowRatePolicy, ShadowRatePolicyConfig


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


def _optional_bounds(
    config: Mapping[str, Any], minimum: str, maximum: str
) -> tuple[float, float] | None:
    low, high = config.get(minimum), config.get(maximum)
    if low is None and high is None:
        return None
    if low is None or high is None:
        raise ValueError(f"gimbal.{minimum} and gimbal.{maximum} must be set together")
    bounds = (float(low), float(high))
    if bounds[0] >= bounds[1]:
        raise ValueError(f"gimbal.{minimum} must be less than gimbal.{maximum}")
    return bounds


def load_runtime_settings(
    config: Mapping[str, Any], *, base_dir: Path, sequence_base: int = 0
) -> tuple[dict[str, Any], ShadowRatePolicyConfig]:
    net = config.get("net")
    gimbal = config.get("gimbal")
    controller = config.get("controller_v2")
    if not isinstance(net, Mapping):
        raise ValueError("configuration requires net mapping")
    if not isinstance(gimbal, Mapping):
        raise ValueError("configuration requires gimbal mapping")
    if not isinstance(controller, Mapping):
        raise ValueError("configuration requires controller_v2 mapping")
    report = Path(str(controller.get("qualified_report", "")))
    if not report.is_absolute():
        report = base_dir / report
    report = report.resolve()
    if not report.is_file():
        raise ValueError(f"qualified controller report does not exist: {report}")
    try:
        valid_for_ms = float(controller["valid_for_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("controller_v2.valid_for_ms must be numeric") from exc
    if not 0.0 < valid_for_ms <= 250.0:
        raise ValueError("controller_v2.valid_for_ms must be in (0, 250]")
    policy = load_qualified_shadow_policy_config(
        report,
        yaw_position_limits_rad=_optional_bounds(gimbal, "yaw_min_rad", "yaw_max_rad"),
        pitch_position_limits_rad=_optional_bounds(gimbal, "pitch_min_rad", "pitch_max_rad"),
        valid_for_ns=int(valid_for_ms * 1_000_000.0),
        intent_mode="live",
        sequence_base=sequence_base,
    )
    endpoints = {
        "snapshot_sub": str(net.get("zmq_perception_v2", "")),
        "gimbal_sub": str(net.get("zmq_gimbal_state", "")),
        "manual_bind": _bind(str(net.get("zmq_manual_state", "")), "net.zmq_manual_state"),
        "intent_bind": _bind(str(net.get("zmq_control", "")), "net.zmq_control"),
        "diagnostics_bind": _bind(
            str(net.get("zmq_control_diagnostics", "")),
            "net.zmq_control_diagnostics",
        ),
    }
    _port(endpoints["snapshot_sub"], "net.zmq_perception_v2")
    _port(endpoints["gimbal_sub"], "net.zmq_gimbal_state")
    return ({
        **endpoints,
        "qualified_report": report,
        "qualified_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    }, policy)


def _write_json(path: Path | None, value: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _build_shutdown_intents(
    last_intent: ControlIntent | None,
    *,
    issued_monotonic_ns: int,
    valid_for_ns: int,
    copies: int = 3,
) -> tuple[ControlIntent, ...]:
    """Build redundant zero-rate stops without replaying an intent sequence."""
    if copies <= 0:
        raise ValueError("copies must be positive")
    first_sequence = 1 if last_intent is None else last_intent.sequence + 1
    observation_sequence = 0 if last_intent is None else last_intent.observation_sequence
    return tuple(
        ControlIntent(
            sequence=first_sequence + offset,
            observation_sequence=observation_sequence,
            issued_monotonic_ns=issued_monotonic_ns,
            valid_until_monotonic_ns=issued_monotonic_ns + valid_for_ns,
            mode="live",
            yaw_rate_rad_s=0.0,
            pitch_rate_rad_s=0.0,
            reason="controller_shutdown",
        )
        for offset in range(copies)
    )


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/network.yaml")
    parser.add_argument("--config-extra", default="configs/perception.yaml,configs/control.yaml,configs/system.yaml")
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument(
        "--diagnostics-trace",
        type=Path,
        help="optional JSONL output for versioned estimator/timing diagnostics",
    )
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--health-file", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--enable-live-intent-publish",
        action="store_true",
        help="required acknowledgement before binding production net.zmq_control",
    )
    args = parser.parse_args(argv)
    if args.duration_s is not None and args.duration_s <= 0.0:
        parser.error("--duration-s must be positive")
    paths = resolve_config_paths(args.config, args.config_extra)
    try:
        bundle = load_config_bundle(
            paths,
            required_sections=("net", "video", "control", "controller_v2", "gimbal"),
        )
        config = bundle.mutable_copy()
        video, _profile = resolve_active_video_profile(config)
        frame_size = (int(video["width"]), int(video["height"]))
        control = ControlConfig.from_raw_config(config, frame_size)
        laser_mount = LaserMountConfig.from_raw_config(config)
        sequence_base = time.time_ns() // 1_000
        settings, policy_config = load_runtime_settings(
            config, base_dir=Path.cwd(), sequence_base=sequence_base
        )
    except (ConfigError, ControlConfigError, LaserConfigError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    startup = {
        "mode": "v2_qualified_fixed_rate_controller",
        "intent_mode": policy_config.intent_mode,
        "loop_hz": 1.0 / policy_config.nominal_period_s,
        "snapshot_sub": settings["snapshot_sub"],
        "gimbal_sub": settings["gimbal_sub"],
        "manual_bind": settings["manual_bind"],
        "intent_bind": settings["intent_bind"],
        "diagnostics_bind": settings["diagnostics_bind"],
        "qualified_report": str(settings["qualified_report"]),
        "qualified_report_sha256": settings["qualified_report_sha256"],
        "serial_access": False,
        "diagnostics_trace_enabled": args.diagnostics_trace is not None,
        **bundle.provenance(),
    }
    if args.check:
        print(json.dumps(startup, sort_keys=True))
        return 0
    if not args.enable_live_intent_publish:
        raise SystemExit("--enable-live-intent-publish is required")

    context = zmq.Context()
    snapshot_sub = _sub(context, settings["snapshot_sub"])
    gimbal_sub = _sub(context, settings["gimbal_sub"])
    manual_pull = context.socket(zmq.PULL)
    manual_pull.setsockopt(zmq.RCVHWM, 10)
    manual_pull.setsockopt(zmq.LINGER, 0)
    manual_pull.bind(settings["manual_bind"])
    intent_pub = context.socket(zmq.PUB)
    intent_pub.setsockopt(zmq.SNDHWM, 1)
    intent_pub.setsockopt(zmq.LINGER, 100)
    intent_pub.bind(settings["intent_bind"])
    diagnostics_pub = context.socket(zmq.PUB)
    diagnostics_pub.setsockopt(zmq.SNDHWM, 1)
    diagnostics_pub.setsockopt(zmq.LINGER, 0)
    diagnostics_pub.bind(settings["diagnostics_bind"])

    assembler = ControlObservationAssembler(
        control, laser_mount=laser_mount, sequence_base=sequence_base
    )
    policy = ShadowRatePolicy(policy_config)
    stop = install_signal_handlers()
    start = time.monotonic()
    deadline = None if args.duration_s is None else start + args.duration_s
    period = policy_config.nominal_period_s
    next_tick = start
    snapshots = gimbal_states = manual_states = invalid = intents = missed = 0
    reasons: Counter[str] = Counter()
    last_intent: ControlIntent | None = None
    last_health_intents = -1
    trace = None
    diagnostics_trace = None
    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        trace = args.trace.open("w", encoding="utf-8", buffering=1)
        trace.write(json.dumps({"type": "meta", **startup}, sort_keys=True) + "\n")
    if args.diagnostics_trace is not None:
        args.diagnostics_trace.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_trace = args.diagnostics_trace.open("w", encoding="utf-8", buffering=1)
    if args.ready_file is not None:
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        args.ready_file.write_text("ready\n", encoding="utf-8")
    try:
        while not stop.is_set() and (deadline is None or time.monotonic() < deadline):
            now = time.monotonic()
            payload = _latest(snapshot_sub)
            if payload is not None:
                try:
                    assembler.update_perception_snapshot(perception_snapshot_from_json(payload), received_at=now)
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
                    gimbal_states += 1
            payload = _latest(manual_pull)
            if payload is not None:
                try:
                    manual = manual_control_state_from_json(payload)
                except (TypeError, ValueError, json.JSONDecodeError):
                    invalid += 1
                else:
                    assembler.update_manual_state(manual, received_at=now)
                    manual_states += 1
            if now >= next_tick:
                missed += int(max(0.0, now - next_tick) / period)
                observation = assembler.build(now=now)
                last_intent = policy.decide(observation)
                intent_pub.send_string(last_intent.model_dump_json(exclude_none=True))
                if policy.last_diagnostics is not None:
                    diagnostics_pub.send_string(
                        policy.last_diagnostics.model_dump_json(exclude_none=True)
                    )
                if diagnostics_trace is not None and policy.last_diagnostics is not None:
                    diagnostics_trace.write(
                        policy.last_diagnostics.model_dump_json(exclude_none=True) + "\n"
                    )
                if trace is not None:
                    trace.write(
                        json.dumps(
                            {
                                "type": "tick",
                                "observation": observation.model_dump(mode="json"),
                                "intent": last_intent.model_dump(mode="json"),
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                        + "\n"
                    )
                intents += 1
                reasons[last_intent.reason] += 1
                next_tick = now + period
            if (
                args.health_file is not None
                and intents
                and intents % 50 == 0
                and intents != last_health_intents
            ):
                _write_json(args.health_file, {
                    **startup,
                    "updated_monotonic_ns": time.monotonic_ns(),
                    "intents": intents,
                    "missed_periods": missed,
                    "invalid_messages": invalid,
                    "reasons": dict(sorted(reasons.items())),
                })
                last_health_intents = intents
            time.sleep(min(0.002, period / 4.0))
    finally:
        now_ns = time.monotonic_ns()
        stop_intents = _build_shutdown_intents(
            last_intent,
            issued_monotonic_ns=now_ns,
            valid_for_ns=min(policy_config.valid_for_ns, 50_000_000),
        )
        for stop_intent in stop_intents:
            intent_pub.send_string(stop_intent.model_dump_json(exclude_none=True))
            if trace is not None:
                trace.write(
                    json.dumps(
                        {
                            "type": "shutdown",
                            "intent": stop_intent.model_dump(mode="json"),
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
            time.sleep(0.01)
        if args.ready_file is not None:
            args.ready_file.unlink(missing_ok=True)
        if trace is not None:
            trace.close()
        if diagnostics_trace is not None:
            diagnostics_trace.close()
        for socket in (snapshot_sub, gimbal_sub, manual_pull, intent_pub, diagnostics_pub):
            socket.close(0)
        context.term()

    report = {
        **startup,
        "duration_s": time.monotonic() - start,
        "snapshot_messages": snapshots,
        "gimbal_states": gimbal_states,
        "manual_states": manual_states,
        "invalid_messages": invalid,
        "intents": intents,
        "missed_periods": missed,
        "reasons": dict(sorted(reasons.items())),
    }
    _write_json(args.report, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
