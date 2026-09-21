#!/usr/bin/env python3
"""Run the simulator baseline V2 controller for pipeline evaluation.

The command publisher is deliberately restricted to a TCP loopback bind and
requires an explicit enable flag.  This tool imports no serial or gimbal
driver, cannot publish to the production control endpoint, and deliberately
does not load real-hardware controller tuning artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, MutableMapping, Optional, Sequence
from urllib.parse import urlsplit

import zmq

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from common.config import (  # noqa: E402
    ConfigError,
    load_config_bundle,
    resolve_active_video_profile,
    resolve_config_paths,
)
from common.control import (  # noqa: E402
    ControlConfig,
    ControlConfigError,
    LaserConfigError,
    LaserMountConfig,
)
from common.perception import (  # noqa: E402
    PerceptionSnapshotV2,
    perception_snapshot_from_json,
)
from common.schemas import (  # noqa: E402
    CamState,
    ControlCmd,
    ControlIntent,
    ControlObservation,
    ManualControlState,
)
from common.shutdown import install_signal_handlers  # noqa: E402
from common.sim_mode import resolve_simulation_motion_mode  # noqa: E402
from jetson.control_observation import ControlObservationAssembler  # noqa: E402
from jetson.shadow_rate_policy import ShadowRatePolicy, ShadowRatePolicyConfig  # noqa: E402


DEFAULT_CONTROL_ENDPOINT = "tcp://127.0.0.1:5571"
DEFAULT_CAMSTATE_ENDPOINT = "tcp://127.0.0.1:5572"

def require_loopback_endpoint(endpoint: str, name: str) -> str:
    value = str(endpoint or "").strip()
    parsed = urlsplit(value)
    if parsed.scheme != "tcp" or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ValueError(f"{name} must be a TCP loopback endpoint")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{name} has an invalid port") from exc
    if port is None or not 1 <= port <= 65535:
        raise ValueError(f"{name} must include a valid port")
    return value


def _mapping(parent: Mapping[str, Any], key: str, path: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}.{key} must be a mapping")
    return value


def apply_sim_camera_intrinsics(
    config: MutableMapping[str, Any], frame_size: tuple[int, int]
) -> dict[str, float]:
    """Make controller intrinsics exactly match the configured renderer FOV."""

    sim = _mapping(config, "sim", "config")
    camera = _mapping(sim, "camera", "sim")
    try:
        fov_y_deg = float(camera["fov_y_deg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("sim.camera.fov_y_deg must be numeric") from exc
    if not math.isfinite(fov_y_deg) or not 1.0 < fov_y_deg < 179.0:
        raise ValueError("sim.camera.fov_y_deg must be finite and between 1 and 179")
    width, height = frame_size
    fy_px = height / (2.0 * math.tan(math.radians(fov_y_deg) * 0.5))
    fov_x_deg = math.degrees(2.0 * math.atan(width / (2.0 * fy_px)))
    control = config.get("control")
    if not isinstance(control, MutableMapping):
        raise ValueError("config.control must be a mutable mapping")
    control["fx_fy_from_fov"] = True
    control["fov_deg"] = {"h": fov_x_deg, "v": fov_y_deg}
    return {
        "fov_x_deg": fov_x_deg,
        "fov_y_deg": fov_y_deg,
        "fx_px": fy_px,
        "fy_px": fy_px,
    }


def load_sim_baseline_policy_config(
    config: Mapping[str, Any],
) -> tuple[ShadowRatePolicyConfig, dict[str, float]]:
    """Load an estimator-free controller that is scoped only to simulation."""

    sim = _mapping(config, "sim", "config")
    baseline = _mapping(sim, "baseline_controller", "sim")
    if baseline.get("type") != "bounded_p":
        raise ValueError("sim.baseline_controller.type must be bounded_p")
    kp = _mapping(baseline, "kp", "sim.baseline_controller")
    rates = _mapping(
        baseline, "rate_limits_rad_s", "sim.baseline_controller"
    )
    accelerations = _mapping(
        baseline, "accel_limits_rad_s2", "sim.baseline_controller"
    )
    acceptance_raw = _mapping(
        baseline, "acceptance", "sim.baseline_controller"
    )
    try:
        loop_hz = float(baseline["loop_hz"])
        valid_for_ms = float(baseline["valid_for_ms"])
        acceptance = {
            str(key): float(value) for key, value in acceptance_raw.items()
        }
        policy = ShadowRatePolicyConfig(
            yaw_kp=float(kp["yaw"]),
            pitch_kp=float(kp["pitch"]),
            yaw_rate_limit_rad_s=float(rates["yaw"]),
            pitch_rate_limit_rad_s=float(rates["pitch"]),
            yaw_accel_limit_rad_s2=float(accelerations["yaw"]),
            pitch_accel_limit_rad_s2=float(accelerations["pitch"]),
            nominal_period_s=1.0 / loop_hz,
            valid_for_ns=int(valid_for_ms * 1_000_000.0),
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid sim.baseline_controller: {exc}") from exc
    required_acceptance = {
        "min_duration_s",
        "warmup_s",
        "max_acquisition_time_s",
        "acquisition_error_px",
        "acquisition_hold_s",
        "min_tracking_fraction",
        "max_rms_error_px",
        "max_p95_error_px",
        "max_rate_limited_fraction",
        "max_command_drops",
        "min_yaw_pose_span_rad",
    }
    missing = sorted(required_acceptance - acceptance.keys())
    if missing:
        raise ValueError(f"sim baseline acceptance is missing: {', '.join(missing)}")
    return policy, acceptance


def load_sim_evaluation_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the boundary between simulation and hardware qualification."""

    sim = _mapping(config, "sim", "config")
    raw = _mapping(sim, "evaluation_contract", "sim")
    contract = {
        "motion_model_role": str(raw.get("motion_model_role", "")),
        "detection_fidelity_goal": str(raw.get("detection_fidelity_goal", "")),
        "hardware_controller_simulation_role": str(
            raw.get("hardware_controller_simulation_role", "")
        ),
        "hardware_tuning_from_sim_allowed": raw.get(
            "hardware_tuning_from_sim_allowed"
        ),
    }
    if contract["motion_model_role"] != "stable_substitute":
        raise ValueError("sim motion_model_role must be stable_substitute")
    if contract["detection_fidelity_goal"] != "real_camera_equivalent":
        raise ValueError("sim detection_fidelity_goal must be real_camera_equivalent")
    if contract["hardware_controller_simulation_role"] != "interface_observation_only":
        raise ValueError(
            "sim hardware_controller_simulation_role must be interface_observation_only"
        )
    if contract["hardware_tuning_from_sim_allowed"] is not False:
        raise ValueError("hardware tuning from simulation must be disabled")
    return contract


def _sub(context: zmq.Context, endpoint: str) -> zmq.Socket:
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVHWM, 1)
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


def control_cmd_from_intent(
    observation: ControlObservation,
    intent: ControlIntent,
    *,
    snapshot: Optional[PerceptionSnapshotV2],
    control_config: ControlConfig,
    laser_mount: LaserMountConfig,
    now_s: float,
) -> ControlCmd:
    """Adapt one simulator-policy intent to the simulator command wire."""

    center_px = (control_config.cx_px, control_config.cy_px)
    target = observation.target
    target_uv = target.target_center_px or center_px
    target_ok = intent.reason == "tracking" and target.valid
    frame_id = observation.source_frame_id or 0
    source_time_ns = observation.source_time_ns or 0
    error_px = target.pixel_error if target_ok and target.pixel_error else (0.0, 0.0)
    error_rad = (
        target.bearing_error_rad
        if target_ok and target.bearing_error_rad
        else (0.0, 0.0)
    )
    parallax = {
        "laser_origin_px": center_px if target.parallax_active else None,
        "laser_dot_px": target.aim_reference_px if target.parallax_active else None,
        "laser_on_target": target.on_target if target_ok and target.parallax_active else None,
        "laser_range_m": target.distance_m,
        "laser_range_source": target.distance_source,
        "parallax_compensation_active": target.parallax_active,
    }
    return ControlCmd(
        frame_id=frame_id,
        src_ts_ms=source_time_ns // 1_000_000,
        cmd_ts_ms=int(now_s * 1000.0),
        target_ok=target_ok,
        target_uv=target_uv,
        err_uv=error_px,
        err_rad=error_rad,
        pan_rate_cmd=(intent.yaw_rate_rad_s if target_ok else 0.0),
        tilt_rate_cmd=(intent.pitch_rate_rad_s if target_ok else 0.0),
        controller_mode="pid",
        **parallax,
    )


def _safe_sim_manual_state(now_s: float) -> ManualControlState:
    return ManualControlState(
        src_ts_ms=int(now_s * 1000.0),
        source="sim_tracking_controller",
        active=False,
        emergency=False,
        control_cmd_enabled=True,
        joystick_raw=(0, 0),
        joystick_rate_cmd=(0.0, 0.0),
        serial_local_mode=False,
        note="loopback simulation authority only",
    )


def _mean_edge(values: list[float], *, first: bool, count: int = 50) -> Optional[float]:
    if not values:
        return None
    sample = values[:count] if first else values[-count:]
    return fmean(sample)


def _rms(values: list[float]) -> Optional[float]:
    return math.sqrt(fmean(value * value for value in values)) if values else None


def _percentile(values: list[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _acquisition_time_s(
    elapsed_s: list[float],
    errors_px: list[float],
    *,
    threshold_px: float,
    hold_s: float,
) -> Optional[float]:
    """Return the first window whose p95 error stays within the acquisition gate."""

    for start_index, start_s in enumerate(elapsed_s):
        end_index = start_index
        while end_index < len(elapsed_s) and elapsed_s[end_index] < start_s + hold_s:
            end_index += 1
        if end_index == len(elapsed_s) and elapsed_s[-1] < start_s + hold_s:
            break
        window = errors_px[start_index:end_index]
        if window and (_percentile(window, 0.95) or math.inf) <= threshold_px:
            return start_s
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/network.yaml")
    parser.add_argument(
        "--config-extra",
        default="configs/perception.yaml,configs/control.yaml,configs/system.yaml,configs/deepstream_pc_moving_tracking.yaml,configs/control_sim.yaml",
    )
    parser.add_argument("--snapshot-sub")
    parser.add_argument("--sim-camstate-sub", default=DEFAULT_CAMSTATE_ENDPOINT)
    parser.add_argument("--sim-control-bind", default=DEFAULT_CONTROL_ENDPOINT)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--status-interval-s", type=float, default=5.0)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--enable-sim-control",
        action="store_true",
        help="required acknowledgement before binding the simulator command PUB",
    )
    args = parser.parse_args(argv)
    if args.duration_s is not None and args.duration_s <= 0.0:
        parser.error("--duration-s must be positive")
    if args.status_interval_s <= 0.0:
        parser.error("--status-interval-s must be positive")
    try:
        command_endpoint = require_loopback_endpoint(
            args.sim_control_bind, "--sim-control-bind"
        )
        camstate_endpoint = require_loopback_endpoint(
            args.sim_camstate_sub, "--sim-camstate-sub"
        )
    except ValueError as exc:
        parser.error(str(exc))
    if command_endpoint == camstate_endpoint:
        parser.error("simulator command and CamState endpoints must be distinct")

    paths = resolve_config_paths(args.config, args.config_extra)
    try:
        bundle = load_config_bundle(paths, required_sections=("net", "video"))
        config = bundle.mutable_copy()
        video, _profile = resolve_active_video_profile(config)
        frame_size = (int(video["width"]), int(video["height"]))
        camera_model = apply_sim_camera_intrinsics(config, frame_size)
        control_config = ControlConfig.from_raw_config(config, frame_size)
        laser_mount = LaserMountConfig.from_raw_config(config)
        baseline_policy_config, acceptance = load_sim_baseline_policy_config(config)
        evaluation_contract = load_sim_evaluation_contract(config)
    except (
        ConfigError,
        ControlConfigError,
        LaserConfigError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise SystemExit(f"invalid simulator controller configuration: {exc}") from exc
    net = config.get("net")
    if not isinstance(net, Mapping):
        raise SystemExit("configuration has no net mapping")
    snapshot_endpoint = str(args.snapshot_sub or net.get("zmq_perception_v2", ""))
    production_control = str(net.get("zmq_control", ""))
    if command_endpoint == production_control:
        raise SystemExit("simulator controller refuses production net.zmq_control")
    sim = config.get("sim")
    if not isinstance(sim, Mapping):
        raise SystemExit("configuration has no sim mapping")
    try:
        motion_mode = resolve_simulation_motion_mode(sim)
    except ValueError as exc:
        raise SystemExit(f"invalid simulator motion mode: {exc}") from exc
    if motion_mode.moves_physical_mount:
        raise SystemExit(
            "sim baseline controller refuses sim.use_jetson_cam_state=true; "
            "hardware-in-loop requires the separately authorized tuned live controller"
        )
    policy_config = baseline_policy_config
    loop_hz = 1.0 / policy_config.nominal_period_s
    startup = {
        "mode": "sim_baseline_tracking_controller",
        "hardware_control_disabled": True,
        "hardware_controller_tuning_loaded": False,
        "hardware_tuning_from_sim_allowed": False,
        "controller_profile": "sim.baseline_controller",
        "controller_artifact": None,
        "evaluation_scope": "system_operation_and_video_pipeline",
        "sim_motion_mode": motion_mode.name,
        **evaluation_contract,
        "camera_model": camera_model,
        "parallax_indicator": {
            # The projection is on by default and is the controller's image
            # reference when aim_mode=laser_point.
            "enabled": True,
            "controller_aim_mode": control_config.aim_mode,
            "mount_offset_m_cv": laser_mount.offset_m.as_tuple(),
            "mount_direction_cv": laser_mount.dir_cam.as_tuple(),
            "range_policy": "selected_known_size_then_config_default",
            "fallback_range_m": control_config.laser.default_distance_m,
            "tolerance_px": control_config.laser.tolerance_px,
        },
        "acceptance": acceptance,
        "snapshot_sub": snapshot_endpoint,
        "sim_camstate_sub": camstate_endpoint,
        "sim_control_bind": command_endpoint,
        "loop_hz": loop_hz,
        **bundle.provenance(),
    }
    print(json.dumps(startup, sort_keys=True))
    if args.check:
        return 0
    if not args.enable_sim_control:
        raise SystemExit("--enable-sim-control is required to publish simulator commands")

    assembler = ControlObservationAssembler(
        control_config, laser_mount=laser_mount
    )
    policy = ShadowRatePolicy(policy_config)
    context = zmq.Context()
    snapshot_sub = _sub(context, snapshot_endpoint)
    camstate_sub = _sub(context, camstate_endpoint)
    command_pub = context.socket(zmq.PUB)
    command_pub.setsockopt(zmq.SNDHWM, 1)
    command_pub.setsockopt(zmq.LINGER, 100)
    command_pub.bind(command_endpoint)
    stop = install_signal_handlers()
    trace = None
    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        trace = args.trace.open("w", encoding="utf-8", buffering=1)
        trace.write(json.dumps({"type": "meta", **startup}, sort_keys=True) + "\n")

    latest_snapshot: Optional[PerceptionSnapshotV2] = None
    latest_camstate: Optional[CamState] = None
    start = time.monotonic()
    deadline = None if args.duration_s is None else start + args.duration_s
    period = policy_config.nominal_period_s
    next_tick = start
    next_status = start + args.status_interval_s
    perception_updates = camstate_updates = commands = tracking = missed_periods = 0
    command_drops = 0
    reasons: Counter[str] = Counter()
    yaw_positions: list[float] = []
    pitch_positions: list[float] = []
    error_norms: list[float] = []
    pixel_error_norms: list[float] = []
    pixel_error_elapsed_s: list[float] = []
    command_norms: list[float] = []
    rate_limited_commands = acceleration_limited_commands = 0
    last_observation: Optional[ControlObservation] = None
    last_snapshot: Optional[PerceptionSnapshotV2] = None
    try:
        while not stop.is_set() and (deadline is None or time.monotonic() < deadline):
            now = time.monotonic()
            payload = _latest(snapshot_sub)
            if payload is not None:
                try:
                    latest_snapshot = perception_snapshot_from_json(payload)
                except (TypeError, ValueError):
                    latest_snapshot = None
                else:
                    assembler.update_perception_snapshot(latest_snapshot, received_at=now)
                    perception_updates += 1
            payload = _latest(camstate_sub)
            if payload is not None:
                try:
                    latest_camstate = CamState(**json.loads(payload))
                except (TypeError, ValueError, json.JSONDecodeError):
                    latest_camstate = None
                else:
                    assembler.update_cam_state(latest_camstate, received_at=now)
                    camstate_updates += 1
            if now >= next_tick:
                lateness = max(0.0, now - next_tick)
                missed_periods += int(lateness / period)
                assembler.update_manual_state(
                    _safe_sim_manual_state(now), received_at=now
                )
                observation = assembler.build(now=now)
                intent = policy.decide(observation)
                command = control_cmd_from_intent(
                    observation,
                    intent,
                    snapshot=latest_snapshot,
                    control_config=control_config,
                    laser_mount=laser_mount,
                    now_s=now,
                )
                try:
                    command_pub.send_json(
                        command.model_dump(mode="json", exclude_none=True),
                        flags=zmq.NOBLOCK,
                    )
                except zmq.Again:
                    command_drops += 1
                else:
                    commands += 1
                    tracking += int(command.target_ok)
                reasons[intent.reason] += 1
                command_norms.append(math.hypot(command.pan_rate_cmd, command.tilt_rate_cmd))
                if command.target_ok and observation.target.bearing_error_rad is not None:
                    error_norms.append(math.hypot(*observation.target.bearing_error_rad))
                    pixel_error_norms.append(math.hypot(*command.err_uv))
                    pixel_error_elapsed_s.append(now - start)
                if intent.limits.yaw_rate_limited or intent.limits.pitch_rate_limited:
                    rate_limited_commands += 1
                if intent.limits.acceleration_limited:
                    acceleration_limited_commands += 1
                if latest_camstate is not None:
                    yaw_positions.append(float(latest_camstate.pan))
                    pitch_positions.append(float(latest_camstate.tilt))
                if trace is not None:
                    trace.write(
                        json.dumps(
                            {
                                "type": "tick",
                                "observation": observation.model_dump(mode="json"),
                                "intent": intent.model_dump(mode="json"),
                                "command": command.model_dump(mode="json", exclude_none=True),
                                "camstate": (
                                    None
                                    if latest_camstate is None
                                    else latest_camstate.model_dump(mode="json", exclude_none=True)
                                ),
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                        + "\n"
                    )
                last_observation = observation
                last_snapshot = latest_snapshot
                next_tick = now + period
            if now >= next_status:
                pose = (
                    None
                    if latest_camstate is None
                    else [float(latest_camstate.pan), float(latest_camstate.tilt)]
                )
                print(
                    json.dumps(
                        {
                            "type": "status",
                            "elapsed_s": now - start,
                            "commands": commands,
                            "command_drops": command_drops,
                            "tracking_commands": tracking,
                            "latest_pose_rad": pose,
                            "latest_reason": (
                                None if not reasons else reasons.most_common(1)[0][0]
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                next_status = now + args.status_interval_s
            time.sleep(min(0.002, period / 4.0))
    finally:
        if last_observation is not None:
            zero = ControlCmd(
                frame_id=last_observation.source_frame_id or 0,
                src_ts_ms=(last_observation.source_time_ns or 0) // 1_000_000,
                cmd_ts_ms=int(time.monotonic() * 1000.0),
                target_ok=False,
                target_uv=(control_config.cx_px, control_config.cy_px),
                err_uv=(0.0, 0.0),
                err_rad=(0.0, 0.0),
                pan_rate_cmd=0.0,
                tilt_rate_cmd=0.0,
                controller_mode="pid",
            )
            for _ in range(3):
                try:
                    command_pub.send_json(
                        zero.model_dump(mode="json", exclude_none=True),
                        flags=zmq.NOBLOCK,
                    )
                except zmq.Again:
                    pass
                time.sleep(0.01)
        if trace is not None:
            trace.close()
        snapshot_sub.close(0)
        camstate_sub.close(0)
        command_pub.close(100)
        context.term()

    elapsed = time.monotonic() - start
    tracking_fraction = tracking / commands if commands else 0.0
    rate_limited_fraction = rate_limited_commands / commands if commands else 0.0
    rms_error_px = _rms(pixel_error_norms)
    p95_error_px = _percentile(pixel_error_norms, 0.95)
    steady_errors_px = [
        error
        for sample_s, error in zip(pixel_error_elapsed_s, pixel_error_norms)
        if sample_s >= acceptance["warmup_s"]
    ]
    steady_rms_error_px = _rms(steady_errors_px)
    steady_p95_error_px = _percentile(steady_errors_px, 0.95)
    acquisition_time_s = _acquisition_time_s(
        pixel_error_elapsed_s,
        pixel_error_norms,
        threshold_px=acceptance["acquisition_error_px"],
        hold_s=acceptance["acquisition_hold_s"],
    )
    yaw_pose_span = max(yaw_positions) - min(yaw_positions) if yaw_positions else 0.0
    acceptance_failures: list[str] = []
    if elapsed < acceptance["min_duration_s"]:
        acceptance_failures.append("duration_below_minimum")
    if tracking_fraction < acceptance["min_tracking_fraction"]:
        acceptance_failures.append("tracking_fraction_below_minimum")
    if acquisition_time_s is None or acquisition_time_s > acceptance["max_acquisition_time_s"]:
        acceptance_failures.append("acquisition_time_above_maximum")
    if steady_rms_error_px is None or steady_rms_error_px > acceptance["max_rms_error_px"]:
        acceptance_failures.append("steady_rms_error_px_above_maximum")
    if steady_p95_error_px is None or steady_p95_error_px > acceptance["max_p95_error_px"]:
        acceptance_failures.append("steady_p95_error_px_above_maximum")
    if rate_limited_fraction > acceptance["max_rate_limited_fraction"]:
        acceptance_failures.append("rate_limited_fraction_above_maximum")
    if command_drops > acceptance["max_command_drops"]:
        acceptance_failures.append("command_drops_above_maximum")
    if yaw_pose_span < acceptance["min_yaw_pose_span_rad"]:
        acceptance_failures.append("yaw_pose_span_below_minimum")
    report = {
        **startup,
        "duration_s": elapsed,
        "commands": commands,
        "command_drops": command_drops,
        "tracking_commands": tracking,
        "hold_commands": commands - tracking,
        "tracking_fraction": tracking_fraction,
        "perception_updates": perception_updates,
        "camstate_updates": camstate_updates,
        "missed_periods": missed_periods,
        "reasons": dict(sorted(reasons.items())),
        "max_command_norm_rad_s": max(command_norms, default=0.0),
        "rate_limited_commands": rate_limited_commands,
        "rate_limited_fraction": rate_limited_fraction,
        "acceleration_limited_commands": acceleration_limited_commands,
        "rms_error_px": rms_error_px,
        "p95_error_px": p95_error_px,
        "steady_rms_error_px": steady_rms_error_px,
        "steady_p95_error_px": steady_p95_error_px,
        "acquisition_time_s": acquisition_time_s,
        "rms_error_rad": _rms(error_norms),
        "p95_error_rad": _percentile(error_norms, 0.95),
        "yaw_pose_span_rad": yaw_pose_span,
        "pitch_pose_span_rad": (
            max(pitch_positions) - min(pitch_positions) if pitch_positions else 0.0
        ),
        "mean_error_first_50_rad": _mean_edge(error_norms, first=True),
        "mean_error_last_50_rad": _mean_edge(error_norms, first=False),
        "last_source_frame_id": (
            None if last_snapshot is None else last_snapshot.frame.frame_id
        ),
        "simulation_acceptance": {
            "passed": not acceptance_failures,
            "failures": acceptance_failures,
        },
        "qualification": {
            "scope": "simulation_integration",
            "qualified": not acceptance_failures,
            "failures": acceptance_failures,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
