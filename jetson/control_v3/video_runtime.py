"""Fixed-rate V3 video controller; live publication requires test-only opt-ins.

This process never opens serial. The default is trace-only shadow. The live
path still requires an externally chosen clock policy, manual safety state,
and a separate explicitly enabled gimbal bridge.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import zmq

from common.config import load_config_bundle, resolve_active_video_profile, resolve_config_paths
from common.control import ControlConfig, LaserMountConfig
from common.perception import perception_snapshot_from_json
from common.schemas import CamState, ControlIntent, manual_control_state_from_json
from common.shutdown import install_signal_handlers
from jetson.control_observation import ControlObservationAssembler
from jetson.control_v3.clock_poller import ClockPoller
from jetson.control_v3.clock_watchdog import ClockWatchdogConfig
from jetson.control_v3.pid import AxisPIDConfig, BasicPID
from jetson.control_v3.video_controller import VideoControllerCore, VideoControllerPolicy
from jetson.control_v3.video_input import stamp_verified_snapshot


def _port(endpoint: str) -> int:
    parsed = urlsplit(endpoint)
    if parsed.scheme != "tcp" or not parsed.hostname or parsed.port is None:
        raise ValueError(f"invalid TCP endpoint: {endpoint}")
    return parsed.port


def _bind(endpoint: str) -> str:
    return f"tcp://0.0.0.0:{_port(endpoint)}"


def _sub(context: zmq.Context, endpoint: str) -> zmq.Socket:
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.connect(endpoint)
    return socket


def _latest(socket: zmq.Socket) -> bytes | None:
    payload = None
    while True:
        try:
            payload = socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return payload


def _decode_sim_frame_pose(
    payload: bytes,
) -> tuple[dict, tuple[float, float] | None, int | None]:
    """Peel HIL-only frame pose before validating against the deployed V2 schema.

    The Jetson candidate's dirty common/perception.py remains untouched; V3
    owns this additive simulator transport field in its isolated runtime.
    """

    raw = json.loads(payload)
    if not isinstance(raw, dict) or not isinstance(raw.get("frame"), dict):
        raise ValueError("invalid simulator snapshot envelope")
    frame = raw["frame"]
    pose = frame.pop("sim_capture_pose_rad", None)
    applied_ns = frame.pop("sim_applied_camstate_ns", None)
    if pose is None and applied_ns is None:
        return raw, None, None
    if (not isinstance(pose, (list, tuple)) or len(pose) != 2
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in pose)
            or not isinstance(applied_ns, int) or applied_ns <= 0):
        raise ValueError("invalid simulator capture-pose metadata")
    return raw, (float(pose[0]), float(pose[1])), applied_ns


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/network.yaml")
    parser.add_argument("--config-extra", required=True)
    parser.add_argument("--snapshot-sub", required=True)
    parser.add_argument("--gimbal-sub", required=True)
    parser.add_argument("--manual-bind", required=True)
    parser.add_argument("--clock-endpoint", required=True)
    parser.add_argument("--intent-bind", required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--feedforward-scale", type=float, choices=(0.0, 0.5))
    parser.add_argument("--feedforward-schedule", choices=("off-on-off", "on-off-on"))
    parser.add_argument("--pitch-kp", type=float, choices=(4.0, 8.0), default=4.0)
    parser.add_argument("--pose-source", choices=("encoder", "render", "frame"), default="encoder")
    parser.add_argument("--sim-camera-fov-y-deg", type=float)
    parser.add_argument("--max-capture-age-ms", type=int, choices=(150, 250), default=150)
    parser.add_argument("--clock-drift-ppm", type=float)
    parser.add_argument("--ack-shadow-only", action="store_true")
    parser.add_argument("--ack-empirical-test-clock", action="store_true")
    parser.add_argument("--acknowledge-unloaded-hardware", action="store_true")
    parser.add_argument("--enable-live-intent-publish", action="store_true")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not 0 < args.duration_s <= 30:
        parser.error("duration must be in (0, 30] seconds")
    if (args.feedforward_scale is None) == (args.feedforward_schedule is None):
        parser.error("select exactly one explicit feedforward scale or crossover schedule")
    if args.feedforward_schedule is not None and args.duration_s != 30:
        parser.error("crossover schedule requires exactly 30 seconds")
    if args.clock_drift_ppm is None or not math.isfinite(args.clock_drift_ppm) or not 0 <= args.clock_drift_ppm <= 1000:
        parser.error("supply a finite test clock drift policy in [0, 1000] ppm")
    if args.enable_live_intent_publish:
        if not args.ack_empirical_test_clock or not args.acknowledge_unloaded_hardware:
            parser.error("live V3 video requires test-clock and unloaded-hardware acknowledgements")
        if args.pose_source != "frame":
            parser.error("simulator HIL live mode requires exact source-frame camera pose")
        if args.ack_shadow_only:
            parser.error("shadow-only acknowledgement conflicts with live publication")
    elif not args.ack_shadow_only:
        parser.error("shadow V3 video requires --ack-shadow-only")
    if args.pose_source in {"render", "frame"}:
        if (args.sim_camera_fov_y_deg is None or not math.isfinite(args.sim_camera_fov_y_deg)
                or not 1 < args.sim_camera_fov_y_deg < 179):
            parser.error("simulator HIL requires explicit --sim-camera-fov-y-deg")
    elif args.sim_camera_fov_y_deg is not None:
        parser.error("simulator FOV override is only valid with simulator HIL")
    endpoints = (
        args.snapshot_sub, args.gimbal_sub, args.manual_bind,
        args.clock_endpoint, args.intent_bind,
    )
    if len({_port(endpoint) for endpoint in endpoints}) != len(endpoints):
        parser.error("V3 endpoint ports must be distinct")
    paths = resolve_config_paths(args.config, args.config_extra)
    bundle = load_config_bundle(paths, required_sections=("net", "video", "control"))
    config = bundle.mutable_copy()
    video, _ = resolve_active_video_profile(config)
    sim_fov_x_deg = None
    if args.sim_camera_fov_y_deg is not None:
        # The simulator derives horizontal FOV from vertical FOV and frame
        # aspect ratio. Keep this scoped to explicit HIL; never alter the
        # real-camera calibration or persisted configuration.
        sim_fov_x_deg = math.degrees(2 * math.atan(
            int(video["width"]) / int(video["height"])
            * math.tan(math.radians(args.sim_camera_fov_y_deg) / 2)
        ))
        config["control"]["fx_fy_from_fov"] = True
        config["control"]["fov_deg"] = {
            "h": sim_fov_x_deg, "v": args.sim_camera_fov_y_deg,
        }
    control_config = ControlConfig.from_raw_config(config, (int(video["width"]), int(video["height"])))
    laser_mount = LaserMountConfig.from_raw_config(config)
    clock_policy = ClockWatchdogConfig(
        max_exchange_age_ns=150_000_000,
        configured_max_drift_ppm=args.clock_drift_ppm,
        max_interval_width_ns=15_000_000,
        max_capture_age_ns=args.max_capture_age_ms * 1_000_000,
        max_mapping_uncertainty_ns=20_000_000,
        required_samples=2,
    )
    startup = {
        "mode": (
            "v3_video_test_live" if args.enable_live_intent_publish and not args.check
            else "v3_video_check" if args.check else "v3_video_shadow"
        ),
        "motor_authority": bool(args.enable_live_intent_publish and not args.check),
        "requested_live_publication": bool(args.enable_live_intent_publish),
        "check_only": bool(args.check),
        "clock_policy_basis": "empirical_test_only",
        "clock_drift_ppm": args.clock_drift_ppm,
        "feedforward_scale": args.feedforward_scale,
        "feedforward_schedule": args.feedforward_schedule,
        "yaw_kp": 8.0,
        "pitch_kp": args.pitch_kp,
        "max_capture_age_ms": args.max_capture_age_ms,
        "max_travel_rad": 0.15,
        "pose_source": args.pose_source,
        "sim_camera_fov_y_deg": args.sim_camera_fov_y_deg,
        "sim_camera_fov_x_deg": sim_fov_x_deg,
        "aim_fx_px": control_config.fx_px,
        "aim_fy_px": control_config.fy_px,
        "duration_s": args.duration_s,
        "snapshot_sub": args.snapshot_sub,
        "gimbal_sub": args.gimbal_sub,
        "manual_bind": _bind(args.manual_bind),
        "clock_endpoint": args.clock_endpoint,
        "intent_bind": _bind(args.intent_bind),
        **bundle.provenance(),
    }
    if args.check:
        print(json.dumps(startup, sort_keys=True))
        return 0

    context = zmq.Context()
    snapshot_sub = _sub(context, args.snapshot_sub)
    gimbal_sub = _sub(context, args.gimbal_sub)
    manual_pull = context.socket(zmq.PULL)
    manual_pull.setsockopt(zmq.LINGER, 0)
    manual_pull.bind(_bind(args.manual_bind))
    intent_pub = None
    if args.enable_live_intent_publish:
        intent_pub = context.socket(zmq.PUB)
        intent_pub.setsockopt(zmq.LINGER, 100)
        intent_pub.bind(_bind(args.intent_bind))
    clock = ClockPoller(args.clock_endpoint, clock_policy, interval_s=0.05)
    assembler = ControlObservationAssembler(control_config, laser_mount=laser_mount)
    core = VideoControllerCore(
        BasicPID(
            AxisPIDConfig(8.0, 0.0, 0.0, 0.0, 0.2, 3.5),
            AxisPIDConfig(args.pitch_kp, 0.0, 0.0, 0.0, 0.2, 3.5),
        ),
        VideoControllerPolicy(
            feedforward_scale=args.feedforward_scale or 0.0,
            live_authorized=args.enable_live_intent_publish,
            max_capture_age_ns=args.max_capture_age_ms * 1_000_000,
            pose_source=args.pose_source,
        ),
    )
    stop = install_signal_handlers()
    reasons: Counter[str] = Counter()
    ff_reasons: Counter[str] = Counter()
    snapshots = gimbal_states = manual_states = invalid = ticks = missed = 0
    latest_frame_pose: tuple[float, float] | None = None
    latest_frame_camstate_ns: int | None = None
    latest_pose_frame_id: int | None = None
    last_observation_sequence = 0
    started = time.monotonic()
    next_tick_ns = time.monotonic_ns()
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    clock.start()
    with args.trace.open("w", encoding="utf-8", buffering=1) as trace:
        trace.write(json.dumps({"type": "meta", **startup}, sort_keys=True) + "\n")
        try:
            while not stop.is_set() and time.monotonic() - started < args.duration_s:
                payload = _latest(snapshot_sub)
                if payload is not None:
                    received_ns = time.monotonic_ns()
                    try:
                        raw_snapshot, sim_pose, sim_applied_ns = _decode_sim_frame_pose(payload)
                        snapshot = stamp_verified_snapshot(
                            perception_snapshot_from_json(raw_snapshot),
                            received_ns=received_ns, observed_ns=time.monotonic_ns(),
                        )
                    except (ValueError, TypeError, json.JSONDecodeError):
                        invalid += 1
                    else:
                        assembler.update_perception_snapshot(snapshot, received_at=received_ns / 1e9)
                        latest_pose_frame_id = snapshot.frame.frame_id
                        latest_frame_pose = sim_pose
                        latest_frame_camstate_ns = sim_applied_ns
                        snapshots += 1
                payload = _latest(gimbal_sub)
                if payload is not None:
                    received_ns = time.monotonic_ns()
                    try:
                        state = CamState.model_validate_json(payload)
                    except ValueError:
                        invalid += 1
                    else:
                        sample_ns = state.state_monotonic_ns
                        if sample_ns is None or not 0 <= received_ns - sample_ns <= 100_000_000:
                            invalid += 1
                        elif core.observe_cam_state(state):
                            assembler.update_cam_state(state, received_at=sample_ns / 1e9)
                            gimbal_states += 1
                payload = _latest(manual_pull)
                if payload is not None:
                    received_ns = time.monotonic_ns()
                    try:
                        manual = manual_control_state_from_json(payload)
                    except (ValueError, TypeError, json.JSONDecodeError):
                        invalid += 1
                    else:
                        assembler.update_manual_state(manual, received_at=received_ns / 1e9)
                        manual_states += 1
                now_ns = time.monotonic_ns()
                if now_ns < next_tick_ns:
                    time.sleep(min((next_tick_ns - now_ns) / 1e9, 0.002))
                    continue
                missed += max(0, (now_ns - next_tick_ns) // 20_000_000)
                observation = assembler.build(now=now_ns / 1e9)
                last_observation_sequence = observation.sequence
                bounds, clock_reason = clock.bounds(now_ns=observation.created_monotonic_ns)
                frame_pose = (
                    latest_frame_pose if observation.source_frame_id == latest_pose_frame_id
                    else None
                )
                frame_camstate_ns = (
                    latest_frame_camstate_ns if observation.source_frame_id == latest_pose_frame_id
                    else None
                )
                schedule_block = None
                active_ff_scale = args.feedforward_scale
                if args.feedforward_schedule is not None:
                    schedule_block = min(int((time.monotonic() - started) // 10), 2)
                    active_ff_scale = (
                        (0.0, 0.5, 0.0)[schedule_block]
                        if args.feedforward_schedule == "off-on-off"
                        else (0.5, 0.0, 0.5)[schedule_block]
                    )
                decision = core.decide(
                    observation, bounds, frame_pose_rad=frame_pose,
                    frame_camstate_ns=frame_camstate_ns,
                    feedforward_scale=active_ff_scale,
                )
                if intent_pub is not None:
                    intent_pub.send_string(decision.intent.model_dump_json(exclude_none=True))
                trace.write(json.dumps({
                    "type": "tick", "sequence": observation.sequence,
                    "source_frame_id": observation.source_frame_id,
                    "clock_reason": clock_reason,
                    "pid_reason": decision.intent.reason,
                    "ff_reason": decision.feedforward.reason,
                    "feedforward_scale": active_ff_scale,
                    "schedule_block": schedule_block,
                    "raw_bearing_error_rad": observation.target.bearing_error_rad,
                    "gimbal_pose_rad": [observation.gimbal.yaw_rad, observation.gimbal.pitch_rad],
                    "gimbal_age_ms": observation.gimbal.sample_age_ms,
                    "feedforward_rad_s": decision.applied_feedforward_rad_s,
                    "estimated_target_rate_rad_s": [
                        decision.feedforward.yaw_rate_rad_s,
                        decision.feedforward.pitch_rate_rad_s,
                    ],
                    "estimated_capture_midpoint_ns": decision.feedforward.capture_midpoint_ns,
                    "capture_camera_pose_rad": decision.feedforward.capture_camera_pose_rad,
                    "measured_target_world_rad": decision.feedforward.measured_target_world_rad,
                    "sim_applied_camstate_ns": frame_camstate_ns,
                    "pid_terms_yaw_rad_s": [
                        decision.pid.pid.yaw.proportional_rad_s,
                        decision.pid.pid.yaw.integral_rad_s,
                        decision.pid.pid.yaw.derivative_rad_s,
                    ],
                    "pid_terms_pitch_rad_s": [
                        decision.pid.pid.pitch.proportional_rad_s,
                        decision.pid.pid.pitch.integral_rad_s,
                        decision.pid.pid.pitch.derivative_rad_s,
                    ],
                    "capture_age_ns": (
                        None if decision.pid.timing.capture_age_ns is None
                        else [decision.pid.timing.capture_age_ns.earliest_ns,
                              decision.pid.timing.capture_age_ns.latest_ns]
                    ),
                    "intent": decision.intent.model_dump(mode="json"),
                }, separators=(",", ":"), sort_keys=True) + "\n")
                reasons[decision.intent.reason] += 1
                ff_reasons[decision.feedforward.reason] += 1
                ticks += 1
                next_tick_ns += 20_000_000
                if next_tick_ns < now_ns:
                    next_tick_ns = now_ns + 20_000_000
        finally:
            if intent_pub is not None:
                for offset in range(3):
                    now_ns = time.monotonic_ns()
                    stop_intent = ControlIntent(
                        sequence=last_observation_sequence + offset + 1,
                        observation_sequence=last_observation_sequence,
                        issued_monotonic_ns=now_ns,
                        valid_until_monotonic_ns=now_ns + 50_000_000,
                        mode="live", yaw_rate_rad_s=0.0, pitch_rate_rad_s=0.0,
                        reason="controller_shutdown",
                    )
                    intent_pub.send_string(stop_intent.model_dump_json(exclude_none=True))
                    time.sleep(0.01)
            clock.close()
            snapshot_sub.close(0)
            gimbal_sub.close(0)
            manual_pull.close(0)
            if intent_pub is not None:
                intent_pub.close(0)
            context.term()
    report = {
        **startup, "ticks": ticks, "missed_periods": missed,
        "snapshots": snapshots, "gimbal_states": gimbal_states,
        "manual_states": manual_states, "invalid_messages": invalid,
        "pid_reasons": dict(reasons), "ff_reasons": dict(ff_reasons),
        "clock_events": clock.stats(),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
