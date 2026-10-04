"""Fixed-rate video controller service.

Policy, gains, endpoints and the clock basis come only from the validated
``controller`` config section (``runtime_config``). ``mode: shadow`` traces
decisions without publishing; ``mode: live`` publishes short-lease intents to
the gimbal bridge, which alone owns serial. Stopping (SIGTERM/SIGINT or
``--duration-s``) publishes explicit zero-rate intents.
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
from jetson.control.clock_poller import ClockPoller
from jetson.control.clock_watchdog import ClockWatchdogConfig
from jetson.control.pid import AxisPIDConfig, BasicPID
from jetson.control.diagnostics import build_diagnostics
from jetson.control.engagement import EngagementMonitor
from jetson.control.runtime_config import ControlRuntimeConfig
from jetson.control.timing import ClockBounds
from jetson.control.video_controller import VideoControllerCore, VideoControllerPolicy
from jetson.control.video_input import stamp_verified_snapshot


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


def _publish_record(pub: "zmq.Socket | None", record: dict) -> None:
    """Send one flight-recorder record; drop it rather than block control."""
    if pub is None:
        return
    try:
        pub.send_string(json.dumps(record, separators=(",", ":"), sort_keys=True), flags=zmq.NOBLOCK)
    except zmq.Again:
        pass


def _write_json(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base")
    parser.add_argument("--config-extra", required=True,
                        help="comma-separated config files; must include a controller section")
    parser.add_argument("--duration-s", type=float,
                        help="stop after this long; default runs until SIGTERM/SIGINT")
    parser.add_argument("--trace", type=Path, help="per-tick JSONL trace (optional)")
    parser.add_argument("--report", type=Path, help="final report JSON (optional)")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--health-file", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.duration_s is not None and not (math.isfinite(args.duration_s) and args.duration_s > 0):
        parser.error("--duration-s must be positive")
    paths = resolve_config_paths(args.config, args.config_extra)
    bundle = load_config_bundle(paths, required_sections=("net", "video", "control"))
    config = bundle.mutable_copy()
    try:
        cfg = ControlRuntimeConfig.from_config(config)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(f"invalid controller configuration: {exc}")
    endpoints = (cfg.snapshot_endpoint, cfg.gimbal_endpoint, cfg.manual_bind,
                 cfg.clock_endpoint, cfg.intent_bind, cfg.diagnostics_bind)
    if cfg.record_bind is not None:
        endpoints = (*endpoints, cfg.record_bind)
    if len({_port(endpoint) for endpoint in endpoints}) != len(endpoints):
        parser.error("controller endpoint ports must be distinct")
    live = cfg.mode == "live"
    video, _ = resolve_active_video_profile(config)
    camera_fov_x_deg = None
    if cfg.camera_fov_y_deg is not None:
        # Intrinsics of the camera actually producing frames (e.g. a simulated
        # camera): horizontal FOV follows from vertical FOV and aspect ratio.
        # In-memory only; persisted calibration is not rewritten.
        camera_fov_x_deg = math.degrees(2 * math.atan(
            int(video["width"]) / int(video["height"])
            * math.tan(math.radians(cfg.camera_fov_y_deg) / 2)
        ))
        config["control"]["fx_fy_from_fov"] = True
        config["control"]["fov_deg"] = {"h": camera_fov_x_deg, "v": cfg.camera_fov_y_deg}
    control_config = ControlConfig.from_raw_config(config, (int(video["width"]), int(video["height"])))
    laser_mount = LaserMountConfig.from_raw_config(config)
    clock_policy = ClockWatchdogConfig(
        max_exchange_age_ns=150_000_000,
        configured_max_drift_ppm=cfg.clock_drift_ppm,
        max_interval_width_ns=15_000_000,
        max_capture_age_ns=cfg.max_capture_age_ms * 1_000_000,
        max_mapping_uncertainty_ns=20_000_000,
        required_samples=2,
    )
    startup = {
        "mode": "video_check" if args.check else f"video_{cfg.mode}",
        "motor_authority": live and not args.check,
        "check_only": bool(args.check),
        "controller": cfg.describe(),
        "camera_fov_x_deg": camera_fov_x_deg,
        "aim_fx_px": control_config.fx_px,
        "aim_fy_px": control_config.fy_px,
        # Field of view implied by the aim intrinsics; a simulated camera must match it.
        "aim_fov_deg": [
            math.degrees(2 * math.atan(int(video["width"]) / (2 * control_config.fx_px))),
            math.degrees(2 * math.atan(int(video["height"]) / (2 * control_config.fy_px))),
        ],
        "duration_s": args.duration_s,
        **bundle.provenance(),
    }
    if args.check:
        print(json.dumps(startup, sort_keys=True))
        return 0

    context = zmq.Context()
    snapshot_sub = _sub(context, cfg.snapshot_endpoint)
    gimbal_sub = _sub(context, cfg.gimbal_endpoint)
    manual_pull = context.socket(zmq.PULL)
    manual_pull.setsockopt(zmq.LINGER, 0)
    manual_pull.bind(_bind(cfg.manual_bind))
    intent_pub = None
    if live:
        intent_pub = context.socket(zmq.PUB)
        intent_pub.setsockopt(zmq.LINGER, 100)
        intent_pub.bind(_bind(cfg.intent_bind))
    # Read-only display diagnostics (HUD); published in every mode.
    diagnostics_pub = context.socket(zmq.PUB)
    diagnostics_pub.setsockopt(zmq.LINGER, 0)
    diagnostics_pub.setsockopt(zmq.SNDHWM, 2)
    diagnostics_pub.bind(_bind(cfg.diagnostics_bind))
    # Flight-recorder stream: every tick and panel state; never blocks control.
    record_pub = None
    if cfg.record_bind is not None:
        record_pub = context.socket(zmq.PUB)
        record_pub.setsockopt(zmq.LINGER, 0)
        record_pub.setsockopt(zmq.SNDHWM, 1000)
        record_pub.bind(_bind(cfg.record_bind))
    clock = ClockPoller(cfg.clock_endpoint, clock_policy, interval_s=0.05)
    # A camera on this host timestamps frames on this host's clock: no exchange.
    same_host_source = cfg.source_clock == "jetson_monotonic"
    # Intent sequence numbers must keep rising across controller restarts: the
    # bridge drops any intent not newer than the last it accepted. The base is
    # monotonic milliseconds, which advances faster than the 50 Hz sequence.
    assembler = ControlObservationAssembler(
        control_config, laser_mount=laser_mount, sequence_base=time.monotonic_ns() // 1_000_000,
    )
    core = VideoControllerCore(
        BasicPID(
            AxisPIDConfig(cfg.yaw_kp, 0.0, 0.0, 0.0, cfg.rate_limit_rad_s, cfg.accel_limit_rad_s2),
            AxisPIDConfig(cfg.pitch_kp, 0.0, 0.0, 0.0, cfg.rate_limit_rad_s, cfg.accel_limit_rad_s2),
        ),
        VideoControllerPolicy(
            feedforward_scale=cfg.feedforward_scale,
            predict=cfg.predict,
            feedforward_accel_sigma_rad_s2=cfg.feedforward_accel_sigma_rad_s2,
            live_authorized=live,
            max_capture_age_ns=cfg.max_capture_age_ms * 1_000_000,
            max_travel_rad=cfg.max_travel_rad,
            source_clock_domain=cfg.source_clock,
            idle_return_after_ns=(None if cfg.idle_return_s is None
                                  else int(cfg.idle_return_s * 1e9)),
            idle_return_rate_rad_s=cfg.idle_return_rate_rad_s,
            coast_ns=None if cfg.coast_s is None else int(cfg.coast_s * 1e9),
            manual_rate_limit_rad_s=cfg.manual_rate_limit_rad_s,
            manual_accel_limit_rad_s2=cfg.manual_accel_limit_rad_s2,
        ),
    )
    engagement = EngagementMonitor()
    stop = install_signal_handlers()
    reasons: Counter[str] = Counter()
    ff_reasons: Counter[str] = Counter()
    snapshots = gimbal_states = manual_states = invalid = ticks = missed = 0
    last_observation_sequence = 0
    frame_size_px: tuple[int, int] | None = None
    started = time.monotonic()
    next_tick_ns = time.monotonic_ns()
    last_health_s = 0.0
    for path in (args.trace, args.report, args.ready_file, args.health_file):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
    if not same_host_source:
        clock.start()
    trace = args.trace.open("w", encoding="utf-8", buffering=1) if args.trace else None
    if trace is not None:
        trace.write(json.dumps({"type": "meta", **startup}, sort_keys=True) + "\n")
    _write_json(args.ready_file, {"ready": True, "started_monotonic_s": started, **startup})
    try:
        while not stop.is_set() and (args.duration_s is None or time.monotonic() - started < args.duration_s):
            payload = _latest(snapshot_sub)
            if payload is not None:
                received_ns = time.monotonic_ns()
                try:
                    snapshot = stamp_verified_snapshot(
                        perception_snapshot_from_json(payload),
                        received_ns=received_ns, observed_ns=time.monotonic_ns(),
                        keep_upstream_receipt=cfg.local_clock == "jetson",
                        source_clock_domain=cfg.source_clock,
                    )
                except (ValueError, TypeError, json.JSONDecodeError):
                    invalid += 1
                else:
                    assembler.update_perception_snapshot(snapshot, received_at=received_ns / 1e9)
                    frame_size_px = (snapshot.frame.width, snapshot.frame.height)
                    snapshots += 1
            payload = _latest(gimbal_sub)
            if payload is not None:
                received_ns = time.monotonic_ns()
                try:
                    state = CamState.model_validate_json(payload)
                except ValueError:
                    invalid += 1
                else:
                    published_ns = state.state_monotonic_ns
                    measured = [value for value in (state.pan_sample_monotonic_ns,
                                                    state.tilt_sample_monotonic_ns) if value]
                    if (published_ns is None or len(measured) != 2
                            or not 0 <= received_ns - published_ns <= 100_000_000):
                        invalid += 1
                    elif core.observe_cam_state(state):
                        # Gimbal freshness is the age of the older axis measurement.
                        assembler.update_cam_state(state, received_at=min(measured) / 1e9)
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
                    _publish_record(record_pub, {"type": "manual", "monotonic_ns": received_ns,
                                                 "state": manual.model_dump(mode="json")})
                    manual_states += 1
            now_ns = time.monotonic_ns()
            if now_ns < next_tick_ns:
                time.sleep(min((next_tick_ns - now_ns) / 1e9, 0.002))
                continue
            missed += max(0, (now_ns - next_tick_ns) // 20_000_000)
            observation = assembler.build(now=now_ns / 1e9)
            last_observation_sequence = observation.sequence
            if same_host_source:
                bounds, clock_reason = ClockBounds.identity(observation.created_monotonic_ns), "same_host"
            else:
                bounds, clock_reason = clock.bounds(now_ns=observation.created_monotonic_ns)
            decision = core.decide(observation, bounds)
            engage_record = engagement.update(observation, decision.intent)
            if engage_record is not None:
                log_event = {k: v for k, v in engage_record.items() if k != "type"}
                print(json.dumps({"event": engage_record["type"], **log_event}, sort_keys=True), flush=True)
                _publish_record(record_pub, engage_record)
                if trace is not None:
                    trace.write(json.dumps(engage_record, separators=(",", ":"), sort_keys=True) + "\n")
            if intent_pub is not None:
                intent_pub.send_string(decision.intent.model_dump_json(exclude_none=True))
            try:
                diagnostics_pub.send_string(build_diagnostics(
                    observation, decision, feedforward_scale=cfg.feedforward_scale,
                    created_monotonic_ns=time.monotonic_ns(), frame_size_px=frame_size_px,
                    engagement=engagement.last,
                ).model_dump_json(exclude_none=True), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            if trace is not None or record_pub is not None:
                tick_record = {
                    "type": "tick", "sequence": observation.sequence,
                    "source_frame_id": observation.source_frame_id,
                    "clock_reason": clock_reason,
                    "pid_reason": decision.intent.reason,
                    "ff_reason": decision.feedforward.reason,
                    "feedforward_scale": cfg.feedforward_scale,
                    "raw_bearing_error_rad": observation.target.bearing_error_rad,
                    "pid_error_source": decision.pid_error_source,
                    "predicted_bearing_error_rad": decision.feedforward.predicted_bearing_error_rad,
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
                    "travel_held": list(decision.travel_held),
                }
                # The complete controller input: target, gimbal, safety/panel state.
                tick_record["observation"] = observation.model_dump(mode="json")
                tick_record["monotonic_ns"] = observation.created_monotonic_ns
                if trace is not None:
                    trace.write(json.dumps(tick_record, separators=(",", ":"), sort_keys=True) + "\n")
                _publish_record(record_pub, tick_record)
            reasons[decision.intent.reason] += 1
            ff_reasons[decision.feedforward.reason] += 1
            ticks += 1
            next_tick_ns += 20_000_000
            if next_tick_ns < now_ns:
                next_tick_ns = now_ns + 20_000_000
            if time.monotonic() - last_health_s >= 1.0:
                last_health_s = time.monotonic()
                _write_json(args.health_file, {
                    "monotonic_s": last_health_s, "ticks": ticks, "missed_periods": missed,
                    "snapshots": snapshots, "gimbal_states": gimbal_states,
                    "manual_states": manual_states, "invalid_messages": invalid,
                    "last_reason": decision.intent.reason,
                    "pid_reasons": dict(reasons), "ff_reasons": dict(ff_reasons),
                })
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
        if record_pub is not None:
            record_pub.close(0)
        diagnostics_pub.close(0)
        if intent_pub is not None:
            intent_pub.close(0)
        context.destroy(linger=0)
        if trace is not None:
            trace.close()
        if args.ready_file is not None:
            args.ready_file.unlink(missing_ok=True)
    report = {
        **startup, "ticks": ticks, "missed_periods": missed,
        "snapshots": snapshots, "gimbal_states": gimbal_states,
        "manual_states": manual_states, "invalid_messages": invalid,
        "pid_reasons": dict(reasons), "ff_reasons": dict(ff_reasons),
        "clock_events": clock.stats(),
    }
    _write_json(args.report, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
