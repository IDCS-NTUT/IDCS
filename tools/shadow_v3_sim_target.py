"""Read-only V3 simulator-snapshot timing and target-rate feedforward study.

The camera pose and manual safety values are deliberately synthetic; this
script neither publishes ControlIntent nor opens serial. PC clock drift is
an explicit empirical *shadow-only* assumption, never motor authority.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import zmq

from common.config import load_config_bundle, resolve_active_video_profile, resolve_config_paths
from common.control import ControlConfig, LaserMountConfig
from common.perception import perception_snapshot_from_json
from common.schemas import CamState, ManualControlState
from jetson.control_observation import ControlObservationAssembler
from jetson.control_v3.clock_watchdog import ClockWatchdog, ClockWatchdogConfig
from jetson.control_v3.pid import AxisPIDConfig, BasicPID
from jetson.control_v3.shadow_pid import ShadowPIDController
from jetson.control_v3.video_feedforward import VideoTargetRateEstimator
from jetson.control_v3.video_input import stamp_verified_snapshot
from tools.shadow_v3_verified_video import _exchange_clock


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/network.yaml")
    parser.add_argument("--config-extra", required=True)
    parser.add_argument("--snapshot-endpoint", required=True)
    parser.add_argument("--clock-endpoint", required=True)
    parser.add_argument("--gimbal-sub", help="subscribe to measured encoder CamState instead of synthetic pose")
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--empirical-drift-ppm", type=float, default=1000.0)
    parser.add_argument("--ack-shadow-only", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not 0 < args.duration_s <= 30:
        parser.error("duration must be in (0, 30] seconds")
    if not args.ack_shadow_only:
        parser.error("empirical clock policy requires --ack-shadow-only")
    if not math.isfinite(args.empirical_drift_ppm) or not 0 <= args.empirical_drift_ppm <= 1000:
        parser.error("empirical drift must be in [0, 1000] ppm")
    paths = resolve_config_paths(args.config, args.config_extra)
    bundle = load_config_bundle(paths, required_sections=("net", "video", "control"))
    config = bundle.mutable_copy()
    video, _ = resolve_active_video_profile(config)
    control = ControlConfig.from_raw_config(config, (int(video["width"]), int(video["height"])))
    laser = LaserMountConfig.from_raw_config(config)
    policy = ClockWatchdogConfig(
        max_exchange_age_ns=150_000_000,
        configured_max_drift_ppm=args.empirical_drift_ppm,
        max_interval_width_ns=15_000_000,
        max_capture_age_ns=150_000_000,
        max_mapping_uncertainty_ns=20_000_000,
        required_samples=2,
    )
    startup = {
        "mode": "v3_sim_target_shadow_only", "motor_authority": False,
        "synthetic_pose": args.gimbal_sub is None,
        "synthetic_safety": True,
        "gimbal_sub": args.gimbal_sub,
        "empirical_drift_ppm": args.empirical_drift_ppm,
        "mapping_uncertainty_policy_ms": 20,
        "snapshot_endpoint": args.snapshot_endpoint,
        "clock_endpoint": args.clock_endpoint,
        **bundle.provenance(),
    }
    if args.check:
        print(json.dumps(startup, sort_keys=True))
        return 0
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(args.snapshot_endpoint)
    gimbal_sub = None
    if args.gimbal_sub:
        gimbal_sub = context.socket(zmq.SUB)
        gimbal_sub.setsockopt(zmq.LINGER, 0)
        gimbal_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        gimbal_sub.connect(args.gimbal_sub)
    assembler = ControlObservationAssembler(control, laser_mount=laser)
    watchdog = ClockWatchdog(policy)
    def new_shadow_pid() -> ShadowPIDController:
        return ShadowPIDController(BasicPID(
            AxisPIDConfig(8.0, 0.0, 0.0, 0.0, 0.2, 3.5),
            AxisPIDConfig(4.0, 0.0, 0.0, 0.0, 0.2, 3.5),
        ), max_clock_sample_age_ns=150_000_000,
            max_capture_age_ns=150_000_000,
            max_gimbal_age_ns=100_000_000,
            max_safety_age_ns=750_000_000)
    baseline = new_shadow_pid()
    combined = new_shadow_pid()
    feedforward = VideoTargetRateEstimator()
    reasons: Counter[str] = Counter()
    ff_reasons: Counter[str] = Counter()
    clock_reasons: Counter[str] = Counter()
    received = verified = ff_valid = cam_states = invalid_cam_states = 0
    started = time.monotonic()
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.trace.open("w", encoding="utf-8", buffering=1) as trace:
        trace.write(json.dumps({"type": "meta", **startup}, sort_keys=True) + "\n")
        try:
            while time.monotonic() - started < args.duration_s:
                if not sub.poll(100):
                    continue
                raw = sub.recv()
                received_ns = time.monotonic_ns()
                received += 1
                try:
                    snapshot = perception_snapshot_from_json(raw)
                    snapshot = stamp_verified_snapshot(
                        snapshot, received_ns=received_ns,
                        observed_ns=time.monotonic_ns(),
                    )
                except (TypeError, ValueError) as exc:
                    reasons[f"snapshot_invalid:{type(exc).__name__}:{str(exc)[:160]}"] += 1
                    continue
                verified += 1
                exchange = _exchange_clock(context, args.clock_endpoint)
                if exchange is not None:
                    clock_reasons[watchdog.observe(exchange)] += 1
                else:
                    clock_reasons["exchange_failed"] += 1
                decision_ns = time.monotonic_ns()
                bounds, clock_reason = watchdog.bounds(jetson_now_ns=decision_ns)
                clock_reasons[clock_reason] += 1
                state = None
                if gimbal_sub is not None:
                    while True:
                        try:
                            payload = gimbal_sub.recv(flags=zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        try:
                            candidate = CamState.model_validate_json(payload)
                        except ValueError:
                            invalid_cam_states += 1
                            continue
                        if (
                            candidate.state_monotonic_ns is None
                            or candidate.state_monotonic_ns > decision_ns
                            or decision_ns - candidate.state_monotonic_ns > 100_000_000
                        ):
                            invalid_cam_states += 1
                            continue
                        if feedforward.observe_cam_state(candidate):
                            state = candidate
                            cam_states += 1
                else:
                    # No measured pose is claimed or used for motor authority.
                    state = CamState(
                        frame_id=snapshot.frame.frame_id, src_ts_ms=0,
                        state_monotonic_ns=decision_ns,
                        pan=0.0, tilt=0.0, pan_rate=0.0, tilt_rate=0.0,
                    )
                    feedforward.observe_cam_state(state)
                manual = ManualControlState(
                    src_ts_ms=0, source="shadow_fixture", active=False,
                    emergency=False, control_cmd_enabled=True,
                    joystick_raw=(0, 0), joystick_rate_cmd=(0.0, 0.0),
                )
                assembler.update_perception_snapshot(snapshot, received_at=decision_ns / 1e9)
                if state is not None:
                    assembler.update_cam_state(
                        state, received_at=(state.state_monotonic_ns or 0) / 1e9,
                    )
                assembler.update_manual_state(manual, received_at=decision_ns / 1e9)
                observation = assembler.build(now=decision_ns / 1e9)
                ff_result = feedforward.estimate(observation, bounds)
                baseline_result = baseline.decide(observation, bounds)
                applied_ff = (
                    (0.5 * ff_result.yaw_rate_rad_s, 0.5 * ff_result.pitch_rate_rad_s)
                    if ff_result.valid else (0.0, 0.0)
                )
                combined_result = combined.decide(
                    observation, bounds, feedforward_rad_s=applied_ff,
                )
                reasons[combined_result.intent.reason] += 1
                ff_reasons[ff_result.reason] += 1
                ff_valid += ff_result.valid
                trace.write(json.dumps({
                    "type": "tick", "source_frame_id": snapshot.frame.frame_id,
                    "pid_reason": combined_result.intent.reason,
                    "ff_reason": ff_result.reason,
                    "ff_rate_rad_s": [ff_result.yaw_rate_rad_s, ff_result.pitch_rate_rad_s],
                    "applied_ff_rad_s": applied_ff,
                    "baseline_command_rad_s": [
                        baseline_result.intent.yaw_rate_rad_s,
                        baseline_result.intent.pitch_rate_rad_s,
                    ],
                    "combined_command_rad_s": [
                        combined_result.intent.yaw_rate_rad_s,
                        combined_result.intent.pitch_rate_rad_s,
                    ],
                    "combined_ff_term_rad_s": [
                        combined_result.pid.yaw.feedforward_rad_s,
                        combined_result.pid.pitch.feedforward_rad_s,
                    ],
                    "clock_reason": clock_reason,
                    "capture_width_ns": ff_result.clock_width_ns,
                    "target_valid": observation.target.valid,
                    "gimbal_valid": observation.gimbal.valid,
                    "gimbal_age_ms": observation.gimbal.sample_age_ms,
                    "gimbal_rate_present": [
                        observation.gimbal.yaw_rate_rad_s is not None,
                        observation.gimbal.pitch_rate_rad_s is not None,
                    ],
                }, separators=(",", ":"), sort_keys=True) + "\n")
        finally:
            sub.close(0)
            if gimbal_sub is not None:
                gimbal_sub.close(0)
            context.term()
    report = {
        **startup, "duration_s": time.monotonic() - started,
        "received_snapshots": received, "verified_snapshots": verified,
        "cam_states": cam_states, "invalid_cam_states": invalid_cam_states,
        "feedforward_valid": ff_valid,
        "pid_reasons": dict(reasons), "ff_reasons": dict(ff_reasons),
        "clock_reasons": dict(clock_reasons),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if received > 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())
