"""Bounded unloaded raw-PID hardware trial with a Jetson-local target.

This isolates feedback, serial execution, and encoder motion from detection,
rendering, and PC/Jetson clock mapping. It does not qualify the video path.
The bridge remains the exclusive serial owner and firmware-timed F6 commands
provide the independent command-expiry stop path.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import zmq

from common.schemas import CamState, ControlIntent, ControlIntentLimits, ManualControlState
from jetson.control.feedforward import TargetRateKalman
from jetson.control.pid import AxisPIDConfig, BasicPID, PIDInput
from jetson.control.timing import TimingVerdict


PERIOD_S = 0.02
MAX_DURATION_S = 30.0
MAX_YAW_RATE_RAD_S = 0.2
MAX_PITCH_RATE_RAD_S = 0.2
MAX_YAW_TRAVEL_RAD = 0.15
MAX_PITCH_TRAVEL_RAD = 0.03
MAX_ACTIVE_PITCH_TRAVEL_RAD = 0.15
MAX_GIMBAL_AGE_NS = 100_000_000
MAX_SAFETY_AGE_NS = 750_000_000


def reference_offset_rad(elapsed_s: float) -> float:
    """Small, symmetric step/reversal; never exceeds 0.06 rad from home."""
    if elapsed_s < 3.0:
        return 0.0
    if elapsed_s < 8.0:
        return 0.06
    if elapsed_s < 13.0:
        return -0.06
    return 0.0


def moving_reference_offset_rad(elapsed_s: float) -> float:
    """Controlled, continuous 0.06-rad sine target; max rate 0.126 rad/s."""
    if elapsed_s < 3.0:
        return 0.0
    return 0.06 * math.sin(2.0 * math.pi * (elapsed_s - 3.0) / 3.0)


def safe_state(
    *, now_ns: int, gimbal: CamState | None, gimbal_receipt_ns: int | None,
    manual: ManualControlState | None, manual_receipt_ns: int | None,
    home_yaw_rad: float | None, home_pitch_rad: float | None,
    max_yaw_travel_rad: float = MAX_YAW_TRAVEL_RAD,
    max_pitch_travel_rad: float = MAX_PITCH_TRAVEL_RAD,
) -> str:
    if manual is None or manual_receipt_ns is None or not 0 <= now_ns - manual_receipt_ns <= MAX_SAFETY_AGE_NS:
        return "safety_stale"
    if manual.active or manual.emergency or not manual.control_cmd_enabled:
        return "safety_hold"
    if gimbal is None or gimbal_receipt_ns is None or not 0 <= now_ns - gimbal_receipt_ns <= MAX_GIMBAL_AGE_NS:
        return "gimbal_stale"
    # The bridge reports encoder rates only when its independent rate window
    # has enough samples. Missing rates must not invalidate a fresh pose;
    # derivative feedback falls back to zero for that sample.
    values = (gimbal.pan, gimbal.tilt)
    if any(value is None or not math.isfinite(value) for value in values):
        return "gimbal_invalid"
    if gimbal.state_monotonic_ns is not None and not 0 <= now_ns - gimbal.state_monotonic_ns <= MAX_GIMBAL_AGE_NS:
        return "gimbal_sample_stale"
    if home_yaw_rad is not None and abs(gimbal.pan - home_yaw_rad) > max_yaw_travel_rad:
        return "yaw_travel_limit"
    if home_pitch_rad is not None and abs(gimbal.tilt - home_pitch_rad) > max_pitch_travel_rad:
        return "pitch_travel_limit"
    return "ready"


def _latest(socket: zmq.Socket) -> bytes | None:
    payload = None
    while True:
        try:
            payload = socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return payload


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gimbal-sub", required=True)
    parser.add_argument("--manual-bind", required=True)
    parser.add_argument("--intent-bind", required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--axis", choices=("yaw", "pitch_a"), default="yaw")
    parser.add_argument("--kp", type=float, default=8.0)
    parser.add_argument("--target-profile", choices=("step", "sine"), default="step")
    parser.add_argument("--observation-delay-ms", type=int, default=60)
    parser.add_argument("--feedforward-scale", type=float, default=0.0)
    parser.add_argument("--yaw-kd", type=float, default=0.1)
    parser.add_argument("--firmware-runtime-ms", type=int, default=100)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--enable-live-intent-publish", action="store_true")
    parser.add_argument("--acknowledge-unloaded-hardware", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not 3.0 <= args.duration_s <= MAX_DURATION_S:
        parser.error("duration must be in [3, 30] seconds")
    if not math.isfinite(args.yaw_kd) or not 0.0 <= args.yaw_kd <= 0.2:
        parser.error("yaw Kd must be in [0, 0.2]")
    if not math.isfinite(args.kp) or not 0.0 <= args.kp <= 10.0:
        parser.error("Kp must be in [0, 10]")
    if not math.isfinite(args.feedforward_scale) or not 0.0 <= args.feedforward_scale <= 1.0:
        parser.error("feedforward scale must be in [0, 1]")
    if not 0 <= args.observation_delay_ms <= 100:
        parser.error("observation delay must be in [0, 100] ms")
    if args.feedforward_scale and args.target_profile != "sine":
        parser.error("nonzero feedforward requires a moving sine target")
    if args.firmware_runtime_ms not in (20, 100):
        parser.error("firmware runtime must be 20 or 100 ms")
    if args.enable_live_intent_publish != args.acknowledge_unloaded_hardware:
        parser.error("live trial requires both explicit acknowledgements")
    if not args.gimbal_sub.startswith("tcp://") or not args.manual_bind.startswith("tcp://") or not args.intent_bind.startswith("tcp://"):
        parser.error("all endpoints must be TCP")
    config = {
        "mode": "v3_local_synthetic_pid_hardware_trial",
        "live": args.enable_live_intent_publish,
        "duration_s": args.duration_s,
        "period_s": PERIOD_S,
        "axis": args.axis,
        "active_axis_gains": {"kp": args.kp, "ki": 0.0, "kd": args.yaw_kd},
        "target_profile": args.target_profile,
        "observation_delay_ms": args.observation_delay_ms if args.target_profile == "sine" else 0,
        "feedforward_scale": args.feedforward_scale,
        "yaw_gains": {"kp": args.kp if args.axis == "yaw" else 0.0, "ki": 0.0,
                      "kd": args.yaw_kd if args.axis == "yaw" else 0.0},
        "pitch_gains": {"kp": args.kp if args.axis == "pitch_a" else 0.0, "ki": 0.0,
                        "kd": args.yaw_kd if args.axis == "pitch_a" else 0.0},
        "yaw_rate_limit_rad_s": MAX_YAW_RATE_RAD_S,
        "yaw_acceleration_limit_rad_s2": 3.5,
        "pitch_command_rad_s": 0.0 if args.axis == "yaw" else None,
        "max_yaw_travel_rad": MAX_YAW_TRAVEL_RAD if args.axis == "yaw" else MAX_PITCH_TRAVEL_RAD,
        "max_pitch_travel_rad": MAX_PITCH_TRAVEL_RAD if args.axis == "yaw" else MAX_ACTIVE_PITCH_TRAVEL_RAD,
        "intent_valid_for_ms": 50,
        "bridge_firmware_command_runtime_required_ms": args.firmware_runtime_ms,
        "target": f"Jetson-local deterministic {args.target_profile} {args.axis}",
        "video_timing_qualified": False,
    }
    if args.check:
        print(json.dumps(config, sort_keys=True))
        return 0

    context = zmq.Context()
    gimbal_sub = context.socket(zmq.SUB)
    gimbal_sub.setsockopt(zmq.LINGER, 0)
    gimbal_sub.setsockopt(zmq.CONFLATE, 1)
    gimbal_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    gimbal_sub.connect(args.gimbal_sub)
    manual_pull = context.socket(zmq.PULL)
    manual_pull.setsockopt(zmq.LINGER, 0)
    manual_pull.setsockopt(zmq.CONFLATE, 1)
    manual_pull.bind(args.manual_bind)
    intent_pub = context.socket(zmq.PUB)
    intent_pub.setsockopt(zmq.LINGER, 100)
    intent_pub.bind(args.intent_bind)
    yaw_config = AxisPIDConfig(
        args.kp if args.axis == "yaw" else 0.0, 0.0,
        args.yaw_kd if args.axis == "yaw" else 0.0, 0.0,
        MAX_YAW_RATE_RAD_S if args.axis == "yaw" else 0.01, 3.5,
    )
    pitch_config = AxisPIDConfig(
        args.kp if args.axis == "pitch_a" else 0.0, 0.0,
        args.yaw_kd if args.axis == "pitch_a" else 0.0, 0.0,
        MAX_PITCH_RATE_RAD_S if args.axis == "pitch_a" else 0.01, 3.5,
    )
    pid = BasicPID(yaw_config, pitch_config)
    target_estimator = TargetRateKalman()
    gimbal = None
    manual = None
    gimbal_receipt_ns = manual_receipt_ns = None
    home_yaw_rad = home_pitch_rad = None
    reason_counts: Counter[str] = Counter()
    sequence_base = time.time_ns() // 1000
    start_ns = time.monotonic_ns()
    next_tick_ns = start_ns
    sequence = 0
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    with args.trace.open("w", encoding="utf-8", buffering=1) as trace:
        trace.write(json.dumps({"type": "meta", **config}, sort_keys=True) + "\n")
        try:
            while time.monotonic_ns() - start_ns < args.duration_s * 1e9:
                now_ns = time.monotonic_ns()
                payload = _latest(gimbal_sub)
                if payload is not None:
                    try:
                        gimbal = CamState.model_validate_json(payload)
                        gimbal_receipt_ns = now_ns
                    except ValueError:
                        gimbal = None
                payload = _latest(manual_pull)
                if payload is not None:
                    try:
                        manual = ManualControlState.model_validate_json(payload)
                        manual_receipt_ns = now_ns
                    except ValueError:
                        manual = None
                if now_ns < next_tick_ns:
                    time.sleep(min(0.001, (next_tick_ns - now_ns) / 1e9))
                    continue
                sequence += 1
                status = safe_state(
                    now_ns=now_ns, gimbal=gimbal, gimbal_receipt_ns=gimbal_receipt_ns,
                    manual=manual, manual_receipt_ns=manual_receipt_ns,
                    home_yaw_rad=home_yaw_rad, home_pitch_rad=home_pitch_rad,
                    max_yaw_travel_rad=(MAX_YAW_TRAVEL_RAD if args.axis == "yaw" else MAX_PITCH_TRAVEL_RAD),
                    max_pitch_travel_rad=(MAX_PITCH_TRAVEL_RAD if args.axis == "yaw" else MAX_ACTIVE_PITCH_TRAVEL_RAD),
                )
                if status == "ready" and home_yaw_rad is None:
                    assert gimbal is not None
                    home_yaw_rad, home_pitch_rad = gimbal.pan, gimbal.tilt
                elapsed_s = (now_ns - start_ns) / 1e9
                home_active = home_yaw_rad if args.axis == "yaw" else home_pitch_rad
                measured_active = None if gimbal is None else (gimbal.pan if args.axis == "yaw" else gimbal.tilt)
                delay_s = args.observation_delay_ms / 1000 if args.target_profile == "sine" else 0.0
                observed_elapsed_s = max(0.0, elapsed_s - delay_s)
                target_offset = (moving_reference_offset_rad(observed_elapsed_s)
                                 if args.target_profile == "sine" else reference_offset_rad(elapsed_s))
                reference = None if home_active is None else home_active + target_offset
                error = 0.0 if reference is None or measured_active is None else reference - measured_active
                error_pair = (error, 0.0) if args.axis == "yaw" else (0.0, error)
                sample_ns = now_ns - round(delay_s * 1e9)
                if reference is not None and args.target_profile == "sine":
                    target_estimator.observe(track_id=1, angle_rad=reference, sample_ns=sample_ns)
                estimate = target_estimator.estimate(decision_ns=now_ns, track_id=1)
                feedforward = (args.feedforward_scale * estimate.rate_rad_s
                               if status == "ready" and estimate.valid else 0.0)
                feedforward_pair = ((feedforward, 0.0) if args.axis == "yaw"
                                    else (0.0, feedforward))
                decision = pid.decide(PIDInput(
                    decision_ns=now_ns,
                    track_id=1,
                    error_rad=error_pair,
                    gimbal_rate_rad_s=(
                        0.0 if gimbal is None or gimbal.pan_rate is None
                        or not math.isfinite(gimbal.pan_rate) else gimbal.pan_rate,
                        0.0 if args.axis == "yaw" or gimbal is None or gimbal.tilt_rate is None
                        or not math.isfinite(gimbal.tilt_rate) else gimbal.tilt_rate,
                    ),
                    timing=TimingVerdict(status == "ready", "local_synthetic" if status == "ready" else status),
                    safety_allowed=status == "ready",
                    gimbal_valid=status == "ready",
                    feedforward_rad_s=feedforward_pair,
                ))
                reason = status if status != "ready" else decision.reason
                active_decision = decision.yaw if args.axis == "yaw" else decision.pitch
                pid_command = active_decision.final_rad_s if status == "ready" else 0.0
                command = pid_command
                intent = ControlIntent(
                    sequence=sequence_base + sequence,
                    observation_sequence=sequence_base + sequence,
                    issued_monotonic_ns=now_ns,
                    valid_until_monotonic_ns=now_ns + (50_000_000 if args.enable_live_intent_publish else 0),
                    mode="live" if args.enable_live_intent_publish else "shadow",
                    yaw_rate_rad_s=command if args.axis == "yaw" else 0.0,
                    pitch_rate_rad_s=command if args.axis == "pitch_a" else 0.0,
                    limits=ControlIntentLimits(
                        yaw_rate_limited=decision.yaw.rate_limited,
                        pitch_rate_limited=decision.pitch.rate_limited,
                        acceleration_limited=active_decision.acceleration_limited,
                    ),
                    reason=reason,
                )
                if args.enable_live_intent_publish:
                    intent_pub.send_string(intent.model_dump_json(exclude_none=True))
                trace.write(json.dumps({
                    "type": "tick", "elapsed_s": elapsed_s,
                    "status": status,
                    "reference_yaw_rad": reference if args.axis == "yaw" else None,
                    "reference_pitch_rad": reference if args.axis == "pitch_a" else None,
                    "measured_yaw_rad": None if gimbal is None else gimbal.pan,
                    "measured_yaw_rate_rad_s": None if gimbal is None else gimbal.pan_rate,
                    "yaw_rate_fallback_zero": gimbal is None or gimbal.pan_rate is None
                    or not math.isfinite(gimbal.pan_rate),
                    "measured_pitch_rad": None if gimbal is None else gimbal.tilt,
                    "error_yaw_rad": error if args.axis == "yaw" else 0.0,
                    "error_pitch_rad": error if args.axis == "pitch_a" else 0.0,
                    "intent": intent.model_dump(mode="json"),
                    "raw_pid_yaw_rate_rad_s": (
                        active_decision.proportional_rad_s + active_decision.integral_rad_s
                        + active_decision.derivative_rad_s if args.axis == "yaw" else 0.0
                    ),
                    "raw_pid_pitch_rate_rad_s": (
                        active_decision.proportional_rad_s + active_decision.integral_rad_s
                        + active_decision.derivative_rad_s if args.axis == "pitch_a" else 0.0
                    ),
                    "combined_final_rate_rad_s": pid_command,
                    "target_sample_monotonic_ns": sample_ns if args.target_profile == "sine" else None,
                    "target_rate_estimate_rad_s": estimate.rate_rad_s if estimate.valid else None,
                    "target_estimate_reason": estimate.reason,
                    "feedforward_rad_s": feedforward,
                    "pid": {"p": active_decision.proportional_rad_s,
                            "d": active_decision.derivative_rad_s,
                            "feedforward": active_decision.feedforward_rad_s,
                            "pre_limit": active_decision.pre_limit_rad_s,
                            "rate_limited": active_decision.rate_limited,
                            "acceleration_limited": active_decision.acceleration_limited},
                }, separators=(",", ":"), sort_keys=True) + "\n")
                reason_counts[reason] += 1
                next_tick_ns += round(PERIOD_S * 1e9)
                if next_tick_ns < now_ns:
                    next_tick_ns = now_ns + round(PERIOD_S * 1e9)
        finally:
            if args.enable_live_intent_publish:
                for offset in range(3):
                    stopped_ns = time.monotonic_ns()
                    stop_intent = ControlIntent(
                        sequence=sequence_base + sequence + offset + 1,
                        observation_sequence=sequence_base + sequence,
                        issued_monotonic_ns=stopped_ns,
                        valid_until_monotonic_ns=stopped_ns + 50_000_000,
                        mode="live", yaw_rate_rad_s=0.0, pitch_rate_rad_s=0.0,
                        reason="trial_shutdown",
                    )
                    intent_pub.send_string(stop_intent.model_dump_json(exclude_none=True))
                    time.sleep(0.01)
            gimbal_sub.close(0)
            manual_pull.close(0)
            intent_pub.close(0)
            context.term()
    print(json.dumps({"ticks": sequence, "reasons": dict(reason_counts), **config}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
