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

from common.gimbal.mks_servo42_rs485 import SpeedCommandDither
from common.schemas import CamState, ControlIntent, ControlIntentLimits, ManualControlState
from jetson.control_v3.pid import AxisPIDConfig, BasicPID, PIDInput
from jetson.control_v3.timing import TimingVerdict


PERIOD_S = 0.02
MAX_DURATION_S = 30.0
MAX_YAW_RATE_RAD_S = 0.2
MAX_YAW_TRAVEL_RAD = 0.15
MAX_PITCH_TRAVEL_RAD = 0.03
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


def safe_state(
    *, now_ns: int, gimbal: CamState | None, gimbal_receipt_ns: int | None,
    manual: ManualControlState | None, manual_receipt_ns: int | None,
    home_yaw_rad: float | None, home_pitch_rad: float | None,
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
    if home_yaw_rad is not None and abs(gimbal.pan - home_yaw_rad) > MAX_YAW_TRAVEL_RAD:
        return "yaw_travel_limit"
    if home_pitch_rad is not None and abs(gimbal.tilt - home_pitch_rad) > MAX_PITCH_TRAVEL_RAD:
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
    parser.add_argument("--yaw-kd", type=float, default=0.1)
    parser.add_argument("--speed-dither", action="store_true")
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
    if args.firmware_runtime_ms not in (20, 100):
        parser.error("firmware runtime must be 20 or 100 ms")
    if args.speed_dither != (args.firmware_runtime_ms == 20):
        parser.error("speed dither requires the 20-ms firmware timer")
    if args.enable_live_intent_publish != args.acknowledge_unloaded_hardware:
        parser.error("live trial requires both explicit acknowledgements")
    if not args.gimbal_sub.startswith("tcp://") or not args.manual_bind.startswith("tcp://") or not args.intent_bind.startswith("tcp://"):
        parser.error("all endpoints must be TCP")
    config = {
        "mode": "v3_local_synthetic_pid_hardware_trial",
        "live": args.enable_live_intent_publish,
        "duration_s": args.duration_s,
        "period_s": PERIOD_S,
        "yaw_gains": {"kp": 8.0, "ki": 0.0, "kd": args.yaw_kd},
        "speed_dither": args.speed_dither,
        "yaw_rate_limit_rad_s": MAX_YAW_RATE_RAD_S,
        "yaw_acceleration_limit_rad_s2": 3.5,
        "pitch_command_rad_s": 0.0,
        "max_yaw_travel_rad": MAX_YAW_TRAVEL_RAD,
        "max_pitch_travel_rad": MAX_PITCH_TRAVEL_RAD,
        "intent_valid_for_ms": 50,
        "bridge_firmware_command_runtime_required_ms": args.firmware_runtime_ms,
        "target": "Jetson-local deterministic 0,+0.06,-0.06,0 rad yaw",
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
    yaw_config = AxisPIDConfig(8.0, 0.0, args.yaw_kd, 0.0, MAX_YAW_RATE_RAD_S, 3.5)
    pitch_config = AxisPIDConfig(0.0, 0.0, 0.0, 0.0, 0.01, 3.5)
    pid = BasicPID(yaw_config, pitch_config)
    speed_dither = SpeedCommandDither(gear_ratio=1.0)
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
                )
                if status == "ready" and home_yaw_rad is None:
                    assert gimbal is not None
                    home_yaw_rad, home_pitch_rad = gimbal.pan, gimbal.tilt
                elapsed_s = (now_ns - start_ns) / 1e9
                reference = None if home_yaw_rad is None else home_yaw_rad + reference_offset_rad(elapsed_s)
                error = 0.0 if reference is None or gimbal is None else reference - gimbal.pan
                decision = pid.decide(PIDInput(
                    decision_ns=now_ns,
                    track_id=1,
                    error_rad=(error, 0.0),
                    gimbal_rate_rad_s=(
                        0.0 if gimbal is None or gimbal.pan_rate is None
                        or not math.isfinite(gimbal.pan_rate) else gimbal.pan_rate,
                        0.0,
                    ),
                    timing=TimingVerdict(status == "ready", "local_synthetic" if status == "ready" else status),
                    safety_allowed=status == "ready",
                    gimbal_valid=status == "ready",
                ))
                reason = status if status != "ready" else decision.reason
                pid_command = decision.yaw.final_rad_s if status == "ready" else 0.0
                command = speed_dither.quantize(pid_command) if args.speed_dither else pid_command
                intent = ControlIntent(
                    sequence=sequence_base + sequence,
                    observation_sequence=sequence_base + sequence,
                    issued_monotonic_ns=now_ns,
                    valid_until_monotonic_ns=now_ns + (50_000_000 if args.enable_live_intent_publish else 0),
                    mode="live" if args.enable_live_intent_publish else "shadow",
                    yaw_rate_rad_s=command,
                    pitch_rate_rad_s=0.0,
                    limits=ControlIntentLimits(
                        yaw_rate_limited=decision.yaw.rate_limited,
                        acceleration_limited=decision.yaw.acceleration_limited,
                    ),
                    reason=reason,
                )
                if args.enable_live_intent_publish:
                    intent_pub.send_string(intent.model_dump_json(exclude_none=True))
                trace.write(json.dumps({
                    "type": "tick", "elapsed_s": elapsed_s,
                    "status": status, "reference_yaw_rad": reference,
                    "measured_yaw_rad": None if gimbal is None else gimbal.pan,
                    "measured_yaw_rate_rad_s": None if gimbal is None else gimbal.pan_rate,
                    "yaw_rate_fallback_zero": gimbal is None or gimbal.pan_rate is None
                    or not math.isfinite(gimbal.pan_rate),
                    "measured_pitch_rad": None if gimbal is None else gimbal.tilt,
                    "error_yaw_rad": error, "intent": intent.model_dump(mode="json"),
                    "raw_pid_yaw_rate_rad_s": pid_command,
                    "pid": {"p": decision.yaw.proportional_rad_s,
                            "d": decision.yaw.derivative_rad_s,
                            "pre_limit": decision.yaw.pre_limit_rad_s,
                            "rate_limited": decision.yaw.rate_limited,
                            "acceleration_limited": decision.yaw.acceleration_limited},
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
