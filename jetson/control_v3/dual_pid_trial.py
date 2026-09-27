"""Bounded unloaded V3 yaw-plus-pitch-A synthetic moving-target trial.

The target is generated in Jetson monotonic time with an explicit 60-ms
observation delay. It isolates hardware control from PC video timing.
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
from jetson.control_v3.feedforward import TargetRateKalman
from jetson.control_v3.local_pid_trial import _latest, safe_state
from jetson.control_v3.pid import AxisPIDConfig, BasicPID, PIDInput
from jetson.control_v3.timing import TimingVerdict


def target_offset(elapsed_s: float, *, period_s: float = 3.0) -> float:
    return 0.0 if elapsed_s < 3.0 else 0.06 * math.sin(2 * math.pi * (elapsed_s - 3.0) / period_s)


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gimbal-sub", required=True)
    parser.add_argument("--manual-bind", required=True)
    parser.add_argument("--intent-bind", required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--feedforward-scale", type=float, required=True)
    parser.add_argument("--target-period-s", type=float, default=3.0)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--enable-live-intent-publish", action="store_true")
    parser.add_argument("--acknowledge-unloaded-hardware", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not 10 <= args.duration_s <= 30:
        parser.error("duration must be in [10, 30] seconds")
    if args.feedforward_scale not in (0.0, 0.5):
        parser.error("feedforward scale must be 0 or 0.5")
    if args.target_period_s not in (3.0, 4.0):
        parser.error("target period must be 3 or 4 seconds")
    if args.enable_live_intent_publish != args.acknowledge_unloaded_hardware:
        parser.error("live trial requires both hardware acknowledgements")
    if not all(endpoint.startswith("tcp://") for endpoint in
               (args.gimbal_sub, args.manual_bind, args.intent_bind)):
        parser.error("all endpoints must be TCP")
    config = {
        "mode": "v3_dual_synthetic_pid_feedforward_hardware_trial",
        "duration_s": args.duration_s, "period_s": 0.02,
        "yaw_gains": {"kp": 8.0, "ki": 0.0, "kd": 0.0},
        "pitch_gains": {"kp": 4.0, "ki": 0.0, "kd": 0.0},
        "feedforward_scale": args.feedforward_scale,
        "observation_delay_ms": 60,
        "target": f"Jetson-local 0.06-rad {args.target_period_s:g}-second sine on both axes",
        "target_period_s": args.target_period_s,
        "rate_cap_rad_s": 0.2, "travel_guard_rad": 0.15,
        "live": args.enable_live_intent_publish,
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
    pid = BasicPID(
        AxisPIDConfig(8.0, 0.0, 0.0, 0.0, 0.2, 3.5),
        AxisPIDConfig(4.0, 0.0, 0.0, 0.0, 0.2, 3.5),
    )
    estimators = (TargetRateKalman(), TargetRateKalman())
    gimbal = manual = None
    gimbal_receipt_ns = manual_receipt_ns = None
    home = None
    reasons: Counter[str] = Counter()
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
                    home_yaw_rad=None if home is None else home[0],
                    home_pitch_rad=None if home is None else home[1],
                    max_yaw_travel_rad=0.15, max_pitch_travel_rad=0.15,
                )
                if status == "ready" and home is None:
                    assert gimbal is not None
                    home = (gimbal.pan, gimbal.tilt)
                elapsed_s = (now_ns - start_ns) / 1e9
                observed_elapsed_s = max(0.0, elapsed_s - 0.06)
                sample_ns = now_ns - 60_000_000
                reference = None if home is None else tuple(
                    value + target_offset(observed_elapsed_s, period_s=args.target_period_s)
                    for value in home
                )
                measured = None if gimbal is None else (gimbal.pan, gimbal.tilt)
                error = ((0.0, 0.0) if reference is None or measured is None else
                         tuple(reference[index] - measured[index] for index in (0, 1)))
                estimates = []
                for index, estimator in enumerate(estimators):
                    if reference is not None:
                        estimator.observe(track_id=index + 1, angle_rad=reference[index],
                                          sample_ns=sample_ns)
                    estimates.append(estimator.estimate(decision_ns=now_ns, track_id=index + 1))
                feedforward = tuple(
                    args.feedforward_scale * item.rate_rad_s
                    if status == "ready" and item.valid else 0.0 for item in estimates
                )
                gimbal_rates = (
                    0.0 if gimbal is None or gimbal.pan_rate is None
                    or not math.isfinite(gimbal.pan_rate) else gimbal.pan_rate,
                    0.0 if gimbal is None or gimbal.tilt_rate is None
                    or not math.isfinite(gimbal.tilt_rate) else gimbal.tilt_rate,
                )
                decision = pid.decide(PIDInput(
                    decision_ns=now_ns, track_id=1, error_rad=error,
                    gimbal_rate_rad_s=gimbal_rates, feedforward_rad_s=feedforward,
                    timing=TimingVerdict(status == "ready", "local_synthetic" if status == "ready" else status),
                    safety_allowed=status == "ready", gimbal_valid=status == "ready",
                ))
                reason = status if status != "ready" else decision.reason
                intent = ControlIntent(
                    sequence=sequence_base + sequence,
                    observation_sequence=sequence_base + sequence,
                    issued_monotonic_ns=now_ns,
                    valid_until_monotonic_ns=now_ns + (50_000_000 if args.enable_live_intent_publish else 0),
                    mode="live" if args.enable_live_intent_publish else "shadow",
                    yaw_rate_rad_s=decision.yaw.final_rad_s if status == "ready" else 0.0,
                    pitch_rate_rad_s=decision.pitch.final_rad_s if status == "ready" else 0.0,
                    limits=ControlIntentLimits(
                        yaw_rate_limited=decision.yaw.rate_limited,
                        pitch_rate_limited=decision.pitch.rate_limited,
                        acceleration_limited=decision.yaw.acceleration_limited or decision.pitch.acceleration_limited,
                    ),
                    reason=reason,
                )
                if args.enable_live_intent_publish:
                    intent_pub.send_string(intent.model_dump_json(exclude_none=True))
                trace.write(json.dumps({
                    "type": "tick", "elapsed_s": elapsed_s, "status": status,
                    "reference_rad": reference, "measured_rad": measured,
                    "error_rad": error, "sample_monotonic_ns": sample_ns,
                    "estimator_rate_rad_s": [item.rate_rad_s if item.valid else None for item in estimates],
                    "estimator_reasons": [item.reason for item in estimates],
                    "feedforward_rad_s": feedforward,
                    "feedback_rad_s": [decision.yaw.proportional_rad_s, decision.pitch.proportional_rad_s],
                    "intent": intent.model_dump(mode="json"),
                }, separators=(",", ":"), sort_keys=True) + "\n")
                reasons[reason] += 1
                next_tick_ns += 20_000_000
                if next_tick_ns < now_ns:
                    next_tick_ns = now_ns + 20_000_000
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
    print(json.dumps({"ticks": sequence, "reasons": dict(reasons), **config}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
