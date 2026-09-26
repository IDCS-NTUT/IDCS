"""Bounded V3 raw-PID shadow canary using verified video timing and a synthetic target.

This process only subscribes to perception and queries the PC clock. It never
opens serial, motor, laser, or control-publish sockets. The target/gimbal/safety
inputs are synthetic and must not be used to qualify hardware gains.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import zmq

from common.perception import PerceptionSnapshotV2, perception_snapshot_from_json
from common.schemas import (
    ControlGimbalObservation,
    ControlObservation,
    ControlSafetyObservation,
    ControlTargetObservation,
    ControlTransportObservation,
)
from jetson.control_v3.clock_watchdog import ClockWatchdog, ClockWatchdogConfig
from jetson.control_v3.pid import AxisPIDConfig, BasicPID
from jetson.control_v3.shadow_pid import ShadowPIDController
from jetson.control_v3.timing import ClockBounds


def synthetic_observation(
    snapshot: PerceptionSnapshotV2, *, sequence: int, decision_ns: int
) -> ControlObservation:
    """Stable guaranteed target; only frame identity/timing comes from video."""

    phase = 2 * math.pi * (sequence - 1) / 60
    yaw_error = 0.08 * math.sin(phase)
    pitch_error = 0.04 * math.cos(phase)
    aim = (snapshot.frame.width / 2, snapshot.frame.height / 2)
    pixel_error = (yaw_error * 300, pitch_error * 300)
    return ControlObservation(
        sequence=sequence,
        created_monotonic_ns=decision_ns,
        source_frame_id=snapshot.frame.frame_id,
        source_time_ns=snapshot.frame.source_time_ns,
        source_clock_domain=snapshot.frame.source_clock_domain,
        source_identity_verified=snapshot.frame.source_identity_verified,
        frame_received_time_ns=snapshot.frame.received_time_ns,
        frame_receive_clock_domain=snapshot.frame.receive_clock_domain,
        frame_observed_time_ns=snapshot.frame.observed_time_ns,
        frame_observation_clock_domain=snapshot.frame.observation_clock_domain,
        target=ControlTargetObservation(
            valid=True,
            track_id=1,
            class_id="synthetic_guaranteed_target",
            confidence=1.0,
            target_center_px=(aim[0] + pixel_error[0], aim[1] + pixel_error[1]),
            aim_reference_px=aim,
            pixel_error=pixel_error,
            bearing_error_rad=(yaw_error, pitch_error),
            source_age_ms=0.0,
        ),
        gimbal=ControlGimbalObservation(
            valid=True,
            yaw_rad=0.0,
            pitch_rad=0.0,
            yaw_rate_rad_s=0.0,
            pitch_rate_rad_s=0.0,
            sample_age_ms=0.0,
        ),
        transport=ControlTransportObservation(),
        safety=ControlSafetyObservation(
            valid=True,
            auto_allowed=True,
            manual_active=False,
            emergency_active=False,
            sample_age_ms=0.0,
        ),
    )


def _exchange_clock(context: zmq.Context, endpoint: str) -> ClockBounds | None:
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 50)
    socket.setsockopt(zmq.SNDTIMEO, 50)
    try:
        socket.connect(endpoint)
        sent_ns = time.monotonic_ns()
        socket.send_json({"version": 1, "jetson_send_ns": sent_ns})
        payload = socket.recv_json()
        received_ns = time.monotonic_ns()
        if not isinstance(payload, dict) or payload.get("version") != 1 or payload.get("jetson_send_ns") != sent_ns:
            return None
        return ClockBounds.from_exchange(
            jetson_send_ns=sent_ns,
            pc_receive_ns=int(payload["pc_receive_ns"]),
            pc_send_ns=int(payload["pc_send_ns"]),
            jetson_receive_ns=received_ns,
        )
    except (KeyError, TypeError, ValueError, zmq.ZMQError):
        return None
    finally:
        socket.close(0)


def _p95(values: list[float]) -> float | None:
    return None if not values else sorted(values)[math.ceil(0.95 * len(values)) - 1]


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-endpoint", required=True)
    parser.add_argument("--clock-endpoint", required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--empirical-drift-ppm", type=float)
    parser.add_argument("--ack-empirical-bound-shadow-only", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not 0 < args.duration_s <= 30:
        parser.error("bounded duration must be in (0, 30]")
    if args.empirical_drift_ppm is not None and (
        not args.ack_empirical_bound_shadow_only
        or not math.isfinite(args.empirical_drift_ppm)
        or not 0 <= args.empirical_drift_ppm < 1_000_000
    ):
        parser.error("empirical drift policy requires explicit shadow-only acknowledgement")
    for endpoint in (args.snapshot_endpoint, args.clock_endpoint):
        if not endpoint.startswith("tcp://"):
            parser.error("canary endpoints must be TCP")
        try:
            port = int(endpoint.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            parser.error("canary endpoint requires a numeric port")
        if port < 50000:
            parser.error("canary endpoints must use isolated ports >= 50000")
    if args.check:
        print(json.dumps({
            "mode": "shadow_only",
            "target": "guaranteed_synthetic",
            "motor_authority": False,
            "drift_policy": args.empirical_drift_ppm,
        }, sort_keys=True))
        return 0

    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.LINGER, 0)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
    subscriber.connect(args.snapshot_endpoint)
    watchdog = ClockWatchdog(ClockWatchdogConfig(
        max_exchange_age_ns=100_000_000,
        configured_max_drift_ppm=args.empirical_drift_ppm,
        required_samples=2,
    ))
    axis = AxisPIDConfig(
        kp=1.0, ki=0.2, kd=0.0,
        integral_limit_rad_s=0.05,
        rate_limit_rad_s=0.2,
        acceleration_limit_rad_s2=1.0,
    )
    controller = ShadowPIDController(
        BasicPID(axis, axis),
        max_clock_sample_age_ns=100_000_000,
        max_capture_age_ns=500_000_000,
        max_gimbal_age_ns=100_000_000,
        max_safety_age_ns=750_000_000,
    )
    reasons: Counter[str] = Counter()
    clock_status: Counter[str] = Counter()
    source_age_ms: list[float] = []
    snapshots = verified_snapshots = nonzero_intents = 0
    first_frame_id = last_frame_id = None
    deadline = time.monotonic() + args.duration_s
    try:
        while time.monotonic() < deadline:
            if not subscriber.poll(100):
                continue
            try:
                snapshot = perception_snapshot_from_json(subscriber.recv())
            except (TypeError, ValueError):
                reasons["snapshot_invalid"] += 1
                continue
            snapshots += 1
            verified_snapshots += snapshot.frame.source_identity_verified is True
            first_frame_id = snapshot.frame.frame_id if first_frame_id is None else first_frame_id
            last_frame_id = snapshot.frame.frame_id
            exchange = _exchange_clock(context, args.clock_endpoint)
            if exchange is not None:
                clock_status[watchdog.observe(exchange)] += 1
            else:
                clock_status["exchange_failed"] += 1
            decision_ns = time.monotonic_ns()
            bounds, status = watchdog.bounds(jetson_now_ns=decision_ns)
            clock_status[status] += 1
            obs = synthetic_observation(snapshot, sequence=snapshots, decision_ns=decision_ns)
            result = controller.decide(obs, bounds)
            reasons[result.intent.reason] += 1
            nonzero_intents += (
                abs(result.intent.yaw_rate_rad_s) > 1e-9
                or abs(result.intent.pitch_rate_rad_s) > 1e-9
            )
            if result.timing.capture_age_ns is not None:
                source_age_ms.append(result.timing.capture_age_ns.latest_ns / 1e6)
    finally:
        subscriber.close(0)
        context.term()
    report = {
        "mode": "shadow_only",
        "motor_authority": False,
        "target": "guaranteed_synthetic",
        "empirical_drift_policy_ppm": args.empirical_drift_ppm,
        "drift_policy_qualified_for_live": False,
        "snapshots": snapshots,
        "verified_snapshots": verified_snapshots,
        "first_frame_id": first_frame_id,
        "last_frame_id": last_frame_id,
        "nonzero_shadow_intents": nonzero_intents,
        "reasons": dict(reasons),
        "clock_status": dict(clock_status),
        "source_age_upper_p50_ms": None if not source_age_ms else statistics.median(source_age_ms),
        "source_age_upper_p95_ms": _p95(source_age_ms),
    }
    if args.report is not None:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if snapshots > 0 and verified_snapshots == snapshots else 1


if __name__ == "__main__":
    raise SystemExit(run())
