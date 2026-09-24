from __future__ import annotations

import json
from pathlib import Path

from common.schemas import (
    ControlGimbalObservation,
    ControlObservation,
    ControlSafetyObservation,
    ControlTargetObservation,
    ControlTransportObservation,
)
from jetson.qualified_controller_profile import load_qualified_shadow_policy_config
from jetson.shadow_rate_policy import ShadowRatePolicy
from tools.analyze_estimator_feedforward_trace import analyze_trace


def _report() -> dict:
    def axis(kp: float, kd: float, q: float) -> dict:
        return {
            "pid_gains": {"kp": kp, "ki": 0.0, "kd": kd},
            "kalman_config": {
                "acceleration_spectral_density": q,
                "measurement_variance_rad2": 1e-6,
                "initial_position_variance_rad2": 1e-4,
                "initial_rate_variance_rad2_s2": 0.25,
                "innovation_gate_nis": 16.0,
                "max_gap_s": 0.25,
                "max_consecutive_rejections": 2,
            },
            "feedforward_gain": 0.5,
            "qualification": {"qualified": True},
        }

    return {
        "format": "idcs.offline_los_estimator_validation",
        "qualification": {"qualified": True},
        "measurement_scenario": {"controller_hz": 50.0, "vision_hz": 30.0},
        "axes": {"yaw": axis(2.0, 0.1, 0.01), "pitch": axis(2.0, 0.0, 0.01)},
    }


def _observation(index: int) -> ControlObservation:
    now_ns = 1_000_000_000 + index * 40_000_000
    return ControlObservation(
        sequence=index + 1,
        created_monotonic_ns=now_ns,
        source_frame_id=index + 1,
        source_time_ns=now_ns - 20_000_000,
        source_clock_domain="pc_monotonic",
        frame_received_time_ns=now_ns - 15_000_000,
        frame_receive_clock_domain="jetson_monotonic",
        frame_observed_time_ns=now_ns - 5_000_000,
        frame_observation_clock_domain="jetson_monotonic",
        target=ControlTargetObservation(
            valid=True,
            track_id=7,
            target_center_px=(640.0 + index * 3.0, 360.0),
            aim_reference_px=(640.0, 360.0),
            pixel_error=(index * 3.0, 0.0),
            bearing_error_rad=(index * 0.01, 0.0),
            source_age_ms=0.0,
        ),
        gimbal=ControlGimbalObservation(
            valid=True,
            yaw_rad=0.0,
            pitch_rad=0.0,
            yaw_rate_rad_s=0.0,
            pitch_rate_rad_s=0.0,
            sample_age_ms=2.0,
        ),
        transport=ControlTransportObservation(),
        safety=ControlSafetyObservation(
            valid=True,
            auto_allowed=True,
            manual_active=False,
            emergency_active=False,
        ),
    )


def test_ablation_replays_qualified_policy_exactly_and_is_marked_noncausal(tmp_path: Path) -> None:
    report_path = tmp_path / "qualified.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")
    config = load_qualified_shadow_policy_config(
        report_path,
        valid_for_ns=80_000_000,
        intent_mode="live",
        sequence_base=100,
    )
    policy = ShadowRatePolicy(config)
    trace_path = tmp_path / "trace.jsonl"
    records = []
    for index in range(8):
        observation = _observation(index)
        intent = policy.decide(observation)
        records.append({
            "type": "tick",
            "observation": observation.model_dump(mode="json"),
            "intent": intent.model_dump(mode="json"),
        })
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    result = analyze_trace(trace_path, report_path)

    assert result["causal_performance_claim_allowed"] is False
    assert result["ticks"] == 8
    assert result["recorded_qualified_exact_fraction"] == 1.0
    assert result["variants"]["qualified_estimator_feedforward"]["measurement_updates"] == 8
    assert result["comparisons_to_qualified"]["estimated_no_feedforward"][
        "command_authority_mismatches"
    ] == 0
    assert result["comparisons_to_qualified"]["estimated_no_feedforward"][
        "yaw_delta"
    ]["max_abs_rad_s"] > 0.0
