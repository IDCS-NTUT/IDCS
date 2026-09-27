from __future__ import annotations

import pytest

from common.clock_sync import ClockOffsetSample
from common.schemas import (
    CamState,
    ControlGimbalObservation,
    ControlObservation,
    ControlSafetyObservation,
    ControlTargetObservation,
    ControlTransportObservation,
)
from jetson.shadow_rate_policy import ShadowRatePolicy, ShadowRatePolicyConfig
from jetson.los_kalman import LOSKalmanConfig


def _observation(sequence: int, timestamp: int, *, track_id: int | None = 7,
                 error: tuple[float, float] | None = (0.2, -0.1),
                 rate: tuple[float, float] | None = None, auto_allowed: bool = True,
                 target_valid: bool = True, yaw: float = 0.0, pitch: float = 0.0) -> ControlObservation:
    return ControlObservation(
        sequence=sequence,
        created_monotonic_ns=timestamp,
        target=ControlTargetObservation(
            valid=target_valid,
            track_id=track_id,
            confidence=0.9,
            target_center_px=(700.0, 380.0) if target_valid else None,
            aim_reference_px=(640.0, 360.0) if target_valid else None,
            pixel_error=(60.0, 20.0) if target_valid else None,
            bearing_error_rad=error,
            bearing_rate_rad_s=rate,
        ),
        gimbal=ControlGimbalObservation(valid=True, yaw_rad=yaw, pitch_rad=pitch),
        transport=ControlTransportObservation(),
        safety=ControlSafetyObservation(valid=True, auto_allowed=auto_allowed,
                                        manual_active=False, emergency_active=False),
    )


def test_policy_is_shadow_only_and_uses_bearing_rate_feedforward() -> None:
    policy = ShadowRatePolicy(ShadowRatePolicyConfig(yaw_kp=2.0, pitch_kp=3.0, yaw_kd=1.0, pitch_kd=2.0,
                                                       yaw_accel_limit_rad_s2=100.0, pitch_accel_limit_rad_s2=100.0))
    intent = policy.decide(_observation(1, 1_000_000_000, rate=(0.1, 0.2)))
    assert intent.mode == "shadow" and intent.reason == "tracking"
    assert intent.yaw_rate_rad_s == pytest.approx(0.5)
    assert intent.pitch_rate_rad_s == pytest.approx(0.1)


def test_policy_requires_explicit_live_intent_mode() -> None:
    shadow = ShadowRatePolicy(
        ShadowRatePolicyConfig(yaw_kp=1.0, pitch_kp=1.0)
    ).decide(_observation(1, 1_000_000_000))
    live = ShadowRatePolicy(
        ShadowRatePolicyConfig(
            yaw_kp=1.0,
            pitch_kp=1.0,
            intent_mode="live",
        )
    ).decide(_observation(1, 1_000_000_000))

    assert shadow.mode == "shadow"
    assert live.mode == "live"


def test_policy_sequence_base_survives_runtime_restarts() -> None:
    intent = ShadowRatePolicy(
        ShadowRatePolicyConfig(
            yaw_kp=1.0,
            pitch_kp=1.0,
            sequence_base=1_700_000_000_000,
        )
    ).decide(_observation(1, 1_000_000_000))

    assert intent.sequence == 1_700_000_000_001


def test_policy_holds_and_resets_for_disallowed_or_lost_target() -> None:
    policy = ShadowRatePolicy(ShadowRatePolicyConfig(yaw_kp=1.0, pitch_kp=1.0, yaw_accel_limit_rad_s2=100.0,
                                                       pitch_accel_limit_rad_s2=100.0))
    policy.decide(_observation(1, 1_000_000_000))
    held = policy.decide(_observation(2, 1_020_000_000, auto_allowed=False))
    reacquired = policy.decide(_observation(3, 1_040_000_000))
    assert (held.yaw_rate_rad_s, held.pitch_rate_rad_s, held.reason) == (0.0, 0.0, "auto_disallowed")
    assert reacquired.reason == "tracking" and reacquired.yaw_rate_rad_s == pytest.approx(0.2)


def test_policy_enforces_rate_acceleration_and_position_limits() -> None:
    policy = ShadowRatePolicy(ShadowRatePolicyConfig(
        yaw_kp=10.0, pitch_kp=10.0, yaw_rate_limit_rad_s=0.5, pitch_rate_limit_rad_s=0.5,
        yaw_accel_limit_rad_s2=2.0, pitch_accel_limit_rad_s2=2.0, nominal_period_s=0.02,
        yaw_position_limits_rad=(-0.5, 0.5),
    ))
    bounded = policy.decide(_observation(1, 1_000_000_000))
    assert bounded.yaw_rate_rad_s == pytest.approx(0.04)
    assert bounded.limits.yaw_rate_limited and bounded.limits.acceleration_limited
    stopped = policy.decide(_observation(2, 1_020_000_000, yaw=0.5))
    assert stopped.reason == "position_limit_hold"
    assert stopped.yaw_rate_rad_s == 0.0 and stopped.limits.position_limited
    assert stopped.pitch_rate_rad_s == pytest.approx(-0.08)


def test_policy_holds_on_identity_change_and_out_of_order_observations() -> None:
    policy = ShadowRatePolicy(ShadowRatePolicyConfig(yaw_kp=1.0, pitch_kp=1.0))
    policy.decide(_observation(2, 2_000_000_000, track_id=7))
    switched = policy.decide(_observation(3, 2_020_000_000, track_id=8))
    out_of_order = policy.decide(_observation(2, 2_040_000_000, track_id=8))
    assert switched.reason == "target_switch_hold"
    assert out_of_order.reason == "observation_out_of_order"


def test_opt_in_los_estimator_produces_absolute_rate_feedforward() -> None:
    kalman = LOSKalmanConfig(acceleration_spectral_density=0.01, measurement_variance_rad2=1e-6)
    policy = ShadowRatePolicy(ShadowRatePolicyConfig(
        yaw_kp=0.0,
        pitch_kp=0.0,
        yaw_los_kalman=kalman,
        pitch_los_kalman=kalman,
        yaw_feedforward_gain=0.5,
        pitch_feedforward_gain=0.5,
        yaw_accel_limit_rad_s2=100.0,
        pitch_accel_limit_rad_s2=100.0,
    ))
    policy.decide(_observation(1, 1_000_000_000, error=(0.0, 0.0)))
    policy.decide(_observation(2, 1_100_000_000, error=(0.1, 0.0)))
    intent = policy.decide(_observation(3, 1_200_000_000, error=(0.2, 0.0)))

    assert intent.yaw_rate_rad_s > 0.3
    assert intent.pitch_rate_rad_s == pytest.approx(0.0)


def test_opt_in_diagnostics_decompose_command_without_changing_intent() -> None:
    kalman = LOSKalmanConfig(acceleration_spectral_density=0.01, measurement_variance_rad2=1e-6)
    policy = ShadowRatePolicy(ShadowRatePolicyConfig(
        yaw_kp=2.0,
        pitch_kp=3.0,
        yaw_kd=0.1,
        pitch_kd=0.2,
        yaw_los_kalman=kalman,
        pitch_los_kalman=kalman,
        yaw_feedforward_gain=0.5,
        pitch_feedforward_gain=0.5,
        yaw_accel_limit_rad_s2=100.0,
        pitch_accel_limit_rad_s2=100.0,
    ))
    observation = _observation(1, 1_000_000_000, error=(0.1, -0.05))
    observation = observation.model_copy(update={
        "source_frame_id": 10,
        "source_time_ns": 900_000_000,
        "source_clock_domain": "pc_monotonic",
        "frame_received_time_ns": 960_000_000,
        "frame_receive_clock_domain": "jetson_monotonic",
        "frame_observed_time_ns": 980_000_000,
        "frame_observation_clock_domain": "jetson_monotonic",
    })

    intent = policy.decide(observation)
    diagnostics = policy.last_diagnostics

    assert diagnostics is not None
    assert diagnostics.intent_sequence == intent.sequence
    assert diagnostics.observation_sequence == observation.sequence
    assert diagnostics.timing.frame_receive_to_tick_ms == pytest.approx(40.0)
    assert diagnostics.timing.frame_receive_to_observe_ms == pytest.approx(20.0)
    assert diagnostics.timing.frame_observe_to_tick_ms == pytest.approx(20.0)
    assert diagnostics.timing.source_to_local_mapping_available is False
    assert diagnostics.yaw.estimator_enabled
    assert diagnostics.yaw.measurement_updated
    assert diagnostics.yaw.measurement_accepted is True
    assert diagnostics.yaw.feedback_term_rad_s == pytest.approx(0.2)
    assert diagnostics.yaw.feedforward_term_rad_s == pytest.approx(0.0)
    assert diagnostics.yaw.final_rate_rad_s == intent.yaw_rate_rad_s
    assert diagnostics.yaw.angle_variance_rad2 is not None


def test_hold_diagnostics_are_zero_and_clear_estimator_state() -> None:
    kalman = LOSKalmanConfig()
    policy = ShadowRatePolicy(ShadowRatePolicyConfig(
        yaw_kp=1.0,
        pitch_kp=1.0,
        yaw_los_kalman=kalman,
        pitch_los_kalman=kalman,
    ))

    intent = policy.decide(_observation(1, 1_000_000_000, target_valid=False, error=None))
    diagnostics = policy.last_diagnostics

    assert intent.reason == "target_invalid"
    assert diagnostics is not None
    assert diagnostics.reason == "target_invalid"
    assert diagnostics.yaw.final_rate_rad_s == 0.0
    assert diagnostics.pitch.final_rate_rad_s == 0.0
    assert diagnostics.yaw.estimated_target_rate_rad_s is None


def test_offline_raw_pd_ablation_uses_gimbal_rate_damping() -> None:
    policy = ShadowRatePolicy(ShadowRatePolicyConfig(
        yaw_kp=2.0,
        pitch_kp=3.0,
        yaw_kd=0.5,
        pitch_kd=0.25,
        raw_gimbal_damping=True,
        yaw_accel_limit_rad_s2=100.0,
        pitch_accel_limit_rad_s2=100.0,
    ))
    observation = _observation(1, 1_000_000_000, error=(0.1, -0.1))
    observation = observation.model_copy(update={
        "gimbal": observation.gimbal.model_copy(update={
            "yaw_rate_rad_s": 0.2,
            "pitch_rate_rad_s": -0.4,
        })
    })

    intent = policy.decide(observation)
    diagnostics = policy.last_diagnostics

    assert intent.yaw_rate_rad_s == pytest.approx(0.1)
    assert intent.pitch_rate_rad_s == pytest.approx(-0.2)
    assert diagnostics is not None
    assert diagnostics.yaw.damping_term_rad_s == pytest.approx(-0.1)
    assert diagnostics.pitch.damping_term_rad_s == pytest.approx(0.1)


def test_mapped_source_time_includes_upstream_frame_latency() -> None:
    kalman = LOSKalmanConfig()
    policy = ShadowRatePolicy(ShadowRatePolicyConfig(
        yaw_kp=0.0, pitch_kp=0.0,
        yaw_los_kalman=kalman, pitch_los_kalman=kalman,
        source_clock_mapping_enabled=True,
    ))
    # PC clock is 100 ms ahead. Frame was captured 50 ms before the Jetson
    # decision, although the snapshot arrived only 5 ms before it.
    observation = _observation(1, 1_000_000_000, error=(0.0, 0.0))
    observation = observation.model_copy(update={
        "source_frame_id": 10,
        "source_time_ns": 1_050_000_000,
        "source_clock_domain": "pc_monotonic",
        "target": observation.target.model_copy(update={"source_age_ms": 5.0}),
    })
    policy.set_source_clock_sample(ClockOffsetSample(
        offset_ns=100_000_000,
        round_trip_ns=2_000_000,
        uncertainty_ns=1_000_000,
        observed_jetson_ns=990_000_000,
    ))
    policy.record_cam_state(CamState(
        frame_id=1, src_ts_ms=0, state_monotonic_ns=940_000_000,
        pan=0.2, tilt=0.0,
    ), received_at_ns=941_000_000)
    policy.record_cam_state(CamState(
        frame_id=2, src_ts_ms=0, state_monotonic_ns=960_000_000,
        pan=0.3, tilt=0.0,
    ), received_at_ns=961_000_000)

    policy.decide(observation)
    diagnostics = policy.last_diagnostics

    assert diagnostics is not None
    assert diagnostics.timing.estimator_time_source == "mapped_pc_source"
    assert diagnostics.timing.source_frame_age_ms == pytest.approx(50.0)
    assert diagnostics.timing.snapshot_receipt_age_ms == pytest.approx(5.0)
    assert diagnostics.timing.source_to_local_mapping_available
    assert diagnostics.yaw.estimate_sample_time_s == pytest.approx(0.95)
    assert diagnostics.yaw.estimated_target_angle_rad == pytest.approx(0.25)
    assert diagnostics.timing.frame_gimbal_pose_age_ms == pytest.approx(10.0)


def test_clock_mapping_loss_disables_estimated_velocity() -> None:
    kalman = LOSKalmanConfig()
    policy = ShadowRatePolicy(ShadowRatePolicyConfig(
        yaw_kp=1.0, pitch_kp=1.0,
        yaw_los_kalman=kalman, pitch_los_kalman=kalman,
        yaw_feedforward_gain=0.5, pitch_feedforward_gain=0.5,
        source_clock_mapping_enabled=True,
    ))
    observation = _observation(1, 1_000_000_000)
    observation = observation.model_copy(update={
        "source_frame_id": 10,
        "source_time_ns": 1_050_000_000,
        "source_clock_domain": "pc_monotonic",
    })

    intent = policy.decide(observation)

    assert intent.reason == "tracking"
    assert policy.last_diagnostics is not None
    assert policy.last_diagnostics.timing.estimator_time_source == "unavailable"
    assert policy.last_diagnostics.yaw.feedforward_term_rad_s == 0.0
    assert policy.last_diagnostics.yaw.estimator_enabled


def test_clock_offset_refresh_does_not_update_same_source_frame_twice() -> None:
    kalman = LOSKalmanConfig()
    policy = ShadowRatePolicy(ShadowRatePolicyConfig(
        yaw_kp=0.0, pitch_kp=0.0,
        yaw_los_kalman=kalman, pitch_los_kalman=kalman,
        source_clock_mapping_enabled=True,
    ))
    for sample_ns, pan in ((940_000_000, 0.1), (960_000_000, 0.2)):
        policy.record_cam_state(CamState(
            frame_id=1, src_ts_ms=0, state_monotonic_ns=sample_ns,
            pan=pan, tilt=0.0,
        ), received_at_ns=sample_ns)
    first = _observation(1, 1_000_000_000, error=(0.0, 0.0))
    first = first.model_copy(update={
        "source_frame_id": 42,
        "source_time_ns": 1_050_000_000,
        "source_clock_domain": "pc_monotonic",
    })
    policy.set_source_clock_sample(ClockOffsetSample(
        offset_ns=100_000_000, round_trip_ns=2_000_000,
        uncertainty_ns=1_000_000, observed_jetson_ns=990_000_000,
    ))
    policy.decide(first)
    assert policy.last_diagnostics is not None
    assert policy.last_diagnostics.yaw.measurement_updated
    second = first.model_copy(update={"sequence": 2, "created_monotonic_ns": 1_020_000_000})
    policy.set_source_clock_sample(ClockOffsetSample(
        offset_ns=99_500_000, round_trip_ns=1_500_000,
        uncertainty_ns=750_000, observed_jetson_ns=1_010_000_000,
    ))
    policy.decide(second)
    assert policy.last_diagnostics is not None
    assert not policy.last_diagnostics.yaw.measurement_updated
