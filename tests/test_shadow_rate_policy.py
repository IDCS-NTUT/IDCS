from __future__ import annotations

import pytest

from common.schemas import (
    ControlGimbalObservation,
    ControlObservation,
    ControlSafetyObservation,
    ControlTargetObservation,
    ControlTransportObservation,
)
from jetson.shadow_rate_policy import ShadowRatePolicy, ShadowRatePolicyConfig


def _observation(sequence: int, timestamp: int, *, track_id: int | None = 7,
                 error: tuple[float, float] | None = (0.2, -0.1),
                 rate: tuple[float, float] | None = None, auto_allowed: bool = True,
                 target_valid: bool = True, yaw: float = 0.0, pitch: float = 0.0) -> ControlObservation:
    return ControlObservation(
        sequence=sequence,
        created_monotonic_ns=timestamp,
        target=ControlTargetObservation(valid=target_valid, track_id=track_id, confidence=0.9,
                                        bearing_error_rad=error, bearing_rate_rad_s=rate),
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
