"""Deterministic, non-actuating rate policy for controller-overhaul replay.

This module deliberately has no ZMQ, serial, gimbal, or wall-clock dependency.
It consumes one validated :class:`ControlObservation` and emits a short-lived
``shadow`` :class:`ControlIntent`.  It is suitable for replay and parity work,
not for hardware command authority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from common.schemas import ControlIntent, ControlIntentLimits, ControlObservation
from jetson.los_kalman import AxisLOSKalman, LOSKalmanConfig


@dataclass(frozen=True)
class ShadowRatePolicyConfig:
    """Explicit, local-monotonic bounds for :class:`ShadowRatePolicy`."""

    yaw_kp: float
    pitch_kp: float
    yaw_kd: float = 0.0
    pitch_kd: float = 0.0
    yaw_rate_limit_rad_s: float = 1.0
    pitch_rate_limit_rad_s: float = 1.0
    yaw_accel_limit_rad_s2: float = 2.0
    pitch_accel_limit_rad_s2: float = 2.0
    nominal_period_s: float = 0.02
    valid_for_ns: int = 50_000_000
    yaw_position_limits_rad: Optional[Tuple[float, float]] = None
    pitch_position_limits_rad: Optional[Tuple[float, float]] = None
    yaw_los_kalman: Optional[LOSKalmanConfig] = None
    pitch_los_kalman: Optional[LOSKalmanConfig] = None
    yaw_feedforward_gain: float = 0.0
    pitch_feedforward_gain: float = 0.0

    def __post_init__(self) -> None:
        finite_positive = (
            self.yaw_rate_limit_rad_s,
            self.pitch_rate_limit_rad_s,
            self.yaw_accel_limit_rad_s2,
            self.pitch_accel_limit_rad_s2,
            self.nominal_period_s,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in finite_positive):
            raise ValueError("rate, acceleration, and period limits must be finite and > 0")
        if any(not math.isfinite(value) for value in (
            self.yaw_kp, self.pitch_kp, self.yaw_kd, self.pitch_kd,
            self.yaw_feedforward_gain, self.pitch_feedforward_gain,
        )):
            raise ValueError("gains must be finite")
        if (self.yaw_los_kalman is None) != (self.pitch_los_kalman is None):
            raise ValueError("yaw and pitch LOS Kalman configs must be enabled together")
        if self.valid_for_ns <= 0:
            raise ValueError("valid_for_ns must be > 0")
        for bounds in (self.yaw_position_limits_rad, self.pitch_position_limits_rad):
            if bounds is not None and (not all(math.isfinite(value) for value in bounds) or bounds[0] >= bounds[1]):
                raise ValueError("position limits must be finite (min, max) pairs")


class ShadowRatePolicy:
    """Bounded PD plus bearing-rate-feedforward policy with fail-safe holds.

    Positive yaw/pitch command is defined to increase the matching gimbal angle.
    ``bearing_rate_rad_s`` is the selected target's camera-relative bearing
    rate, so it is added directly as the derivative/feedforward term.  This
    convention is explicit for replay; sign/plant validation is still required
    before a live policy is designed.
    """

    def __init__(self, config: ShadowRatePolicyConfig) -> None:
        self._config = config
        self._intent_sequence = 0
        self._last_observation_sequence = -1
        self._last_issued_ns: Optional[int] = None
        self._last_rates = (0.0, 0.0)
        self._last_track_id: Optional[int] = None
        self._yaw_los = AxisLOSKalman(config.yaw_los_kalman) if config.yaw_los_kalman is not None else None
        self._pitch_los = AxisLOSKalman(config.pitch_los_kalman) if config.pitch_los_kalman is not None else None
        self._last_los_source: Optional[tuple[Optional[int], int]] = None

    def _intent(self, observation: ControlObservation, *, yaw: float, pitch: float,
                reason: str, limits: Optional[ControlIntentLimits] = None) -> ControlIntent:
        self._intent_sequence += 1
        issued = observation.created_monotonic_ns
        return ControlIntent(
            sequence=self._intent_sequence,
            observation_sequence=observation.sequence,
            issued_monotonic_ns=issued,
            valid_until_monotonic_ns=issued + self._config.valid_for_ns,
            mode="shadow",
            yaw_rate_rad_s=yaw,
            pitch_rate_rad_s=pitch,
            limits=limits or ControlIntentLimits(),
            reason=reason,
        )

    def _reset(self) -> None:
        self._last_issued_ns = None
        self._last_rates = (0.0, 0.0)
        self._last_track_id = None
        self._last_los_source = None
        if self._yaw_los is not None:
            self._yaw_los.reset()
        if self._pitch_los is not None:
            self._pitch_los.reset()

    def _estimated_target_terms(
        self,
        observation: ControlObservation,
        raw_error: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return current bearing error and absolute LOS rate for opt-in KF mode."""

        if self._yaw_los is None or self._pitch_los is None:
            return raw_error, observation.target.bearing_rate_rad_s or (0.0, 0.0)
        now_s = observation.created_monotonic_ns / 1_000_000_000.0
        target_age_s = (observation.target.source_age_ms or 0.0) / 1000.0
        target_sample_s = now_s - target_age_s
        source_key = (observation.source_frame_id, int(round(target_sample_s * 1_000_000_000.0)))
        yaw_position = observation.gimbal.yaw_rad or 0.0
        pitch_position = observation.gimbal.pitch_rad or 0.0
        yaw_rate = observation.gimbal.yaw_rate_rad_s or 0.0
        pitch_rate = observation.gimbal.pitch_rate_rad_s or 0.0
        gimbal_age_s = (observation.gimbal.sample_age_ms or 0.0) / 1000.0
        gimbal_sample_s = now_s - gimbal_age_s
        if source_key != self._last_los_source:
            yaw_at_target = yaw_position + yaw_rate * (target_sample_s - gimbal_sample_s)
            pitch_at_target = pitch_position + pitch_rate * (target_sample_s - gimbal_sample_s)
            try:
                self._yaw_los.update(yaw_at_target + raw_error[0], sample_time_s=target_sample_s)
                self._pitch_los.update(pitch_at_target + raw_error[1], sample_time_s=target_sample_s)
            except ValueError:
                # A regressed local sample timestamp cannot be fused.  Reset
                # both axes atomically and treat this sample as reacquisition.
                self._yaw_los.reset()
                self._pitch_los.reset()
                self._yaw_los.update(yaw_at_target + raw_error[0], sample_time_s=target_sample_s)
                self._pitch_los.update(pitch_at_target + raw_error[1], sample_time_s=target_sample_s)
            self._last_los_source = source_key
        yaw_estimate = self._yaw_los.estimate(query_time_s=now_s)
        pitch_estimate = self._pitch_los.estimate(query_time_s=now_s)
        if yaw_estimate is None or pitch_estimate is None:
            return raw_error, (0.0, 0.0)
        return (
            (yaw_estimate.angle_rad - yaw_position, pitch_estimate.angle_rad - pitch_position),
            (yaw_estimate.rate_rad_s, pitch_estimate.rate_rad_s),
        )

    def _hold(self, observation: ControlObservation, reason: str) -> ControlIntent:
        self._reset()
        return self._intent(observation, yaw=0.0, pitch=0.0, reason=reason)

    @staticmethod
    def _clamp(value: float, limit: float) -> tuple[float, bool]:
        clipped = max(-limit, min(limit, value))
        return clipped, clipped != value

    @staticmethod
    def _outward(position: float, command: float, limits: Optional[Tuple[float, float]]) -> bool:
        return limits is not None and ((position <= limits[0] and command < 0.0) or (position >= limits[1] and command > 0.0))

    def _acceleration_limit(self, target: float, previous: float, limit: float, now_ns: int) -> tuple[float, bool]:
        if self._last_issued_ns is None:
            elapsed_s = self._config.nominal_period_s
        else:
            elapsed_s = max(self._config.nominal_period_s, (now_ns - self._last_issued_ns) / 1_000_000_000.0)
        maximum_delta = limit * elapsed_s
        bounded = max(previous - maximum_delta, min(previous + maximum_delta, target))
        return bounded, bounded != target

    def decide(self, observation: ControlObservation) -> ControlIntent:
        """Return one bounded shadow intent for ``observation``.

        Invalid, stale, manual, emergency, identity-switch, or out-of-order
        inputs yield an immediate zero-rate hold and reset limiter memory.
        """

        if observation.sequence <= self._last_observation_sequence:
            return self._hold(observation, "observation_out_of_order")
        self._last_observation_sequence = observation.sequence

        safety = observation.safety
        if safety.emergency_active:
            return self._hold(observation, "emergency_active")
        if safety.manual_active:
            return self._hold(observation, "manual_active")
        if not safety.valid:
            return self._hold(observation, "safety_invalid")
        if not safety.auto_allowed:
            return self._hold(observation, "auto_disallowed")
        if not observation.target.valid:
            return self._hold(observation, "target_invalid")
        if not observation.gimbal.valid:
            return self._hold(observation, "gimbal_invalid")
        if observation.target.bearing_error_rad is None:
            return self._hold(observation, "target_incomplete")

        track_id = observation.target.track_id
        if self._last_track_id is not None and track_id is not None and track_id != self._last_track_id:
            return self._hold(observation, "target_switch_hold")

        raw_error = observation.target.bearing_error_rad
        (yaw_error, pitch_error), (yaw_rate, pitch_rate) = self._estimated_target_terms(observation, raw_error)
        if self._yaw_los is None:
            # Compatibility path: historical ``kd`` multiplies the externally
            # supplied camera-relative rate.
            desired_yaw = self._config.yaw_kp * yaw_error + self._config.yaw_kd * yaw_rate
            desired_pitch = self._config.pitch_kp * pitch_error + self._config.pitch_kd * pitch_rate
        else:
            desired_yaw = (
                self._config.yaw_kp * yaw_error
                - self._config.yaw_kd * (observation.gimbal.yaw_rate_rad_s or 0.0)
                + self._config.yaw_feedforward_gain * yaw_rate
            )
            desired_pitch = (
                self._config.pitch_kp * pitch_error
                - self._config.pitch_kd * (observation.gimbal.pitch_rate_rad_s or 0.0)
                + self._config.pitch_feedforward_gain * pitch_rate
            )
        desired_yaw, yaw_limited = self._clamp(desired_yaw, self._config.yaw_rate_limit_rad_s)
        desired_pitch, pitch_limited = self._clamp(desired_pitch, self._config.pitch_rate_limit_rad_s)

        yaw_position_limited = self._outward(
            observation.gimbal.yaw_rad or 0.0, desired_yaw, self._config.yaw_position_limits_rad
        )
        if yaw_position_limited:
            desired_yaw = 0.0
            position_limited = True
        else:
            position_limited = False
        pitch_position_limited = self._outward(
            observation.gimbal.pitch_rad or 0.0, desired_pitch, self._config.pitch_position_limits_rad
        )
        if pitch_position_limited:
            desired_pitch = 0.0
            position_limited = True

        # A blocked axis must stop immediately, but the other axis still has
        # to honor its own acceleration bound.
        if yaw_position_limited:
            yaw, yaw_accel_limited = 0.0, False
        else:
            yaw, yaw_accel_limited = self._acceleration_limit(
                desired_yaw, self._last_rates[0], self._config.yaw_accel_limit_rad_s2, observation.created_monotonic_ns
            )
        if pitch_position_limited:
            pitch, pitch_accel_limited = 0.0, False
        else:
            pitch, pitch_accel_limited = self._acceleration_limit(
                desired_pitch, self._last_rates[1], self._config.pitch_accel_limit_rad_s2, observation.created_monotonic_ns
            )
        self._last_rates = (yaw, pitch)
        self._last_issued_ns = observation.created_monotonic_ns
        self._last_track_id = track_id
        return self._intent(
            observation, yaw=yaw, pitch=pitch, reason="position_limit_hold" if position_limited else "tracking",
            limits=ControlIntentLimits(yaw_rate_limited=yaw_limited, pitch_rate_limited=pitch_limited,
                                       acceleration_limited=yaw_accel_limited or pitch_accel_limited,
                                       position_limited=position_limited),
        )
