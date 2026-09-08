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
        if any(not math.isfinite(value) for value in (self.yaw_kp, self.pitch_kp, self.yaw_kd, self.pitch_kd)):
            raise ValueError("gains must be finite")
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

        yaw_error, pitch_error = observation.target.bearing_error_rad
        yaw_rate, pitch_rate = observation.target.bearing_rate_rad_s or (0.0, 0.0)
        desired_yaw = self._config.yaw_kp * yaw_error + self._config.yaw_kd * yaw_rate
        desired_pitch = self._config.pitch_kp * pitch_error + self._config.pitch_kd * pitch_rate
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
