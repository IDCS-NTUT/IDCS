"""Raw-bearing feedback PID, independent of estimation and actuation.

The derivative term is on measured gimbal position (negative gimbal rate),
not on target-error differences.  A later target-velocity feedforward stage
therefore has its own explicit contribution and cannot be hidden inside D.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from jetson.control_v3.timing import TimingVerdict


@dataclass(frozen=True)
class AxisPIDConfig:
    kp: float
    ki: float
    kd: float
    integral_limit_rad_s: float
    rate_limit_rad_s: float
    acceleration_limit_rad_s2: float

    def __post_init__(self) -> None:
        values = (
            self.kp, self.ki, self.kd, self.integral_limit_rad_s,
            self.rate_limit_rad_s, self.acceleration_limit_rad_s2,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("PID parameters must be finite")
        if min(self.kp, self.ki, self.kd, self.integral_limit_rad_s) < 0:
            raise ValueError("PID gains and integral limit must be nonnegative")
        if self.rate_limit_rad_s <= 0 or self.acceleration_limit_rad_s2 <= 0:
            raise ValueError("rate and acceleration limits must be positive")


@dataclass(frozen=True)
class PIDInput:
    decision_ns: int
    track_id: int
    error_rad: tuple[float, float]
    gimbal_rate_rad_s: tuple[float, float]
    timing: TimingVerdict
    safety_allowed: bool
    gimbal_valid: bool


@dataclass(frozen=True)
class AxisPIDDecision:
    proportional_rad_s: float = 0.0
    integral_rad_s: float = 0.0
    derivative_rad_s: float = 0.0
    pre_limit_rad_s: float = 0.0
    rate_limited_rad_s: float = 0.0
    final_rad_s: float = 0.0
    rate_limited: bool = False
    acceleration_limited: bool = False


@dataclass(frozen=True)
class PIDDecision:
    reason: str
    yaw: AxisPIDDecision = AxisPIDDecision()
    pitch: AxisPIDDecision = AxisPIDDecision()


class BasicPID:
    """Deterministic two-axis baseline. It has no socket or motor access."""

    def __init__(self, yaw: AxisPIDConfig, pitch: AxisPIDConfig) -> None:
        self._config = (yaw, pitch)
        self.reset()

    def reset(self) -> None:
        self._integral = [0.0, 0.0]
        self._last_rate = [0.0, 0.0]
        self._last_ns: int | None = None
        self._track_id: int | None = None

    def _axis(self, axis: int, error: float, measured_rate: float, dt_s: float) -> AxisPIDDecision:
        config = self._config[axis]
        proportional = config.kp * error
        derivative = -config.kd * measured_rate
        trial_integral = max(
            -config.integral_limit_rad_s,
            min(config.integral_limit_rad_s, self._integral[axis] + config.ki * error * dt_s),
        )
        trial = proportional + trial_integral + derivative
        # Conditional integration: do not wind up while pushing farther into
        # the rate bound. Recompute demand after freezing the integral.
        if abs(trial) > config.rate_limit_rad_s and trial * error > 0:
            integral = self._integral[axis]
        else:
            integral = trial_integral
        self._integral[axis] = integral
        pre_limit = proportional + integral + derivative
        rate_limited = max(-config.rate_limit_rad_s, min(config.rate_limit_rad_s, pre_limit))
        max_delta = config.acceleration_limit_rad_s2 * dt_s
        final = max(
            self._last_rate[axis] - max_delta,
            min(self._last_rate[axis] + max_delta, rate_limited),
        )
        self._last_rate[axis] = final
        return AxisPIDDecision(
            proportional, integral, derivative, pre_limit, rate_limited, final,
            rate_limited != pre_limit, final != rate_limited,
        )

    def decide(self, sample: PIDInput) -> PIDDecision:
        if not sample.safety_allowed or not sample.gimbal_valid or not sample.timing.valid:
            self.reset()
            reason = (
                "safety_hold" if not sample.safety_allowed else
                "gimbal_invalid" if not sample.gimbal_valid else sample.timing.reason
            )
            return PIDDecision(reason)
        if not all(math.isfinite(value) for value in (*sample.error_rad, *sample.gimbal_rate_rad_s)):
            self.reset()
            return PIDDecision("nonfinite_input")
        if self._last_ns is not None and sample.decision_ns <= self._last_ns:
            self.reset()
            return PIDDecision("nonmonotonic_decision_time")
        if self._track_id is not None and sample.track_id != self._track_id:
            self.reset()
            return PIDDecision("target_switch_hold")
        dt_s = 0.0 if self._last_ns is None else (sample.decision_ns - self._last_ns) / 1e9
        self._last_ns = sample.decision_ns
        self._track_id = sample.track_id
        return PIDDecision(
            "tracking",
            self._axis(0, sample.error_rad[0], sample.gimbal_rate_rad_s[0], dt_s),
            self._axis(1, sample.error_rad[1], sample.gimbal_rate_rad_s[1], dt_s),
        )
