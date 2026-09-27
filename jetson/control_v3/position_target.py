"""Rate command to F5 absolute-axis target; no serial or motor access.

F6 speed commands carry integer RPM, so at 1:1 gearing any rate below
~0.105 rad/s encodes as zero. F5 targets are multi-turn encoder counts
(16384 per motor turn), so this adapter integrates the controller's rate
into a target angle and lets the motor's position loop track it. F5 speed
is still integer RPM; it is chosen per tick as the smallest value that
covers the remaining distance within one nominal tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MAX_F5_SPEED_RPM = 3000


@dataclass(frozen=True)
class PositionTargetConfig:
    travel_limit_rad: float
    max_speed_rpm: int
    tick_s: float
    max_lead_rad: float
    max_step_dt_s: float
    acc: int = 10
    counts_per_rev: int = 16384
    gear_ratio: float = 1.0
    motor_sign: int = 1

    def __post_init__(self) -> None:
        if min(self.travel_limit_rad, self.tick_s, self.max_lead_rad,
               self.max_step_dt_s, self.gear_ratio) <= 0:
            raise ValueError("limits, periods, and gear ratio must be positive")
        if not all(math.isfinite(v) for v in (
                self.travel_limit_rad, self.tick_s, self.max_lead_rad,
                self.max_step_dt_s, self.gear_ratio)):
            raise ValueError("limits, periods, and gear ratio must be finite")
        if not 1 <= self.max_speed_rpm <= MAX_F5_SPEED_RPM:
            raise ValueError("max_speed_rpm must be 1-3000")
        if not 0 <= self.acc <= 255:
            raise ValueError("acc must be 0-255")
        if self.counts_per_rev <= 0:
            raise ValueError("counts_per_rev must be positive")
        if self.motor_sign not in (-1, 1):
            raise ValueError("motor_sign must be +1 or -1")

    @property
    def counts_per_axis_rad(self) -> float:
        return self.counts_per_rev * self.gear_ratio / (2.0 * math.pi)


@dataclass(frozen=True)
class PositionCommand:
    valid: bool
    reason: str
    target_counts: int = 0
    speed_rpm: int = 0
    acc: int = 0
    target_axis_rad: float = 0.0
    measured_axis_rad: float = 0.0
    travel_clamped: bool = False
    lead_clamped: bool = False


class RateToPositionTarget:
    """One-axis rate integrator producing bounded F5 targets.

    Angles are camera-axis radians relative to ``home_counts``; ``motor_sign``
    maps a positive axis rate to the motor's count direction.
    """

    def __init__(self, config: PositionTargetConfig) -> None:
        self.config = config
        self._home_counts: int | None = None
        self._target_axis = 0.0
        self._last_ns: int | None = None

    @property
    def started(self) -> bool:
        return self._home_counts is not None

    def start(self, *, home_counts: int, measured_counts: int, now_ns: int) -> None:
        """Anchor the frame at ``home_counts`` and the target at the measured pose."""

        self._home_counts = int(home_counts)
        self._target_axis = self._axis_rad(measured_counts)
        self._last_ns = int(now_ns)

    def reset(self) -> None:
        self._home_counts = None
        self._target_axis = 0.0
        self._last_ns = None

    def _axis_rad(self, counts: int) -> float:
        assert self._home_counts is not None
        return (self.config.motor_sign * (int(counts) - self._home_counts)
                / self.config.counts_per_axis_rad)

    def _counts(self, axis_rad: float) -> int:
        assert self._home_counts is not None
        return self._home_counts + self.config.motor_sign * round(
            axis_rad * self.config.counts_per_axis_rad)

    def step(self, *, rate_rad_s: float, measured_counts: int, now_ns: int) -> PositionCommand:
        cfg = self.config
        if not self.started or self._last_ns is None:
            return PositionCommand(False, "not_started")
        if not math.isfinite(rate_rad_s):
            return PositionCommand(False, "rate_not_finite")
        if now_ns <= self._last_ns:
            return PositionCommand(False, "time_not_advancing")
        measured = self._axis_rad(measured_counts)
        if abs(measured) > cfg.travel_limit_rad + cfg.max_lead_rad:
            return PositionCommand(False, "measured_outside_travel", measured_axis_rad=measured)

        dt = (now_ns - self._last_ns) / 1e9
        self._last_ns = now_ns
        if dt > cfg.max_step_dt_s:
            # A stalled caller must not catch up in one leap: hold where we are.
            reason = "reanchored_after_gap"
            target = measured
        else:
            reason = "ok"
            target = self._target_axis + rate_rad_s * dt

        lead_clamped = abs(target - measured) > cfg.max_lead_rad
        if lead_clamped:
            target = measured + math.copysign(cfg.max_lead_rad, target - measured)
        travel_clamped = abs(target) > cfg.travel_limit_rad
        if travel_clamped:
            target = math.copysign(cfg.travel_limit_rad, target)
        self._target_axis = target

        target_counts = self._counts(target)
        remaining_motor_revs = abs(target_counts - int(measured_counts)) / cfg.counts_per_rev
        needed_rpm = remaining_motor_revs * 60.0 / cfg.tick_s
        speed_rpm = min(max(math.ceil(needed_rpm - 1e-9), 1), cfg.max_speed_rpm)
        return PositionCommand(
            True, reason,
            target_counts=target_counts,
            speed_rpm=speed_rpm,
            acc=cfg.acc,
            target_axis_rad=target,
            measured_axis_rad=measured,
            travel_clamped=travel_clamped,
            lead_clamped=lead_clamped,
        )
