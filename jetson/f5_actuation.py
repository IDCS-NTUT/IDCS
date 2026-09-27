"""Plan F5 absolute-axis serial commands from accepted rate intents.

Pure planning for ``gimbal_bridge``'s ``f5_position`` actuation mode: no
serial, ZMQ, or clock access. Each moving intent becomes one F5 target per
axis (``RateToPositionTarget``); anything that cannot be planned safely
becomes an immediate F5 stop on every axis, sent at critical priority so the
serial service treats it as an emergency.

Travel is bounded by the tighter of ``travel_limit_rad`` around the encoder
position seen at the first fresh reading ("home") and the bridge's hard
angle limits. F5 has no firmware command expiry like timed F6; if the
bridge stops sending, the motor finishes its current target, which is at
most ``max_lead_rad`` beyond the last measured position, and holds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from common.gimbal.mks_servo42_rs485 import MksServo42Axis
from jetson.control_v3.position_target import (
    MAX_F5_SPEED_RPM,
    PositionTargetConfig,
    RateToPositionTarget,
)


@dataclass(frozen=True)
class F5AxisSpec:
    name: str
    addr: int
    motor_sign: int
    camstate_sign: int
    gear_ratio: float
    counts_per_rev: int
    accel: int
    rate_limit_rad_s: float
    hard_min_rad: Optional[float] = None
    hard_max_rad: Optional[float] = None

    @property
    def counts_per_axis_rad(self) -> float:
        return self.counts_per_rev * self.gear_ratio / (2.0 * math.pi)

    @property
    def max_speed_rpm(self) -> int:
        """Same integer-RPM ceiling the F6 path encodes for this rate limit."""
        rpm = self.rate_limit_rad_s * 60.0 / (2.0 * math.pi) * self.gear_ratio
        return min(max(1, math.floor(rpm + 1e-9)), MAX_F5_SPEED_RPM)


@dataclass(frozen=True)
class F5PlannerConfig:
    travel_limit_rad: float
    max_lead_rad: float
    tick_s: float
    max_step_dt_s: float
    max_encoder_age_s: float

    def __post_init__(self) -> None:
        values = (self.travel_limit_rad, self.max_lead_rad, self.tick_s,
                  self.max_step_dt_s, self.max_encoder_age_s)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("F5 planner limits and periods must be positive and finite")


@dataclass(frozen=True)
class EncoderReading:
    counts: int
    age_s: float
    timing_ok: bool = True


@dataclass(frozen=True)
class F5AxisCommand:
    axis: str
    addr: int
    payload: tuple[int, ...]
    priority: str
    target_counts: Optional[int]
    speed_rpm: int
    requested_rate_rad_s: float


@dataclass(frozen=True)
class F5Plan:
    stop: bool
    reason: str
    commands: tuple[F5AxisCommand, ...]


class F5IntentPlanner:
    def __init__(self, axes: Sequence[F5AxisSpec], config: F5PlannerConfig) -> None:
        names = [axis.name for axis in axes]
        if not axes or len(set(names)) != len(names):
            raise ValueError("F5 axes must be non-empty with unique names")
        for axis in axes:
            if axis.motor_sign not in (-1, 1) or axis.camstate_sign not in (-1, 1):
                raise ValueError(f"{axis.name}: signs must be +1 or -1")
            if not (math.isfinite(axis.rate_limit_rad_s) and axis.rate_limit_rad_s > 0):
                raise ValueError(f"{axis.name}: rate limit must be positive")
        self.axes = tuple(axes)
        self.config = config
        self._home: dict[str, int] = {}
        self._adapters: dict[str, RateToPositionTarget] = {}

    def home_counts(self, axis: str) -> Optional[int]:
        return self._home.get(axis)

    def _bounds(self, axis: F5AxisSpec, home_counts: int) -> tuple[float, float]:
        """Adapter-frame bounds: travel around home within the hard limits.

        Adapter angle a = motor_sign * (counts - home) / k; the bridge's
        CamState angle c = camstate_sign * counts / k.
        """

        k = axis.counts_per_axis_rad
        s = axis.motor_sign * axis.camstate_sign
        h = axis.motor_sign * home_counts / k
        c_min = -math.inf if axis.hard_min_rad is None else axis.hard_min_rad
        c_max = math.inf if axis.hard_max_rad is None else axis.hard_max_rad
        if s > 0:
            hard_lo, hard_hi = c_min - h, c_max - h
        else:
            hard_lo, hard_hi = -c_max - h, -c_min - h
        travel = self.config.travel_limit_rad
        return max(-travel, hard_lo), min(travel, hard_hi)

    def _adapter(self, axis: F5AxisSpec, home_counts: int) -> Optional[RateToPositionTarget]:
        lo, hi = self._bounds(axis, home_counts)
        try:
            config = PositionTargetConfig(
                min_axis_rad=lo,
                max_axis_rad=hi,
                max_speed_rpm=axis.max_speed_rpm,
                tick_s=self.config.tick_s,
                max_lead_rad=self.config.max_lead_rad,
                max_step_dt_s=self.config.max_step_dt_s,
                acc=axis.accel,
                counts_per_rev=axis.counts_per_rev,
                gear_ratio=axis.gear_ratio,
                motor_sign=axis.motor_sign,
            )
        except ValueError:
            return None
        return RateToPositionTarget(config)

    def stop_plan(self, reason: str) -> F5Plan:
        """Immediate F5 stop (acc 0) on every axis; next motion re-anchors."""

        for adapter in self._adapters.values():
            adapter.reset()
        payload = MksServo42Axis._encode_absolute_axis_stop_payload(0)
        return F5Plan(True, reason, tuple(
            F5AxisCommand(axis.name, axis.addr, payload, "critical", None, 0, 0.0)
            for axis in self.axes
        ))

    def plan(
        self,
        rates: Mapping[str, float],
        encoders: Mapping[str, Optional[EncoderReading]],
        *,
        now_ns: int,
    ) -> F5Plan:
        values = [float(rates.get(axis.name, 0.0)) for axis in self.axes]
        if not all(math.isfinite(v) for v in values):
            return self.stop_plan("non_finite_rate")
        if all(abs(v) <= 1e-12 for v in values):
            return self.stop_plan("zero_rate")

        commands = []
        for axis, rate in zip(self.axes, values):
            reading = encoders.get(axis.name)
            if (
                reading is None
                or not reading.timing_ok
                or not 0.0 <= reading.age_s <= self.config.max_encoder_age_s
            ):
                return self.stop_plan(f"encoder_unfresh:{axis.name}")
            home = self._home.get(axis.name)
            if home is None:
                adapter = self._adapter(axis, reading.counts)
                if adapter is None:
                    return self.stop_plan(f"home_outside_hard_limits:{axis.name}")
                self._home[axis.name] = home = reading.counts
                self._adapters[axis.name] = adapter
            adapter = self._adapters[axis.name]
            if not adapter.started:
                # Integration starts from the next intent; hold where we are now.
                adapter.start(home_counts=home, measured_counts=reading.counts, now_ns=now_ns)
                commands.append(F5AxisCommand(
                    axis.name,
                    axis.addr,
                    MksServo42Axis._encode_absolute_axis_payload(
                        reading.counts, 1, axis.accel),
                    "high",
                    reading.counts,
                    1,
                    rate,
                ))
                continue
            step = adapter.step(rate_rad_s=rate, measured_counts=reading.counts, now_ns=now_ns)
            if not step.valid:
                return self.stop_plan(f"{step.reason}:{axis.name}")
            commands.append(F5AxisCommand(
                axis.name,
                axis.addr,
                MksServo42Axis._encode_absolute_axis_payload(
                    step.target_counts, step.speed_rpm, step.acc),
                "high",
                step.target_counts,
                step.speed_rpm,
                rate,
            ))
        return F5Plan(False, "tracking", tuple(commands))
