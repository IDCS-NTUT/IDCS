from __future__ import annotations

import math

import pytest

from common.gimbal.mks_servo42_rs485 import MksServo42Axis
from jetson.control_v3.position_target import (
    PositionTargetConfig,
    RateToPositionTarget,
)

TICK_NS = 20_000_000
START_NS = 1_000_000_000
COUNTS_PER_RAD = 16384 / (2.0 * math.pi)


def _config(**overrides) -> PositionTargetConfig:
    values = dict(
        min_axis_rad=-0.15,
        max_axis_rad=0.15,
        max_speed_rpm=2,
        tick_s=0.02,
        max_lead_rad=0.02,
        max_step_dt_s=0.1,
    )
    values.update(overrides)
    return PositionTargetConfig(**values)


class _F5Plant:
    """Motor that follows its latest F5 target at the commanded speed."""

    def __init__(self, counts: float = 0.0, *, stuck: bool = False) -> None:
        self.counts = counts
        self.target = counts
        self.speed_rpm = 0
        self.stuck = stuck

    @property
    def measured(self) -> int:
        return round(self.counts)

    def command(self, target_counts: int, speed_rpm: int) -> None:
        self.target = target_counts
        self.speed_rpm = speed_rpm

    def advance(self, seconds: float, substep_s: float = 0.001) -> None:
        for _ in range(round(seconds / substep_s)):
            if self.stuck:
                continue
            max_move = self.speed_rpm / 60.0 * 16384 * substep_s
            error = self.target - self.counts
            self.counts += math.copysign(min(abs(error), max_move), error)


def _run(adapter, plant, rates, *, start_ns=START_NS, tick_ns=TICK_NS):
    commands = []
    now = start_ns
    for rate in rates:
        now += tick_ns
        plant.advance(tick_ns / 1e9)
        cmd = adapter.step(rate_rad_s=rate, measured_counts=plant.measured, now_ns=now)
        assert cmd.valid, cmd.reason
        plant.command(cmd.target_counts, cmd.speed_rpm)
        commands.append(cmd)
    plant.advance(tick_ns / 1e9)
    return commands


def _started(config=None, plant=None):
    plant = plant or _F5Plant()
    adapter = RateToPositionTarget(config or _config())
    adapter.start(home_counts=0, measured_counts=plant.measured, now_ns=START_NS)
    return adapter, plant


def test_sub_rpm_rate_moves_that_f6_would_encode_as_zero() -> None:
    rate = 0.01  # rad/s; about 0.1 RPM at 1:1
    assert MksServo42Axis.quantized_speed_rad_s(rate, 1.0) == 0.0
    adapter, plant = _started()
    _run(adapter, plant, [rate] * 250)  # 5 s
    assert plant.counts / COUNTS_PER_RAD == pytest.approx(0.05, abs=2 / COUNTS_PER_RAD)


@pytest.mark.parametrize("rate", [0.003, 0.02, 0.07, -0.15])
def test_average_rate_is_preserved_without_drift(rate) -> None:
    adapter, plant = _started()
    seconds = min(4.0, 0.12 / abs(rate))  # stay inside the travel limit
    ticks = round(seconds / 0.02)
    _run(adapter, plant, [rate] * ticks)
    expected = rate * ticks * 0.02
    assert plant.counts / COUNTS_PER_RAD == pytest.approx(expected, abs=2 / COUNTS_PER_RAD)


def test_speed_is_smallest_integer_rpm_covering_one_tick() -> None:
    adapter, plant = _started(_config(max_speed_rpm=50))
    slow = _run(adapter, plant, [0.01] * 5)
    assert {cmd.speed_rpm for cmd in slow} == {1}
    # 1 rad/s is ~9.55 RPM, so a steady ramp needs 10 RPM per tick.
    adapter, plant = _started(_config(max_speed_rpm=50, max_lead_rad=0.5))
    fast = _run(adapter, plant, [1.0] * 5)
    assert all(9 <= cmd.speed_rpm <= 11 for cmd in fast[1:])


def test_speed_never_exceeds_cap() -> None:
    adapter, plant = _started(_config(max_speed_rpm=2, max_lead_rad=0.1))
    commands = _run(adapter, plant, [5.0] * 20)
    assert max(cmd.speed_rpm for cmd in commands) == 2


def test_target_never_leaves_travel_limit() -> None:
    adapter, plant = _started(_config(max_speed_rpm=20))
    commands = _run(adapter, plant, [0.5] * 100 + [-0.5] * 200)
    limit_counts = 0.15 * COUNTS_PER_RAD
    assert all(abs(cmd.target_counts) <= math.ceil(limit_counts) for cmd in commands)
    assert any(cmd.travel_clamped for cmd in commands)
    assert abs(plant.counts) <= limit_counts + 1


def test_stalled_motor_limits_target_lead_and_does_not_leap_when_freed() -> None:
    adapter, plant = _started(plant=_F5Plant(stuck=True))
    commands = _run(adapter, plant, [0.1] * 100)
    assert all(abs(cmd.target_axis_rad - cmd.measured_axis_rad) <= 0.02 + 1e-12 for cmd in commands)
    assert commands[-1].lead_clamped


def test_gap_reanchors_to_measured_instead_of_catching_up() -> None:
    adapter, plant = _started()
    _run(adapter, plant, [0.05] * 10)
    held = plant.measured
    late = adapter.step(rate_rad_s=0.05, measured_counts=held,
                        now_ns=START_NS + 11 * TICK_NS + 500_000_000)
    assert late.valid and late.reason == "reanchored_after_gap"
    assert late.target_counts == held


def test_motor_sign_inverts_count_direction() -> None:
    adapter, plant = _started(_config(motor_sign=-1))
    _run(adapter, plant, [0.05] * 50)
    assert plant.counts < 0
    assert adapter.step(rate_rad_s=0.0, measured_counts=plant.measured,
                        now_ns=START_NS + 60 * TICK_NS).measured_axis_rad > 0


def test_home_offset_is_respected() -> None:
    plant = _F5Plant(counts=100_000)
    adapter = RateToPositionTarget(_config())
    adapter.start(home_counts=100_000, measured_counts=plant.measured, now_ns=START_NS)
    commands = _run(adapter, plant, [1.0] * 100)
    assert max(cmd.target_counts for cmd in commands) <= 100_000 + math.ceil(0.15 * COUNTS_PER_RAD)


def test_invalid_inputs_are_rejected_without_changing_target() -> None:
    adapter = RateToPositionTarget(_config())
    assert adapter.step(rate_rad_s=0.0, measured_counts=0, now_ns=START_NS).reason == "not_started"
    adapter.start(home_counts=0, measured_counts=0, now_ns=START_NS)
    assert adapter.step(rate_rad_s=math.nan, measured_counts=0, now_ns=START_NS + TICK_NS).reason == "rate_not_finite"
    assert adapter.step(rate_rad_s=0.0, measured_counts=0, now_ns=START_NS).reason == "time_not_advancing"
    outside = round(0.2 * COUNTS_PER_RAD)
    assert adapter.step(rate_rad_s=0.0, measured_counts=outside,
                        now_ns=START_NS + TICK_NS).reason == "measured_outside_travel"


@pytest.mark.parametrize(
    "overrides",
    [dict(max_speed_rpm=0), dict(max_speed_rpm=3001), dict(tick_s=0.0),
     dict(max_axis_rad=math.inf), dict(motor_sign=0), dict(acc=256),
     dict(min_axis_rad=0.01), dict(max_axis_rad=-0.01), dict(min_axis_rad=0.0, max_axis_rad=0.0)],
)
def test_config_rejects_unsafe_values(overrides) -> None:
    with pytest.raises(ValueError):
        _config(**overrides)


def test_asymmetric_bounds_clamp_each_side_separately() -> None:
    adapter, plant = _started(_config(min_axis_rad=-0.02, max_axis_rad=0.05, max_speed_rpm=20))
    up = _run(adapter, plant, [0.5] * 30)
    assert max(cmd.target_axis_rad for cmd in up) == pytest.approx(0.05)
    down = _run(adapter, plant, [-0.5] * 60, start_ns=START_NS + 30 * TICK_NS)
    assert min(cmd.target_axis_rad for cmd in down) == pytest.approx(-0.02)
