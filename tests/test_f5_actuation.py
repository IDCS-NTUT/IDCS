from __future__ import annotations

import math
import time

import pytest

from jetson.f5_actuation import (
    EncoderReading,
    F5AxisSpec,
    F5IntentPlanner,
    F5PlannerConfig,
)
from tools import serial_io_service

K = 16384 / (2.0 * math.pi)  # counts per rad at 1:1
T0 = 1_000_000_000
TICK = 20_000_000


def _axis(name="yaw", addr=1, **overrides) -> F5AxisSpec:
    values = dict(name=name, addr=addr, motor_sign=1, camstate_sign=1,
                  gear_ratio=1.0, counts_per_rev=16384, accel=10,
                  rate_limit_rad_s=0.2)
    values.update(overrides)
    return F5AxisSpec(**values)


def _planner(*axes, **overrides) -> F5IntentPlanner:
    values = dict(travel_limit_rad=0.15, max_lead_rad=0.01, tick_s=0.02,
                  max_step_dt_s=0.1, max_encoder_age_s=0.1)
    values.update(overrides)
    return F5IntentPlanner(axes or (_axis(),), F5PlannerConfig(**values))


def _fresh(counts: int) -> EncoderReading:
    return EncoderReading(counts=counts, age_s=0.01)


def _target(payload) -> int:
    return int.from_bytes(bytes(payload[3:7]), "big", signed=True)


def _speed(payload) -> int:
    return (payload[0] << 8) | payload[1]


def _serial_command(cmd) -> serial_io_service.SerialCommand:
    return serial_io_service.SerialCommand(
        cmd_id=f"f5:{cmd.axis}", func="F5", addr=cmd.addr, payload=cmd.payload,
        expect_reply=False, expected_len=None, priority=cmd.priority,
        target="gimbal", timeout_ms=None, retry=None,
        sent_ts_ms=int(time.time() * 1000), enqueued_monotonic_ns=time.monotonic_ns())


def test_zero_rates_stop_every_axis_as_serial_emergency() -> None:
    planner = _planner(_axis(), _axis("pitch_a", addr=2))
    plan = planner.plan({"yaw": 0.0, "pitch_a": 0.0},
                        {"yaw": _fresh(0), "pitch_a": _fresh(0)}, now_ns=T0)
    assert plan.stop and plan.reason == "zero_rate"
    assert [cmd.addr for cmd in plan.commands] == [1, 2]
    for cmd in plan.commands:
        assert cmd.payload == (0, 0, 0, 0, 0, 0, 0)
        assert serial_io_service._is_emergency_command(_serial_command(cmd))


def test_tracking_commands_are_coalescible_not_emergencies() -> None:
    planner = _planner()
    plan = planner.plan({"yaw": 0.05}, {"yaw": _fresh(500)}, now_ns=T0)
    cmd = plan.commands[0]
    assert not plan.stop and cmd.priority == "high"
    serial_cmd = _serial_command(cmd)
    assert not serial_io_service._is_emergency_command(serial_cmd)
    assert serial_io_service._coalesce_key(serial_cmd) is not None


def test_first_motion_holds_then_integrates_from_measured_home() -> None:
    planner = _planner()
    first = planner.plan({"yaw": 0.1}, {"yaw": _fresh(500)}, now_ns=T0)
    assert _target(first.commands[0].payload) == 500
    assert planner.home_counts("yaw") == 500
    second = planner.plan({"yaw": 0.1}, {"yaw": _fresh(500)}, now_ns=T0 + TICK)
    assert _target(second.commands[0].payload) == 500 + round(0.1 * 0.02 * K)
    assert _speed(second.commands[0].payload) == 1


@pytest.mark.parametrize(
    ("reading", "reason"),
    [(None, "encoder_unfresh:yaw"),
     (EncoderReading(0, age_s=0.5), "encoder_unfresh:yaw"),
     (EncoderReading(0, age_s=-0.01), "encoder_unfresh:yaw"),
     (EncoderReading(0, age_s=0.01, timing_ok=False), "encoder_unfresh:yaw")],
)
def test_unfresh_encoder_fails_closed_to_stop(reading, reason) -> None:
    plan = _planner().plan({"yaw": 0.05}, {"yaw": reading}, now_ns=T0)
    assert plan.stop and plan.reason == reason
    assert all(cmd.priority == "critical" for cmd in plan.commands)


def test_one_unfresh_axis_stops_all_axes() -> None:
    planner = _planner(_axis(), _axis("pitch_a", addr=2))
    plan = planner.plan({"yaw": 0.05, "pitch_a": 0.0},
                        {"yaw": _fresh(0), "pitch_a": None}, now_ns=T0)
    assert plan.stop and plan.reason == "encoder_unfresh:pitch_a"
    assert {cmd.addr for cmd in plan.commands} == {1, 2}


def test_non_finite_rate_stops() -> None:
    plan = _planner().plan({"yaw": math.inf}, {"yaw": _fresh(0)}, now_ns=T0)
    assert plan.stop and plan.reason == "non_finite_rate"


def test_home_outside_hard_limits_refuses_and_keeps_home_unset() -> None:
    planner = _planner(_axis(hard_min_rad=-0.1, hard_max_rad=0.1))
    plan = planner.plan({"yaw": 0.05}, {"yaw": _fresh(round(0.2 * K))}, now_ns=T0)
    assert plan.stop and plan.reason == "home_outside_hard_limits:yaw"
    assert planner.home_counts("yaw") is None


def _drive(planner, rate, *, start_counts, ticks):
    """Ideal plant: measured counts follow the last target each tick."""
    counts = start_counts
    targets = []
    for index in range(ticks):
        plan = planner.plan({"yaw": rate}, {"yaw": _fresh(counts)}, now_ns=T0 + index * TICK)
        assert not plan.stop, plan.reason
        counts = _target(plan.commands[0].payload)
        targets.append(counts)
    return targets


@pytest.mark.parametrize(("motor_sign", "camstate_sign"), [(1, 1), (-1, -1), (1, -1), (-1, 1)])
def test_hard_limit_tighter_than_travel_is_never_crossed(motor_sign, camstate_sign) -> None:
    # Home at CamState +0.05 rad; hard max 0.08 rad leaves 0.03 rad of travel.
    home = round(camstate_sign * 0.05 * K)
    axis = _axis(motor_sign=motor_sign, camstate_sign=camstate_sign,
                 hard_min_rad=-0.5, hard_max_rad=0.08, rate_limit_rad_s=5.0)
    # Drive toward increasing CamState angle whatever the signs are.
    rate = 1.0 * motor_sign * camstate_sign
    targets = _drive(_planner(axis, max_lead_rad=0.05), rate, start_counts=home, ticks=200)
    camstate = [camstate_sign * t / K for t in targets]
    assert max(camstate) == pytest.approx(0.08, abs=1.0 / K)
    assert max(camstate) <= 0.08 + 1.0 / K


def test_travel_limit_applies_around_home() -> None:
    home = 10_000
    axis = _axis(rate_limit_rad_s=5.0)
    targets = _drive(_planner(axis, max_lead_rad=0.05), -1.0, start_counts=home, ticks=200)
    assert min(targets) == home - round(0.15 * K)


def test_stop_resets_integration_and_next_motion_holds_first() -> None:
    planner = _planner()
    _drive(planner, 0.1, start_counts=0, ticks=5)
    planner.plan({"yaw": 0.0}, {"yaw": _fresh(40)}, now_ns=T0 + 10 * TICK)
    resumed = planner.plan({"yaw": 0.1}, {"yaw": _fresh(40)}, now_ns=T0 + 11 * TICK)
    assert not resumed.stop and _target(resumed.commands[0].payload) == 40
    assert planner.home_counts("yaw") == 0  # home is kept across stops


def test_speed_ceiling_matches_f6_rate_limit_encoding() -> None:
    assert _axis(rate_limit_rad_s=0.2).max_speed_rpm == 1
    assert _axis(rate_limit_rad_s=10.0).max_speed_rpm == 95
    assert _axis(rate_limit_rad_s=0.05).max_speed_rpm == 1


@pytest.mark.parametrize(
    "overrides",
    [dict(travel_limit_rad=0.0), dict(max_lead_rad=math.nan), dict(tick_s=-0.02),
     dict(max_encoder_age_s=math.inf)],
)
def test_planner_config_rejects_bad_values(overrides) -> None:
    with pytest.raises(ValueError):
        _planner(**overrides)


def test_duplicate_or_bad_axes_are_rejected() -> None:
    with pytest.raises(ValueError):
        _planner(_axis(), _axis())
    with pytest.raises(ValueError):
        _planner(_axis(motor_sign=0))


# --- gimbal_bridge startup checks for f5_position mode ---

from jetson.gimbal_bridge import _build_f5_planner  # noqa: E402


def _bridge_kwargs(**overrides):
    values = dict(
        pitch_b_enabled=False, pitch_authority="a", render_prediction_source="publication",
        yaw_addr=1, pitch_a_addr=2, yaw_sign=1.0, pitch_a_sign=-1.0,
        camstate_yaw_sign=1.0, camstate_pitch_sign=-1.0, yaw_ratio=1.0, pitch_ratio=1.0,
        counts_per_rev=16384, yaw_accel=10, pitch_accel=10,
        yaw_rate_limit=0.2, pitch_rate_limit=0.2,
        yaw_limits=(-0.5, 0.5), pitch_limits=(None, None),
    )
    values.update(overrides)
    return values


def test_bridge_builds_yaw_and_pitch_a_planner_with_defaults() -> None:
    planner = _build_f5_planner({"travel_limit_rad": 0.15}, **_bridge_kwargs())
    assert [(axis.name, axis.addr, axis.motor_sign) for axis in planner.axes] == [
        ("yaw", 1, 1), ("pitch_a", 2, -1)]
    assert planner.config.max_lead_rad == 0.01
    assert planner.config.tick_s == pytest.approx(0.02)
    assert planner.axes[0].hard_min_rad == -0.5


@pytest.mark.parametrize(
    ("raw", "overrides", "message"),
    [
        (None, {}, "must be a mapping"),
        ({}, {}, "travel_limit_rad is required"),
        ({"travel_limit_rad": 0.15}, {"pitch_b_enabled": True}, "pitch-B disabled"),
        ({"travel_limit_rad": 0.15}, {"pitch_authority": "b"}, "authority a"),
        ({"travel_limit_rad": 0.15}, {"render_prediction_source": "wire_execution"}, "wire_execution"),
        ({"travel_limit_rad": 0.15}, {"yaw_sign": 0.5}, "yaw_motor_sign"),
        ({"travel_limit_rad": -1.0}, {}, "invalid F5"),
        ({"travel_limit_rad": 0.15, "max_lead_rad": 0}, {}, "invalid F5"),
    ],
)
def test_bridge_refuses_f5_configs_it_cannot_bound(raw, overrides, message) -> None:
    with pytest.raises(SystemExit, match=message):
        _build_f5_planner(raw, **_bridge_kwargs(**overrides))
