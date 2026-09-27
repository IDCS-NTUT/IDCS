"""Measured F6 speed model (bench probes 2026-09-27) and cap-honouring encoding."""

from __future__ import annotations

import math

import pytest

from common.gimbal.mks_servo42_rs485 import (
    F6_MEASURED_MICROSTEPS_PER_S,
    MksServo42Axis,
    f6_level_for_rate,
    f6_level_speed_rad_s,
    min_f6_speed_rad_s,
)

RAD_PER_STEP = 2.0 * math.pi / 3200


@pytest.mark.parametrize("level", sorted(F6_MEASURED_MICROSTEPS_PER_S))
def test_measured_levels_return_table_speed(level) -> None:
    assert f6_level_speed_rad_s(level) == pytest.approx(F6_MEASURED_MICROSTEPS_PER_S[level] * RAD_PER_STEP)
    assert f6_level_speed_rad_s(-level) == pytest.approx(-F6_MEASURED_MICROSTEPS_PER_S[level] * RAD_PER_STEP)


def test_level_nine_interpolates_and_levels_above_ten_extrapolate_n_plus_one() -> None:
    assert f6_level_speed_rad_s(8) < f6_level_speed_rad_s(9) < f6_level_speed_rad_s(10)
    assert f6_level_speed_rad_s(21) == pytest.approx(611.0 * 2 * RAD_PER_STEP)


def test_gear_ratio_divides_axis_speed() -> None:
    assert f6_level_speed_rad_s(3, 2.0) == pytest.approx(f6_level_speed_rad_s(3) / 2.0)


def test_slowest_speed_is_level_one() -> None:
    assert min_f6_speed_rad_s() == pytest.approx(114.0 * RAD_PER_STEP)
    assert min_f6_speed_rad_s() > 0.2  # a 0.2 rad/s cap cannot be met by any nonzero level


@pytest.mark.parametrize(("request_rad_s", "level"), [
    (0.0, 0), (0.05, 0), (0.11, 0), (0.12, 1), (0.25, 1), (0.3, 2),
    (0.5, 4), (1.0, 8), (-0.3, -2),
])
def test_nearest_measured_level_is_selected(request_rad_s, level) -> None:
    assert f6_level_for_rate(request_rad_s) == level


def test_cap_is_never_exceeded_by_actual_speed() -> None:
    for cap in (0.1, 0.2, 0.25, 0.4, 0.8, 1.5):
        for request in [i * 0.013 for i in range(-160, 161)]:
            level = f6_level_for_rate(request, 1.0, cap)
            assert abs(f6_level_speed_rad_s(level)) <= cap + 1e-12
            assert level == 0 or math.copysign(1, level) == math.copysign(1, request)


def test_cap_below_slowest_speed_encodes_zero() -> None:
    assert f6_level_for_rate(0.5, 1.0, 0.2) == 0
    assert MksServo42Axis.quantized_speed_rad_s(0.5, 1.0, 0.2) == 0.0


def test_quantized_speed_reports_measured_not_nominal() -> None:
    # 0.3 rad/s selects level 2, which actually runs 164 microsteps/s.
    assert MksServo42Axis.quantized_speed_rad_s(0.3, 1.0) == pytest.approx(164.0 * RAD_PER_STEP)


@pytest.mark.parametrize(("level", "expected"), [
    (0, (0x00, 0x00, 10)), (2, (0x00, 0x02, 10)), (-2, (0x80, 0x02, 10)), (300, (0x01, 0x2C, 10)),
])
def test_level_payload_bytes(level, expected) -> None:
    assert MksServo42Axis._encode_speed_level_payload(level, 10) == expected


def test_non_finite_request_is_rejected() -> None:
    with pytest.raises(ValueError):
        f6_level_for_rate(math.nan)
