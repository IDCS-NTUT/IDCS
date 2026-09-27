from __future__ import annotations

import math

import pytest

from common.gimbal.mks_servo42_rs485 import MksServo42Axis
from jetson.tools.home_axes import home_targets

BENCH = {"yaw_addr": 1, "pitch_motor_a_addr": 2, "pitch_motor_b_enabled": False,
         "camstate_yaw_sign": -1.0, "camstate_pitch_sign": -1.0,
         "yaw_min_rad": -2.9, "yaw_max_rad": -1.9, "pitch_min_rad": 0.3, "pitch_max_rad": 1.95}


def test_targets_are_envelope_centres_in_encoder_counts() -> None:
    targets = home_targets(BENCH)
    assert targets["yaw"]["target_counts"] == round(2.4 * 16384 / (2 * math.pi))  # the toolkit's 6259
    assert targets["pitch"]["centre_rad"] == pytest.approx(1.125)
    geared = home_targets({**BENCH, "yaw_gear_ratio": 5.0})
    assert geared["yaw"]["target_counts"] == pytest.approx(5 * targets["yaw"]["target_counts"], abs=2)


def test_paired_pitch_and_missing_envelope_are_refused() -> None:
    with pytest.raises(ValueError, match="pitch-B"):
        home_targets({**BENCH, "pitch_motor_b_enabled": True})
    with pytest.raises(ValueError, match="yaw_min_rad"):
        home_targets({**BENCH, "yaw_min_rad": None})


def test_f4_payload_matches_the_manual_layout() -> None:
    assert MksServo42Axis._encode_relative_axis_payload(-2869, 60, 2) == [0, 60, 2, *(-2869).to_bytes(4, "big", signed=True)]
    assert MksServo42Axis._encode_relative_axis_payload(100, 5000, 300)[:3] == [0x0B, 0xB8, 255]  # clamped
