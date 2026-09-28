"""Step-count (0x33) position feedback: serial parsing and bridge contract."""

from __future__ import annotations

import time

import pytest

from jetson.gimbal_bridge import _require_position_feedback_polled, _scheduled_addrs
from tools import serial_io_service


def _cmd(func: str, expected_len: int) -> serial_io_service.SerialCommand:
    return serial_io_service.SerialCommand(
        cmd_id="poll", func=func, addr=1, payload=(), expect_reply=True,
        expected_len=expected_len, priority="low", target="gimbal", timeout_ms=None,
        retry=None, sent_ts_ms=int(time.time() * 1000), enqueued_monotonic_ns=time.monotonic_ns())


def test_step_count_reply_is_parsed_as_signed_steps() -> None:
    assert serial_io_service._parse_reply("0x33", (-1234).to_bytes(4, "big", signed=True)) == {"steps": -1234}
    assert serial_io_service._parse_reply("0x33", b"\x00\x00\x04\x51") == {"steps": 1105}


def test_step_count_reply_length_is_validated() -> None:
    assert serial_io_service._validate_reply(_cmd("0x33", 4), b"\x00\x00\x00\x01")
    assert not serial_io_service._validate_reply(_cmd("0x33", None), b"\x00\x01")


def _schedule(*entries):
    return {"serial_io": {"schedule": [dict(name=n, func=f, addr=a) for n, f, a in entries]}}


def test_scheduled_addresses_are_found_per_function() -> None:
    cfg = _schedule(("s1", "0x33", 1), ("s2", "0x33", 2), ("e1", "0x31", 1), ("st", "F1", 1))
    assert _scheduled_addrs(cfg, 0x33) == {1, 2}
    assert _scheduled_addrs(cfg, 0x31) == {1}


def test_steps_feedback_requires_step_polls_for_every_controlled_motor() -> None:
    cfg = _schedule(("s1", "0x33", 1), ("e2", "0x31", 2))
    with pytest.raises(SystemExit, match=r"0x33 for motor addresses \[2\]"):
        _require_position_feedback_polled(cfg, "steps", [1, 2])
    _require_position_feedback_polled(_schedule(("s1", "0x33", 1), ("s2", "0x33", 2)), "steps", [1, 2])


def test_encoder_feedback_mode_and_bad_mode() -> None:
    _require_position_feedback_polled(_schedule(("e1", "0x31", 1)), "encoder", [1])
    with pytest.raises(SystemExit, match="steps' or 'encoder"):
        _require_position_feedback_polled({}, "imu", [1])


def test_repo_control_config_polls_steps_for_all_pitch_and_yaw_motors() -> None:
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/base/gimbal.yaml").read_text())
    assert cfg["gimbal"]["position_feedback"] == "steps"
    _require_position_feedback_polled(cfg, "steps", [1, 2, 3])


def test_step_count_is_anchored_to_the_encoder_frame():
    from jetson.gimbal_bridge import StepAnchor

    anchor = StepAnchor()
    assert anchor.observe_steps(1, 170_000) is None      # unanchored: no position yet
    assert not anchor.observe_encoder(2, 500)             # no step reading for addr 2 yet
    assert anchor.observe_encoder(1, 6_250)                # encoder says the axis is at 6250
    assert anchor.observe_steps(1, 170_000) == 6_250
    assert anchor.observe_steps(1, 170_512) == 6_762       # step changes carry over
    assert not anchor.observe_encoder(1, 9_999)            # anchored once
