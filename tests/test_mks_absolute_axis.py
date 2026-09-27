"""F5 absolute-axis and 0x98 heartbeat encoding, checked against the manual."""

from __future__ import annotations

import pytest

from common.gimbal.mks_servo42_rs485 import MksServo42Axis, RS485Bus, RS485Error


def _frame(addr: int, func: int, payload) -> bytes:
    body = bytes([0xFA, addr, func, *payload])
    return body + bytes([RS485Bus._crc8(body)])


class _RecordingBus:
    def __init__(self, reply: bytes = b"") -> None:
        self.calls = []
        self.reply = reply

    def send_command(self, addr, func, data=None, **kwargs):
        self.calls.append((addr, func, tuple(data or ()), kwargs))
        return self.reply


# Manual V1.0.9, section 11.4.1 / 11.4.2 worked examples (full frames with CRC).
@pytest.mark.parametrize(
    ("payload", "expected_hex"),
    [
        (MksServo42Axis._encode_absolute_axis_payload(0x4000, 600, 2),
         "FA 01 F5 02 58 02 00 00 40 00 8C"),
        (MksServo42Axis._encode_absolute_axis_payload(-0x4000, 600, 2),
         "FA 01 F5 02 58 02 FF FF C0 00 0A"),
        (MksServo42Axis._encode_absolute_axis_stop_payload(2),
         "FA 01 F5 00 00 02 00 00 00 00 F2"),
        (MksServo42Axis._encode_absolute_axis_stop_payload(0),
         "FA 01 F5 00 00 00 00 00 00 00 F0"),
    ],
)
def test_f5_frames_match_manual_examples(payload, expected_hex) -> None:
    assert _frame(0x01, 0xF5, payload) == bytes.fromhex(expected_hex)


def test_heartbeat_payload_is_big_endian_milliseconds() -> None:
    assert MksServo42Axis._encode_heartbeat_payload(0) == (0, 0, 0, 0)
    assert MksServo42Axis._encode_heartbeat_payload(250) == (0, 0, 0, 0xFA)
    assert MksServo42Axis._encode_heartbeat_payload(0x01020304) == (1, 2, 3, 4)


@pytest.mark.parametrize("speed_rpm", [0, -1, 3001])
def test_f5_move_rejects_speed_outside_1_to_3000(speed_rpm) -> None:
    # Speed 0 would be read by the firmware as a stop, not a move.
    with pytest.raises(ValueError):
        MksServo42Axis._encode_absolute_axis_payload(0, speed_rpm, 2)


@pytest.mark.parametrize("target", [2**31, -(2**31) - 1])
def test_f5_rejects_targets_outside_int32(target) -> None:
    with pytest.raises(ValueError):
        MksServo42Axis._encode_absolute_axis_payload(target, 10, 2)


@pytest.mark.parametrize(
    ("target", "speed", "acc"),
    [(1.0, 10, 2), (0, 1.5, 2), (True, 10, 2), (0, 10, 256), (0, 10, -1)],
)
def test_f5_rejects_non_integer_or_out_of_range_fields(target, speed, acc) -> None:
    with pytest.raises((TypeError, ValueError)):
        MksServo42Axis._encode_absolute_axis_payload(target, speed, acc)


def test_int32_extremes_encode_exactly() -> None:
    assert MksServo42Axis._encode_absolute_axis_payload(2**31 - 1, 1, 0)[3:] == (0x7F, 0xFF, 0xFF, 0xFF)
    assert MksServo42Axis._encode_absolute_axis_payload(-(2**31), 1, 0)[3:] == (0x80, 0, 0, 0)


def test_axis_methods_send_f5_to_individual_address() -> None:
    bus = _RecordingBus()
    axis = MksServo42Axis(bus=bus, addr=2, group_addr=0x50)
    axis.command_absolute_axis(123, 5, acc=3, use_group=False)
    axis.stop_absolute_axis(use_group=False)
    assert [call[:3] for call in bus.calls] == [
        (2, 0xF5, (0, 5, 3, 0, 0, 0, 123)),
        (2, 0xF5, (0, 0, 0, 0, 0, 0, 0)),
    ]


def test_heartbeat_configuration_requires_success_status() -> None:
    ok = _RecordingBus(reply=b"\x01")
    MksServo42Axis(bus=ok, addr=1).set_heartbeat_timeout_ms(200)
    assert ok.calls[0][:3] == (1, 0x98, (0, 0, 0, 200))
    assert ok.calls[0][3]["expected_response_len"] == 1

    with pytest.raises(RS485Error):
        MksServo42Axis(bus=_RecordingBus(reply=b"\x00"), addr=1).set_heartbeat_timeout_ms(200)
