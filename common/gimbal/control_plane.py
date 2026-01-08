"""Control-plane handshake framing for RS-485 authority negotiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .mks_servo42_rs485 import MASTER_START, SLAVE_START, RS485Bus

CONTROL_ADDR = 0x7E
CONTROL_FUNC = 0x7D
CONTROL_VERSION = 0x01

ROLE_JETSON_ACTIVE = 0x00
ROLE_PI_ACTIVE = 0x01

FLAG_TAKEOVER = 0x01
FLAG_RETURN = 0x02
FLAG_ACK_YIELD = 0x04
FLAG_MASK_ALL = FLAG_TAKEOVER | FLAG_RETURN | FLAG_ACK_YIELD

PAYLOAD_LEN = 6


@dataclass(frozen=True)
class ControlPlaneFrame:
    """Decoded control-plane payload for a handshake frame."""

    version: int
    role: int
    flags: int
    counter: int
    reserved: int = 0

    @classmethod
    def from_payload(cls, payload: Iterable[int]) -> "ControlPlaneFrame":
        data = list(payload)
        if len(data) != PAYLOAD_LEN:
            raise ValueError(f"control-plane payload must be {PAYLOAD_LEN} bytes")
        version, role, flags, counter_hi, counter_lo, reserved = data
        counter = ((counter_hi & 0xFF) << 8) | (counter_lo & 0xFF)
        return cls(
            version=version & 0xFF,
            role=role & 0xFF,
            flags=flags & 0xFF,
            counter=counter,
            reserved=reserved & 0xFF,
        )

    def to_payload(self) -> bytes:
        counter_hi = (self.counter >> 8) & 0xFF
        counter_lo = self.counter & 0xFF
        return bytes(
            [
                self.version & 0xFF,
                self.role & 0xFF,
                self.flags & 0xFF,
                counter_hi,
                counter_lo,
                self.reserved & 0xFF,
            ]
        )

    def validate(self, *, expected_version: int = CONTROL_VERSION) -> None:
        if self.version != (expected_version & 0xFF):
            raise ValueError("control-plane version mismatch")
        if self.role not in (ROLE_JETSON_ACTIVE, ROLE_PI_ACTIVE):
            raise ValueError("control-plane role mismatch")
        if (self.flags & FLAG_MASK_ALL) != (self.flags & 0xFF):
            raise ValueError("control-plane flags mismatch")
        if not (0 <= self.counter <= 0xFFFF):
            raise ValueError("control-plane counter out of range")


def build_control_frame(
    payload: ControlPlaneFrame,
    *,
    start_byte: int = MASTER_START,
    addr: int = CONTROL_ADDR,
    func: int = CONTROL_FUNC,
) -> bytes:
    """Serialize a control-plane frame including start byte, addr, func, and CRC."""

    payload_bytes = payload.to_payload()
    frame_wo_crc = bytearray([start_byte & 0xFF, addr & 0xFF, func & 0xFF])
    frame_wo_crc.extend(payload_bytes)
    crc = RS485Bus._crc8(frame_wo_crc)
    return bytes(frame_wo_crc + bytes([crc]))


def parse_control_frame(
    frame: bytes,
    *,
    expected_start: int,
    expected_addr: int = CONTROL_ADDR,
    expected_func: int = CONTROL_FUNC,
    expected_version: int = CONTROL_VERSION,
) -> ControlPlaneFrame:
    """Parse and validate a control-plane frame payload."""

    if len(frame) != 1 + 2 + PAYLOAD_LEN + 1:
        raise ValueError("control-plane frame length mismatch")
    if frame[0] != (expected_start & 0xFF):
        raise ValueError("control-plane start byte mismatch")
    if frame[1] != (expected_addr & 0xFF):
        raise ValueError("control-plane address mismatch")
    if frame[2] != (expected_func & 0xFF):
        raise ValueError("control-plane function mismatch")
    if RS485Bus._crc8(frame[:-1]) != frame[-1]:
        raise ValueError("control-plane CRC mismatch")
    parsed = ControlPlaneFrame.from_payload(frame[3:-1])
    parsed.validate(expected_version=expected_version)
    return parsed


def build_ping(
    *,
    role: int,
    counter: int,
    flags: int = 0,
    start_byte: int = MASTER_START,
) -> bytes:
    return build_control_frame(
        ControlPlaneFrame(
            version=CONTROL_VERSION,
            role=role,
            flags=flags,
            counter=counter,
        ),
        start_byte=start_byte,
    )


def build_reply(
    *,
    role: int,
    counter: int,
    flags: int = 0,
    start_byte: int = SLAVE_START,
) -> bytes:
    return build_control_frame(
        ControlPlaneFrame(
            version=CONTROL_VERSION,
            role=role,
            flags=flags,
            counter=counter,
        ),
        start_byte=start_byte,
    )
