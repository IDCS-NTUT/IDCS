"""Length-prefixed framed UART protocol shared between Jetson and Raspberry Pi.

Frames use a two-byte preamble (0xAA 0x55), a one-byte version field, a
one-byte message type, a two-byte sequence number, and a two-byte payload
length followed by the payload and a CRC-16/X25 checksum. This design keeps
the protocol distinct from the gimbal RS485 framing (0xFA/0xFB start bytes)
and keeps framing robust against line noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Iterator, Optional


PREAMBLE = b"\xAA\x55"
VERSION = 0x01
HEADER_LEN = 2 + 1 + 1 + 2 + 2  # preamble + version + type + seq + length
CRC_LEN = 2
MAX_PAYLOAD_LEN = 255


class MessageType(IntEnum):
    """Enumerates framed UART message types."""

    HANDSHAKE = 0x01
    HANDSHAKE_ACK = 0x02
    HEARTBEAT = 0x03
    MODE_REQUEST = 0x10
    MODE_ACK = 0x11
    ERROR_REPORT = 0x7E


class LinkMode(IntEnum):
    """Desired or reported link role/mode."""

    STANDBY = 0
    ACTIVE = 1
    DIAGNOSTIC = 2


class FrameDecodeError(Exception):
    """Raised when a frame cannot be decoded."""


class FrameCRCError(FrameDecodeError):
    """Raised when a CRC mismatch is detected."""


class FrameFormatError(FrameDecodeError):
    """Raised when framing or version information is invalid."""


def crc16_x25(data: Iterable[int]) -> int:
    """Compute CRC-16/X25 over the provided data."""

    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
    crc ^= 0xFFFF
    return crc & 0xFFFF


@dataclass
class Frame:
    """Decoded UART frame."""

    msg_type: MessageType
    seq: int
    payload: bytes


def encode_frame(msg_type: MessageType, seq: int, payload: bytes = b"") -> bytes:
    """Encode a :class:`Frame` to wire bytes."""

    if len(payload) > MAX_PAYLOAD_LEN:
        raise ValueError(f"payload too large ({len(payload)} bytes; max={MAX_PAYLOAD_LEN})")

    seq_clamped = seq & 0xFFFF
    header = bytearray()
    header.extend(PREAMBLE)
    header.append(VERSION)
    header.append(int(msg_type) & 0xFF)
    header.extend(seq_clamped.to_bytes(2, byteorder="big", signed=False))
    header.extend(len(payload).to_bytes(2, byteorder="big", signed=False))
    body = bytes(header[2:] + payload)  # CRC covers version..payload
    crc = crc16_x25(body)
    header.extend(payload)
    header.extend(crc.to_bytes(2, byteorder="big", signed=False))
    return bytes(header)


class FrameParser:
    """Incremental framed UART parser."""

    def __init__(self, *, on_crc_error: Optional[callable] = None) -> None:
        self._buf = bytearray()
        self._on_crc_error = on_crc_error

    def feed(self, chunk: bytes) -> Iterator[Frame]:
        """Feed raw bytes and yield decoded frames as they become available."""

        if not chunk:
            return iter(())
        self._buf.extend(chunk)

        frames: list[Frame] = []
        while True:
            start = self._buf.find(PREAMBLE)
            if start == -1:
                self._buf.clear()
                break
            if start > 0:
                del self._buf[:start]

            if len(self._buf) < HEADER_LEN:
                break

            version = self._buf[2]
            if version != VERSION:
                del self._buf[0]
                continue

            msg_type_val = self._buf[3]
            seq = int.from_bytes(self._buf[4:6], byteorder="big", signed=False)
            payload_len = int.from_bytes(self._buf[6:8], byteorder="big", signed=False)
            if payload_len > MAX_PAYLOAD_LEN:
                del self._buf[0]
                continue

            frame_len = HEADER_LEN + payload_len + CRC_LEN
            if len(self._buf) < frame_len:
                break

            frame_bytes = bytes(self._buf[:frame_len])
            payload = frame_bytes[8:-2]
            crc_read = int.from_bytes(frame_bytes[-2:], byteorder="big", signed=False)
            crc_calc = crc16_x25(frame_bytes[2:-2])
            if crc_read != crc_calc:
                if self._on_crc_error:
                    self._on_crc_error(frame_bytes, crc_read, crc_calc)
                del self._buf[0]
                continue

            try:
                msg_type = MessageType(msg_type_val)
            except ValueError as exc:
                raise FrameFormatError(f"Unknown message type: {msg_type_val:#x}") from exc

            frames.append(Frame(msg_type=msg_type, seq=seq, payload=payload))
            del self._buf[:frame_len]

        return iter(frames)


def _pack_str(value: str) -> bytes:
    encoded = value.encode("utf-8", errors="replace")[:255]
    return bytes([len(encoded)]) + encoded


def _unpack_str(buffer: memoryview) -> tuple[str, memoryview]:
    if not buffer:
        raise FrameFormatError("cannot unpack string from empty payload")
    length = buffer[0]
    if len(buffer) < 1 + length:
        raise FrameFormatError("string field truncated")
    raw = buffer[1 : 1 + length]
    remainder = buffer[1 + length :]
    return raw.tobytes().decode("utf-8", errors="replace"), remainder


def build_handshake_payload(*, role: str, heartbeat_ms: int, capabilities: str) -> bytes:
    payload = bytearray()
    payload.append(VERSION)
    payload.extend(int(heartbeat_ms).to_bytes(2, byteorder="big", signed=False))
    payload.extend(_pack_str(role))
    payload.extend(_pack_str(capabilities))
    return bytes(payload)


def parse_handshake_payload(payload: bytes) -> dict[str, object]:
    data = memoryview(payload)
    if len(data) < 3:
        raise FrameFormatError("handshake payload too short")
    version = data[0]
    heartbeat_ms = int.from_bytes(data[1:3], byteorder="big", signed=False)
    role, remainder = _unpack_str(data[3:])
    capabilities, _ = _unpack_str(remainder)
    return {"version": version, "heartbeat_ms": heartbeat_ms, "role": role, "capabilities": capabilities}


def build_handshake_ack_payload(*, role: str, ok: bool, info: str = "") -> bytes:
    payload = bytearray()
    payload.append(VERSION)
    payload.append(0x01 if ok else 0x00)
    payload.extend(_pack_str(role))
    payload.extend(_pack_str(info))
    return bytes(payload)


def parse_handshake_ack_payload(payload: bytes) -> dict[str, object]:
    data = memoryview(payload)
    if len(data) < 2:
        raise FrameFormatError("handshake ack payload too short")
    version = data[0]
    ok = bool(data[1])
    role, remainder = _unpack_str(data[2:])
    info, _ = _unpack_str(remainder)
    return {"version": version, "ok": ok, "role": role, "info": info}


def build_heartbeat_payload(*, uptime_ms: int, last_error: int = 0) -> bytes:
    payload = bytearray()
    payload.append(last_error & 0xFF)
    payload.extend(int(uptime_ms).to_bytes(4, byteorder="big", signed=False))
    return bytes(payload)


def parse_heartbeat_payload(payload: bytes) -> dict[str, int]:
    if len(payload) < 5:
        raise FrameFormatError("heartbeat payload too short")
    last_error = payload[0]
    uptime_ms = int.from_bytes(payload[1:5], byteorder="big", signed=False)
    return {"last_error": last_error, "uptime_ms": uptime_ms}


def build_mode_request_payload(mode: LinkMode, reason: str = "") -> bytes:
    payload = bytearray()
    payload.append(int(mode) & 0xFF)
    payload.extend(_pack_str(reason))
    return bytes(payload)


def parse_mode_request_payload(payload: bytes) -> dict[str, object]:
    if not payload:
        raise FrameFormatError("mode request payload empty")
    mode_val = payload[0]
    reason, _ = _unpack_str(memoryview(payload[1:]))
    try:
        mode = LinkMode(mode_val)
    except ValueError as exc:
        raise FrameFormatError(f"unknown mode value {mode_val}") from exc
    return {"mode": mode, "reason": reason}


def build_mode_ack_payload(mode: LinkMode, accepted: bool, info: str = "") -> bytes:
    payload = bytearray()
    payload.append(int(mode) & 0xFF)
    payload.append(0x01 if accepted else 0x00)
    payload.extend(_pack_str(info))
    return bytes(payload)


def parse_mode_ack_payload(payload: bytes) -> dict[str, object]:
    if len(payload) < 2:
        raise FrameFormatError("mode ack payload too short")
    mode_val = payload[0]
    accepted = bool(payload[1])
    info, _ = _unpack_str(memoryview(payload[2:]))
    try:
        mode = LinkMode(mode_val)
    except ValueError as exc:
        raise FrameFormatError(f"unknown mode value {mode_val}") from exc
    return {"mode": mode, "accepted": accepted, "info": info}


def build_error_payload(code: int, info: str = "") -> bytes:
    payload = bytearray()
    payload.append(code & 0xFF)
    payload.extend(_pack_str(info))
    return bytes(payload)


def parse_error_payload(payload: bytes) -> dict[str, object]:
    if not payload:
        raise FrameFormatError("error payload empty")
    code = payload[0]
    info, _ = _unpack_str(memoryview(payload[1:]))
    return {"code": code, "info": info}
