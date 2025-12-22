import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.uart_protocol import (
    FrameParser,
    LinkMode,
    MessageType,
    build_handshake_payload,
    build_mode_request_payload,
    encode_frame,
    parse_handshake_payload,
    parse_mode_request_payload,
)


def test_frame_roundtrip_handshake() -> None:
    payload = build_handshake_payload(role="jetson", heartbeat_ms=500, capabilities="link")
    frame_bytes = encode_frame(MessageType.HANDSHAKE, 42, payload)

    parser = FrameParser()
    frames = list(parser.feed(frame_bytes))
    assert len(frames) == 1
    frame = frames[0]
    assert frame.msg_type is MessageType.HANDSHAKE
    assert frame.seq == 42
    parsed = parse_handshake_payload(frame.payload)
    assert parsed["role"] == "jetson"
    assert parsed["heartbeat_ms"] == 500


def test_crc_error_callback_invoked() -> None:
    payload = build_handshake_payload(role="jetson", heartbeat_ms=1000, capabilities="cap")
    frame_bytes = bytearray(encode_frame(MessageType.HANDSHAKE, 1, payload))
    frame_bytes[-1] ^= 0xFF  # corrupt CRC

    errors: list[tuple[int, int]] = []

    def _on_crc(frame: bytes, expected: int, actual: int) -> None:
        errors.append((expected, actual))

    parser = FrameParser(on_crc_error=_on_crc)
    list(parser.feed(bytes(frame_bytes)))
    assert errors, "CRC callback should have been invoked"


def test_mode_request_parse() -> None:
    payload = build_mode_request_payload(LinkMode.DIAGNOSTIC, "verify sensors")
    frame_bytes = encode_frame(MessageType.MODE_REQUEST, 3, payload)
    parser = FrameParser()
    frames = list(parser.feed(frame_bytes))
    assert frames[0].msg_type is MessageType.MODE_REQUEST
    parsed = parse_mode_request_payload(frames[0].payload)
    assert parsed["mode"] is LinkMode.DIAGNOSTIC
    assert parsed["reason"] == "verify sensors"
