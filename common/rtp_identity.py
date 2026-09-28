"""Minimal RTP frame-key parser shared by sender and receiver probes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RtpFrameKey:
    ssrc: int
    timestamp: int

    def __post_init__(self) -> None:
        if not 0 <= self.ssrc <= 0xFFFFFFFF or not 0 <= self.timestamp <= 0xFFFFFFFF:
            raise ValueError("RTP key fields must be unsigned 32-bit integers")


@dataclass(frozen=True)
class RtpPacketIdentity:
    key: RtpFrameKey
    sequence: int
    marker: bool


def parse_rtp_identity(header: bytes) -> RtpPacketIdentity:
    """Read only fixed RTP header fields; CSRC/extensions follow these fields."""

    if len(header) < 12 or header[0] >> 6 != 2:
        raise ValueError("invalid RTP version or truncated header")
    return RtpPacketIdentity(
        key=RtpFrameKey(
            int.from_bytes(header[8:12], "big"),
            int.from_bytes(header[4:8], "big"),
        ),
        sequence=int.from_bytes(header[2:4], "big"),
        marker=bool(header[1] & 0x80),
    )
