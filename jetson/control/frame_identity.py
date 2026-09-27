"""Fail-closed join of encoded-frame metadata and decoded frame identity.

The sender binds a source frame to (RTP SSRC, RTP timestamp) at the payloader.
The receiver binds that key to the jitterbuffer marker packet's PTS.  Only
exactly matching keys and decoded PTS values may yield verified source time.
No FIFO/arrival-order fallback exists.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from common.rtp_identity import RtpFrameKey


@dataclass(frozen=True)
class SourceFrameHeader:
    key: RtpFrameKey
    frame_id: int
    source_time_ns: int
    source_clock_domain: str

    def __post_init__(self) -> None:
        if self.frame_id < 0 or self.source_time_ns < 0 or not self.source_clock_domain:
            raise ValueError("invalid source frame header")


@dataclass(frozen=True)
class FrameIdentityResult:
    verified: bool
    reason: str
    header: SourceFrameHeader | None = None


class FrameIdentityJoiner:
    """Bounded, single-consumer correlation with ambiguity poisoning."""

    def __init__(self, *, capacity: int = 256) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._headers: OrderedDict[RtpFrameKey, SourceFrameHeader | None] = OrderedDict()
        self._markers: OrderedDict[int, RtpFrameKey | None] = OrderedDict()
        self._used_pts: OrderedDict[int, None] = OrderedDict()
        self._used_keys: OrderedDict[RtpFrameKey, None] = OrderedDict()
        self.dropped_capacity = 0
        self.ambiguous_headers = 0
        self.ambiguous_markers = 0

    def _bound(self, mapping: OrderedDict) -> None:
        while len(mapping) > self.capacity:
            mapping.popitem(last=False)
            self.dropped_capacity += 1

    def push_header(self, header: SourceFrameHeader) -> None:
        if header.key in self._used_keys:
            self.ambiguous_headers += 1
            return
        existing = self._headers.get(header.key)
        if header.key in self._headers:
            if existing != header:
                self._headers[header.key] = None
                self.ambiguous_headers += 1
            return
        self._headers[header.key] = header
        self._bound(self._headers)

    def push_marker(self, *, decoded_pts_ns: int, key: RtpFrameKey) -> None:
        if decoded_pts_ns < 0:
            raise ValueError("decoded PTS must be nonnegative")
        if decoded_pts_ns in self._used_pts or key in self._used_keys:
            self.ambiguous_markers += 1
            return
        existing = self._markers.get(decoded_pts_ns)
        if decoded_pts_ns in self._markers:
            if existing != key:
                self._markers[decoded_pts_ns] = None
                self.ambiguous_markers += 1
            return
        self._markers[decoded_pts_ns] = key
        self._bound(self._markers)

    def match(self, *, decoded_pts_ns: int) -> FrameIdentityResult:
        if decoded_pts_ns in self._used_pts:
            return FrameIdentityResult(False, "decoded_pts_reused")
        if decoded_pts_ns not in self._markers:
            return FrameIdentityResult(False, "rtp_marker_missing")
        key = self._markers.pop(decoded_pts_ns)
        self._used_pts[decoded_pts_ns] = None
        self._bound(self._used_pts)
        if key is None:
            return FrameIdentityResult(False, "rtp_marker_ambiguous")
        if key in self._used_keys:
            return FrameIdentityResult(False, "rtp_key_reused")
        self._used_keys[key] = None
        self._bound(self._used_keys)
        if key not in self._headers:
            return FrameIdentityResult(False, "source_header_missing")
        header = self._headers.pop(key)
        if header is None:
            return FrameIdentityResult(False, "source_header_ambiguous")
        return FrameIdentityResult(True, "verified", header)
