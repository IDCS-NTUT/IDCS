"""Fail-closed identity for a return-video frame actually displayed on the Pi.

The Jetson binds a source frame to the *return* RTP SSRC/timestamp.  The Pi
binds that key to the jitterbuffer marker PTS and finally to the post-overlay
buffer PTS.  No frame counter, arrival order, or host-clock estimate is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import math
import threading
from typing import Mapping

from common.rtp_identity import RtpFrameKey, RtpPacketIdentity
from common.frame_identity import FrameIdentityJoiner, SourceFrameHeader


@dataclass(frozen=True)
class DisplayedSourceFrame:
    tracker_session: str
    frame_id: int
    source_time_ns: int
    source_clock_domain: str
    displayed_local_s: float


class ReturnFrameIdentity:
    """Bounded, thread-safe join; missing/ambiguous metadata yields no frame."""

    def __init__(self, capacity: int = 128) -> None:
        self._lock = threading.Lock()
        self._joiner = FrameIdentityJoiner(capacity=capacity)
        self._sessions: dict[RtpFrameKey, str] = {}
        self._packet_pts: dict[RtpFrameKey, int | None] = {}
        self._decoded_pts: OrderedDict[int, bool] = OrderedDict()
        self._capacity = capacity
        self._latest: DisplayedSourceFrame | None = None
        self.matched = 0
        self.unmatched = 0

    def push_header(self, payload: Mapping[str, object]) -> bool:
        try:
            key = RtpFrameKey(int(payload["rtp_ssrc"]), int(payload["rtp_timestamp"]))
            session = str(payload["tracker_session"])
            if not session or len(session) > 128:
                return False
            header = SourceFrameHeader(
                key=key,
                frame_id=int(payload["frame_id"]),
                source_time_ns=int(payload["source_time_ns"]),
                source_clock_domain=str(payload["source_clock_domain"]),
            )
        except (KeyError, TypeError, ValueError):
            return False
        with self._lock:
            previous = self._sessions.get(key)
            if previous is not None and previous != session:
                # Poison a conflicting session exactly as the underlying
                # header joiner poisons conflicting frame metadata.
                self._joiner.push_header(header)
                self._joiner.push_header(SourceFrameHeader(
                    key=key, frame_id=header.frame_id + 1,
                    source_time_ns=header.source_time_ns,
                    source_clock_domain=header.source_clock_domain,
                ))
                self._sessions.pop(key, None)
                return False
            self._joiner.push_header(header)
            self._sessions[key] = session
            while len(self._sessions) > self._capacity:
                self._sessions.pop(next(iter(self._sessions)))
        return True

    def push_marker(self, decoded_pts_ns: int, key: RtpFrameKey) -> None:
        with self._lock:
            self._joiner.push_marker(decoded_pts_ns=decoded_pts_ns, key=key)

    def push_rtp_packet(self, pts_ns: int | None, packet: RtpPacketIdentity) -> None:
        """Use a timestamp from the same RTP AU when the marker has no Gst PTS."""
        if pts_ns is not None and pts_ns < 0:
            return
        with self._lock:
            key = packet.key
            if key not in self._packet_pts:
                self._packet_pts[key] = pts_ns
            elif pts_ns is not None and self._packet_pts[key] not in (None, -1, pts_ns):
                self._packet_pts[key] = -1
            elif pts_ns is not None and self._packet_pts[key] is None:
                self._packet_pts[key] = pts_ns
            while len(self._packet_pts) > self._capacity:
                self._packet_pts.pop(next(iter(self._packet_pts)))
            if packet.marker:
                marker_pts = self._packet_pts.pop(key, None)
                if marker_pts is not None and marker_pts >= 0:
                    self._joiner.push_marker(decoded_pts_ns=marker_pts, key=key)

    def push_decoded(self, pts_ns: int) -> None:
        with self._lock:
            self._decoded_pts[pts_ns] = pts_ns not in self._decoded_pts
            while len(self._decoded_pts) > self._capacity:
                self._decoded_pts.popitem(last=False)

    def displayed(self, decoded_pts_ns: int, local_s: float) -> DisplayedSourceFrame | None:
        if not math.isfinite(local_s):
            return None
        with self._lock:
            # A newer output frame supersedes the old one even if its sideband
            # is late or missing.  Never label an already-passed image as live.
            if self._latest is not None and local_s < self._latest.displayed_local_s:
                return None
            self._latest = None
            if self._decoded_pts.pop(decoded_pts_ns, False) is not True:
                self.unmatched += 1
                return None
            return self._resolve(decoded_pts_ns, local_s)

    def _resolve(self, decoded_pts_ns: int, local_s: float) -> DisplayedSourceFrame | None:
        """Resolve with the caller holding ``_lock``."""
        result = self._joiner.match(decoded_pts_ns=decoded_pts_ns)
        session = None if result.header is None else self._sessions.pop(result.header.key, None)
        if not result.verified or not session:
            self.unmatched += 1
            return None
        header = result.header
        frame = DisplayedSourceFrame(
            session, header.frame_id, header.source_time_ns,
            header.source_clock_domain, local_s,
        )
        self._latest = frame
        self.matched += 1
        return frame

    def latest(self, local_s: float, *, max_age_s: float = 0.25) -> DisplayedSourceFrame | None:
        with self._lock:
            frame = self._latest
            if (frame is None or not math.isfinite(local_s) or
                    local_s < frame.displayed_local_s or
                    local_s - frame.displayed_local_s > max_age_s):
                return None
            return frame

    def unidentified_output(self) -> None:
        """An output buffer without usable PTS invalidates the previous image."""
        with self._lock:
            self._latest = None

    def clear(self) -> None:
        with self._lock:
            self._joiner = FrameIdentityJoiner(capacity=self._capacity)
            self._sessions.clear()
            self._packet_pts.clear()
            self._decoded_pts.clear()
            self._latest = None
