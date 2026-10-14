"""Bind a verified source frame to its encoded return RTP frame key.

No FIFO fallback: encoder timestamp changes or dropped frames simply withhold
metadata, so the panel cannot seed a different image accidentally.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading

from common.rtp_identity import RtpPacketIdentity
from common.rtp_identity import RtpFrameKey


@dataclass(frozen=True)
class SourceIdentity:
    tracker_session: str
    frame_id: int
    source_time_ns: int
    source_clock_domain: str


class ReturnRtpMapper:
    def __init__(self, capacity: int = 128) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._lock = threading.Lock()
        self._sources: OrderedDict[int, SourceIdentity | None] = OrderedDict()
        self._decoded: OrderedDict[int, SourceIdentity | None] = OrderedDict()
        self._packet_pts: OrderedDict[RtpFrameKey, int | None] = OrderedDict()
        self.mapped = 0
        self.withheld = 0
        self.ambiguous = 0
        self.source_first_pts = self.source_last_pts = None
        self.encoded_first_pts = self.encoded_last_pts = None

    def record_verified_frame(self, native_frame_id: int, source: SourceIdentity) -> None:
        """Retain verified input metadata through GPU conversion and dropping."""
        if native_frame_id < 0:
            raise ValueError('invalid native frame ID')
        with self._lock:
            if native_frame_id in self._decoded and self._decoded[native_frame_id] != source:
                self._decoded[native_frame_id] = None
                self.ambiguous += 1
            else:
                self._decoded[native_frame_id] = source
            while len(self._decoded) > self.capacity:
                self._decoded.popitem(last=False)

    def record_encoder_input(self, encoder_pts_ns: int, native_frame_id: int) -> bool:
        """Use the final encoder buffer's retained frame metadata; no FIFO."""
        with self._lock:
            source = self._decoded.pop(native_frame_id, None)
            if source is None:
                self.withheld += 1
                return False
        self.record_source(encoder_pts_ns, source)
        return True

    def record_source(self, pts_ns: int, source: SourceIdentity) -> None:
        if pts_ns < 0 or not source.tracker_session or source.frame_id < 0:
            raise ValueError("invalid return source identity")
        with self._lock:
            if self.source_first_pts is None:
                self.source_first_pts = pts_ns
            self.source_last_pts = pts_ns
            if pts_ns in self._sources:
                if self._sources[pts_ns] != source:
                    self._sources[pts_ns] = None
                    self.ambiguous += 1
                return
            self._sources[pts_ns] = source
            while len(self._sources) > self.capacity:
                self._sources.popitem(last=False)
                self.withheld += 1

    def encoded_packet(self, pts_ns: int | None, packet: RtpPacketIdentity) -> dict | None:
        if pts_ns is not None and pts_ns < 0:
            raise ValueError("invalid encoded PTS")
        with self._lock:
            if pts_ns is not None:
                if self.encoded_first_pts is None:
                    self.encoded_first_pts = pts_ns
                self.encoded_last_pts = pts_ns
            key = packet.key
            if key not in self._packet_pts:
                self._packet_pts[key] = pts_ns
            elif pts_ns is not None and self._packet_pts[key] not in (None, -1, pts_ns):
                self._packet_pts[key] = -1  # conflicting AU timestamp, never unpoison
                self.ambiguous += 1
            elif pts_ns is not None and self._packet_pts[key] is None:
                self._packet_pts[key] = pts_ns
            while len(self._packet_pts) > self.capacity:
                self._packet_pts.popitem(last=False)
            if not packet.marker:
                return None
            encoded_pts = self._packet_pts.pop(key, None)
            source = self._sources.pop(encoded_pts, None) if encoded_pts is not None and encoded_pts >= 0 else None
            if source is None:
                self.withheld += 1
                return None
            self.mapped += 1
        return {
            "rtp_ssrc": packet.key.ssrc,
            "rtp_timestamp": packet.key.timestamp,
            "tracker_session": source.tracker_session,
            "frame_id": source.frame_id,
            "source_time_ns": source.source_time_ns,
            "source_clock_domain": source.source_clock_domain,
        }
