"""Control-free transport for correlated PerceptionSnapshot V2 metadata."""

from __future__ import annotations

import threading
from collections import Counter
from typing import Any

import zmq

from common.perception import PerceptionSnapshotV2, perception_snapshot_to_json
from jetson.deepstream.header_correlation import FrameHeader, HeaderCorrelator
from common.rtp_identity import RtpFrameKey
from jetson.control.frame_identity import FrameIdentityJoiner, SourceFrameHeader


class SnapshotTransport:
    """Correlate optional PC headers and publish only immutable V2 snapshots."""

    def __init__(
        self,
        *,
        header_bind: str | None,
        snapshot_bind: str,
        capacity: int = 8,
        verified_rtp_headers: bool = False,
    ) -> None:
        if not snapshot_bind:
            raise ValueError("snapshot_bind is required")
        self._ctx = zmq.Context()
        self._pull = None
        if header_bind is not None:
            self._pull = self._ctx.socket(zmq.PULL)
            self._pull.setsockopt(zmq.RCVHWM, capacity)
            self._pull.setsockopt(zmq.LINGER, 0)
            self._pull.bind(header_bind)
        self._snapshot_pub = self._ctx.socket(zmq.PUB)
        self._snapshot_pub.setsockopt(zmq.SNDHWM, 1)
        self._snapshot_pub.setsockopt(zmq.LINGER, 0)
        self._snapshot_pub.bind(snapshot_bind)
        self.correlator = HeaderCorrelator(capacity=capacity)
        self.verified_rtp_headers = verified_rtp_headers
        self.identity_joiner = FrameIdentityJoiner(capacity=max(capacity, 256)) if verified_rtp_headers else None
        self._identity_lock = threading.Lock()
        self.identity_headers_accepted = 0
        self.identity_markers_seen = 0
        self.identity_matches = 0
        self.identity_failure_reasons: Counter[str] = Counter()
        self.invalid_headers = 0
        self.published = 0
        self.withheld_no_header = 0

    def drain_headers(self) -> None:
        if self._pull is None:
            return
        while True:
            try:
                payload: Any = self._pull.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            if not isinstance(payload, dict):
                self.invalid_headers += 1
                continue
            if self.identity_joiner is not None:
                try:
                    header = SourceFrameHeader(
                        key=RtpFrameKey(int(payload["rtp_ssrc"]), int(payload["rtp_timestamp"])),
                        frame_id=int(payload["frame_id"]),
                        source_time_ns=int(payload["source_time_ns"]),
                        source_clock_domain=str(payload["source_clock_domain"]),
                    )
                    with self._identity_lock:
                        self.identity_joiner.push_header(header)
                        self.identity_headers_accepted += 1
                except (KeyError, TypeError, ValueError):
                    self.invalid_headers += 1
            elif not self.correlator.push_mapping(payload):
                self.invalid_headers += 1

    def push_rtp_marker(self, *, decoded_pts_ns: int, key: RtpFrameKey) -> None:
        if self.identity_joiner is not None:
            with self._identity_lock:
                self.identity_joiner.push_marker(decoded_pts_ns=decoded_pts_ns, key=key)
                self.identity_markers_seen += 1

    def next_header(self, *, decoded_pts_ns: int | None = None) -> FrameHeader | None:
        if self.identity_joiner is not None:
            with self._identity_lock:
                result = (
                    self.identity_joiner.match(decoded_pts_ns=decoded_pts_ns)
                    if decoded_pts_ns is not None else None
                )
                if result is not None:
                    if result.verified:
                        self.identity_matches += 1
                    else:
                        self.identity_failure_reasons[result.reason] += 1
            source = result.header if result is not None and result.verified else None
            header = None if source is None else FrameHeader(
                frame_id=source.frame_id,
                src_ts_ms=source.source_time_ns // 1_000_000,
                source_time_ns=source.source_time_ns,
                source_identity_verified=True,
            )
        else:
            header = self.correlator.match_next().header
        if header is None:
            self.withheld_no_header += 1
        return header

    @property
    def requires_headers(self) -> bool:
        return self._pull is not None

    def publish(self, snapshot: PerceptionSnapshotV2) -> bool:
        try:
            self._snapshot_pub.send_string(
                perception_snapshot_to_json(snapshot), flags=zmq.NOBLOCK
            )
        except zmq.Again:
            return False
        self.published += 1
        return True

    def report(self) -> dict[str, int | bool]:
        return {
            "header_correlation": self.requires_headers,
            "published": self.published,
            "withheld_no_header": self.withheld_no_header,
            "invalid_headers": self.invalid_headers,
            "dropped_overflow": self.correlator.dropped_overflow,
            "dropped_nonmonotonic": self.correlator.dropped_nonmonotonic,
            "verified_rtp_headers": self.verified_rtp_headers,
            "identity_ambiguous_headers": 0 if self.identity_joiner is None else self.identity_joiner.ambiguous_headers,
            "identity_ambiguous_markers": 0 if self.identity_joiner is None else self.identity_joiner.ambiguous_markers,
            "identity_headers_accepted": self.identity_headers_accepted,
            "identity_markers_seen": self.identity_markers_seen,
            "identity_matches": self.identity_matches,
            "identity_failure_reasons": dict(self.identity_failure_reasons),
        }

    def close(self) -> None:
        if self._pull is not None:
            self._pull.close()
        self._snapshot_pub.close()
        self._ctx.destroy(linger=0)
