"""Control-free transport for correlated PerceptionSnapshot V2 metadata."""

from __future__ import annotations

from typing import Any

import zmq

from common.perception import PerceptionSnapshotV2, perception_snapshot_to_json
from jetson.deepstream.header_correlation import FrameHeader, HeaderCorrelator


class SnapshotTransport:
    """Correlate optional PC headers and publish only immutable V2 snapshots."""

    def __init__(
        self,
        *,
        header_bind: str | None,
        snapshot_bind: str,
        capacity: int = 8,
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
            if not isinstance(payload, dict) or not self.correlator.push_mapping(payload):
                self.invalid_headers += 1

    def next_header(self) -> FrameHeader | None:
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
        }

    def close(self) -> None:
        if self._pull is not None:
            self._pull.close()
        self._snapshot_pub.close()
        self._ctx.term()
