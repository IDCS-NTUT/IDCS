"""Control-free V2 transport with an explicit legacy display adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import zmq

from common.perception import (
    PerceptionSnapshotV2,
    perception_snapshot_to_json,
)
from common.perception_compat import detection_msg_from_snapshot
from common.schemas import detection_msg_to_json
from jetson.deepstream.header_correlation import HeaderCorrelator, FrameHeader


@dataclass
class LegacyDetectionJsonlWriter:
    """Write legacy replay JSON only at an explicitly requested sink."""

    path: Path
    handle: Any = field(init=False)
    messages: int = 0

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")

    def write(self, snapshot: PerceptionSnapshotV2) -> None:
        message = detection_msg_from_snapshot(snapshot, use_tracks=None)
        self.handle.write(detection_msg_to_json(message) + "\n")
        self.messages += 1

    def close(self) -> None:
        self.handle.close()


class ShadowTransport:
    """PULL PC frame headers and PUB only correlated detection metadata.

    No control socket is created here by design.  A caller must opt in with
    explicit bind endpoints; this keeps the DeepStream migration path unable
    to actuate hardware.
    """

    def __init__(
        self,
        *,
        header_bind: str | None,
        result_bind: str | None,
        snapshot_bind: str | None = None,
        capacity: int = 8,
    ) -> None:
        if result_bind is None and snapshot_bind is None:
            raise ValueError("at least one result endpoint is required")
        if result_bind is not None and result_bind == snapshot_bind:
            raise ValueError("V2 and legacy result endpoints must be distinct")
        self._ctx = zmq.Context()
        self._pull = None
        if header_bind is not None:
            self._pull = self._ctx.socket(zmq.PULL)
            self._pull.setsockopt(zmq.RCVHWM, capacity)
            self._pull.setsockopt(zmq.LINGER, 0)
            self._pull.bind(header_bind)
        self._legacy_pub = None
        if result_bind is not None:
            self._legacy_pub = self._ctx.socket(zmq.PUB)
            self._legacy_pub.setsockopt(zmq.SNDHWM, 1)
            self._legacy_pub.setsockopt(zmq.LINGER, 0)
            self._legacy_pub.bind(result_bind)
        self._snapshot_pub = None
        if snapshot_bind is not None:
            self._snapshot_pub = self._ctx.socket(zmq.PUB)
            self._snapshot_pub.setsockopt(zmq.SNDHWM, 1)
            self._snapshot_pub.setsockopt(zmq.LINGER, 0)
            self._snapshot_pub.bind(snapshot_bind)
        self.correlator = HeaderCorrelator(capacity=capacity)
        self.invalid_headers = 0
        self.published = 0
        self.legacy_published = 0
        self.snapshot_published = 0
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
        """Publish V2 natively and adapt only for the legacy result socket."""

        sent = False
        if self._snapshot_pub is not None:
            try:
                self._snapshot_pub.send_string(
                    perception_snapshot_to_json(snapshot), flags=zmq.NOBLOCK
                )
            except zmq.Again:
                pass
            else:
                self.snapshot_published += 1
                sent = True
        if self._legacy_pub is not None:
            message = detection_msg_from_snapshot(snapshot, use_tracks=None)
            try:
                self._legacy_pub.send_string(
                    detection_msg_to_json(message), flags=zmq.NOBLOCK
                )
            except zmq.Again:
                pass
            else:
                self.legacy_published += 1
                sent = True
        if sent:
            self.published += 1
        return sent

    def report(self) -> dict[str, int | bool]:
        return {
            "header_correlation": self.requires_headers,
            "published": self.published,
            "legacy_published": self.legacy_published,
            "snapshot_published": self.snapshot_published,
            "withheld_no_header": self.withheld_no_header,
            "invalid_headers": self.invalid_headers,
            "dropped_overflow": self.correlator.dropped_overflow,
            "dropped_nonmonotonic": self.correlator.dropped_nonmonotonic,
        }

    def close(self) -> None:
        if self._pull is not None:
            self._pull.close()
        if self._legacy_pub is not None:
            self._legacy_pub.close()
        if self._snapshot_pub is not None:
            self._snapshot_pub.close()
        self._ctx.term()
