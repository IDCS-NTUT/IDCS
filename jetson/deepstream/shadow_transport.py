"""Control-free ZMQ transport for the DeepStream shadow runtime."""

from __future__ import annotations

from typing import Any

import zmq

from common.schemas import DetectionMsg, detection_msg_to_json
from jetson.deepstream.header_correlation import HeaderCorrelator, FrameHeader


class ShadowTransport:
    """PULL PC frame headers and PUB only correlated detection metadata.

    No control socket is created here by design.  A caller must opt in with
    explicit bind endpoints; this keeps the DeepStream migration path unable
    to actuate hardware.
    """

    def __init__(self, *, header_bind: str | None, result_bind: str, capacity: int = 8) -> None:
        self._ctx = zmq.Context()
        self._pull = None
        if header_bind is not None:
            self._pull = self._ctx.socket(zmq.PULL)
            self._pull.setsockopt(zmq.RCVHWM, capacity)
            self._pull.setsockopt(zmq.LINGER, 0)
            self._pull.bind(header_bind)
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.SNDHWM, 1)
        self._pub.setsockopt(zmq.LINGER, 0)
        self._pub.bind(result_bind)
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

    def publish(self, message: DetectionMsg) -> bool:
        try:
            self._pub.send_string(detection_msg_to_json(message), flags=zmq.NOBLOCK)
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
        self._pub.close()
        self._ctx.term()
