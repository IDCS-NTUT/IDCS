"""Bounded PC-header correlation for the DeepStream migration path.

Video RTP and the ``CamState`` ZMQ channel have no shared transport sequence
number.  IDCS defines both as latest-only streams, so a DeepStream runtime must
not invent timestamps or reuse an arbitrarily old header.  This module keeps a
small ordered queue of validated headers and pairs at most one header with each
processed video frame.  It is deliberately independent of ZMQ/GStreamer so its
drop and freshness behaviour is unit-testable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Mapping


@dataclass(frozen=True)
class FrameHeader:
    frame_id: int
    src_ts_ms: int


@dataclass(frozen=True)
class HeaderCorrelation:
    header: FrameHeader | None
    dropped_stale: int
    pending: int


class HeaderCorrelator:
    """Pair ordered external headers to DeepStream frames without fabrication.

    Headers with a non-increasing frame id are discarded. A bounded queue drops
    its oldest item under pressure, matching the system's latest-only transport
    semantics. ``match_next`` returns ``None`` when no fresh header exists;
    callers must then withhold publication rather than create a fake IDCS
    ``DetectionMsg`` frame identity.
    """

    def __init__(self, *, capacity: int = 8) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._headers: Deque[FrameHeader] = deque()
        self._capacity = int(capacity)
        self._last_enqueued_frame_id: int | None = None
        self.dropped_overflow = 0
        self.dropped_nonmonotonic = 0

    def push_mapping(self, payload: Mapping[str, object]) -> bool:
        """Validate and enqueue a bare header or a ``CamState`` payload."""

        try:
            header = FrameHeader(
                frame_id=int(payload["frame_id"]),
                src_ts_ms=int(payload["src_ts_ms"]),
            )
        except (KeyError, TypeError, ValueError):
            return False
        if self._last_enqueued_frame_id is not None and header.frame_id <= self._last_enqueued_frame_id:
            self.dropped_nonmonotonic += 1
            return False
        if len(self._headers) >= self._capacity:
            self._headers.popleft()
            self.dropped_overflow += 1
        self._headers.append(header)
        self._last_enqueued_frame_id = header.frame_id
        return True

    def match_next(self) -> HeaderCorrelation:
        header = self._headers.popleft() if self._headers else None
        return HeaderCorrelation(
            header=header,
            dropped_stale=self.dropped_overflow,
            pending=len(self._headers),
        )
