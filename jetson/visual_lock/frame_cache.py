"""Small exact-identity cache for panel seeds; never selects by arrival order."""

from __future__ import annotations

from collections import deque

from common.visual_roi_seed import VisualRoiSeed


class ExactFrameCache:
    def __init__(self, capacity: int = 16, max_span_ns: int = 400_000_000):
        if not 1 <= capacity <= 32 or max_span_ns <= 0:
            raise ValueError("invalid exact-frame cache bounds")
        self.capacity = capacity
        self.max_span_ns = max_span_ns
        self._frames = deque()

    def add(self, frame, pixels) -> None:
        if (self._frames and (frame.tracker_session != self._frames[-1][0].tracker_session
                or frame.frame_id <= self._frames[-1][0].frame_id
                or frame.source_time_ns <= self._frames[-1][0].source_time_ns)):
            self.clear()
        self._frames.append((frame, pixels))
        while len(self._frames) > self.capacity or (
                self._frames and frame.source_time_ns - self._frames[0][0].source_time_ns > self.max_span_ns):
            self._release(self._frames.popleft()[1])

    def replay_from(self, request: VisualRoiSeed):
        if not self._frames:
            return None
        for index, (frame, _) in enumerate(self._frames):
            if (frame.tracker_session == request.tracker_session
                    and frame.frame_id == request.frame_id
                    and frame.source_time_ns == request.source_time_ns
                    and frame.source_clock_domain == request.source_clock_domain
                    and (frame.width, frame.height) == (request.width, request.height)
                    and frame.source_identity_verified is True):
                replay = list(self._frames)[index:]
                if replay[-1][0].source_time_ns - frame.source_time_ns > self.max_span_ns:
                    return None
                if any(b.source_time_ns - a.source_time_ns > 150_000_000
                       for (a, _), (b, _) in zip(replay, replay[1:])):
                    return None
                return replay
        return None

    def clear(self) -> None:
        while self._frames:
            self._release(self._frames.popleft()[1])

    @staticmethod
    def _release(pixels) -> None:
        close = getattr(pixels, "close", None)
        if close is not None: close()
