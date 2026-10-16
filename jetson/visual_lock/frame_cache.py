"""Small exact-identity cache for panel seeds; never selects by arrival order."""

from __future__ import annotations

from collections import deque

from common.visual_roi_seed import VisualRoiSeed


def bounded_catchup(replay, max_gap_ns: int):
    """Preserve exact seed/end images; skip only known intermediate frames.

    A real gap or identity violation is never hidden from the tracker. Half its
    existing gap budget bounds sampled source intervals, not transport age.
    No image is synthesized, retimestamped, closed or reseeded here.
    """
    if max_gap_ns <= 0:
        raise ValueError('positive tracker gap budget required')
    frames = list(replay)
    for (a, _), (b, _) in zip(frames, frames[1:]):
        if (a.tracker_session != b.tracker_session or a.source_clock_domain != b.source_clock_domain
                or (a.width,a.height) != (b.width,b.height)
                or not a.source_identity_verified or not b.source_identity_verified
                or b.frame_id <= a.frame_id or not 0 < b.source_time_ns-a.source_time_ns <= max_gap_ns):
            return frames
    if len(frames) < 2:
        return frames
    chosen = [frames[0]]
    index = 0
    while index < len(frames)-1:
        following = index+1
        while (following+1 < len(frames) and
               frames[following+1][0].source_time_ns-frames[index][0].source_time_ns <= max_gap_ns//2):
            following += 1
        chosen.append(frames[following])
        index = following
    return chosen


class ExactFrameCache:
    def __init__(self, capacity: int = 16, max_span_ns: int = 400_000_000):
        if not 1 <= capacity <= 32 or max_span_ns <= 0:
            raise ValueError("invalid exact-frame cache bounds")
        self.capacity = capacity
        self.max_span_ns = max_span_ns
        self._frames = deque()
        self._decoded_pts = {}

    def add(self, frame, pixels, *, decoded_pts_ns=None) -> None:
        if (self._frames and (frame.tracker_session != self._frames[-1][0].tracker_session
                or frame.frame_id <= self._frames[-1][0].frame_id
                or frame.source_time_ns <= self._frames[-1][0].source_time_ns)):
            self.clear()
        self._frames.append((frame, pixels))
        self._decoded_pts[id(frame)] = decoded_pts_ns
        while len(self._frames) > self.capacity or (
                self._frames and frame.source_time_ns - self._frames[0][0].source_time_ns > self.max_span_ns):
            self._evict()

    def decoded_pts(self, frame):
        return self._decoded_pts.get(id(frame))

    def _evict(self):
        frame, pixels = self._frames.popleft()
        self._decoded_pts.pop(id(frame), None)
        self._release(pixels)

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
            self._evict()

    @staticmethod
    def _release(pixels) -> None:
        close = getattr(pixels, "close", None)
        if close is not None: close()
