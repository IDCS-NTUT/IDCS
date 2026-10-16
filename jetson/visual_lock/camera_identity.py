"""Exact Argus PTS -> sensor identity for the isolated camera ROI owner.

Read qdata before conversion/mux; consume it only at that exact frame's GPU
input. Missing or conflicting metadata never becomes a native frame counter.
"""
from collections import OrderedDict
import threading

from .return_identity import SourceIdentity


class CameraIdentityIndex:
    def __init__(self, tracker_session, capacity=256):
        if not tracker_session or capacity < 1:
            raise ValueError("camera session and positive capacity required")
        self.session = tracker_session
        self.capacity = capacity
        self._frames = OrderedDict()
        self._lock = threading.Lock()
        self.recorded = self.missing = self.conflicts = 0
        self._last = None
        self._last_pts = None

    def record(self, pts_ns, sensor):
        if (type(pts_ns) is not int or not 0 <= pts_ns < (1 << 63)-1
                or sensor is None or sensor.frame_number < 0 or sensor.start_ns <= 0):
            with self._lock:
                self.missing += 1
            return
        source = SourceIdentity(self.session, sensor.frame_number, sensor.start_ns, "jetson_monotonic")
        with self._lock:
            if pts_ns in self._frames and self._frames[pts_ns] != source:
                self._frames[pts_ns] = None
                self.conflicts += 1
            else:
                self._frames[pts_ns] = source
            self.recorded += 1
            while len(self._frames) > self.capacity:
                self._frames.popitem(last=False)

    def pop(self, pts_ns):
        with self._lock:
            source = self._frames.pop(pts_ns, None)
            if source is None:
                return None
            if self._last is not None and (pts_ns <= self._last_pts or source.frame_id <= self._last.frame_id
                    or source.source_time_ns <= self._last.source_time_ns):
                raise ValueError("camera sensor identity repeated or out of order")
            self._last = source
            self._last_pts = pts_ns
            return source
