"""Display-only decoded-return cadence watchdog; never grants control authority."""
from collections import deque
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class ReturnVideoHealth:
    threshold_fps: float = 20.0
    hold_s: float = 1.0
    window_s: float = 1.0
    stale_s: float = 1.0
    _frames: deque = field(default_factory=lambda: deque(maxlen=240))
    _started: float | None = None
    _low_since: float | None = None
    _lock: object = field(default_factory=Lock,init=False,repr=False,compare=False)

    def mark(self, now_s: float) -> None:
        # Pi invokes this on the GStreamer callback thread; warning/fps are UI
        # thread reads. Keep reconnect reset and timestamps one coherent sample.
        with self._lock:
            if self._frames and now_s <= self._frames[-1]:
                return  # Duplicate/out-of-order callbacks cannot improve cadence.
            if self._frames and now_s-self._frames[-1] > self.stale_s:
                self._frames.clear()
                self._started = None
                self._low_since = None
            if self._started is None:
                self._started = now_s
            self._frames.append(now_s)

    def fps(self, now_s: float) -> float:
        # Fixed elapsed window includes gaps; inter-frame-only rates hide stalls.
        with self._lock:
            return self._fps(now_s)

    def _fps(self, now_s: float) -> float:
        return sum(now_s-self.window_s < t <= now_s for t in self._frames) / self.window_s

    def warning(self, now_s: float) -> str | None:
        with self._lock:
            if not self._frames:
                self._low_since = None
                return 'NO VIDEO'
            if now_s-self._frames[-1] > self.stale_s:
                self._low_since = None
                return 'VIDEO LOST'
            if now_s-self._started < self.window_s or self._fps(now_s) >= self.threshold_fps:
                self._low_since = None
                return None
            if self._low_since is None:
                self._low_since = now_s
            return 'LOW VIDEO FPS' if now_s-self._low_since > self.hold_s else None
