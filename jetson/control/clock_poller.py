"""Dedicated source-clock polling, off the fixed-rate control loop."""

from __future__ import annotations

import json
import threading
import time
from collections import Counter

import zmq

from jetson.control.clock_watchdog import ClockWatchdog, ClockWatchdogConfig
from jetson.control.timing import ClockBounds


def exchange_clock(context: zmq.Context, endpoint: str) -> ClockBounds | None:
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 50)
    socket.setsockopt(zmq.SNDTIMEO, 50)
    try:
        socket.connect(endpoint)
        sent_ns = time.monotonic_ns()
        socket.send_json({"version": 1, "jetson_send_ns": sent_ns})
        reply = json.loads(socket.recv())
        received_ns = time.monotonic_ns()
        if reply.get("version") != 1 or reply.get("jetson_send_ns") != sent_ns:
            return None
        return ClockBounds.from_exchange(
            jetson_send_ns=sent_ns,
            pc_receive_ns=int(reply["pc_receive_ns"]),
            pc_send_ns=int(reply["pc_send_ns"]),
            jetson_receive_ns=received_ns,
            max_drift_ppm=None,
        )
    except (ValueError, TypeError, KeyError, zmq.ZMQError):
        return None
    finally:
        socket.close(0)


class ClockPoller:
    def __init__(
        self, endpoint: str, policy: ClockWatchdogConfig, *, interval_s: float = 0.05,
    ) -> None:
        if not 0.02 <= interval_s <= 0.1:
            raise ValueError("clock poll interval must be in [20, 100] ms")
        self.endpoint = endpoint
        self.interval_s = interval_s
        self.watchdog = ClockWatchdog(policy)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stats: Counter[str] = Counter()

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def bounds(self, *, now_ns: int) -> tuple[ClockBounds | None, str]:
        with self._lock:
            return self.watchdog.bounds(jetson_now_ns=now_ns)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def _run(self) -> None:
        context = zmq.Context()
        try:
            while not self._stop.is_set():
                sample = exchange_clock(context, self.endpoint)
                self._observe_sample(sample)
                self._stop.wait(self.interval_s)
        finally:
            context.destroy(linger=0)

    def _observe_sample(self, sample: ClockBounds | None) -> None:
        with self._lock:
            if sample is None:
                self._stats["exchange_failed"] += 1
                return
            width_ns = sample.offset_max_ns - sample.offset_min_ns
            self._stats["max_exchange_width_ns"] = max(
                self._stats["max_exchange_width_ns"], width_ns,
            )
            outcome = self.watchdog.observe(sample)
            self._stats[outcome] += 1
            if outcome == "clock_exchange_uncertainty_exceeded":
                # One over-wide software-timestamp exchange is unusable, but
                # does not prove oscillator drift. Drop all previous bounds;
                # the controller holds until fresh clean samples requalify.
                self.watchdog.reset()
                self._stats["clock_requalifications"] += 1
