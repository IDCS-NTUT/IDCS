"""Background source-clock calibration with bounded uncertainty and age."""

from __future__ import annotations

import json
import argparse
import threading
import time
from collections import deque

import zmq

from common.clock_sync import ClockOffsetSample, calculate_clock_offset


class ClockSyncClient:
    def __init__(self, endpoint: str, *, max_rtt_ms: float = 10.0) -> None:
        self.endpoint = endpoint
        self.max_rtt_ns = int(max_rtt_ms * 1_000_000)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._samples: deque[ClockOffsetSample] = deque(maxlen=10)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def best_sample(self, *, now_ns: int) -> ClockOffsetSample | None:
        with self._lock:
            recent = [
                sample for sample in self._samples
                if 0 <= now_ns - sample.observed_jetson_ns <= 5_000_000_000
            ]
        return min(recent, key=lambda sample: sample.round_trip_ns) if recent else None

    def _run(self) -> None:
        context = zmq.Context()
        try:
            while not self._stop.is_set():
                socket = context.socket(zmq.REQ)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVTIMEO, 500)
                try:
                    socket.connect(self.endpoint)
                    sent_ns = time.monotonic_ns()
                    socket.send_json({"version": 1, "jetson_send_ns": sent_ns})
                    reply = json.loads(socket.recv())
                    received_ns = time.monotonic_ns()
                    if reply.get("version") == 1 and reply.get("jetson_send_ns") == sent_ns:
                        sample = calculate_clock_offset(
                            sent_ns,
                            int(reply["pc_receive_ns"]),
                            int(reply["pc_send_ns"]),
                            received_ns,
                        )
                        if sample.round_trip_ns <= self.max_rtt_ns:
                            with self._lock:
                                self._samples.append(sample)
                except (ValueError, TypeError, KeyError, zmq.ZMQError):
                    pass
                finally:
                    socket.close(0)
                self._stop.wait(1.0)
        finally:
            context.term()


def probe() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--wait-s", type=float, default=4.0)
    args = parser.parse_args()
    client = ClockSyncClient(args.endpoint)
    client.start()
    try:
        deadline = time.monotonic() + args.wait_s
        while time.monotonic() < deadline:
            sample = client.best_sample(now_ns=time.monotonic_ns())
            if sample is not None:
                print(json.dumps({
                    "offset_ns": sample.offset_ns,
                    "round_trip_ms": sample.round_trip_ns / 1_000_000.0,
                    "uncertainty_ms": sample.uncertainty_ns / 1_000_000.0,
                }, sort_keys=True))
                return 0
            time.sleep(0.1)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(probe())
