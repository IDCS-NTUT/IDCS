"""PC monotonic clock responder for timestamped video source frames."""

from __future__ import annotations

import argparse
import json
import threading
import time

import zmq

from common.shutdown import install_signal_handlers


class ClockSyncResponder:
    """Own one REP socket entirely within its worker thread."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._startup_error: Exception | None = None
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(2.0):
            raise RuntimeError("clock responder did not start")
        if self._startup_error is not None:
            raise RuntimeError(f"clock responder failed: {self._startup_error}")

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _serve(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            try:
                socket.bind(self.endpoint)
            except Exception as exc:  # noqa: BLE001
                self._startup_error = exc
                return
            finally:
                self._ready.set()
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            while not self._stop.is_set():
                if socket not in dict(poller.poll(100)):
                    continue
                received_ns = time.monotonic_ns()
                try:
                    request = json.loads(socket.recv())
                    sent_ns = time.monotonic_ns()
                    reply = {
                        "version": 1,
                        "jetson_send_ns": int(request["jetson_send_ns"]),
                        "pc_receive_ns": received_ns,
                        "pc_send_ns": sent_ns,
                    }
                except (ValueError, TypeError, KeyError):
                    reply = {"error": "invalid_clock_request"}
                socket.send_json(reply)
        finally:
            socket.close(0)
            context.destroy(linger=0)


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", required=True, help="PC LAN TCP endpoint")
    parser.add_argument("--duration-s", type=float)
    args = parser.parse_args()
    if args.duration_s is not None and args.duration_s <= 0:
        parser.error("--duration-s must be positive")
    responder = ClockSyncResponder(args.bind)
    responder.start()
    stop = install_signal_handlers()
    deadline = None if args.duration_s is None else time.monotonic() + args.duration_s
    try:
        while not stop.wait(0.1) and (deadline is None or time.monotonic() < deadline):
            pass
    finally:
        responder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
