"""Flight recorder: store every movement and control message for analysis.

Passively subscribes to the configured ZMQ PUB streams (perception snapshots
and target selection, controller ticks and panel states, controller intents,
gimbal state, serial commands/replies, simulator truth) and writes each
message, stamped with its receive time, to chronological gzip JSONL segments:

    <root>/<session>/segment-000001.jsonl.gz   one record per line:
    {"rx_ns": monotonic receive, "wall_ns": Unix time, "stream": name,
     "topic": serial topic or null, "msg": the message as published}

Segments rotate every ``segment_s`` and are sync-flushed every 2 s, so an
open segment is readable up to its last flush. When the root exceeds
``max_total_gb`` the oldest closed segments are deleted. Read-only: it
never publishes or binds.

    python -m tools.flight_recorder --config configs/recorder/jetson.yaml
    python -m tools.flight_log summary <root or session>
"""

from __future__ import annotations

import argparse
import gzip
import json
import socket
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
import zmq

from common.shutdown import install_signal_handlers

FLUSH_INTERVAL_S = 2.0


@dataclass(frozen=True)
class StreamSpec:
    name: str
    endpoint: str
    subscribe: str = ""


@dataclass(frozen=True)
class RecorderConfig:
    root: Path
    segment_s: float
    max_total_bytes: int
    streams: tuple[StreamSpec, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RecorderConfig":
        section = raw.get("recorder")
        if not isinstance(section, Mapping):
            raise ValueError("missing recorder section")
        streams = tuple(
            StreamSpec(str(s["name"]), str(s["endpoint"]), str(s.get("subscribe", "")))
            for s in section.get("streams") or ()
        )
        if not streams:
            raise ValueError("recorder.streams must list at least one stream")
        if len({s.name for s in streams}) != len(streams):
            raise ValueError("recorder stream names must be unique")
        segment_s = float(section.get("segment_s", 300))
        max_gb = float(section.get("max_total_gb", 10))
        if not 10 <= segment_s <= 3600:
            raise ValueError("recorder.segment_s must be in [10, 3600]")
        if not 0.1 <= max_gb <= 1000:
            raise ValueError("recorder.max_total_gb must be in [0.1, 1000]")
        return cls(Path(str(section["root"])).expanduser(), segment_s, int(max_gb * 1e9), streams)


def decode(payload: bytes) -> tuple[str | None, Any]:
    """Split a published message into (topic, JSON body); keep text if not JSON."""
    text = payload.decode("utf-8", errors="replace")
    try:
        return None, json.loads(text)
    except ValueError:
        pass
    topic, sep, body = text.partition(" ")
    if sep:
        try:
            return topic, json.loads(body)
        except ValueError:
            pass
    return None, text


def prune(root: Path, max_total_bytes: int, *, keep: Path | None = None) -> list[Path]:
    """Delete the oldest closed segments until the root fits the cap."""
    segments = sorted(root.glob("*/segment-*.jsonl.gz"), key=lambda p: (p.parent.name, p.name))
    total = sum(p.stat().st_size for p in segments)
    removed = []
    for path in segments:
        if total <= max_total_bytes:
            break
        if keep is not None and path == keep:
            continue
        total -= path.stat().st_size
        path.unlink()
        removed.append(path)
        if not any(path.parent.glob("segment-*")):
            for leftover in path.parent.iterdir():
                leftover.unlink()
            path.parent.rmdir()
    return removed


class SegmentWriter:
    def __init__(self, session_dir: Path, segment_s: float) -> None:
        self._dir = session_dir
        self._segment_s = segment_s
        self._index = 0
        self._file: gzip.GzipFile | None = None
        self._opened_s = 0.0
        self._flushed_s = 0.0
        self.path: Path | None = None

    def write(self, record: dict, now_s: float) -> Path | None:
        """Write one record; returns the just-closed segment when it rotated."""
        closed = None
        if self._file is None or now_s - self._opened_s >= self._segment_s:
            closed = self.close()
            self._index += 1
            self.path = self._dir / f"segment-{self._index:06d}.jsonl.gz"
            self._file = gzip.open(self.path, "wb", compresslevel=6)
            self._opened_s = self._flushed_s = now_s
        self._file.write((json.dumps(record, separators=(",", ":")) + "\n").encode())
        if now_s - self._flushed_s >= FLUSH_INTERVAL_S:
            self._file.flush(zlib.Z_SYNC_FLUSH)
            self._flushed_s = now_s
        return closed

    def close(self) -> Path | None:
        if self._file is None:
            return None
        self._file.close()
        self._file = None
        return self.path


def run(cfg: RecorderConfig, *, duration_s: float | None = None, stop=None) -> dict:
    cfg.root.mkdir(parents=True, exist_ok=True)
    started_wall = time.strftime("%Y%m%dT%H%M%S")
    session_dir = cfg.root / f"{started_wall}-{socket.gethostname()}"
    session_dir.mkdir()
    (session_dir / "session.json").write_text(json.dumps({
        "started_wall": started_wall, "host": socket.gethostname(),
        "started_monotonic_ns": time.monotonic_ns(), "started_wall_ns": time.time_ns(),
        "streams": [s.__dict__ for s in cfg.streams],
    }, indent=1))
    context = zmq.Context()
    sockets = {}
    poller = zmq.Poller()
    for spec in cfg.streams:
        sock = context.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, 100_000)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt_string(zmq.SUBSCRIBE, spec.subscribe)
        sock.connect(spec.endpoint)
        sockets[sock] = spec.name
        poller.register(sock, zmq.POLLIN)
    writer = SegmentWriter(session_dir, cfg.segment_s)
    counts: Counter[str] = Counter()
    started = last_report = time.monotonic()
    print(json.dumps({"flight_recorder": "started", "session": str(session_dir),
                      "streams": [s.name for s in cfg.streams]}), flush=True)
    try:
        while (stop is None or not stop.is_set()) and (
                duration_s is None or time.monotonic() - started < duration_s):
            for sock, _ in poller.poll(200):
                # Drain everything waiting on this socket in one go.
                while True:
                    try:
                        payload = sock.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    topic, message = decode(payload)
                    name = sockets[sock]
                    counts[name] += 1
                    closed = writer.write({"rx_ns": time.monotonic_ns(), "wall_ns": time.time_ns(),
                                           "stream": name, "topic": topic, "msg": message},
                                          time.monotonic())
                    if closed is not None:
                        prune(cfg.root, cfg.max_total_bytes, keep=writer.path)
            now = time.monotonic()
            if now - last_report >= 60:
                last_report = now
                print(json.dumps({"flight_recorder": "recording", "segment": str(writer.path),
                                  "messages": dict(counts)}), flush=True)
    finally:
        writer.close()
        for sock in sockets:
            sock.close(0)
        context.term()
        prune(cfg.root, cfg.max_total_bytes)
    summary = {"flight_recorder": "stopped", "session": str(session_dir), "messages": dict(counts)}
    print(json.dumps(summary), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, help="override recorder.root")
    parser.add_argument("--duration-s", type=float)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if args.root is not None:
        raw.setdefault("recorder", {})["root"] = str(args.root)
    cfg = RecorderConfig.from_mapping(raw)
    run(cfg, duration_s=args.duration_s, stop=install_signal_handlers())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
