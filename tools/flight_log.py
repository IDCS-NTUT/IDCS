"""Read flight-recorder sessions (see tools/flight_recorder.py).

    python -m tools.flight_log summary <root|session>       sessions, spans, counts
    python -m tools.flight_log extract <session> --stream controller [--type tick] > ticks.jsonl

Python use: ``for record in iter_records(session_dir, streams={"controller"})``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator


def sessions(path: Path) -> list[Path]:
    """A session directory, or every session under a recorder root."""
    if (path / "session.json").exists():
        return [path]
    return sorted(p for p in path.iterdir() if (p / "session.json").exists())


def _lines(segment: Path) -> Iterator[bytes]:
    # An open or crashed segment ends mid-stream; read up to the last flush.
    with gzip.open(segment, "rb") as handle:
        try:
            yield from handle
        except (EOFError, zlib.error, gzip.BadGzipFile):
            return


def iter_records(session: Path, *, streams: Iterable[str] | None = None) -> Iterator[dict]:
    wanted = set(streams) if streams is not None else None
    for segment in sorted(session.glob("segment-*.jsonl.gz")):
        for line in _lines(segment):
            try:
                record = json.loads(line)
            except ValueError:
                continue  # a line cut by the last flush
            if wanted is None or record.get("stream") in wanted:
                yield record


def summarize(session: Path) -> dict:
    counts: Counter[str] = Counter()
    first = last = None
    for record in iter_records(session):
        counts[record["stream"]] += 1
        first = record["wall_ns"] if first is None else first
        last = record["wall_ns"]
    size = sum(p.stat().st_size for p in session.glob("segment-*"))
    return {"session": session.name, "span_s": None if first is None else round((last - first) / 1e9, 1),
            "size_mb": round(size / 1e6, 1), "messages": dict(counts)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    summary = sub.add_parser("summary")
    summary.add_argument("path", type=Path)
    extract = sub.add_parser("extract")
    extract.add_argument("session", type=Path)
    extract.add_argument("--stream", action="append", required=True)
    extract.add_argument("--type", help="only messages whose msg.type matches (e.g. tick, manual)")
    args = parser.parse_args()
    if args.command == "summary":
        for session in sessions(args.path):
            print(json.dumps(summarize(session)))
        return 0
    for record in iter_records(args.session, streams=args.stream):
        msg = record.get("msg")
        if args.type and not (isinstance(msg, dict) and msg.get("type") == args.type):
            continue
        sys.stdout.write(json.dumps(record, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
