"""Record and summarize serial command lifecycle and actuation-state messages."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional

import zmq

from common.serial_io import SerialReplySubscriber


class ExecutionAudit:
    def __init__(self) -> None:
        self.outcomes: Counter[str] = Counter()
        self.event_count = 0
        self.snapshot_count = 0
        self.sequence_gaps = 0
        self.duplicate_terminal_ids = 0
        self.epochs: set[str] = set()
        self._last_sequence_by_epoch: dict[str, int] = {}
        self._terminal_ids: set[tuple[str, str]] = set()
        self.delivery_ms: list[float] = []
        self.latest_accounting: dict[str, int] = {}
        self.latest_axes: dict[str, Any] = {}

    def ingest(self, message: Mapping[str, Any], *, received_ns: int) -> None:
        message_type = message.get("type")
        if message_type == "SerialCommandEventV1":
            self._ingest_event(message, received_ns=received_ns)
        elif message_type == "SerialActuationStateV1":
            self._ingest_snapshot(message, received_ns=received_ns)

    def _observe_sequence(self, epoch: str, sequence: int) -> None:
        previous = self._last_sequence_by_epoch.get(epoch)
        if previous is not None and sequence > previous + 1:
            self.sequence_gaps += sequence - previous - 1
        if previous is None or sequence > previous:
            self._last_sequence_by_epoch[epoch] = sequence

    def _observe_delivery(self, message: Mapping[str, Any], received_ns: int) -> None:
        timing = message.get("timing")
        event_ns: Optional[int]
        if isinstance(timing, Mapping):
            try:
                event_ns = int(timing.get("event_monotonic_ns"))
            except (TypeError, ValueError):
                event_ns = None
        else:
            try:
                event_ns = int(message.get("event_monotonic_ns"))
            except (TypeError, ValueError):
                event_ns = None
        if event_ns is not None and received_ns >= event_ns:
            self.delivery_ms.append((received_ns - event_ns) / 1e6)

    def _ingest_event(self, message: Mapping[str, Any], *, received_ns: int) -> None:
        try:
            epoch = str(message["service_epoch"])
            sequence = int(message["sequence"])
        except (KeyError, TypeError, ValueError):
            return
        self.epochs.add(epoch)
        self._observe_sequence(epoch, sequence)
        self._observe_delivery(message, received_ns)
        self.event_count += 1
        self.outcomes[str(message.get("event", "unknown"))] += 1
        terminal_key = (epoch, str(message.get("cmd_id", "")))
        if terminal_key in self._terminal_ids:
            self.duplicate_terminal_ids += 1
        self._terminal_ids.add(terminal_key)
        accounting = message.get("accounting")
        if isinstance(accounting, Mapping):
            self.latest_accounting = {
                key: int(accounting.get(key, 0))
                for key in ("admitted", "terminal", "pending")
            }

    def _ingest_snapshot(
        self, message: Mapping[str, Any], *, received_ns: int
    ) -> None:
        try:
            epoch = str(message["service_epoch"])
            sequence = int(message["event_sequence"])
        except (KeyError, TypeError, ValueError):
            return
        self.epochs.add(epoch)
        self._observe_sequence(epoch, sequence)
        self._observe_delivery(message, received_ns)
        self.snapshot_count += 1
        accounting = message.get("accounting")
        if isinstance(accounting, Mapping):
            self.latest_accounting = {
                key: int(accounting.get(key, 0))
                for key in ("admitted", "terminal", "pending")
            }
        axes = message.get("axes")
        if isinstance(axes, Mapping):
            self.latest_axes = dict(axes)

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
        return ordered[index]

    def report(self) -> dict[str, Any]:
        admitted = self.latest_accounting.get("admitted")
        terminal = self.latest_accounting.get("terminal")
        pending = self.latest_accounting.get("pending")
        return {
            "type": "SerialExecutionAuditReportV1",
            "event_count": self.event_count,
            "snapshot_count": self.snapshot_count,
            "outcomes": dict(sorted(self.outcomes.items())),
            "epochs": sorted(self.epochs),
            "sequence_gaps": self.sequence_gaps,
            "duplicate_terminal_ids": self.duplicate_terminal_ids,
            "latest_accounting": self.latest_accounting,
            "accounting_complete": (
                admitted is not None
                and terminal is not None
                and pending == 0
                and admitted == terminal
            ),
            "delivery_ms": {
                "count": len(self.delivery_ms),
                "mean": statistics.fmean(self.delivery_ms)
                if self.delivery_ms
                else None,
                "p95": self._percentile(self.delivery_ms, 0.95),
                "p99": self._percentile(self.delivery_ms, 0.99),
                "max": max(self.delivery_ms) if self.delivery_ms else None,
            },
            "latest_axes": self.latest_axes,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint", default="tcp://127.0.0.1:5572", help="Serial PUB endpoint"
    )
    parser.add_argument("--target", default="gimbal")
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.duration_s <= 0.0:
        raise SystemExit("--duration-s must be positive")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)

    context = zmq.Context()
    subscriber = SerialReplySubscriber(
        args.endpoint,
        topics=[
            f"serial.command.{args.target}",
            f"serial.actuation.{args.target}",
        ],
        ctx=context,
    )
    audit = ExecutionAudit()
    trace_file = (
        args.trace.open("w", encoding="utf-8") if args.trace is not None else None
    )
    deadline = time.monotonic() + args.duration_s
    try:
        while time.monotonic() < deadline:
            messages = subscriber.recv_nowait()
            if not messages:
                time.sleep(0.002)
                continue
            for message in messages:
                received_ns = time.monotonic_ns()
                audit.ingest(message, received_ns=received_ns)
                if trace_file is not None:
                    trace_file.write(
                        json.dumps(
                            {"received_monotonic_ns": received_ns, "message": message},
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
    finally:
        if trace_file is not None:
            trace_file.close()
        subscriber.close()
        context.destroy(linger=0)

    report = audit.report()
    report["duration_s"] = args.duration_s
    report["endpoint"] = args.endpoint
    report["target"] = args.target
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
