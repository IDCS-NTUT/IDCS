"""Passive receiver-side monitor for IDCS DetectionMsg streams."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import zmq

from common.schemas import DetectionMsg, detection_msg_from_json, detection_msg_to_json


@dataclass
class MetadataMetrics:
    messages: int = 0
    boxes: int = 0
    selected: int = 0
    invalid: int = 0
    nonmonotonic_frame_ids: int = 0
    nonmonotonic_source_timestamps: int = 0
    frame_gaps: int = 0
    classes: Counter[str] = field(default_factory=Counter)
    tracker_ids: set[int] = field(default_factory=set)
    tracker_observations: Counter[int] = field(default_factory=Counter)
    selected_tracker_ids: set[int] = field(default_factory=set)
    selected_track_changes: int = 0
    _last_frame_id: int | None = None
    _last_src_ts_ms: int | None = None
    _last_selected_track_id: int | None = None

    def ingest(self, message: DetectionMsg) -> None:
        if self._last_frame_id is not None:
            if message.frame_id <= self._last_frame_id:
                self.nonmonotonic_frame_ids += 1
            else:
                self.frame_gaps += max(0, message.frame_id - self._last_frame_id - 1)
        if self._last_src_ts_ms is not None and message.src_ts_ms <= self._last_src_ts_ms:
            self.nonmonotonic_source_timestamps += 1
        self._last_frame_id, self._last_src_ts_ms = message.frame_id, message.src_ts_ms
        self.messages += 1
        self.boxes += len(message.boxes)
        self.selected += int(message.target_idx is not None)
        for box in message.boxes:
            self.classes[box.cls] += 1
            if box.track_id is not None:
                self.tracker_ids.add(int(box.track_id))
                self.tracker_observations[int(box.track_id)] += 1
        target_track_id = message.target_track_id
        if target_track_id is None and message.target_idx is not None and 0 <= message.target_idx < len(message.boxes):
            target_track_id = message.boxes[message.target_idx].track_id
        if target_track_id is not None:
            target_track_id = int(target_track_id)
            if self._last_selected_track_id is not None and target_track_id != self._last_selected_track_id:
                self.selected_track_changes += 1
            self._last_selected_track_id = target_track_id
            self.selected_tracker_ids.add(target_track_id)

    def report(self) -> dict[str, object]:
        return {"messages": self.messages, "boxes": self.boxes, "selected": self.selected,
                "invalid": self.invalid, "nonmonotonic_frame_ids": self.nonmonotonic_frame_ids,
                "nonmonotonic_source_timestamps": self.nonmonotonic_source_timestamps,
                "frame_gaps": self.frame_gaps, "classes": dict(sorted(self.classes.items())),
                "unique_tracker_ids": len(self.tracker_ids),
                "tracker_observations": dict(sorted(self.tracker_observations.items())),
                "selected_tracker_ids": sorted(self.selected_tracker_ids),
                "selected_track_changes": self.selected_track_changes}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.duration_s <= 0:
        parser.error("--duration-s must be positive")
    context = zmq.Context(); sub = context.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 1); sub.setsockopt(zmq.CONFLATE, 1); sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt_string(zmq.SUBSCRIBE, ""); sub.connect(args.endpoint)
    metrics = MetadataMetrics(); output = args.jsonl.open("w", encoding="utf-8") if args.jsonl else None
    deadline = time.monotonic() + args.duration_s
    try:
        while time.monotonic() < deadline:
            try:
                payload = sub.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.002); continue
            try:
                message = detection_msg_from_json(payload)
            except (TypeError, ValueError):
                metrics.invalid += 1; continue
            metrics.ingest(message)
            if output: output.write(detection_msg_to_json(message) + "\n")
    finally:
        if output: output.close()
        sub.close(0); context.term()
    report = metrics.report(); print(json.dumps(report, indent=2, sort_keys=True))
    if args.report: args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
