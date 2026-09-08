"""Latest-only, out-of-probe learned target-selection service.

DeepStream pad probes must never wait for model lifecycle, range estimation, or
policy inference.  This service owns those operations in a separate CPU
process.  The probe submits compact metadata snapshots and applies only the
most recent completed result for matching NvSORT IDs.
"""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from common.schemas import DetectionMsg, detection_msg_to_json
from jetson.deepstream.target_selection import DeepStreamTargetSelector, _class_labels, load_config, normalize_message_class_labels


_BOX_FIELDS = (
    "cls",
    "distance_m",
    "distance_src",
    "threat_level",
    "threat_confidence",
    "threat_score_benign",
    "threat_score_suspicious",
    "threat_score_threatening",
    "priority_score",
    "engagement_rank",
    "breakthrough_time_s",
    "time_to_engage_s",
    "damage_weight",
    "engageable_now",
    "expected_damage_if_ignored",
    "expected_total_damage_if_selected",
)


def _put_latest(target: Any, value: object) -> None:
    while True:
        try:
            target.put_nowait(value)
            return
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                return


def _worker(config_paths: tuple[str, ...], requests: Any, results: Any) -> None:
    try:
        config = copy.deepcopy(load_config([Path(path) for path in config_paths]))
        learned = config.setdefault("swarm_eval", {}).setdefault("learned_model", {})
        # The service itself provides scheduling; avoid another nested worker.
        learned["async_worker"] = False
        selector = DeepStreamTargetSelector(config)
        rate_hz = float(learned.get("max_update_rate_hz") or 10.0)
        interval_s = 1.0 / max(rate_hz, 0.1)
        next_run_s = 0.0
        pending: Mapping[str, Any] | None = None
        _put_latest(results, {"type": "ready"})
        while True:
            timeout_s = max(0.0, next_run_s - time.monotonic()) if pending is not None else None
            try:
                incoming = requests.get(timeout=timeout_s)
            except queue.Empty:
                incoming = None
            if incoming is None and pending is None:
                continue
            if incoming is None:
                snapshot = pending
                pending = None
            else:
                if incoming.get("type") == "stop":
                    return
                pending = incoming
                # Coalesce every queued frame before executing a policy update.
                while True:
                    try:
                        newer = requests.get_nowait()
                    except queue.Empty:
                        break
                    if newer.get("type") == "stop":
                        return
                    pending = newer
                if time.monotonic() < next_run_s:
                    continue
                snapshot = pending
                pending = None
            if snapshot is None:
                continue
            message = DetectionMsg(**snapshot["message"])
            selector.select(message, now_s=time.monotonic())
            annotations: dict[str, dict[str, Any]] = {}
            for index, box in enumerate(message.boxes):
                key = str(box.track_id) if box.track_id is not None else f"index:{index}"
                annotations[key] = {
                    field: getattr(box, field)
                    for field in _BOX_FIELDS
                    if getattr(box, field) is not None
                }
            _put_latest(
                results,
                {
                    "type": "result",
                    "completed_at_s": time.monotonic(),
                    "target_track_id": message.target_track_id,
                    "annotations": annotations,
                    "selected": message.target_idx is not None,
                },
            )
            next_run_s = time.monotonic() + interval_s
    except Exception as exc:
        _put_latest(results, {"type": "error", "message": str(exc)})


class AsyncDeepStreamTargetSelector:
    """Non-blocking bridge used exclusively by the DeepStream metadata probe."""

    def __init__(self, config_paths: Sequence[Path]) -> None:
        self._ctx = mp.get_context("spawn")
        self._requests = self._ctx.Queue(maxsize=1)
        self._results = self._ctx.Queue(maxsize=1)
        self._latest: Mapping[str, Any] | None = None
        self._labels = _class_labels(load_config(config_paths))
        self._error: str | None = None
        self.submitted = 0
        self.applied = 0
        self._process = self._ctx.Process(
            target=_worker,
            args=(tuple(str(path) for path in config_paths), self._requests, self._results),
            daemon=True,
        )
        self._process.start()

    def submit_and_apply(self, message: DetectionMsg) -> None:
        # This is intentionally synchronous and tiny: stable semantic labels
        # are part of the DetectionMsg contract, while expensive range/policy
        # work remains in the latest-only service process.
        normalize_message_class_labels(message, self._labels)
        snapshot = {"message": json.loads(detection_msg_to_json(message))}
        _put_latest(self._requests, snapshot)
        self.submitted += 1
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                break
            if result.get("type") == "result":
                self._latest = result
            elif result.get("type") == "error":
                self._error = str(result.get("message"))
        result = self._latest
        if result is None:
            return
        annotations = result.get("annotations", {})
        for index, box in enumerate(message.boxes):
            key = str(box.track_id) if box.track_id is not None else f"index:{index}"
            for field, value in annotations.get(key, {}).items():
                setattr(box, field, value)
        target_track_id = result.get("target_track_id")
        if target_track_id is not None:
            for index, box in enumerate(message.boxes):
                if box.track_id is not None and int(box.track_id) == int(target_track_id):
                    message.target_idx = index
                    message.target_track_id = int(target_track_id)
                    self.applied += 1
                    break

    def report(self) -> dict[str, int | bool | str | None]:
        return {
            "enabled": True,
            "mode": "latest_only_service",
            "submitted": self.submitted,
            "applied": self.applied,
            "worker_alive": self._process.is_alive(),
            "error": self._error,
        }

    def close(self) -> None:
        _put_latest(self._requests, {"type": "stop"})
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
