"""Latest-only, out-of-probe learned target-selection service.

DeepStream pad probes must never wait for model lifecycle, range estimation, or
policy inference.  This service owns those operations in a separate CPU
process.  The probe submits compact metadata snapshots and applies only the
most recent completed result for matching NvSORT IDs.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from common.config import load_config_bundle
from common.perception import PerceptionSnapshotV2
from jetson.deepstream.target_selection import (
    DeepStreamTargetSelector,
    _class_labels,
    normalize_snapshot_class_labels,
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


def _worker(config_snapshot: Mapping[str, Any], config_digest: str, requests: Any, results: Any) -> None:
    try:
        config = copy.deepcopy(config_snapshot)
        learned = config.setdefault("swarm_eval", {}).setdefault("learned_model", {})
        # The service itself provides scheduling; avoid another nested worker.
        learned["async_worker"] = False
        selector = DeepStreamTargetSelector(config)
        rate_hz = float(learned.get("max_update_rate_hz") or 10.0)
        interval_s = 1.0 / max(rate_hz, 0.1)
        next_run_s = 0.0
        pending: Mapping[str, Any] | None = None
        _put_latest(results, {"type": "ready", "config_digest": config_digest})
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
            source = PerceptionSnapshotV2.model_validate(snapshot["snapshot"])
            selected = selector.select_snapshot(source, now_s=time.monotonic())
            _put_latest(
                results,
                {
                    "type": "result",
                    "snapshot": selected.model_dump(mode="json"),
                },
            )
            next_run_s = time.monotonic() + interval_s
    except Exception as exc:
        _put_latest(results, {"type": "error", "message": str(exc)})


def _apply_completed_snapshot(
    current: PerceptionSnapshotV2,
    completed: PerceptionSnapshotV2,
) -> tuple[PerceptionSnapshotV2, bool]:
    """Apply an older decision only where tracker identity is still present."""

    current_track_ids = {track.track_id for track in current.tracks}
    assessments = tuple(
        item for item in completed.assessments
        if item.track_id in current_track_ids
    )
    selection = None
    if (
        completed.selection is not None
        and completed.selection.track_id in current_track_ids
    ):
        selection = completed.selection.model_copy(update={
            "applied_frame_id": current.frame.frame_id,
        })
    payload = current.model_dump(mode="json")
    payload.update({
        "assessments": [item.model_dump(mode="json") for item in assessments],
        "selection": None if selection is None else selection.model_dump(mode="json"),
    })
    return PerceptionSnapshotV2.model_validate(payload), selection is not None


class AsyncDeepStreamTargetSelector:
    """Non-blocking bridge used exclusively by the DeepStream metadata probe."""

    def __init__(self, config_paths: Sequence[Path]) -> None:
        bundle = load_config_bundle(config_paths, required_sections=("swarm_eval",))
        self._ctx = mp.get_context("spawn")
        self._requests = self._ctx.Queue(maxsize=1)
        self._results = self._ctx.Queue(maxsize=1)
        self._latest: Mapping[str, Any] | None = None
        self._labels = _class_labels(bundle.data)
        self._config_digest = bundle.digest
        self._error: str | None = None
        self.submitted = 0
        self.applied = 0
        self._process = self._ctx.Process(
            target=_worker,
            args=(bundle.mutable_copy(), bundle.digest, self._requests, self._results),
            daemon=True,
        )
        self._process.start()

    def submit_and_apply_snapshot(
        self,
        snapshot: PerceptionSnapshotV2,
    ) -> PerceptionSnapshotV2:
        # Semantic labels are immediate and deterministic; expensive policy
        # work remains in the latest-only service process.
        current = normalize_snapshot_class_labels(snapshot, self._labels)
        _put_latest(self._requests, {
            "snapshot": current.model_dump(mode="json"),
        })
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
            return current
        completed = PerceptionSnapshotV2.model_validate(result["snapshot"])
        applied, selected = _apply_completed_snapshot(current, completed)
        if selected:
            self.applied += 1
        return applied

    def report(self) -> dict[str, int | bool | str | None]:
        return {
            "enabled": True,
            "mode": "latest_only_service",
            "config_digest": self._config_digest,
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
