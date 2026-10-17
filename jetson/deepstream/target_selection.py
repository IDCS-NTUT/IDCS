"""Control-free learned target selection for DeepStream metadata.

The DeepStream pipeline supplies detector boxes and NvSORT identities.  This
adapter adds IDCS class labels and known-size range, then reuses the trained
swarm-policy runtime to annotate threat scores and select one tracked target.
It deliberately has no ZMQ, GStreamer, or controller dependency.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from common.live_config import runtime_bundle
from common.config import merge_config_layers
from common.control import ControlConfig
from common.perception import (
    PerceptionSnapshotV2,
    TargetSelectionV2,
)
from common.ranging import resolve_class_label
from jetson.deepstream.range_guard import KnownSizeRangeGuard
from jetson.swarm_planner import SwarmPlannerRuntime


def _class_labels(config: Mapping[str, Any]) -> Mapping[str, str]:
    perception = config.get("perception", {})
    raw = (
        perception.get("class_labels", {})
        if isinstance(perception, Mapping)
        else {}
    )
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def normalize_snapshot_class_labels(
    snapshot: PerceptionSnapshotV2,
    labels: Mapping[str, str],
) -> PerceptionSnapshotV2:
    """Return a V2 snapshot with semantic classes and unchanged identities."""

    detections = tuple(
        item.model_copy(update={"class_id": resolve_class_label(item.class_id, labels)})
        for item in snapshot.detections
    )
    tracks = tuple(
        item.model_copy(update={"class_id": resolve_class_label(item.class_id, labels)})
        for item in snapshot.tracks
    )
    return snapshot.model_copy(update={"detections": detections, "tracks": tracks})


class DeepStreamTargetSelector:
    """Assess and select tracked objects through the immutable V2 contract.

    The runtime is created lazily once actual source dimensions are known.  A
    selector only returns metadata; callers remain responsible for deciding
    whether any control path exists (the DeepStream shadow runtime has none).
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        planner_factory: Callable[[ControlConfig], Any] = SwarmPlannerRuntime,
    ) -> None:
        # Legacy control parsers still contain concrete ``dict`` checks. Keep
        # the authoritative bundle immutable and materialize one detached copy
        # at this compatibility boundary.
        self._config = merge_config_layers(config)
        self._planner_factory = planner_factory
        self._labels = _class_labels(config)
        self._range_guard = KnownSizeRangeGuard(config, self._labels)
        self._ranging = self._range_guard.ranging
        swarm = config.get("swarm_eval", {})
        excluded = swarm.get("excluded_target_classes", ()) if isinstance(swarm, Mapping) else ()
        self._excluded_classes = {
            str(value).strip().lower()
            for value in excluded
            if str(value).strip()
        }
        self._class_names = {str(v).strip().lower() for v in self._labels.values() if str(v).strip()}
        self._lock_track_id: int | None = None
        self._planner: SwarmPlannerRuntime | None = None
        self._frame_size: tuple[int, int] | None = None
        self._previous_target_id: int | None = None
        self.frames = 0
        self.selected = 0

    @classmethod
    def from_paths(cls, paths: Sequence[Path]) -> "DeepStreamTargetSelector":
        bundle = runtime_bundle(paths, required_sections=("swarm_eval",))
        return cls(bundle.data)

    def reconfigure(self, config: Mapping[str, Any]) -> None:
        """Worker safe point; keep the model and track/selection history."""
        size = self._frame_size or (1280, 720)
        control = ControlConfig.from_raw_config(config, size)
        guard = KnownSizeRangeGuard(config, self._labels)
        if self._planner is not None:
            self._planner.reconfigure(control)
        self._config = merge_config_layers(config)
        self._range_guard, self._ranging = guard, guard.ranging
        self._excluded_classes = set(control.swarm_eval.excluded_target_classes)

    def _ensure_runtime(self, snapshot: PerceptionSnapshotV2) -> None:
        frame_size = (int(snapshot.frame.width), int(snapshot.frame.height))
        if self._planner is not None and self._frame_size == frame_size:
            return
        self._frame_size = frame_size
        control_config = ControlConfig.from_raw_config(self._config, frame_size)
        if not control_config.swarm_eval.enabled:
            raise ValueError("swarm_eval.enabled must be true for DeepStream target selection")
        self._planner = self._planner_factory(control_config)
        if hasattr(self._planner, "set_excluded_target_classes"):
            self._planner.set_excluded_target_classes(self._excluded_classes)

    def set_operator(self, *, target_classes: Sequence[str] | None, lock_track_id: int | None) -> None:
        """The operator's choices: selectable classes (None keeps config) and a locked track.

        A lock only steers the planner's switching hysteresis here; the
        caller overrides the selection on the frame it applies to.
        """
        self._lock_track_id = lock_track_id
        if target_classes is None or not self._class_names:
            return
        wanted = {str(c).strip().lower() for c in target_classes}
        excluded = {c for c in self._class_names if c not in wanted}
        if excluded == self._excluded_classes:
            return
        self._excluded_classes = excluded
        if self._planner is not None and hasattr(self._planner, "set_excluded_target_classes"):
            self._planner.set_excluded_target_classes(excluded)

    def _with_range_assessments(
        self,
        snapshot: PerceptionSnapshotV2,
        *,
        now_s: float,
    ) -> PerceptionSnapshotV2:
        """Qualify current boxes and retain only a bounded last-good range."""

        return self._range_guard.apply(snapshot, now_s=now_s)

    def select_snapshot(
        self,
        snapshot: PerceptionSnapshotV2,
        *,
        now_s: float | None = None,
    ) -> PerceptionSnapshotV2:
        """Select from guaranteed V2 tracks without invoking detector/tracker code."""

        selected_at_s = time.monotonic() if now_s is None else float(now_s)
        normalized = normalize_snapshot_class_labels(snapshot, self._labels)
        if not self._config.get("swarm_eval", {}).get("enabled", False):
            self._previous_target_id = None
            self.frames += 1
            return self._range_guard.apply(normalized.model_copy(update={"selection": None}), now_s=selected_at_s)
        # Avoid loading a second model context for frames whose classes are all
        # intentionally excluded from policy selection.
        if not any(
            track.class_id.strip().lower() not in self._excluded_classes
            for track in normalized.tracks
        ):
            payload = normalized.model_dump(mode="json")
            payload["selection"] = None
            self._previous_target_id = None
            self.frames += 1
            return PerceptionSnapshotV2.model_validate(payload)

        self._ensure_runtime(normalized)
        assert self._planner is not None
        ranged = self._with_range_assessments(normalized, now_s=selected_at_s)
        locked = self._lock_track_id
        if locked is not None and any(track.track_id == locked for track in ranged.tracks):
            self._previous_target_id = locked
        result = self._planner.update_and_select_snapshot(
            ranged,
            current_time_s=selected_at_s,
            previous_target_id=self._previous_target_id,
        )
        selection = None
        if result.selected_track_id is not None:
            selection = TargetSelectionV2(
                track_id=int(result.selected_track_id),
                source_frame_id=snapshot.frame.frame_id,
                applied_frame_id=snapshot.frame.frame_id,
                selected_time_ns=max(0, round(selected_at_s * 1_000_000_000)),
                selection_clock_domain=snapshot.frame.observation_clock_domain,
                policy="swarm_planner",
            )
            self._previous_target_id = int(result.selected_track_id)
            self.selected += 1
        # With no selection keep the last choice: a one-frame dropout must not
        # reset switching hysteresis (the planner ages it out itself).
        self.frames += 1
        payload = ranged.model_dump(mode="json")
        payload.update({
            "assessments": [item.model_dump(mode="json") for item in result.assessments],
            "selection": None if selection is None else selection.model_dump(mode="json"),
        })
        return self._with_range_assessments(
            PerceptionSnapshotV2.model_validate(payload), now_s=selected_at_s)

    def report(self) -> dict[str, int | bool]:
        return {
            "enabled": True,
            "frames": self.frames,
            "selected": self.selected,
            "ranging_enabled": self._ranging.enabled,
        }
