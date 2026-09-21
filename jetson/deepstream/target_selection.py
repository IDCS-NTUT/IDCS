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

from common.camera import CameraIntrinsics
from common.config import load_config_bundle, merge_config_layers
from common.control import ControlConfig
from common.perception import (
    PerceptionSnapshotV2,
    TargetSelectionV2,
)
from common.ranging import (
    KnownSizeRangingConfig,
    estimate_normalized_distance,
    resolve_class_label,
)
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
        self._ranging = KnownSizeRangingConfig.from_raw_config(config)
        swarm = config.get("swarm_eval", {})
        excluded = swarm.get("excluded_target_classes", ()) if isinstance(swarm, Mapping) else ()
        self._excluded_classes = {
            str(value).strip().lower()
            for value in excluded
            if str(value).strip()
        }
        self._planner: SwarmPlannerRuntime | None = None
        self._intrinsics: CameraIntrinsics | None = None
        self._frame_size: tuple[int, int] | None = None
        self._previous_target_id: int | None = None
        self.frames = 0
        self.selected = 0

    @classmethod
    def from_paths(cls, paths: Sequence[Path]) -> "DeepStreamTargetSelector":
        bundle = load_config_bundle(paths, required_sections=("swarm_eval",))
        return cls(bundle.data)

    def _ensure_runtime(self, snapshot: PerceptionSnapshotV2) -> None:
        frame_size = (int(snapshot.frame.width), int(snapshot.frame.height))
        if self._planner is not None and self._frame_size == frame_size:
            return
        self._frame_size = frame_size
        self._intrinsics = CameraIntrinsics.from_raw_config(self._config, frame_size)
        control_config = ControlConfig.from_raw_config(self._config, frame_size)
        if not control_config.swarm_eval.enabled:
            raise ValueError("swarm_eval.enabled must be true for DeepStream target selection")
        self._planner = self._planner_factory(control_config)

    def _with_range_assessments(
        self,
        snapshot: PerceptionSnapshotV2,
    ) -> PerceptionSnapshotV2:
        """Add immutable known-size measurements without legacy box mutation."""

        assert self._intrinsics is not None
        if not self._ranging.enabled:
            return snapshot
        frame_size = (snapshot.frame.width, snapshot.frame.height)
        assessments = {
            item.track_id: item.model_dump(mode="json", exclude_none=True)
            for item in snapshot.assessments
        }
        for track in snapshot.tracks:
            estimate = estimate_normalized_distance(
                class_id=track.class_id,
                width_norm=track.box.w,
                height_norm=track.box.h,
                frame_size=frame_size,
                label_map=self._labels,
                intrinsics=self._intrinsics,
                config=self._ranging,
            )
            if estimate is None:
                continue
            values = assessments.setdefault(track.track_id, {"track_id": track.track_id})
            values.update({
                "distance_m": estimate.distance_m,
                "distance_src": estimate.source,
            })
        payload = snapshot.model_dump(mode="json")
        payload["assessments"] = list(assessments.values())
        return PerceptionSnapshotV2.model_validate(payload)

    def select_snapshot(
        self,
        snapshot: PerceptionSnapshotV2,
        *,
        now_s: float | None = None,
    ) -> PerceptionSnapshotV2:
        """Select from guaranteed V2 tracks without invoking detector/tracker code."""

        selected_at_s = time.monotonic() if now_s is None else float(now_s)
        normalized = normalize_snapshot_class_labels(snapshot, self._labels)
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
        ranged = self._with_range_assessments(normalized)
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
        else:
            self._previous_target_id = None
        self.frames += 1
        payload = ranged.model_dump(mode="json")
        payload.update({
            "assessments": [item.model_dump(mode="json") for item in result.assessments],
            "selection": None if selection is None else selection.model_dump(mode="json"),
        })
        return PerceptionSnapshotV2.model_validate(payload)

    def report(self) -> dict[str, int | bool]:
        return {
            "enabled": True,
            "frames": self.frames,
            "selected": self.selected,
            "ranging_enabled": self._ranging.enabled,
        }
