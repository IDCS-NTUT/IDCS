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
    TrackAssessmentV2,
    detection_msg_from_snapshot,
)
from common.ranging import KnownSizeRangingConfig, iter_distance_estimates, iter_ranging_candidates, resolve_class_label
from common.schemas import DetectionMsg
from jetson.swarm_planner import SwarmPlannerRuntime


_ASSESSMENT_FIELDS = (
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


def _class_labels(config: Mapping[str, Any]) -> Mapping[str, str]:
    yolo = config.get("yolo", {})
    raw = yolo.get("class_labels", {}) if isinstance(yolo, Mapping) else {}
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def normalize_message_class_labels(message: DetectionMsg, labels: Mapping[str, str]) -> None:
    """Apply IDCS semantic labels before any asynchronous metadata handoff."""
    for box in message.boxes:
        box.cls = resolve_class_label(box.cls, labels)


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
    """Annotate a ``DetectionMsg`` and set its target fields using the learned policy.

    The runtime is created lazily once actual source dimensions are known.  A
    selector only mutates metadata; callers remain responsible for deciding
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

    def _ensure_runtime(self, message: DetectionMsg) -> None:
        frame_size = (int(message.img_w), int(message.img_h))
        if self._planner is not None and self._frame_size == frame_size:
            return
        self._frame_size = frame_size
        self._intrinsics = CameraIntrinsics.from_raw_config(self._config, frame_size)
        control_config = ControlConfig.from_raw_config(self._config, frame_size)
        if not control_config.swarm_eval.enabled:
            raise ValueError("swarm_eval.enabled must be true for DeepStream target selection")
        self._planner = self._planner_factory(control_config)

    def select(self, message: DetectionMsg, *, now_s: float | None = None) -> None:
        """Populate range/threat annotations and choose the current target."""

        normalize_message_class_labels(message, self._labels)
        # Do not create a second TensorRT context for frames that cannot ever
        # be selected.  In the current policy, people are intentional
        # non-targets; keeping this fast path CPU-only preserves detector FPS
        # while waiting for an eligible drone.
        if not any(str(box.cls).strip().lower() not in self._excluded_classes for box in message.boxes):
            message.target_idx = None
            message.target_track_id = None
            self._previous_target_id = None
            self.frames += 1
            return
        self._ensure_runtime(message)
        assert self._planner is not None
        assert self._intrinsics is not None
        if self._ranging.enabled:
            candidates = iter_ranging_candidates(
                message.boxes,
                (message.img_w, message.img_h),
                self._labels,
                self._ranging,
            )
            for estimate in iter_distance_estimates(candidates, self._intrinsics, self._ranging):
                estimate.candidate.box.distance_m = estimate.distance_m
                estimate.candidate.box.distance_src = estimate.source

        decision = self._planner.update_and_select(
            message,
            current_time_s=time.monotonic() if now_s is None else float(now_s),
            cam_state=None,
            previous_target_id=self._previous_target_id,
        )
        message.target_idx = decision.chosen_box_index
        if decision.chosen_box_index is None:
            message.target_track_id = None
            self._previous_target_id = None
        else:
            box = message.boxes[decision.chosen_box_index]
            message.target_track_id = box.track_id
            self._previous_target_id = (
                int(box.track_id) if box.track_id is not None else -(decision.chosen_box_index + 1)
            )
            self.selected += 1
        self.frames += 1

    def select_snapshot(
        self,
        snapshot: PerceptionSnapshotV2,
        *,
        now_s: float | None = None,
    ) -> PerceptionSnapshotV2:
        """Select from guaranteed V2 tracks without invoking detector/tracker code."""

        selected_at_s = time.monotonic() if now_s is None else float(now_s)
        normalized = normalize_snapshot_class_labels(snapshot, self._labels)
        message = detection_msg_from_snapshot(normalized, use_tracks=True)
        self.select(message, now_s=selected_at_s)
        assessments = []
        for box in message.boxes:
            if box.track_id is None:
                continue
            values = {
                field: getattr(box, field)
                for field in _ASSESSMENT_FIELDS
                if getattr(box, field) is not None
            }
            if values:
                assessments.append(TrackAssessmentV2(track_id=box.track_id, **values))
        selection = None
        if message.target_track_id is not None:
            selection = TargetSelectionV2(
                track_id=int(message.target_track_id),
                source_frame_id=snapshot.frame.frame_id,
                applied_frame_id=snapshot.frame.frame_id,
                selected_time_ns=max(0, round(selected_at_s * 1_000_000_000)),
                selection_clock_domain=snapshot.frame.observation_clock_domain,
                policy="swarm_planner",
            )
        payload = normalized.model_dump(mode="json")
        payload.update({
            "assessments": [item.model_dump(mode="json") for item in assessments],
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
