"""Control-free learned target selection for DeepStream metadata.

The DeepStream pipeline supplies detector boxes and NvSORT identities.  This
adapter adds IDCS class labels and known-size range, then reuses the trained
swarm-policy runtime to annotate threat scores and select one tracked target.
It deliberately has no ZMQ, GStreamer, or controller dependency.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from common.camera import CameraIntrinsics
from common.config_sync import merge_config_maps, parse_config_text
from common.control import ControlConfig
from common.ranging import KnownSizeRangingConfig, iter_distance_estimates, iter_ranging_candidates, resolve_class_label
from common.schemas import DetectionMsg
from jetson.swarm_planner import SwarmPlannerRuntime


def load_config(paths: Sequence[Path]) -> Mapping[str, Any]:
    """Load and merge IDCS YAML configuration files without config-sync I/O."""

    if not paths:
        raise ValueError("at least one IDCS configuration path is required")
    return merge_config_maps(
        *(parse_config_text(path.read_text(encoding="utf-8"), str(path)) for path in paths)
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


class DeepStreamTargetSelector:
    """Annotate a ``DetectionMsg`` and set its target fields using the learned policy.

    The runtime is created lazily once actual source dimensions are known.  A
    selector only mutates metadata; callers remain responsible for deciding
    whether any control path exists (the DeepStream shadow runtime has none).
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = config
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
        return cls(load_config(paths))

    def _ensure_runtime(self, message: DetectionMsg) -> None:
        frame_size = (int(message.img_w), int(message.img_h))
        if self._planner is not None and self._frame_size == frame_size:
            return
        self._frame_size = frame_size
        self._intrinsics = CameraIntrinsics.from_raw_config(self._config, frame_size)
        control_config = ControlConfig.from_raw_config(self._config, frame_size)
        if not control_config.swarm_eval.enabled:
            raise ValueError("swarm_eval.enabled must be true for DeepStream target selection")
        self._planner = SwarmPlannerRuntime(control_config)

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

    def report(self) -> dict[str, int | bool]:
        return {
            "enabled": True,
            "frames": self.frames,
            "selected": self.selected,
            "ranging_enabled": self._ranging.enabled,
        }
