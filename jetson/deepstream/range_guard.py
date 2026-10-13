"""Per-track known-size range qualification for current V2 video frames.

This is deliberately independent of controller authority. A held value is
bounded, source-session scoped, and never refreshed by a suspect box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from common.camera import CameraIntrinsics
from common.perception import PerceptionSnapshotV2
from common.ranging import KnownSizeRangingConfig, estimate_normalized_distance


@dataclass(frozen=True)
class RangeGuardPolicy:
    hold_s: float = 0.25
    edge_margin_px: float = 2.0
    jump_slack_fraction: float = 0.10
    max_fractional_change_per_s: float = 1.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RangeGuardPolicy":
        camera = config.get("camera", {})
        ranging = camera.get("known_size_ranging", {}) if isinstance(camera, Mapping) else {}
        raw = ranging.get("quality", {}) if isinstance(ranging, Mapping) else {}
        if not isinstance(raw, Mapping):
            raise ValueError("camera.known_size_ranging.quality must be a mapping")
        values = {}
        for name in ("hold_s", "edge_margin_px", "jump_slack_fraction",
                     "max_fractional_change_per_s"):
            try:
                value = float(raw.get(name, getattr(cls, name)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"camera.known_size_ranging.quality.{name} must be numeric") from exc
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"camera.known_size_ranging.quality.{name} must be finite and nonnegative")
            values[name] = value
        return cls(**values)


@dataclass(frozen=True)
class _QualifiedSample:
    class_id: str
    distance_m: float
    source: str
    accepted_s: float


class KnownSizeRangeGuard:
    def __init__(self, config: Mapping[str, Any], labels: Mapping[str, str]) -> None:
        self._config = config
        self._labels = labels
        self.ranging = KnownSizeRangingConfig.from_raw_config(config)
        self.policy = RangeGuardPolicy.from_config(config)
        self._frame_size: tuple[int, int] | None = None
        self._source: tuple[str, str | None] | None = None
        self._intrinsics: CameraIntrinsics | None = None
        self._samples: dict[int, _QualifiedSample] = {}
        self._last_now_s: float | None = None

    def _reset_if_source_changed(self, snapshot: PerceptionSnapshotV2, now_s: float) -> None:
        frame = snapshot.frame
        frame_size = (frame.width, frame.height)
        source = (frame.source_clock_domain, frame.tracker_session)
        if (frame_size != self._frame_size or source != self._source or
                (self._last_now_s is not None and now_s < self._last_now_s)):
            self._samples.clear()
            self._intrinsics = CameraIntrinsics.from_raw_config(self._config, frame_size)
            self._frame_size = frame_size
            self._source = source
        self._last_now_s = now_s

    def _touches_edge(self, box: Any, frame_size: tuple[int, int]) -> bool:
        width, height = frame_size
        margin = self.policy.edge_margin_px
        return (box.x * width <= margin or box.y * height <= margin or
                (1.0 - box.x - box.w) * width <= margin or
                (1.0 - box.y - box.h) * height <= margin)

    def apply(self, snapshot: PerceptionSnapshotV2, *, now_s: float) -> PerceptionSnapshotV2:
        """Replace stale range fields with measured, briefly held, or no range."""

        if not math.isfinite(now_s) or now_s < 0.0:
            raise ValueError("range guard now_s must be finite and nonnegative")
        if not self.ranging.enabled:
            return snapshot
        self._reset_if_source_changed(snapshot, now_s)
        assert self._intrinsics is not None and self._frame_size is not None
        assessments = {
            item.track_id: item.model_dump(mode="json", exclude_none=True)
            for item in snapshot.assessments
        }
        for track in snapshot.tracks:
            values = assessments.get(track.track_id, {"track_id": track.track_id})
            for field in ("distance_m", "distance_src", "distance_quality", "distance_age_ms"):
                values.pop(field, None)
            previous = self._samples.get(track.track_id)
            if previous is not None and previous.class_id != track.class_id:
                previous = None
                self._samples.pop(track.track_id, None)
            candidate = None
            if track.missed_frames == 0 and not self._touches_edge(track.box, self._frame_size):
                candidate = estimate_normalized_distance(
                    class_id=track.class_id,
                    width_norm=track.box.w,
                    height_norm=track.box.h,
                    frame_size=self._frame_size,
                    label_map=self._labels,
                    intrinsics=self._intrinsics,
                    config=self.ranging,
                )
            if candidate is not None and previous is not None:
                # Do not let an indefinitely occluded but still tracked box
                # become acceptable solely because wall time has passed.
                elapsed = min(max(0.0, now_s - previous.accepted_s), self.policy.hold_s)
                permitted = (self.policy.jump_slack_fraction +
                             self.policy.max_fractional_change_per_s * elapsed)
                if abs(candidate.distance_m / previous.distance_m - 1.0) > permitted:
                    candidate = None
            if candidate is not None:
                previous = _QualifiedSample(track.class_id, candidate.distance_m,
                                            candidate.source, now_s)
                self._samples[track.track_id] = previous
                values.update(distance_m=previous.distance_m, distance_src=previous.source,
                              distance_quality="measured", distance_age_ms=0.0)
            elif previous is not None and 0.0 <= now_s - previous.accepted_s <= self.policy.hold_s:
                values.update(distance_m=previous.distance_m, distance_src=previous.source,
                              distance_quality="held",
                              distance_age_ms=(now_s - previous.accepted_s) * 1000.0)
            if len(values) > 1:
                assessments[track.track_id] = values
            else:
                assessments.pop(track.track_id, None)
        live_ids = {track.track_id for track in snapshot.tracks}
        assessments = {track_id: item for track_id, item in assessments.items() if track_id in live_ids}
        # Keep an expired sample only while its track is still present; it
        # remains a continuity reference but never becomes a held range again.
        self._samples = {track_id: item for track_id, item in self._samples.items()
                         if track_id in live_ids}
        payload = snapshot.model_dump(mode="json")
        payload["assessments"] = list(assessments.values())
        return PerceptionSnapshotV2.model_validate(payload)
