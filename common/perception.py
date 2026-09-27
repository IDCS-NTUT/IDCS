"""Strict perception contracts separating detections, tracks, and selection."""
from __future__ import annotations

import math
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _PerceptionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _finite_numbers_only(self):
        def check(value: Any) -> None:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("perception fields must be finite")
            if isinstance(value, Mapping):
                for item in value.values():
                    check(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    check(item)

        check(self.__dict__)
        return self


class NormalizedBoxV2(_PerceptionModel):
    """Top-left/size box normalized to the complete source frame."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _inside_frame(self):
        tolerance = 1e-12
        if self.x + self.w > 1.0 + tolerance or self.y + self.h > 1.0 + tolerance:
            raise ValueError("normalized box must remain inside the source frame")
        return self


class PerceptionFrameV2(_PerceptionModel):
    frame_id: int = Field(ge=0)
    source_time_ns: int = Field(ge=0)
    received_time_ns: int | None = Field(default=None, ge=0)
    observed_time_ns: int = Field(ge=0)
    source_clock_domain: str = Field(min_length=1, max_length=80)
    source_identity_verified: bool | None = None
    receive_clock_domain: str | None = Field(default=None, min_length=1, max_length=80)
    observation_clock_domain: str = Field(min_length=1, max_length=80)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    # Simulator HIL only: the exact camera pose used to render this frame,
    # relative to the simulator's startup-home frame. Real video omits it.
    sim_capture_pose_rad: tuple[float, float] | None = None
    sim_applied_camstate_ns: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _receive_time_names_its_clock(self):
        if (self.received_time_ns is None) != (self.receive_clock_domain is None):
            raise ValueError("received_time_ns and receive_clock_domain must be set together")
        if (self.sim_capture_pose_rad is None) != (self.sim_applied_camstate_ns is None):
            raise ValueError("simulator capture pose and applied CamState time must be paired")
        return self


class PerceptionDetectionV2(_PerceptionModel):
    """One raw detector registration; it deliberately has no track identity."""

    detection_id: int = Field(ge=0)
    box: NormalizedBoxV2
    class_id: str = Field(min_length=1, max_length=80)
    confidence: float = Field(ge=0.0, le=1.0)


class PerceptionTrackV2(_PerceptionModel):
    """One tracker output; identity and lifecycle state belong only here."""

    track_id: int = Field(ge=0)
    box: NormalizedBoxV2
    class_id: str = Field(min_length=1, max_length=80)
    confidence: float = Field(ge=0.0, le=1.0)
    age_frames: int | None = Field(default=None, ge=1)
    missed_frames: int = Field(ge=0)


class TrackAssessmentV2(_PerceptionModel):
    """Selector/risk annotations keyed to a tracker identity."""

    track_id: int = Field(ge=0)
    distance_m: float | None = Field(default=None, ge=0.0)
    distance_src: Literal["height", "width", "average"] | None = None
    threat_level: Literal["benign", "suspicious", "threatening"] | None = None
    threat_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    threat_score_benign: float | None = None
    threat_score_suspicious: float | None = None
    threat_score_threatening: float | None = None
    priority_score: float | None = None
    engagement_rank: int | None = Field(default=None, ge=0)
    breakthrough_time_s: float | None = None
    time_to_engage_s: float | None = None
    damage_weight: float | None = None
    engageable_now: bool | None = None
    expected_damage_if_ignored: float | None = None
    expected_total_damage_if_selected: float | None = None


class TargetSelectionV2(_PerceptionModel):
    """Selector output tied explicitly to the snapshot it evaluated."""

    track_id: int = Field(ge=0)
    source_frame_id: int = Field(ge=0)
    applied_frame_id: int = Field(ge=0)
    selected_time_ns: int = Field(ge=0)
    selection_clock_domain: str = Field(min_length=1, max_length=80)
    policy: str = Field(min_length=1, max_length=80)


class PerceptionSnapshotV2(_PerceptionModel):
    """Atomic boundary between perception, selection, and fixed-rate control."""

    type: Literal["PerceptionSnapshot"] = "PerceptionSnapshot"
    version: Literal[2] = 2
    sequence: int = Field(ge=0)
    frame: PerceptionFrameV2
    detections: tuple[PerceptionDetectionV2, ...] = ()
    tracks: tuple[PerceptionTrackV2, ...] = ()
    assessments: tuple[TrackAssessmentV2, ...] = ()
    selection: TargetSelectionV2 | None = None

    @model_validator(mode="after")
    def _consistent_identity(self):
        detection_ids = [item.detection_id for item in self.detections]
        track_ids = [item.track_id for item in self.tracks]
        assessment_ids = [item.track_id for item in self.assessments]
        if len(set(detection_ids)) != len(detection_ids):
            raise ValueError("detection_id values must be unique within a snapshot")
        if len(set(track_ids)) != len(track_ids):
            raise ValueError("track_id values must be unique within a snapshot")
        if len(set(assessment_ids)) != len(assessment_ids):
            raise ValueError("assessment track_id values must be unique within a snapshot")
        if not set(assessment_ids).issubset(track_ids):
            raise ValueError("assessments must identify tracks in the snapshot")
        if self.selection is not None:
            if self.selection.applied_frame_id != self.frame.frame_id:
                raise ValueError("selection applied_frame_id must match the snapshot frame")
            if self.selection.track_id not in set(track_ids):
                raise ValueError("selection track_id must identify a track in the snapshot")
        return self


def perception_snapshot_to_json(snapshot: PerceptionSnapshotV2) -> str:
    """Serialize one strict V2 snapshot for an internal transport boundary."""

    return snapshot.model_dump_json(exclude_none=True)


def perception_snapshot_from_json(
    payload: str | bytes | bytearray | Mapping[str, Any],
) -> PerceptionSnapshotV2:
    """Validate a V2 snapshot received from JSON or a decoded mapping."""

    if isinstance(payload, Mapping):
        return PerceptionSnapshotV2.model_validate(payload)
    return PerceptionSnapshotV2.model_validate_json(payload)
