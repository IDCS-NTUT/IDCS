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
    observed_time_ns: int = Field(ge=0)
    source_clock_domain: str = Field(min_length=1, max_length=80)
    observation_clock_domain: str = Field(min_length=1, max_length=80)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


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
    age_frames: int = Field(ge=1)
    missed_frames: int = Field(ge=0)


class TargetSelectionV2(_PerceptionModel):
    """Selector output tied explicitly to the snapshot it evaluated."""

    track_id: int = Field(ge=0)
    source_frame_id: int = Field(ge=0)
    selected_time_ns: int = Field(ge=0)
    policy: str = Field(min_length=1, max_length=80)


class PerceptionSnapshotV2(_PerceptionModel):
    """Atomic boundary between perception, selection, and fixed-rate control."""

    type: Literal["PerceptionSnapshot"] = "PerceptionSnapshot"
    version: Literal[2] = 2
    sequence: int = Field(ge=0)
    frame: PerceptionFrameV2
    detections: tuple[PerceptionDetectionV2, ...] = ()
    tracks: tuple[PerceptionTrackV2, ...] = ()
    selection: TargetSelectionV2 | None = None

    @model_validator(mode="after")
    def _consistent_identity(self):
        detection_ids = [item.detection_id for item in self.detections]
        track_ids = [item.track_id for item in self.tracks]
        if len(set(detection_ids)) != len(detection_ids):
            raise ValueError("detection_id values must be unique within a snapshot")
        if len(set(track_ids)) != len(track_ids):
            raise ValueError("track_id values must be unique within a snapshot")
        if self.selection is not None:
            if self.selection.source_frame_id != self.frame.frame_id:
                raise ValueError("selection source_frame_id must match the snapshot frame")
            if self.selection.track_id not in set(track_ids):
                raise ValueError("selection track_id must identify a track in the snapshot")
        return self
