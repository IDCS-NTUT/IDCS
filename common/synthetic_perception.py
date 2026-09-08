"""Deterministic perception scenarios that bypass learned detection quality."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from common.perception import (
    NormalizedBoxV2,
    PerceptionDetectionV2,
    PerceptionFrameV2,
    PerceptionSnapshotV2,
    PerceptionTrackV2,
)
from common.schemas import Box, DetectionMsg


class SyntheticScenarioError(ValueError):
    """Raised when a deterministic scenario cannot produce valid snapshots."""


class _ScenarioModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SyntheticTrackSpec(_ScenarioModel):
    logical_id: str = Field(min_length=1, max_length=80)
    track_id: int = Field(ge=0)
    class_id: str = Field(min_length=1, max_length=80)
    first_frame: int = Field(ge=0)
    last_frame: int = Field(ge=0)
    start_center_norm: tuple[float, float]
    velocity_norm_per_frame: tuple[float, float] = (0.0, 0.0)
    size_norm: tuple[float, float]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    hidden_frames: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _valid_span(self):
        if self.last_frame < self.first_frame:
            raise ValueError("last_frame must not precede first_frame")
        if any(frame < self.first_frame or frame > self.last_frame for frame in self.hidden_frames):
            raise ValueError("hidden_frames must lie within the target frame span")
        if len(set(self.hidden_frames)) != len(self.hidden_frames):
            raise ValueError("hidden_frames must be unique")
        numeric = self.start_center_norm + self.velocity_norm_per_frame + self.size_norm
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("target geometry values must be finite")
        if not all(0.0 <= value <= 1.0 for value in self.start_center_norm):
            raise ValueError("start_center_norm must lie inside the frame")
        if not all(0.0 < value <= 1.0 for value in self.size_norm):
            raise ValueError("size_norm must be positive and no larger than the frame")
        return self


class SyntheticDeliveryFaults(_ScenarioModel):
    dropped_frames: tuple[int, ...] = ()
    duplicate_frames: tuple[int, ...] = ()
    delay_ms_by_frame: dict[int, float] = Field(default_factory=dict)
    duplicate_spacing_ms: float = Field(default=1.0, gt=0.0)


class SyntheticPerceptionScenario(_ScenarioModel):
    type: Literal["SyntheticPerceptionScenario"] = "SyntheticPerceptionScenario"
    version: Literal[1] = 1
    seed: int = Field(ge=0)
    frame_count: int = Field(gt=0)
    fps: float = Field(gt=0.0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    start_time_ns: int = Field(default=0, ge=0)
    observation_latency_ms: float = Field(default=5.0, ge=0.0)
    targets: tuple[SyntheticTrackSpec, ...]
    faults: SyntheticDeliveryFaults = Field(default_factory=SyntheticDeliveryFaults)

    @model_validator(mode="after")
    def _valid_frame_references(self):
        referenced = (
            set(self.faults.dropped_frames)
            | set(self.faults.duplicate_frames)
            | set(self.faults.delay_ms_by_frame)
        )
        if any(frame < 0 or frame >= self.frame_count for frame in referenced):
            raise ValueError("delivery fault frames must lie within the scenario")
        if len(set(self.faults.dropped_frames)) != len(self.faults.dropped_frames):
            raise ValueError("dropped_frames must be unique")
        if len(set(self.faults.duplicate_frames)) != len(self.faults.duplicate_frames):
            raise ValueError("duplicate_frames must be unique")
        if any(delay < 0.0 for delay in self.faults.delay_ms_by_frame.values()):
            raise ValueError("delivery delays must be non-negative")
        for target in self.targets:
            if target.last_frame >= self.frame_count:
                raise ValueError("target frame spans must lie within the scenario")
        return self


@dataclass(frozen=True)
class SyntheticDelivery:
    snapshot: PerceptionSnapshotV2
    arrival_time_ns: int
    duplicate_index: int = 0


def load_synthetic_scenario(path: Path | str) -> SyntheticPerceptionScenario:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return SyntheticPerceptionScenario.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SyntheticScenarioError(f"invalid synthetic perception scenario {path}: {exc}") from exc


def synthetic_scenario_digest(scenario: SyntheticPerceptionScenario) -> str:
    """Hash canonical scenario content for test-result provenance."""

    canonical = json.dumps(
        scenario.model_dump(mode="json"),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def snapshot_at(scenario: SyntheticPerceptionScenario, frame_index: int) -> PerceptionSnapshotV2:
    if frame_index < 0 or frame_index >= scenario.frame_count:
        raise SyntheticScenarioError(f"frame {frame_index} is outside the scenario")
    frame_period_ns = round(1_000_000_000 / scenario.fps)
    source_time_ns = scenario.start_time_ns + frame_index * frame_period_ns
    observed_time_ns = source_time_ns + round(scenario.observation_latency_ms * 1_000_000)
    detections: list[PerceptionDetectionV2] = []
    tracks: list[PerceptionTrackV2] = []

    active = sorted(
        (
            target for target in scenario.targets
            if target.first_frame <= frame_index <= target.last_frame
            and frame_index not in target.hidden_frames
        ),
        key=lambda target: (target.track_id, target.logical_id),
    )
    for detection_id, target in enumerate(active):
        elapsed = frame_index - target.first_frame
        center_x = target.start_center_norm[0] + target.velocity_norm_per_frame[0] * elapsed
        center_y = target.start_center_norm[1] + target.velocity_norm_per_frame[1] * elapsed
        try:
            box = NormalizedBoxV2(
                x=center_x - target.size_norm[0] / 2.0,
                y=center_y - target.size_norm[1] / 2.0,
                w=target.size_norm[0],
                h=target.size_norm[1],
            )
        except ValueError as exc:
            raise SyntheticScenarioError(
                f"target {target.logical_id!r} leaves the frame at frame {frame_index}: {exc}"
            ) from exc
        detections.append(PerceptionDetectionV2(
            detection_id=detection_id,
            box=box,
            class_id=target.class_id,
            confidence=target.confidence,
        ))
        tracks.append(PerceptionTrackV2(
            track_id=target.track_id,
            box=box,
            class_id=target.class_id,
            confidence=target.confidence,
            age_frames=elapsed + 1,
            missed_frames=0,
        ))

    return PerceptionSnapshotV2(
        sequence=frame_index,
        frame=PerceptionFrameV2(
            frame_id=frame_index,
            source_time_ns=source_time_ns,
            observed_time_ns=observed_time_ns,
            source_clock_domain="synthetic",
            observation_clock_domain="synthetic",
            width=scenario.width,
            height=scenario.height,
        ),
        detections=tuple(detections),
        tracks=tuple(tracks),
    )


def iter_deliveries(scenario: SyntheticPerceptionScenario) -> tuple[SyntheticDelivery, ...]:
    """Return a stable arrival-ordered trace including declared transport faults."""

    dropped = set(scenario.faults.dropped_frames)
    duplicates = set(scenario.faults.duplicate_frames)
    deliveries: list[SyntheticDelivery] = []
    for frame_index in range(scenario.frame_count):
        if frame_index in dropped:
            continue
        snapshot = snapshot_at(scenario, frame_index)
        arrival = snapshot.frame.observed_time_ns + round(
            scenario.faults.delay_ms_by_frame.get(frame_index, 0.0) * 1_000_000
        )
        deliveries.append(SyntheticDelivery(snapshot, arrival, 0))
        if frame_index in duplicates:
            deliveries.append(SyntheticDelivery(
                snapshot,
                arrival + round(scenario.faults.duplicate_spacing_ms * 1_000_000),
                1,
            ))
    return tuple(sorted(
        deliveries,
        key=lambda item: (item.arrival_time_ns, item.snapshot.frame.frame_id, item.duplicate_index),
    ))


def to_legacy_detection_msg(
    snapshot: PerceptionSnapshotV2,
    *,
    use_tracks: bool = True,
) -> DetectionMsg:
    """Adapt guaranteed registrations to the legacy downstream test boundary."""

    objects = snapshot.tracks if use_tracks else snapshot.detections
    boxes = [Box(
        x=item.box.x,
        y=item.box.y,
        w=item.box.w,
        h=item.box.h,
        cls=item.class_id,
        conf=item.confidence,
        track_id=item.track_id if isinstance(item, PerceptionTrackV2) else None,
    ) for item in objects]
    return DetectionMsg(
        frame_id=snapshot.frame.frame_id,
        src_ts_ms=snapshot.frame.source_time_ns // 1_000_000,
        rx_ts_ms=snapshot.frame.observed_time_ns // 1_000_000,
        infer_ts_ms=snapshot.frame.observed_time_ns // 1_000_000,
        img_w=snapshot.frame.width,
        img_h=snapshot.frame.height,
        boxes=boxes,
    )
