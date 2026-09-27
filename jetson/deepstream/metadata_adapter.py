"""Pure DeepStream metadata to strict perception V2 helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from common.perception import (
    NormalizedBoxV2,
    PerceptionDetectionV2,
    PerceptionFrameV2,
    PerceptionSnapshotV2,
    PerceptionTrackV2,
)

UNTRACKED_OBJECT_ID = (1 << 64) - 1


@dataclass(frozen=True)
class FrameTiming:
    """Source and observation timing used to construct a V2 frame contract."""

    frame_id: int
    src_ts_ms: int
    rx_ts_ms: int
    infer_ts_ms: int
    img_w: int
    img_h: int
    source_clock_domain: str = "unspecified"
    observation_clock_domain: str = "jetson_monotonic"
    src_ts_ns: int | None = None
    source_identity_verified: bool | None = None


def pts_ns_to_ms(pts_ns: int) -> int:
    """Convert a non-negative GStreamer PTS to relative integer milliseconds."""

    return max(int(pts_ns), 0) // 1_000_000


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


@dataclass(frozen=True)
class ObjectObservationV2:
    box: NormalizedBoxV2
    class_id: str
    confidence: float
    track_id: int | None
    # DeepStream sets detector confidence to -0.1 on objects the tracker
    # carries without a matching detection in this frame (NvDCF shadow/visual
    # tracking). Such an object is a tracker estimate, not a detection.
    detector_matched: bool = True


def object_meta_to_observation_v2(
    object_meta: Any, *, img_w: int, img_h: int
) -> ObjectObservationV2 | None:
    """Normalize one ``NvDsObjectMeta`` directly into immutable V2 geometry."""

    if img_w <= 0 or img_h <= 0:
        raise ValueError("image dimensions must be positive")
    rect = object_meta.rect_params
    left = _clamp(float(rect.left), 0.0, float(img_w))
    top = _clamp(float(rect.top), 0.0, float(img_h))
    right = _clamp(float(rect.left) + float(rect.width), 0.0, float(img_w))
    bottom = _clamp(float(rect.top) + float(rect.height), 0.0, float(img_h))
    if right <= left or bottom <= top:
        return None

    object_id = int(object_meta.object_id)
    track_id = None if object_id == UNTRACKED_OBJECT_ID else object_id
    return ObjectObservationV2(
        box=NormalizedBoxV2(
            x=left / img_w,
            y=top / img_h,
            w=(right - left) / img_w,
            h=(bottom - top) / img_h,
        ),
        class_id=str(int(object_meta.class_id)),
        confidence=_clamp(float(object_meta.confidence), 0.0, 1.0),
        track_id=track_id,
        detector_matched=float(object_meta.confidence) >= 0.0,
    )


class MissedFrameCounter:
    """Consecutive frames each track has gone without a detector match."""

    def __init__(self) -> None:
        self._missed: dict[int, int] = {}

    def update(self, observations: Iterable[ObjectObservationV2]) -> dict[int, int]:
        current: dict[int, int] = {}
        for observation in observations:
            if observation.track_id is None:
                continue
            previous = self._missed.get(observation.track_id, 0)
            current[observation.track_id] = 0 if observation.detector_matched else previous + 1
        self._missed = current  # tracks absent this frame are forgotten
        return current


def perception_snapshot_from_metadata(
    timing: FrameTiming,
    object_metas: Iterable[Any],
    missed_frames: MissedFrameCounter | None = None,
    raw_detections: Iterable[ObjectObservationV2] | None = None,
) -> PerceptionSnapshotV2:
    """Build a strict V2 snapshot from DeepStream detector/tracker metadata.

    With a ``missed_frames`` counter, each track's ``missed_frames`` is the
    number of consecutive frames it has been carried by the tracker alone.
    With ``raw_detections`` (the detector's output for this frame, captured
    before the tracker), ``detections`` holds exactly what the detector saw
    and ``tracks`` what the tracker made of it.
    """

    detections: list[PerceptionDetectionV2] = []
    tracks: list[PerceptionTrackV2] = []
    observations = [
        observation for observation in (
            object_meta_to_observation_v2(object_meta, img_w=timing.img_w, img_h=timing.img_h)
            for object_meta in object_metas
        ) if observation is not None
    ]
    missed = missed_frames.update(observations) if missed_frames is not None else {}
    if raw_detections is not None:
        for observation in raw_detections:
            detections.append(PerceptionDetectionV2(
                detection_id=len(detections), box=observation.box,
                class_id=observation.class_id, confidence=observation.confidence,
            ))
    for observation in observations:
        if observation.track_id is None:
            if raw_detections is not None:
                continue  # the detector's view of this frame is already recorded
            detections.append(
                PerceptionDetectionV2(
                    detection_id=len(detections),
                    box=observation.box,
                    class_id=observation.class_id,
                    confidence=observation.confidence,
                )
            )
        else:
            tracks.append(
                PerceptionTrackV2(
                    track_id=observation.track_id,
                    box=observation.box,
                    class_id=observation.class_id,
                    confidence=observation.confidence,
                    age_frames=None,
                    missed_frames=missed.get(observation.track_id, 0),
                )
            )
    return PerceptionSnapshotV2(
        sequence=int(timing.frame_id),
        frame=PerceptionFrameV2(
            frame_id=int(timing.frame_id),
            source_time_ns=(
                int(timing.src_ts_ns) if timing.src_ts_ns is not None
                else int(timing.src_ts_ms) * 1_000_000
            ),
            received_time_ns=int(timing.rx_ts_ms) * 1_000_000,
            observed_time_ns=int(timing.infer_ts_ms) * 1_000_000,
            source_clock_domain=timing.source_clock_domain,
            source_identity_verified=timing.source_identity_verified,
            receive_clock_domain=timing.observation_clock_domain,
            observation_clock_domain=timing.observation_clock_domain,
            width=int(timing.img_w),
            height=int(timing.img_h),
        ),
        detections=tuple(detections),
        tracks=tuple(tracks),
    )
