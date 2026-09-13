"""Pure DeepStream-metadata to strict perception V2 helpers.

This module deliberately has no GStreamer, PyDS, ZMQ, or control dependency so
its coordinate and schema behavior can be tested off the Jetson.
"""

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
    """Source and observation timing used to construct a V2 frame contract.

    For a file replay, ``src_ts_ms`` is source-PTS-relative milliseconds;
    ``rx_ts_ms`` and ``infer_ts_ms`` are Jetson monotonic timestamps.  They
    must not be subtracted across clock domains.  A live adapter will instead
    match the PC-supplied ``CamState`` header by frame ID.
    """

    frame_id: int
    src_ts_ms: int
    rx_ts_ms: int
    infer_ts_ms: int
    img_w: int
    img_h: int
    source_clock_domain: str = "unspecified"
    observation_clock_domain: str = "jetson_monotonic"


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


def object_meta_to_observation_v2(
    object_meta: Any, *, img_w: int, img_h: int
) -> ObjectObservationV2 | None:
    """Normalize one ``NvDsObjectMeta`` without creating a legacy schema."""

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
    )

def perception_snapshot_from_metadata(
    timing: FrameTiming,
    object_metas: Iterable[Any],
) -> PerceptionSnapshotV2:
    """Build a strict V2 snapshot from DeepStream detector/tracker metadata."""

    detections: list[PerceptionDetectionV2] = []
    tracks: list[PerceptionTrackV2] = []
    for object_meta in object_metas:
        observation = object_meta_to_observation_v2(
            object_meta,
            img_w=timing.img_w,
            img_h=timing.img_h,
        )
        if observation is None:
            continue
        if observation.track_id is None:
            detections.append(PerceptionDetectionV2(
                detection_id=len(detections),
                box=observation.box,
                class_id=observation.class_id,
                confidence=observation.confidence,
            ))
        else:
            tracks.append(PerceptionTrackV2(
                track_id=observation.track_id,
                box=observation.box,
                class_id=observation.class_id,
                confidence=observation.confidence,
                age_frames=None,
                missed_frames=0,
            ))
    return PerceptionSnapshotV2(
        sequence=int(timing.frame_id),
        frame=PerceptionFrameV2(
            frame_id=int(timing.frame_id),
            source_time_ns=int(timing.src_ts_ms) * 1_000_000,
            received_time_ns=int(timing.rx_ts_ms) * 1_000_000,
            observed_time_ns=int(timing.infer_ts_ms) * 1_000_000,
            source_clock_domain=timing.source_clock_domain,
            receive_clock_domain=timing.observation_clock_domain,
            observation_clock_domain=timing.observation_clock_domain,
            width=int(timing.img_w),
            height=int(timing.img_h),
        ),
        detections=tuple(detections),
        tracks=tuple(tracks),
    )
