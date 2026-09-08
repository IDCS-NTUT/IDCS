"""Pure DeepStream-metadata to IDCS detection-message conversion helpers.

This module deliberately has no GStreamer, PyDS, ZMQ, or control dependency so
its coordinate and schema behavior can be tested off the Jetson.  The replay
verifier is the only current caller.  It writes shadow records and cannot
publish a control command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from common.schemas import Box, DetectionMsg


UNTRACKED_OBJECT_ID = (1 << 64) - 1


@dataclass(frozen=True)
class FrameTiming:
    """Timing values in the existing ``DetectionMsg`` clock fields.

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


def pts_ns_to_ms(pts_ns: int) -> int:
    """Convert a non-negative GStreamer PTS to relative integer milliseconds."""

    return max(int(pts_ns), 0) // 1_000_000


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def object_meta_to_box(object_meta: Any, *, img_w: int, img_h: int) -> Box | None:
    """Convert one ``NvDsObjectMeta``-like object to a clipped IDCS ``Box``.

    The helper only relies on the small PyDS surface used by DeepStream:
    ``class_id``, ``confidence``, ``object_id``, and ``rect_params``.
    Invalid or entirely out-of-frame boxes are omitted.
    """

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
    return Box(
        x=left / img_w,
        y=top / img_h,
        w=(right - left) / img_w,
        h=(bottom - top) / img_h,
        cls=str(int(object_meta.class_id)),
        conf=_clamp(float(object_meta.confidence), 0.0, 1.0),
        track_id=track_id,
    )


def detection_msg_from_metadata(
    timing: FrameTiming, object_metas: Iterable[Any],
) -> DetectionMsg:
    """Build a schema-valid, target-free detection message from metadata."""

    boxes = [
        box
        for object_meta in object_metas
        if (box := object_meta_to_box(object_meta, img_w=timing.img_w, img_h=timing.img_h))
        is not None
    ]
    return DetectionMsg(
        frame_id=int(timing.frame_id),
        src_ts_ms=int(timing.src_ts_ms),
        rx_ts_ms=int(timing.rx_ts_ms),
        infer_ts_ms=int(timing.infer_ts_ms),
        img_w=int(timing.img_w),
        img_h=int(timing.img_h),
        boxes=boxes,
    )
