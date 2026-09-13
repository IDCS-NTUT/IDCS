"""Explicit adapters between perception V2 and the legacy display contract.

Core perception, selection, tracking, and controller code must not import this
module.  It exists only at compatibility sinks that still publish or consume
``DetectionMsg``.
"""

from __future__ import annotations

from common.perception import PerceptionSnapshotV2, PerceptionTrackV2
from common.schemas import Box, DetectionMsg


def detection_msg_from_snapshot(
    snapshot: PerceptionSnapshotV2,
    *,
    use_tracks: bool | None = True,
) -> DetectionMsg:
    """Adapt a V2 snapshot to the legacy downstream display boundary.

    ``None`` includes raw detections followed by tracks for migration points
    where tracker availability is determined per object.
    """

    if use_tracks is None:
        objects = snapshot.detections + snapshot.tracks
    else:
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
    assessments = {item.track_id: item for item in snapshot.assessments}
    for box in boxes:
        if box.track_id is None or box.track_id not in assessments:
            continue
        values = assessments[box.track_id].model_dump(
            exclude={"track_id"},
            exclude_none=True,
        )
        for field, value in values.items():
            setattr(box, field, value)
    target_idx = None
    target_track_id = None
    if use_tracks is not False and snapshot.selection is not None:
        target_track_id = snapshot.selection.track_id
        target_idx = next(
            index
            for index, item in enumerate(objects)
            if isinstance(item, PerceptionTrackV2)
            and item.track_id == target_track_id
        )
    return DetectionMsg(
        frame_id=snapshot.frame.frame_id,
        src_ts_ms=snapshot.frame.source_time_ns // 1_000_000,
        rx_ts_ms=(
            snapshot.frame.received_time_ns
            if snapshot.frame.received_time_ns is not None
            else snapshot.frame.observed_time_ns
        ) // 1_000_000,
        infer_ts_ms=snapshot.frame.observed_time_ns // 1_000_000,
        img_w=snapshot.frame.width,
        img_h=snapshot.frame.height,
        boxes=boxes,
        target_idx=target_idx,
        target_track_id=target_track_id,
    )
