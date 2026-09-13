"""Legacy DeepStream metadata adapters retained for display compatibility."""

from __future__ import annotations

from typing import Any, Iterable

from common.perception_compat import detection_msg_from_snapshot
from common.schemas import Box, DetectionMsg
from jetson.deepstream.shadow_adapter import (
    FrameTiming,
    object_meta_to_observation_v2,
    perception_snapshot_from_metadata,
)


def object_meta_to_box(object_meta: Any, *, img_w: int, img_h: int) -> Box | None:
    """Adapt one normalized V2 metadata observation to legacy ``Box``."""

    observation = object_meta_to_observation_v2(
        object_meta,
        img_w=img_w,
        img_h=img_h,
    )
    if observation is None:
        return None
    return Box(
        x=observation.box.x,
        y=observation.box.y,
        w=observation.box.w,
        h=observation.box.h,
        cls=observation.class_id,
        conf=observation.confidence,
        track_id=observation.track_id,
    )


def detection_msg_from_metadata(
    timing: FrameTiming,
    object_metas: Iterable[Any],
) -> DetectionMsg:
    """Build a legacy message through the explicit V2 compatibility edge."""

    return detection_msg_from_snapshot(
        perception_snapshot_from_metadata(timing, object_metas),
        use_tracks=None,
    )
