"""Pure DeepStream metadata to strict perception V2 helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
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
    detector_matched = float(object_meta.confidence) >= 0.0
    # A tracker-only object's detector confidence is DeepStream's -0.1
    # placeholder; its tracker confidence is what says whether the box is
    # still on the target.
    confidence = (float(object_meta.confidence) if detector_matched
                  else float(getattr(object_meta, "tracker_confidence", 0.0)))
    return ObjectObservationV2(
        box=NormalizedBoxV2(
            x=left / img_w,
            y=top / img_h,
            w=(right - left) / img_w,
            h=(bottom - top) / img_h,
        ),
        class_id=str(int(object_meta.class_id)),
        confidence=_clamp(confidence, 0.0, 1.0),
        track_id=track_id,
        detector_matched=detector_matched,
    )


@dataclass
class _StableTrack:
    stable_id: int
    tracker_id: int
    centre: tuple[float, float]  # pixels
    size: tuple[float, float]    # pixels
    velocity: tuple[float, float] = (0.0, 0.0)  # pixels per frame
    last_frame: int = 0


def _box_px(box: NormalizedBoxV2, img_w: int, img_h: int) -> tuple[tuple[float, float], tuple[float, float]]:
    return ((box.x + box.w / 2) * img_w, (box.y + box.h / 2) * img_h), (box.w * img_w, box.h * img_h)


def _iou(a: NormalizedBoxV2, b: NormalizedBoxV2) -> float:
    ix = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


class TrackIdStitcher:
    """Stable track ids across tracker re-identification.

    NvDCF gives a drone a new id when its detections drop out long enough for
    the coasting box to drift from where the next detection lands (swarm
    recording 2026-09-28: 2.7 ids per drone, 488 frames with two tracks on
    one drone). A new tracker id inherits the stable id of a track lost within
    ``max_gap_frames`` whose predicted centre is within ``gate_box_sizes``
    box sizes and whose size is similar; two tracks on one drone (IoU >=
    ``duplicate_iou``) are published once, preferring the detection-backed
    box, under the older id. Stable ids are tracker ids, so a track that is
    never stitched keeps its own id.
    """

    def __init__(self, *, max_gap_frames: int = 45, gate_box_sizes: float = 1.5,
                 max_size_ratio: float = 2.0, duplicate_iou: float = 0.4) -> None:
        self._max_gap = max_gap_frames
        self._gate = gate_box_sizes
        self._max_size_ratio = max_size_ratio
        self._duplicate_iou = duplicate_iou
        self._tracks: dict[int, _StableTrack] = {}  # by stable id
        self._stable_of: dict[int, int] = {}         # tracker id -> stable id
        self._frame = 0
        self.stitched = 0
        self.duplicates_dropped = 0

    def apply(self, observations: list[ObjectObservationV2], *, img_w: int, img_h: int
              ) -> list[ObjectObservationV2]:
        self._frame += 1
        frame = self._frame
        tracked = [o for o in observations if o.track_id is not None]
        untracked = [o for o in observations if o.track_id is None]
        present = {o.track_id for o in tracked}
        claimed: set[int] = set()
        out: list[ObjectObservationV2] = []
        # Known tracker ids first, so a new id cannot take their stable id.
        for observation in sorted(tracked, key=lambda o: o.track_id not in self._stable_of):
            tracker_id = observation.track_id
            centre, size = _box_px(observation.box, img_w, img_h)
            stable_id = self._stable_of.get(tracker_id)
            if stable_id is None or stable_id in claimed:
                stable_id = self._stitch(centre, size, claimed, present, frame) or tracker_id
                if stable_id != tracker_id:
                    self.stitched += 1
                self._stable_of[tracker_id] = stable_id
            claimed.add(stable_id)
            previous = self._tracks.get(stable_id)
            velocity = (0.0, 0.0)
            if previous is not None and frame > previous.last_frame:
                gap = frame - previous.last_frame
                velocity = ((centre[0] - previous.centre[0]) / gap, (centre[1] - previous.centre[1]) / gap)
            self._tracks[stable_id] = _StableTrack(stable_id, tracker_id, centre, size, velocity, frame)
            out.append(replace(observation, track_id=stable_id))
        out = self._drop_duplicates(out)
        self._forget(frame)
        return out + untracked

    def _stitch(self, centre, size, claimed, present, frame) -> int | None:
        best, best_distance = None, math.inf
        for track in self._tracks.values():
            gap = frame - track.last_frame
            if (track.stable_id in claimed or track.tracker_id in present
                    or gap < 1 or gap > self._max_gap):
                continue
            ratio = max(size[0] / max(track.size[0], 1e-6), track.size[0] / max(size[0], 1e-6))
            if ratio > self._max_size_ratio:
                continue
            predicted = (track.centre[0] + track.velocity[0] * gap, track.centre[1] + track.velocity[1] * gap)
            distance = math.hypot(centre[0] - predicted[0], centre[1] - predicted[1])
            if distance <= self._gate * max(track.size) and distance < best_distance:
                best, best_distance = track.stable_id, distance
        return best

    def _drop_duplicates(self, observations: list[ObjectObservationV2]) -> list[ObjectObservationV2]:
        """One track per drone: keep the detection-backed box, under the older id."""
        ranked = sorted(observations, key=lambda o: (not o.detector_matched, o.track_id))
        kept: list[ObjectObservationV2] = []
        for observation in ranked:
            index = next((i for i, k in enumerate(kept)
                          if _iou(k.box, observation.box) >= self._duplicate_iou), None)
            if index is None:
                kept.append(observation)
                continue
            self.duplicates_dropped += 1
            twin = kept[index]
            older, newer = sorted((twin.track_id, observation.track_id))
            if twin.track_id != older:
                # The kept box continues under the older id from now on.
                state = self._tracks.pop(twin.track_id)
                self._tracks[older] = replace(state, stable_id=older)
                self._stable_of[state.tracker_id] = older
                kept[index] = replace(twin, track_id=older)
            else:
                self._tracks.pop(newer, None)
        return kept

    def _forget(self, frame: int) -> None:
        for stable_id in [s for s, t in self._tracks.items() if frame - t.last_frame > self._max_gap]:
            del self._tracks[stable_id]
        live = set(self._tracks)
        self._stable_of = {t: s for t, s in self._stable_of.items() if s in live}


class MissedFrameCounter:
    """Consecutive frames each track has gone without a detector match."""

    def __init__(self) -> None:
        self._missed: dict[int, int] = {}

    def missed(self, track_id: int) -> int:
        """Consecutive frames ``track_id`` has gone without a detector match so far."""
        return self._missed.get(track_id, 0)

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
    shadow_tracks: Iterable[ObjectObservationV2] = (),
    coast_min_confidence: float = 0.0,
    coast_max_frames: int | None = None,
    stitcher: TrackIdStitcher | None = None,
) -> PerceptionSnapshotV2:
    """Build a strict V2 snapshot from DeepStream detector/tracker metadata.

    With a ``missed_frames`` counter, each track's ``missed_frames`` is the
    number of consecutive frames it has been carried by the tracker alone.
    With ``raw_detections`` (the detector's output for this frame, captured
    before the tracker), ``detections`` holds exactly what the detector saw
    and ``tracks`` what the tracker made of it. ``shadow_tracks`` are the
    tracker's own estimates for targets it held back this frame (NvDCF shadow
    tracking), already gated by the caller; they are published as tracks with
    ``missed_frames`` > 0.

    Every coasting box (tracker-only object or shadow estimate) is published
    only while its tracker confidence is at least ``coast_min_confidence``
    and, with ``coast_max_frames``, it has coasted fewer frames than that.
    Gated boxes still count toward ``missed_frames``.
    """

    detections: list[PerceptionDetectionV2] = []
    tracks: list[PerceptionTrackV2] = []
    observations = [
        observation for observation in (
            object_meta_to_observation_v2(object_meta, img_w=timing.img_w, img_h=timing.img_h)
            for object_meta in object_metas
        ) if observation is not None
    ]
    reported = {o.track_id for o in observations if o.track_id is not None}
    observations += [o for o in shadow_tracks if o.track_id not in reported]
    if stitcher is not None:
        observations = stitcher.apply(observations, img_w=timing.img_w, img_h=timing.img_h)
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
            if not observation.detector_matched and (
                    observation.confidence < coast_min_confidence
                    or (coast_max_frames is not None
                        and missed.get(observation.track_id, 0) >= coast_max_frames)):
                continue
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
