"""Search-mode multi-target tracking adapter built around Ultralytics BoT-SORT."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Set, Tuple

import numpy as np

from common.schemas import Box

try:
    from ultralytics.trackers.bot_sort import BOTSORT
except ModuleNotFoundError as exc:  # pragma: no cover - dependency/environment specific
    BOTSORT = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class TrackObservation:
    track_id: int
    cls_id: int
    conf: float
    xyxy: Tuple[float, float, float, float]


class _Detections:
    """Minimal Ultralytics-compatible detection container for tracker.update()."""

    def __init__(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
    ) -> None:
        self.xyxy = self._as_2d(xyxy, 4)
        self.conf = self._as_1d(conf)
        self.cls = self._as_1d(cls)
        if not (len(self.xyxy) == len(self.conf) == len(self.cls)):
            raise ValueError("tracker detection arrays must share the same length")

    @staticmethod
    def _as_2d(values: np.ndarray, width: int) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.size == 0:
            return np.empty((0, width), dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[1] != width:
            raise ValueError(f"expected {width} columns, got {array.shape[1]}")
        return array

    @staticmethod
    def _as_1d(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.size == 0:
            return np.empty((0,), dtype=np.float32)
        return array

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    def __getitem__(self, item: Any) -> "_Detections":
        return _Detections(self.xyxy[item], self.conf[item], self.cls[item])

    @property
    def xywh(self) -> np.ndarray:
        if len(self) == 0:
            return np.empty((0, 4), dtype=np.float32)
        xyxy = self.xyxy
        xywh = np.empty_like(xyxy)
        xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
        xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
        xywh[:, 2] = np.maximum(0.0, xyxy[:, 2] - xyxy[:, 0])
        xywh[:, 3] = np.maximum(0.0, xyxy[:, 3] - xyxy[:, 1])
        return xywh


class BotSortSearchTracker:
    """Thin adapter around Ultralytics BOTSORT for search-mode use in this repo."""

    def __init__(self, *, frame_rate: float, config: dict[str, Any]) -> None:
        if BOTSORT is None:
            raise RuntimeError(
                "BoT-SORT dependency is not available; install ultralytics in the Jetson environment"
            ) from _IMPORT_ERROR

        args = SimpleNamespace(
            tracker_type="botsort",
            track_high_thresh=float(config["track_high_thresh"]),
            track_low_thresh=float(config["track_low_thresh"]),
            new_track_thresh=float(config["new_track_thresh"]),
            track_buffer=int(config["track_buffer"]),
            match_thresh=float(config["match_thresh"]),
            fuse_score=bool(config["fuse_score"]),
            gmc_method=str(config["gmc_method"]),
            proximity_thresh=float(config["proximity_thresh"]),
            appearance_thresh=float(config["appearance_thresh"]),
            with_reid=bool(config["with_reid"]),
            model=str(config["reid_model"]),
        )

        self._tracker = BOTSORT(args=args, frame_rate=max(1, int(round(frame_rate))))
    self._reuse_timeout_frames = max(1, int(config["track_buffer"]))
        self._frame_index = 0
        self._next_display_track_id = 1
        self._free_display_track_ids: List[int] = []
        self._raw_to_display_track_id: dict[int, int] = {}
        self._last_seen_frame: dict[int, int] = {}

    def reset(self) -> None:
        self._tracker.reset()
        self._frame_index = 0
        self._next_display_track_id = 1
        self._free_display_track_ids.clear()
        self._raw_to_display_track_id.clear()
        self._last_seen_frame.clear()

    def _assign_display_track_id(self, raw_track_id: int) -> int:
        display_track_id = self._raw_to_display_track_id.get(raw_track_id)
        if display_track_id is not None:
            return display_track_id
        if self._free_display_track_ids:
            display_track_id = heapq.heappop(self._free_display_track_ids)
        else:
            display_track_id = self._next_display_track_id
            self._next_display_track_id += 1
        self._raw_to_display_track_id[raw_track_id] = display_track_id
        return display_track_id

    def _reclaim_stale_display_track_ids(self, active_raw_track_ids: Set[int]) -> None:
        stale_raw_track_ids: List[int] = []
        for raw_track_id, last_seen_frame in self._last_seen_frame.items():
            if raw_track_id in active_raw_track_ids:
                continue
            if (self._frame_index - int(last_seen_frame)) > self._reuse_timeout_frames:
                stale_raw_track_ids.append(raw_track_id)

        for raw_track_id in stale_raw_track_ids:
            display_track_id = self._raw_to_display_track_id.pop(raw_track_id, None)
            self._last_seen_frame.pop(raw_track_id, None)
            if display_track_id is not None:
                heapq.heappush(self._free_display_track_ids, int(display_track_id))

    def update(
        self,
        *,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
        frame: np.ndarray,
        warp_override: Optional[np.ndarray] = None,
    ) -> List[TrackObservation]:
        self._frame_index += 1
        detections = _Detections(xyxy=xyxy, conf=conf, cls=cls)

        gmc = getattr(self._tracker, "gmc", None)
        original_apply = getattr(gmc, "apply", None)
        if warp_override is not None and gmc is not None and original_apply is not None:
            warp_matrix = np.asarray(warp_override, dtype=np.float32)

            def _apply_override(_img: np.ndarray, _detections: Any = None) -> np.ndarray:
                return warp_matrix

            gmc.apply = _apply_override  # type: ignore[assignment]

        try:
            tracked = self._tracker.update(detections, img=frame)
        finally:
            if warp_override is not None and gmc is not None and original_apply is not None:
                gmc.apply = original_apply  # type: ignore[assignment]

        if tracked is None:
            return []

        tracked_np = np.asarray(tracked, dtype=np.float32)
        if tracked_np.size == 0:
            return []
        if tracked_np.ndim == 1:
            tracked_np = tracked_np.reshape(1, -1)

        observations: List[TrackObservation] = []
        active_raw_track_ids: Set[int] = set()
        for row in tracked_np:
            if row.shape[0] < 7:
                continue
            x1, y1, x2, y2, track_id, score, cls_id = row[:7]
            raw_track_id = int(round(float(track_id)))
            active_raw_track_ids.add(raw_track_id)
            display_track_id = self._assign_display_track_id(raw_track_id)
            self._last_seen_frame[raw_track_id] = self._frame_index
            observations.append(
                TrackObservation(
                    track_id=display_track_id,
                    cls_id=int(round(float(cls_id))),
                    conf=float(score),
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
        self._reclaim_stale_display_track_ids(active_raw_track_ids)
        return observations


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(float(ax1), float(bx1))
    inter_y1 = max(float(ay1), float(by1))
    inter_x2 = min(float(ax2), float(bx2))
    inter_y2 = min(float(ay2), float(by2))
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(ax2 - ax1)) * max(0.0, float(ay2 - ay1))
    area_b = max(0.0, float(bx2 - bx1)) * max(0.0, float(by2 - by1))
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def assign_track_ids_to_boxes(
    boxes: Sequence[Box],
    tracks: Sequence[TrackObservation],
    *,
    img_w: int,
    img_h: int,
    min_iou: float,
) -> List[Optional[int]]:
    """Greedily match tracker outputs to current detections and return per-box track IDs."""

    if not boxes:
        return []

    if not tracks:
        return [None for _ in boxes]

    det_xyxy = []
    for box in boxes:
        x1 = float(box.x) * float(img_w)
        y1 = float(box.y) * float(img_h)
        x2 = (float(box.x) + float(box.w)) * float(img_w)
        y2 = (float(box.y) + float(box.h)) * float(img_h)
        det_xyxy.append(np.asarray((x1, y1, x2, y2), dtype=np.float32))

    candidate_pairs: List[Tuple[float, int, int]] = []
    for det_idx, det_box in enumerate(det_xyxy):
        box_cls_id: Optional[int]
        try:
            box_cls_id = int(str(boxes[det_idx].cls).strip())
        except (TypeError, ValueError):
            box_cls_id = None

        for track_idx, track in enumerate(tracks):
            if box_cls_id is not None and int(track.cls_id) != box_cls_id:
                continue
            iou = _iou_xyxy(det_box, np.asarray(track.xyxy, dtype=np.float32))
            if iou >= min_iou:
                candidate_pairs.append((iou, det_idx, track_idx))

    candidate_pairs.sort(key=lambda item: item[0], reverse=True)

    assigned_det: set[int] = set()
    assigned_track: set[int] = set()
    track_ids: List[Optional[int]] = [None for _ in boxes]

    for _iou, det_idx, track_idx in candidate_pairs:
        if det_idx in assigned_det or track_idx in assigned_track:
            continue
        assigned_det.add(det_idx)
        assigned_track.add(track_idx)
        track_ids[det_idx] = int(tracks[track_idx].track_id)

    return track_ids


def boxes_to_tracker_arrays(
    boxes: Sequence[Box],
    *,
    img_w: int,
    img_h: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert normalized Box entries into xyxy/conf/cls arrays for tracker input."""

    if not boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    xyxy_rows: List[Tuple[float, float, float, float]] = []
    conf_rows: List[float] = []
    cls_rows: List[float] = []

    for box in boxes:
        x1 = float(box.x) * float(img_w)
        y1 = float(box.y) * float(img_h)
        x2 = (float(box.x) + float(box.w)) * float(img_w)
        y2 = (float(box.y) + float(box.h)) * float(img_h)
        xyxy_rows.append((x1, y1, x2, y2))
        conf_rows.append(float(box.conf))
        try:
            cls_rows.append(float(int(str(box.cls).strip())))
        except (TypeError, ValueError):
            cls_rows.append(0.0)

    return (
        np.asarray(xyxy_rows, dtype=np.float32),
        np.asarray(conf_rows, dtype=np.float32),
        np.asarray(cls_rows, dtype=np.float32),
    )
