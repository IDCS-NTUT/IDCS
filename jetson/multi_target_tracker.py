from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from common.schemas import Box


@dataclass
class TrackState:
    track_id: int
    cls: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    velocity: Tuple[float, float]
    age: int
    missed: int
    hits: int
    confirmed: bool


@dataclass
class TrackerUpdateResult:
    detection_to_track: Dict[int, TrackState]
    active_tracks: List[TrackState]


class MultiTargetTracker:
    """Lightweight online multi-object tracker for detector bounding boxes."""

    def __init__(
        self,
        *,
        min_hits: int = 2,
        max_missed: int = 5,
        iou_gate: float = 0.1,
        center_dist_gate_px: float = 160.0,
        use_hungarian: bool = False,
    ) -> None:
        self._min_hits = max(1, int(min_hits))
        self._max_missed = max(1, int(max_missed))
        self._iou_gate = float(max(0.0, min(1.0, iou_gate)))
        self._center_dist_gate_px = float(max(1.0, center_dist_gate_px))
        self._use_hungarian = bool(use_hungarian)
        self._tracks: List[TrackState] = []
        self._next_track_id = 1
        self._last_timestamp_s: Optional[float] = None

    def update(
        self,
        detections: Sequence[Box],
        *,
        img_w: int,
        img_h: int,
        timestamp_s: Optional[float] = None,
    ) -> TrackerUpdateResult:
        dt = self._resolve_dt(timestamp_s)

        for track in self._tracks:
            self._predict_track(track, dt, img_w, img_h)

        matches, unmatched_track_indices, unmatched_det_indices = self._match_tracks(
            detections, img_w=img_w, img_h=img_h
        )

        detection_to_track: Dict[int, TrackState] = {}

        for track_idx, det_idx in matches:
            track = self._tracks[track_idx]
            det = detections[det_idx]
            self._update_track_with_detection(track, det, img_w=img_w, img_h=img_h, dt=dt)
            detection_to_track[det_idx] = track

        for track_idx in unmatched_track_indices:
            self._tracks[track_idx].missed += 1

        for det_idx in unmatched_det_indices:
            det = detections[det_idx]
            track = self._spawn_track(det, img_w=img_w, img_h=img_h)
            self._tracks.append(track)
            detection_to_track[det_idx] = track

        self._tracks = [track for track in self._tracks if track.missed <= self._max_missed]

        active_tracks = [track for track in self._tracks if track.confirmed]
        return TrackerUpdateResult(
            detection_to_track=detection_to_track,
            active_tracks=[self._snapshot(track) for track in active_tracks],
        )

    def _resolve_dt(self, timestamp_s: Optional[float]) -> float:
        if timestamp_s is None:
            return 1.0

        if self._last_timestamp_s is None:
            self._last_timestamp_s = float(timestamp_s)
            return 1.0

        dt = float(timestamp_s) - self._last_timestamp_s
        self._last_timestamp_s = float(timestamp_s)
        if not math.isfinite(dt) or dt <= 0.0:
            return 1.0
        return min(1.0, max(1e-3, dt))

    def _spawn_track(self, det: Box, *, img_w: int, img_h: int) -> TrackState:
        cx, cy = _box_center_px(det, img_w=img_w, img_h=img_h)
        track = TrackState(
            track_id=self._next_track_id,
            cls=str(det.cls),
            confidence=float(det.conf),
            bbox=(float(det.x), float(det.y), float(det.w), float(det.h)),
            center=(cx, cy),
            velocity=(0.0, 0.0),
            age=1,
            missed=0,
            hits=1,
            confirmed=self._min_hits <= 1,
        )
        self._next_track_id += 1
        return track

    def _predict_track(self, track: TrackState, dt: float, img_w: int, img_h: int) -> None:
        vx, vy = track.velocity
        cx = track.center[0] + vx * dt
        cy = track.center[1] + vy * dt
        bw = max(1e-6, track.bbox[2] * float(img_w))
        bh = max(1e-6, track.bbox[3] * float(img_h))
        x = (cx - 0.5 * bw) / float(img_w)
        y = (cy - 0.5 * bh) / float(img_h)
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        track.bbox = (
            x,
            y,
            max(1e-6, min(1.0, track.bbox[2])),
            max(1e-6, min(1.0, track.bbox[3])),
        )
        track.center = (cx, cy)
        track.age += 1

    def _update_track_with_detection(
        self,
        track: TrackState,
        det: Box,
        *,
        img_w: int,
        img_h: int,
        dt: float,
    ) -> None:
        cx, cy = _box_center_px(det, img_w=img_w, img_h=img_h)
        prev_cx, prev_cy = track.center
        safe_dt = max(1e-3, dt)
        raw_vx = (cx - prev_cx) / safe_dt
        raw_vy = (cy - prev_cy) / safe_dt
        alpha = 0.6
        track.velocity = (
            alpha * raw_vx + (1.0 - alpha) * track.velocity[0],
            alpha * raw_vy + (1.0 - alpha) * track.velocity[1],
        )
        track.center = (cx, cy)
        track.bbox = (float(det.x), float(det.y), float(det.w), float(det.h))
        track.cls = str(det.cls)
        track.confidence = float(det.conf)
        track.missed = 0
        track.hits += 1
        if not track.confirmed and track.hits >= self._min_hits:
            track.confirmed = True

    def _match_tracks(
        self,
        detections: Sequence[Box],
        *,
        img_w: int,
        img_h: int,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not self._tracks or not detections:
            return [], list(range(len(self._tracks))), list(range(len(detections)))

        candidate_pairs: List[Tuple[float, int, int]] = []
        for track_idx, track in enumerate(self._tracks):
            tx, ty = track.center
            tb = track.bbox
            for det_idx, det in enumerate(detections):
                if str(det.cls) != track.cls:
                    continue
                db = (float(det.x), float(det.y), float(det.w), float(det.h))
                iou = _iou_xywh(tb, db)
                dcx, dcy = _box_center_px(det, img_w=img_w, img_h=img_h)
                dist = math.hypot(dcx - tx, dcy - ty)
                if iou < self._iou_gate and dist > self._center_dist_gate_px:
                    continue
                dist_norm = min(1.0, dist / self._center_dist_gate_px)
                cost = (1.0 - iou) + (0.25 * dist_norm)
                candidate_pairs.append((cost, track_idx, det_idx))

        if not candidate_pairs:
            return [], list(range(len(self._tracks))), list(range(len(detections)))

        if self._use_hungarian:
            hungarian = self._try_hungarian(candidate_pairs, len(self._tracks), len(detections))
            if hungarian is not None:
                matches = hungarian
            else:
                matches = _greedy_assign(candidate_pairs)
        else:
            matches = _greedy_assign(candidate_pairs)

        matched_tracks = {t for t, _ in matches}
        matched_dets = {d for _, d in matches}
        unmatched_track_indices = [idx for idx in range(len(self._tracks)) if idx not in matched_tracks]
        unmatched_det_indices = [idx for idx in range(len(detections)) if idx not in matched_dets]
        return matches, unmatched_track_indices, unmatched_det_indices

    def _try_hungarian(
        self,
        candidate_pairs: Sequence[Tuple[float, int, int]],
        n_tracks: int,
        n_dets: int,
    ) -> Optional[List[Tuple[int, int]]]:
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore
            import numpy as np
        except Exception:
            return None

        INF = 1e6
        matrix = np.full((n_tracks, n_dets), INF, dtype=float)
        for cost, track_idx, det_idx in candidate_pairs:
            matrix[track_idx, det_idx] = min(matrix[track_idx, det_idx], float(cost))

        row_idx, col_idx = linear_sum_assignment(matrix)
        matches: List[Tuple[int, int]] = []
        for t_idx, d_idx in zip(row_idx.tolist(), col_idx.tolist()):
            if matrix[t_idx, d_idx] >= INF:
                continue
            matches.append((int(t_idx), int(d_idx)))
        return matches

    @staticmethod
    def _snapshot(track: TrackState) -> TrackState:
        return TrackState(
            track_id=int(track.track_id),
            cls=str(track.cls),
            confidence=float(track.confidence),
            bbox=(
                float(track.bbox[0]),
                float(track.bbox[1]),
                float(track.bbox[2]),
                float(track.bbox[3]),
            ),
            center=(float(track.center[0]), float(track.center[1])),
            velocity=(float(track.velocity[0]), float(track.velocity[1])),
            age=int(track.age),
            missed=int(track.missed),
            hits=int(track.hits),
            confirmed=bool(track.confirmed),
        )


def _box_center_px(det: Box, *, img_w: int, img_h: int) -> Tuple[float, float]:
    return (
        (float(det.x) + (float(det.w) * 0.5)) * float(img_w),
        (float(det.y) + (float(det.h) * 0.5)) * float(img_h),
    )


def _iou_xywh(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2 = ax1 + max(0.0, aw)
    ay2 = ay1 + max(0.0, ah)
    bx2 = bx1 + max(0.0, bw)
    by2 = by1 + max(0.0, bh)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


def _greedy_assign(candidate_pairs: Sequence[Tuple[float, int, int]]) -> List[Tuple[int, int]]:
    matches: List[Tuple[int, int]] = []
    used_tracks = set()
    used_dets = set()
    for _, track_idx, det_idx in sorted(candidate_pairs, key=lambda item: item[0]):
        if track_idx in used_tracks or det_idx in used_dets:
            continue
        used_tracks.add(track_idx)
        used_dets.add(det_idx)
        matches.append((track_idx, det_idx))
    return matches
