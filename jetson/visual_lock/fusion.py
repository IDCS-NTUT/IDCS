"""Deterministic quality gates for detector-independent visual lock.

DCF handles appearance/position. Forward-backward checked LK features estimate
translation/scale. Agreement is required: feature flow is not an authority by
itself, and DCF cannot silently reacquire a different object. No depth is inferred
from the operator's arbitrarily sized seed rectangle.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np

from common.perception import (NormalizedBoxV2, PerceptionFrameV2, PerceptionSnapshotV2,
                               PerceptionTrackV2, TargetSelectionV2)


@dataclass(frozen=True)
class Proposal:
    box: tuple[float, float, float, float]
    psr: float
    peak: float
    before: np.ndarray
    after: np.ndarray
    fb: np.ndarray
    native_ms: float = 0.0
    seed_features: int = 0


@dataclass(frozen=True)
class Gates:
    min_points: int = 8
    min_psr: float = 6.0
    max_fb_px: float = 1.0
    max_residual_px: float = 1.5
    min_inlier_fraction: float = .65
    max_disagreement_fraction: float = .15
    min_scale: float = .9
    max_scale: float = 1.1
    max_rotation_rad: float = .25
    acquire_frames: int = 3
    lost_after_frames: int = 3
    max_gap_ns: int = 150_000_000

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid gate {name}")
        for name in ("min_points", "acquire_frames", "lost_after_frames", "max_gap_ns"):
            if type(getattr(self,name)) is not int:
                raise ValueError(f"integer gate required: {name}")
        if self.min_points < 3 or self.min_inlier_fraction > 1:
            raise ValueError("invalid feature gates")
        if not self.min_scale <= 1 <= self.max_scale:
            raise ValueError("scale gates must include unity")


@dataclass(frozen=True)
class Decision:
    state: str
    reason: str
    box: tuple[float, float, float, float] | None = None
    inliers: int = 0
    scale: float | None = None
    disagreement_px: float | None = None
    accepted: bool = False
    update_template: bool = False


def _similarity(a: np.ndarray, b: np.ndarray):
    """Least-squares 2D similarity, disallowing reflection."""
    x = a - a.mean(axis=0)
    y = b - b.mean(axis=0)
    denom = float(np.sum(x*x))
    if denom < 1e-8:
        return None
    real = float(np.sum(x*y)) / denom
    imag = float(np.sum(x[:, 0]*y[:, 1] - x[:, 1]*y[:, 0])) / denom
    matrix = np.array([[real, -imag], [imag, real]])
    offset = b.mean(axis=0) - matrix @ a.mean(axis=0)
    return matrix, offset


def robust_similarity(a, b, threshold):
    """Bounded deterministic RANSAC: at most 64 points / 96 spaced pairs."""
    pairs = list(combinations(range(len(a)), 2))
    if not pairs:
        return None
    pairs = [pairs[i] for i in np.linspace(0, len(pairs)-1, min(96, len(pairs)), dtype=int)]
    # Evaluate the bounded hypothesis set in one NumPy operation. Python loops
    # over tiny fits otherwise dominate GPU tracking time on the Jetson CPU.
    indices = np.asarray(pairs)
    da = a[indices[:,1]]-a[indices[:,0]]
    db = b[indices[:,1]]-b[indices[:,0]]
    good = np.sum(da*da,axis=1) >= 36
    indices, da, db = indices[good], da[good], db[good]
    if not len(indices):
        return None
    denom = np.sum(da*da,axis=1)
    real = np.sum(da*db,axis=1)/denom
    imag = (da[:,0]*db[:,1]-da[:,1]*db[:,0])/denom
    matrix = np.stack((real,-imag,imag,real),axis=1).reshape(-1,2,2)
    offset = b[indices[:,0]]-np.einsum("nij,nj->ni",matrix,a[indices[:,0]])
    predicted = np.einsum("nij,kj->nki",matrix,a)+offset[:,None,:]
    masks = np.linalg.norm(predicted-b,axis=2) <= threshold
    counts = masks.sum(axis=1)
    best = masks[int(np.argmax(counts))]
    if best.sum() < 3:
        return None
    fit = _similarity(a[best], b[best])
    matrix, offset = fit
    mask = np.linalg.norm(a @ matrix.T + offset - b, axis=1) <= threshold
    return matrix, offset, mask


class VisualLock:
    """One explicit seed, one source session, no automatic detector fallback.

    Frame ordering uses ONLY source timestamps in their named clock domain.
    observed_time_ns is retained for V2 publication, not subtracted from a remote
    source clock. Native backend exceptions must call fail() before propagation.
    """
    def __init__(self, gates: Gates = Gates()):
        self.gates = gates
        self.state = "idle"
        self.box = None
        self.frame = None
        self.seed_frame_id = 0
        self.track_id = 0
        self.good = self.bad = self.age = 0

    def seed(self, frame: PerceptionFrameV2, box: NormalizedBoxV2, feature_count: int):
        if not frame.tracker_session or not frame.source_identity_verified:
            raise ValueError("visual lock requires verified source/session identity")
        self.track_id += 1
        self.frame = frame
        self.seed_frame_id = frame.frame_id
        self.box = (box.x*frame.width, box.y*frame.height, box.w*frame.width, box.h*frame.height)
        self.good = self.bad = 0
        self.age = 1
        if min(self.box[2:]) < 12 or feature_count < self.gates.min_points:
            self.state = "lost"
            return Decision("lost", "seed lacks texture or pixel size")
        self.state = "acquiring"
        return Decision(self.state, "operator seed", self.box)

    def fail(self, reason: str) -> Decision:
        self.state = "lost"
        self.good = 0
        return Decision(self.state, reason)

    def _reject(self, reason: str, **details):
        self.bad += 1
        self.good = 0
        self.state = "lost" if self.bad >= self.gates.lost_after_frames else "uncertain"
        return Decision(self.state, reason, **details)

    def check_frame(self, frame: PerceptionFrameV2) -> Decision | None:
        """Reject invalid frame identity before spending GPU work on it."""
        if self.state in {"idle", "lost"}:
            return Decision(self.state, "explicit new seed required")
        old = self.frame
        if (not frame.source_identity_verified or frame.tracker_session != old.tracker_session
                or frame.source_clock_domain != old.source_clock_domain
                or (frame.width, frame.height) != (old.width, old.height)):
            return self.fail("source identity/session/geometry changed")
        if frame.frame_id <= old.frame_id or frame.source_time_ns <= old.source_time_ns:
            # Never accept a replayed command/observation, even while acquiring.
            return self.fail("repeated or out-of-order frame")
        if frame.source_time_ns - old.source_time_ns > self.gates.max_gap_ns:
            return self.fail("source frame gap")
        return None

    def evaluate(self, frame: PerceptionFrameV2, p: Proposal) -> Decision:
        invalid = self.check_frame(frame)
        if invalid is not None:
            return invalid
        self.frame = frame
        self.age += 1
        arrays = (np.asarray(p.before), np.asarray(p.after), np.asarray(p.fb))
        a, b, fb = arrays
        if (len(p.box) != 4 or a.ndim != 2 or a.shape[1] != 2 or b.shape != a.shape or fb.shape != (len(a),)
                or len(a) > 64 or any(not np.isfinite(x).all() for x in arrays)
                or not all(math.isfinite(x) for x in (*p.box, p.psr, p.peak))):
            return self.fail("invalid backend proposal")
        if p.psr < self.gates.min_psr or p.peak <= 0:
            return self._reject("weak DCF response")
        valid = (fb >= 0) & (fb <= self.gates.max_fb_px)
        a, b = a[valid], b[valid]
        if len(a) < self.gates.min_points:
            return self._reject("insufficient forward/backward features", inliers=len(a))
        fit = robust_similarity(a, b, self.gates.max_residual_px)
        if fit is None:
            return self._reject("degenerate feature geometry")
        matrix, offset, mask = fit
        n = int(mask.sum())
        if n < self.gates.min_points or n / len(a) < self.gates.min_inlier_fraction:
            return self._reject("inconsistent feature motion", inliers=n)
        # Require spatial support rather than eight almost coincident corners.
        extent = np.ptp(a[mask], axis=0)
        if np.any(extent < np.array(self.box[2:]) * .2):
            return self._reject("insufficient feature spread", inliers=n)
        scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        rotation = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
        if not self.gates.min_scale <= scale <= self.gates.max_scale or abs(rotation) > self.gates.max_rotation_rad:
            return self._reject("unsupported scale/rotation", inliers=n, scale=scale)
        center = np.array(self.box[:2]) + np.array(self.box[2:]) / 2
        flow_center = matrix @ center + offset
        dcf_center = np.array(p.box[:2]) + np.array(p.box[2:]) / 2
        disagreement = float(np.linalg.norm(flow_center-dcf_center))
        if disagreement > max(3.0, min(self.box[2:])*self.gates.max_disagreement_fraction):
            return self._reject("DCF/feature disagreement", inliers=n, scale=scale,
                                disagreement_px=disagreement)
        # Feature scale refines DCF's fixed-size box; agreeing feature centre has
        # subpixel precision whereas the DCF response is sampled on a HOG grid.
        size = np.array(self.box[2:]) * scale
        xy = flow_center - size / 2
        if np.any(size < 12) or np.any(xy < 0) or np.any(xy+size > [frame.width, frame.height]):
            return self._reject("target reaches image boundary", inliers=n, scale=scale)
        self.box = tuple(float(v) for v in (*xy, *size))
        self.bad = 0
        self.good += 1
        self.state = "tracking" if self.good >= self.gates.acquire_frames else "acquiring"
        return Decision(self.state, "appearance/flow agree", self.box, n, scale, disagreement,
                        True, self.state == "tracking")

    def snapshot(self, sequence: int, decision: Decision) -> PerceptionSnapshotV2:
        """Only verified tracking becomes selectable. No fabricated class/depth."""
        tracks = ()
        selection = None
        if decision.accepted and decision.state == "tracking":
            x, y, w, h = decision.box
            box = NormalizedBoxV2(x=x/self.frame.width, y=y/self.frame.height,
                                  w=w/self.frame.width, h=h/self.frame.height)
            tracks = (PerceptionTrackV2(track_id=self.track_id, box=box, class_id="visual",
                                        confidence=1.0, age_frames=self.age, missed_frames=0),)
            selection = TargetSelectionV2(track_id=self.track_id, source_frame_id=self.frame.frame_id,
                applied_frame_id=self.frame.frame_id, selected_time_ns=self.frame.observed_time_ns,
                selection_clock_domain=self.frame.observation_clock_domain, policy="operator_visual_lock")
        return PerceptionSnapshotV2(sequence=sequence, frame=self.frame, tracks=tracks, selection=selection)
