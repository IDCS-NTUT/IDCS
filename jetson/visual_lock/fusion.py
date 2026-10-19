"""Deterministic quality gates for detector-independent visual lock.

DCF handles appearance/position. Forward-backward checked LK features estimate
translation/scale. Agreement is required: feature flow is not an authority by
itself, and DCF cannot silently reacquire a different object. No depth is inferred
from the operator's arbitrarily sized seed rectangle.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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
    point_ids: np.ndarray | None = None
    input_features: int = 0
    forward_features: int = 0
    backward_features: int = 0
    reference_before: np.ndarray | None = None
    reference_after: np.ndarray | None = None
    reference_fb: np.ndarray | None = None
    seed_report: object | None = None
    seed_gray: np.ndarray | None = None


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
    max_anchor_residual_px: float = .75
    max_geometry_median_fb_px: float = .5

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
    retain_mask: int = 0
    geometry_confident: bool = False
    anchor_scale: float | None = None
    anchor_residual_px: float | None = None


def _similarity(a: np.ndarray, b: np.ndarray):
    """Least-squares 2D similarity, disallowing reflection."""
    a_mean, b_mean = a.mean(axis=0), b.mean(axis=0)
    x = a - a_mean
    y = b - b_mean
    denom = float(np.sum(x*x))
    if denom < 1e-8:
        return None
    real = float(np.sum(x*y)) / denom
    imag = float(np.sum(x[:, 0]*y[:, 1] - x[:, 1]*y[:, 0])) / denom
    matrix = np.array([[real, -imag], [imag, real]])
    offset = b_mean - matrix @ a_mean
    return matrix, offset


@lru_cache(maxsize=None)
def _spaced_pairs(n):
    """The fixed hypothesis pairs for n points (read-only; at most 65 sizes)."""
    pairs = list(combinations(range(n), 2))
    if not pairs:
        return None
    pairs = [pairs[i] for i in np.linspace(0, len(pairs)-1, min(96, len(pairs)), dtype=int)]
    indices = np.asarray(pairs)
    indices.setflags(write=False)
    return indices


def robust_similarity(a, b, threshold):
    """Bounded deterministic RANSAC: at most 64 points / 96 spaced pairs."""
    indices = _spaced_pairs(len(a))
    if indices is None:
        return None
    # Evaluate the bounded hypothesis set in one NumPy operation. Python loops
    # over tiny fits otherwise dominate GPU tracking time on the Jetson CPU.
    da = a[indices[:,1]]-a[indices[:,0]]
    db = b[indices[:,1]]-b[indices[:,0]]
    good = np.sum(da*da,axis=1) >= 36
    indices, da, db = indices[good], da[good], db[good]
    if not len(indices):
        return None
    denom = np.sum(da*da,axis=1)
    real = np.sum(da*db,axis=1)/denom
    imag = (da[:,0]*db[:,1]-da[:,1]*db[:,0])/denom
    # Apply each [[real,-imag],[imag,real]] hypothesis explicitly: einsum's
    # generic path dominated fusion time. Two-term sums are exact in either
    # order, so results are bit-identical to the matrix form.
    a0 = a[indices[:,0]]
    offset_x = b[indices[:,0],0]-(real*a0[:,0]-imag*a0[:,1])
    offset_y = b[indices[:,0],1]-(imag*a0[:,0]+real*a0[:,1])
    ax, ay = a[:,0], a[:,1]
    dx = real[:,None]*ax-imag[:,None]*ay+offset_x[:,None]-b[:,0]
    dy = imag[:,None]*ax+real[:,None]*ay+offset_y[:,None]-b[:,1]
    masks = np.sqrt(dx*dx+dy*dy) <= threshold
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
        self.anchors = {}
        self.anchor_matrix = np.eye(2)
        self.anchor_offset = np.zeros(2)
        self.seed_size = None
        self.seed_center = None

    def seed(self, frame: PerceptionFrameV2, box: NormalizedBoxV2, feature_count: int):
        if not frame.tracker_session or not frame.source_identity_verified:
            raise ValueError("visual lock requires verified source/session identity")
        self.track_id += 1
        self.frame = frame
        self.seed_frame_id = frame.frame_id
        self.box = (box.x*frame.width, box.y*frame.height, box.w*frame.width, box.h*frame.height)
        self.good = self.bad = 0
        self.age = 1
        self.anchors = {}
        self.anchor_matrix = np.eye(2)
        self.anchor_offset = np.zeros(2)
        self.seed_size = np.array(self.box[2:])
        self.seed_center = np.array(self.box[:2])+self.seed_size/2
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
        ids = np.arange(len(a),dtype=np.int64) if p.point_ids is None else np.asarray(p.point_ids)
        if (ids.shape != (len(a),) or not np.issubdtype(ids.dtype,np.integer)
                or len(set(ids.tolist())) != len(a) or np.any(ids < 0)):
            return self.fail("invalid backend feature identity")
        valid = (fb >= 0) & (fb <= self.gates.max_fb_px)
        indices = np.flatnonzero(valid)
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
        # Persistent identities keep geometry relative to the original trusted
        # reference, not an indefinite product of noisy adjacent-frame scales.
        # New feature anchors may be born only under a supported absolute fit.
        kept = indices[mask]
        known = np.array([int(ids[i]) in self.anchors for i in kept])
        initial = not self.anchors
        anchor_a = (a[mask] if initial else
                    np.array([self.anchors[int(ids[i])] for i in kept[known]]).reshape(-1,2))
        anchor_b = b[mask] if initial else b[mask][known]
        reference_mode = p.reference_before is not None
        reference_fb = None
        if reference_mode:
            ra,rb,rfb = (np.asarray(v) for v in (p.reference_before,p.reference_after,p.reference_fb))
            if (ra.ndim != 2 or ra.shape[1] != 2 or rb.shape != ra.shape or rfb.shape != (len(ra),)
                    or len(ra)>64 or any(not np.isfinite(v).all() for v in (ra,rb,rfb))):
                return self.fail("invalid backend reference proposal")
            valid_ref = (rfb >= 0) & (rfb <= self.gates.max_fb_px)
            anchor_a,anchor_b,reference_fb = ra[valid_ref],rb[valid_ref],rfb[valid_ref]
        absolute = robust_similarity(anchor_a,anchor_b,self.gates.max_residual_px) if len(anchor_a) >= self.gates.min_points else None
        geometry = False
        absolute_scale = residual = None
        anchor_matrix,anchor_offset = self.anchor_matrix,self.anchor_offset
        size = np.array(self.box[2:])
        if absolute is not None:
            am,ao,ok = absolute
            absolute_scale = float(np.hypot(am[0,0],am[1,0]))
            residual = float(np.percentile(np.linalg.norm(anchor_a[ok] @ am.T+ao-anchor_b[ok],axis=1),90))
            supported = int(ok.sum()) >= self.gates.min_points and ok.mean() >= self.gates.min_inlier_fraction
            spread = supported and np.all(np.ptp(anchor_a[ok],axis=0) >= self.seed_size*.2)
            # Adjacent flow already enforces rate-of-scale/rotation gates. An
            # immutable reference may legitimately differ substantially after
            # a period of uncertain geometry; do not mistake accumulated true
            # depth change for a one-frame jump.
            ratio = 1. if reference_mode else absolute_scale / float(np.hypot(self.anchor_matrix[0,0],self.anchor_matrix[1,0]))
            geometry_fb = np.median(reference_fb[ok]) if reference_mode else np.median(fb[kept])
            geometry = bool(spread and residual <= self.gates.max_anchor_residual_px
                            and geometry_fb <= self.gates.max_geometry_median_fb_px
                            and self.gates.min_scale <= ratio <= self.gates.max_scale)
            if reference_mode and spread:
                reference_center = am @ self.seed_center+ao
                reference_disagreement = float(np.linalg.norm(reference_center-dcf_center))
                if reference_disagreement <= max(3.,min(self.box[2:])*self.gates.max_disagreement_fraction):
                    flow_center = reference_center
                else:
                    geometry = False
            if geometry:
                size = self.seed_size*absolute_scale
                anchor_matrix,anchor_offset = am,ao
        if reference_mode and not geometry:
            # Do not integrate noisy adjacent flow indefinitely while its
            # absolute geometry disagrees. DCF anchors appearance position;
            # adjacent flow still has to pass ALL original position gates.
            flow_center = .75*dcf_center+.25*flow_center
        xy = flow_center - size / 2
        if np.any(size < 12) or np.any(xy < 0) or np.any(xy+size > [frame.width, frame.height]):
            return self._reject("target reaches image boundary", inliers=n, scale=scale)
        self.box = tuple(float(v) for v in (*xy, *size))
        if geometry and not reference_mode:
            # Retain original references. Added features inherit this trusted
            # transform; existing anchors are NEVER replaced every frame.
            inverse = np.linalg.inv(anchor_matrix)
            refs = {}
            for i in kept:
                key = int(ids[i])
                refs[key] = self.anchors.get(key, inverse @ (p.after[i]-anchor_offset))
            self.anchors = refs
            self.anchor_matrix,self.anchor_offset = anchor_matrix,anchor_offset
        elif geometry:
            self.anchor_matrix,self.anchor_offset = anchor_matrix,anchor_offset
        self.bad = 0
        self.good += 1
        self.state = "tracking" if self.good >= self.gates.acquire_frames else "acquiring"
        return Decision(self.state, "appearance/flow agree" if geometry else "position agrees; scale held",
                        self.box, n, scale, disagreement, True, self.state == "tracking" and geometry,
                        sum(1 << int(i) for i in kept),geometry,absolute_scale,residual)

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
