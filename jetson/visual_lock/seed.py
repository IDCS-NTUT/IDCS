"""Seed-only geometry from native Harris evidence; no image/truth heuristics."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SeedRefinement:
    operator_box: tuple
    refined_box: tuple | None
    reason: str
    corners: list
    scores: list


def refine_seed(box, corners, scores, *, min_points=8):
    """Refuse ambiguous/weak evidence; pad one connected support, never grow.

    A second component with >=25% of the main support (and >=min_points)
    is ambiguous. No centre preference: a target may touch any box edge.
    Only clearly loose axes (support <=65%) tighten, preserving tight seeds.
    """
    box = tuple(float(v) for v in box)
    p, s = np.asarray(corners, dtype=float), np.asarray(scores, dtype=float)
    if (len(box) != 4 or not np.isfinite(box).all() or min(box[2:]) < 12
            or min(box[:2]) < 0):
        raise ValueError("invalid operator box")
    if p.size == 0: p = p.reshape(0, 2)
    if p.ndim != 2 or p.shape[1] != 2 or s.shape != (len(p),) or len(p) > 20000:
        raise ValueError("invalid seed evidence")
    valid = (np.isfinite(p).all(axis=1) & np.isfinite(s) & (s > 0)
             & (p[:, 0] >= box[0]+3) & (p[:, 0] < box[0]+box[2]-3)
             & (p[:, 1] >= box[1]+3) & (p[:, 1] < box[1]+box[3]-3))
    p, s = p[valid], s[valid]
    order = np.lexsort((p[:, 0], p[:, 1], -s))
    p, s = p[order], s[order]
    def result(refined, reason):
        return SeedRefinement(box, refined, reason, p.tolist(), s.tolist())
    if len(p) < min_points:
        return result(None, "seed lacks texture")
    # Spatial hash bounds comparisons for dense evidence; deterministic traversal.
    radius = max(12., min(box[2:])*.18)
    cells = {}
    for i, point in enumerate(p):
        cell = tuple(np.floor(point/radius).astype(int))
        cells.setdefault(cell, []).append(i)
    unseen = set(range(len(p)))
    groups = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        group, queue = [], [first]
        while queue:
            i = queue.pop()
            group.append(i)
            cx, cy = np.floor(p[i]/radius).astype(int)
            for x in range(cx-1, cx+2):
                for y in range(cy-1, cy+2):
                    for j in cells.get((x, y), ()):
                        if j in unseen and np.linalg.norm(p[i]-p[j]) <= radius:
                            unseen.remove(j)
                            queue.append(j)
        groups.append(group)
    groups.sort(key=lambda g: (-len(g), min(g)))
    main = groups[0]
    if len(main) < min_points or len(main) < .75*len(p):
        return result(None, "seed has weak or divided support")
    if len(groups) > 1 and len(groups[1]) >= max(min_points, .25*len(main)):
        return result(None, "seed has competing textured regions")
    support = p[main]
    extent = np.ptp(support, axis=0)
    if np.any(extent < 6):
        return result(None, "seed support is too compact")
    refined = list(box)
    for axis in (0, 1):
        if extent[axis] > .65*box[axis+2]: continue
        margin = min(12., max(4., .15*extent[axis]))
        low = max(box[axis], float(support[:, axis].min())-margin)
        high = min(box[axis]+box[axis+2], float(support[:, axis].max())+margin)
        size = max(12., high-low)
        low = max(box[axis], min(low, box[axis]+box[axis+2]-size))
        refined[axis], refined[axis+2] = low, size
    return result(tuple(refined), "refined" if tuple(refined) != box else "already tight")


class SeedDiagnostics:
    """Fixed 64-slot seed-only ring; never prunes unrelated artifacts."""
    def __init__(self, output):
        self.output = Path(output)/"seeds"
        self.sequence = 0

    def write(self, report, gray, frame, decision):
        self.output.mkdir(parents=True, exist_ok=True)
        slot = self.output/f"seed-{self.sequence % 64:02d}"
        self.sequence += 1
        h, w = gray.shape
        slot.with_suffix(".pgm").write_bytes(f"P5\n{w} {h}\n255\n".encode()+gray.tobytes())
        data = dict(sequence=self.sequence, **asdict(report), frame=frame.model_dump(mode="json"),
                    decision=asdict(decision),
                    crop_origin=[int(np.float32(v)) for v in report.operator_box[:2]])
        slot.with_suffix(".json").write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
