"""Geometry helpers shared between CPU renderer components."""

from __future__ import annotations

from typing import Optional, Tuple

Point = Tuple[float, float]

def clip_segment_to_rect(
    start: Point,
    end: Point,
    width: int,
    height: int,
) -> Optional[Tuple[Point, Point]]:
    """Clip a line segment to the pixel rectangle using Liang–Barsky.

    Inputs are in image pixel coordinates where (0, 0) is the top-left corner.
    The rectangle bounds are inclusive: x in [0, width-1], y in [0, height-1].
    Returns ``None`` when the segment does not intersect the rectangle or when
    width/height are non-positive.

    Notes: we use an epsilon for parallel checks and clamp the computed
    parameters to [0, 1], then clamp final points to the rectangle to improve
    numeric stability near edges.
    """
    if width <= 0 or height <= 0:
        return None

    x_min, y_min = 0.0, 0.0
    x_max, y_max = float(width - 1), float(height - 1)

    dx = end[0] - start[0]
    dy = end[1] - start[1]

    p = (-dx,  dx, -dy,  dy)
    q = (start[0] - x_min, x_max - start[0], start[1] - y_min, y_max - start[1])

    u1, u2 = 0.0, 1.0
    eps = 1e-12

    for pi, qi in zip(p, q):
        if abs(pi) < eps:              # segment parallel to this boundary
            if qi < 0.0:               # outside & parallel -> reject
                return None
            continue

        t = qi / pi
        if pi < 0.0:                    # entering
            if t > u2:
                return None
            if t > u1:
                u1 = t
        else:                           # leaving
            if t < u1:
                return None
            if t < u2:
                u2 = t

    if u2 < u1:
        return None

    # Clamp to [0,1] for numeric safety
    u1 = max(0.0, min(1.0, u1))
    u2 = max(0.0, min(1.0, u2))

    x0 = start[0] + u1 * dx
    y0 = start[1] + u1 * dy
    x1 = start[0] + u2 * dx
    y1 = start[1] + u2 * dy

    # Final clamp to the rectangle bounds (handles roundoff at edges)
    x0 = min(max(x0, x_min), x_max)
    y0 = min(max(y0, y_min), y_max)
    x1 = min(max(x1, x_min), x_max)
    y1 = min(max(y1, y_min), y_max)

    return (x0, y0), (x1, y1)



__all__ = ["Point", "clip_segment_to_rect"]
