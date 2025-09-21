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


    if width <= 0 or height <= 0:
        return None

    x_min = 0.0
    y_min = 0.0
    x_max = float(width - 1)
    y_max = float(height - 1)

    dx = end[0] - start[0]
    dy = end[1] - start[1]

    p = (-dx, dx, -dy, dy)
    q = (start[0] - x_min, x_max - start[0], start[1] - y_min, y_max - start[1])

    u1 = 0.0
    u2 = 1.0

    for pi, qi in zip(p, q):
        if pi == 0.0:
            if qi < 0.0:
                return None
            continue

        t = qi / pi

        if pi < 0.0:
            if t > u2:
                return None
            if t > u1:
                u1 = t
        else:
            if t < u1:
                return None
            if t < u2:
                u2 = t

    clipped_start: Point = (start[0] + u1 * dx, start[1] + u1 * dy)
    clipped_end: Point = (start[0] + u2 * dx, start[1] + u2 * dy)

    return clipped_start, clipped_end


__all__ = ["Point", "clip_segment_to_rect"]