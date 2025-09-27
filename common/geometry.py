"""Geometric helpers for projecting between image pixels and camera-frame rays."""

from __future__ import annotations

import math
from typing import Optional, Tuple

Vector3 = Tuple[float, float, float]


def pixel_to_camera_ray(
    u_px: float,
    v_px: float,
    *,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
) -> Vector3:
    """Return a unit ray in the camera frame that passes through ``(u, v)``.

    The camera frame follows the conventional computer-vision axes with
    ``+X`` pointing right, ``+Y`` pointing down, and ``+Z`` pointing forward.
    """

    x = (u_px - cx_px) / fx_px
    y = (v_px - cy_px) / fy_px
    z = 1.0

    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("pixel_to_camera_ray received a zero-length direction")

    return (x / norm, y / norm, z / norm)


def project_point_to_pixel(
    point: Vector3,
    *,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
) -> Tuple[float, float]:
    """Project a 3D point in the camera frame onto the image plane."""

    x, y, z = point
    if z <= 0.0:
        raise ValueError("project_point_to_pixel requires point.z > 0")

    u = fx_px * (x / z) + cx_px
    v = fy_px * (y / z) + cy_px
    return (u, v)


def intersect_ray_with_depth(
    origin: Vector3,
    direction: Vector3,
    depth_m: float,
) -> Optional[Vector3]:
    """Return the point where a ray intersects the ``z = depth_m`` plane."""

    if depth_m <= 0.0:
        raise ValueError("depth_m must be positive")

    ox, oy, oz = origin
    dx, dy, dz = direction

    if dz == 0.0:
        return None

    t = (depth_m - oz) / dz
    if t < 0.0:
        return None

    return (ox + dx * t, oy + dy * t, oz + dz * t)


def laser_ray_to_pixel(
    origin: Vector3,
    direction: Vector3,
    *,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
    depth_m: float,
) -> Optional[Tuple[float, float]]:
    """Project the laser ray onto the image plane at a given target depth."""

    hit_point = intersect_ray_with_depth(origin, direction, depth_m)
    if hit_point is None:
        return None

    return project_point_to_pixel(hit_point, fx_px=fx_px, fy_px=fy_px, cx_px=cx_px, cy_px=cy_px)

