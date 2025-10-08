"""Geometric helpers for projecting between image pixels and camera-frame rays."""

from __future__ import annotations

import math
from typing import Optional, Tuple

Vector3 = Tuple[float, float, float]
Matrix3 = Tuple[Vector3, Vector3, Vector3]


def _matrix_multiply(a: Matrix3, b: Matrix3) -> Matrix3:
    """Return ``a * b`` for 3×3 matrices."""

    return (
        (
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ),
        (
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ),
        (
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ),
    )


def matrix_vector_mul(matrix: Matrix3, vector: Vector3) -> Vector3:
    """Multiply a 3×3 matrix by a 3D vector."""

    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def rotation_matrix_yaw_pitch(yaw: float, pitch: float) -> Matrix3:
    """Return a rotation matrix applying yaw then pitch (right-handed frame)."""

    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)

    yaw_matrix: Matrix3 = (
        (cy, 0.0, sy),
        (0.0, 1.0, 0.0),
        (-sy, 0.0, cy),
    )
    pitch_matrix: Matrix3 = (
        (1.0, 0.0, 0.0),
        (0.0, cp, -sp),
        (0.0, sp, cp),
    )

    return _matrix_multiply(yaw_matrix, pitch_matrix)


def camera_cv_to_world_matrix(yaw: float, pitch: float) -> Matrix3:
    """Return a matrix that converts CV-frame camera vectors into world space."""

    # Convert CV frame (+Y down) to a camera body frame (+Y up) before applying
    # gimbal rotations so consumers can work in a right-handed world frame.
    cv_to_body: Matrix3 = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
    body_to_world = rotation_matrix_yaw_pitch(yaw, pitch)
    return _matrix_multiply(body_to_world, cv_to_body)


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

