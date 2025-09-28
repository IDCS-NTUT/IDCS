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


def pixel_motion_from_camera_rotation(
    u_px: float,
    v_px: float,
    *,
    yaw_rate_rad_s: float,
    pitch_rate_rad_s: float,
    dt_s: float,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
) -> Tuple[float, float]:
    """Return the pixel shift caused by camera yaw/pitch rotation.

    The helper assumes a pinhole camera with focal lengths ``fx_px``/``fy_px``
    and principal point ``(cx_px, cy_px)``. Positive yaw rotates the camera to
    the right (clockwise when looking down), which makes a static target appear
    to move left on the image plane. Positive pitch rotates the camera upwards,
    causing features to drift downward.
    """

    if dt_s == 0.0 or (yaw_rate_rad_s == 0.0 and pitch_rate_rad_s == 0.0):
        return (0.0, 0.0)

    yaw_delta = float(yaw_rate_rad_s) * float(dt_s)
    pitch_delta = float(pitch_rate_rad_s) * float(dt_s)

    # Convert the pixel back to a unit ray in camera coordinates.
    ray = pixel_to_camera_ray(u_px, v_px, fx_px=fx_px, fy_px=fy_px, cx_px=cx_px, cy_px=cy_px)

    # Apply the inverse of the camera rotation to the ray so that we model how
    # a static world point appears to move on the image plane.
    cos_y = math.cos(-yaw_delta)
    sin_y = math.sin(-yaw_delta)
    x1 = cos_y * ray[0] + sin_y * ray[2]
    y1 = ray[1]
    z1 = -sin_y * ray[0] + cos_y * ray[2]

    cos_x = math.cos(-pitch_delta)
    sin_x = math.sin(-pitch_delta)
    x2 = x1
    y2 = cos_x * y1 - sin_x * z1
    z2 = sin_x * y1 + cos_x * z1

    if z2 <= 0.0:
        # The camera rotated past 90° which should not happen for the small
        # intervals we use. Fall back to no shift to avoid exploding values.
        return (0.0, 0.0)

    new_u = fx_px * (x2 / z2) + cx_px
    new_v = fy_px * (y2 / z2) + cy_px

    return (new_u - u_px, new_v - v_px)


def camera_vector_to_world(
    vector: Vector3,
    *,
    pan_rad: float,
    tilt_rad: float,
) -> Vector3:
    """Rotate a camera-frame vector into the world frame."""

    x, y, z = vector
    # Convert camera ``+Y`` down to world ``+Y`` up before applying rotations.
    base_y = -y

    # Pitch about the camera's X axis (negative tilt lifts the camera up).
    cos_pitch = math.cos(-tilt_rad)
    sin_pitch = math.sin(-tilt_rad)
    pitch_x = x
    pitch_y = cos_pitch * base_y - sin_pitch * z
    pitch_z = sin_pitch * base_y + cos_pitch * z

    # Yaw about the world up axis.
    cos_yaw = math.cos(pan_rad)
    sin_yaw = math.sin(pan_rad)
    world_x = cos_yaw * pitch_x + sin_yaw * pitch_z
    world_y = pitch_y
    world_z = -sin_yaw * pitch_x + cos_yaw * pitch_z
    return (world_x, world_y, world_z)


def world_vector_to_camera(
    vector: Vector3,
    *,
    pan_rad: float,
    tilt_rad: float,
) -> Vector3:
    """Rotate a world-frame vector into the camera frame."""

    x, y, z = vector

    cos_yaw = math.cos(-pan_rad)
    sin_yaw = math.sin(-pan_rad)
    yaw_x = cos_yaw * x + sin_yaw * z
    yaw_y = y
    yaw_z = -sin_yaw * x + cos_yaw * z

    cos_pitch = math.cos(tilt_rad)
    sin_pitch = math.sin(tilt_rad)
    pitch_x = yaw_x
    pitch_y = cos_pitch * yaw_y - sin_pitch * yaw_z
    pitch_z = sin_pitch * yaw_y + cos_pitch * yaw_z

    cam_y = -pitch_y
    return (pitch_x, cam_y, pitch_z)


def pixel_to_world_point(
    u_px: float,
    v_px: float,
    *,
    distance_m: float,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
    pan_rad: float,
    tilt_rad: float,
) -> Vector3:
    """Convert an image pixel and range into a 3D world coordinate."""

    ray = pixel_to_camera_ray(
        u_px,
        v_px,
        fx_px=fx_px,
        fy_px=fy_px,
        cx_px=cx_px,
        cy_px=cy_px,
    )
    point_cam = (ray[0] * distance_m, ray[1] * distance_m, ray[2] * distance_m)
    return camera_vector_to_world(point_cam, pan_rad=pan_rad, tilt_rad=tilt_rad)


def project_world_point_to_pixel(
    point_world: Vector3,
    *,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
    pan_rad: float,
    tilt_rad: float,
) -> Tuple[float, float]:
    """Project a world-frame point into pixel coordinates."""

    point_cam = world_vector_to_camera(
        point_world,
        pan_rad=pan_rad,
        tilt_rad=tilt_rad,
    )
    return project_point_to_pixel(
        point_cam,
        fx_px=fx_px,
        fy_px=fy_px,
        cx_px=cx_px,
        cy_px=cy_px,
    )


def world_velocity_to_pixel_velocity(
    position_world: Vector3,
    velocity_world: Vector3,
    *,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
    pan_rad: float,
    tilt_rad: float,
) -> Tuple[float, float]:
    """Project world-frame velocity into instantaneous pixel velocity."""

    pos_cam = world_vector_to_camera(
        position_world,
        pan_rad=pan_rad,
        tilt_rad=tilt_rad,
    )
    vel_cam = world_vector_to_camera(
        velocity_world,
        pan_rad=pan_rad,
        tilt_rad=tilt_rad,
    )

    x, y, z = pos_cam
    vx, vy, vz = vel_cam
    if z <= 0.0:
        raise ValueError("world_velocity_to_pixel_velocity requires positive depth")

    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z
    du = fx_px * ((vx * z - x * vz) * inv_z2)
    dv = fy_px * ((vy * z - y * vz) * inv_z2)
    return (du, dv)

