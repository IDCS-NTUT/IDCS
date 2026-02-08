"""Shared renderer helpers for camera math and transforms.

These helpers keep projection, camera basis, and orientation handling identical
between the CPU and OpenGL renderers.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


NEAR_CLIP: float = 0.05


def vector_length(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def normalise(vec: np.ndarray) -> np.ndarray:
    length = vector_length(vec)
    if length <= 1e-6:
        return vec
    return vec / length


def rotate_vector(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32)
    axis_vec = np.asarray(axis, dtype=np.float32)
    if abs(angle) <= 1e-6:
        return vec.copy()

    axis_length = vector_length(axis_vec)
    if axis_length <= 1e-6:
        return vec.copy()

    axis_norm = axis_vec / axis_length
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cross = np.cross(axis_norm, vec)
    dot = float(np.dot(axis_norm, vec))
    rotated = vec * cos_a + cross * sin_a + axis_norm * dot * (1.0 - cos_a)
    return rotated.astype(np.float32)


def parse_orientation(orientation: Any) -> Optional[Tuple[float, float, float]]:
    if isinstance(orientation, dict):
        try:
            yaw = float(orientation.get("yaw", 0.0))
            pitch = float(orientation.get("pitch", 0.0))
            roll = float(orientation.get("roll", 0.0))
        except (TypeError, ValueError):
            return None
        return yaw, pitch, roll

    try:
        values = np.asarray(orientation, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None

    if values.size < 2:
        return None

    yaw = float(values[0])
    pitch = float(values[1])
    roll = float(values[2]) if values.size >= 3 else 0.0
    return yaw, pitch, roll


def camera_basis_from_orientation(
    orientation: Any,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    parsed = parse_orientation(orientation)
    if parsed is None:
        return None
    yaw_deg, pitch_deg, roll_deg = parsed

    if not (
        math.isfinite(yaw_deg)
        and math.isfinite(pitch_deg)
        and math.isfinite(roll_deg)
    ):
        return None

    pitch_deg = max(-89.9, min(89.9, pitch_deg))

    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)
    roll_rad = math.radians(roll_deg)

    forward = np.array((0.0, 0.0, -1.0), dtype=np.float32)
    up = np.array((0.0, 1.0, 0.0), dtype=np.float32)
    right = np.array((1.0, 0.0, 0.0), dtype=np.float32)

    if abs(yaw_rad) > 1e-6:
        yaw_axis = np.array((0.0, 1.0, 0.0), dtype=np.float32)
        forward = rotate_vector(forward, yaw_axis, yaw_rad)
        right = rotate_vector(right, yaw_axis, yaw_rad)
        up = rotate_vector(up, yaw_axis, yaw_rad)

    if abs(pitch_rad) > 1e-6:
        pitch_axis = right
        forward = rotate_vector(forward, pitch_axis, pitch_rad)
        up = rotate_vector(up, pitch_axis, pitch_rad)

    if abs(roll_rad) > 1e-6:
        roll_axis = forward
        right = rotate_vector(right, roll_axis, roll_rad)
        up = rotate_vector(up, roll_axis, roll_rad)

    forward = normalise(forward)
    if vector_length(forward) < 1e-6:
        return None

    up = normalise(up)
    if vector_length(up) < 1e-6:
        up = np.array((0.0, 1.0, 0.0), dtype=np.float32)

    right = normalise(np.cross(up, forward))
    if vector_length(right) < 1e-6:
        right = np.cross(forward, np.array((0.0, 1.0, 0.0), dtype=np.float32))
        if vector_length(right) < 1e-6:
            right = np.cross(forward, np.array((1.0, 0.0, 0.0), dtype=np.float32))
            if vector_length(right) < 1e-6:
                return None
    right = normalise(right)
    true_up = normalise(np.cross(forward, right))
    if vector_length(true_up) < 1e-6:
        return None

    return forward, right, true_up


def build_camera(
    camera_state: Dict[str, Any],
    *,
    context: Any,
    width: int,
    height: int,
) -> Optional[Dict[str, Any]]:
    try:
        position = np.asarray(camera_state["position"], dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return None

    orientation = camera_state.get("orientation")
    if orientation is not None:
        basis = camera_basis_from_orientation(orientation)
        if basis is None:
            return None
        forward, right, true_up = basis
    else:
        try:
            target = np.asarray(camera_state["target"], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            return None

        up = camera_state.get("up")
        if up is None:
            up = getattr(context, "world_up", (0.0, 1.0, 0.0))
        up_vec = normalise(np.asarray(up, dtype=np.float32))
        forward = normalise(target - position)
        if vector_length(forward) < 1e-6:
            return None

        right = np.cross(up_vec, forward)  # up กั forward ก๗ +X
        if vector_length(right) < 1e-6:
            right = np.cross(np.array((0.0, 1.0, 0.0), np.float32), forward)
            if vector_length(right) < 1e-6:
                right = np.cross(np.array((1.0, 0.0, 0.0), np.float32), forward)
        right = normalise(right)
        true_up = normalise(np.cross(forward, right))  # F กั R ก๗ U

    try:
        fov_y = float(camera_state.get("fov_y", 60.0))
    except (TypeError, ValueError):
        fov_y = 60.0

    return {
        "position": position,
        "forward": forward,
        "right": right,
        "up": true_up,
        "fov_y": fov_y,
        "aspect": float(width) / float(height),
    }


def projection_matrix(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    nf = 1.0 / (near - far)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) * nf
    m[2, 3] = (2.0 * far * near) * nf
    m[3, 2] = -1.0
    return m


def view_matrix(eye: Sequence[float], forward: Sequence[float], up: Sequence[float]) -> np.ndarray:
    f = np.asarray(forward, dtype=np.float32)
    f = f / (np.linalg.norm(f) + 1e-12)
    u = np.asarray(up, dtype=np.float32)
    u = u / (np.linalg.norm(u) + 1e-12)
    s = np.cross(u, f)
    s = s / (np.linalg.norm(s) + 1e-12)
    u2 = np.cross(f, s)

    m = np.eye(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u2
    m[2, 0:3] = -f
    t = np.eye(4, dtype=np.float32)
    t[0:3, 3] = -np.asarray(eye, dtype=np.float32)
    return m @ t


__all__ = [
    "NEAR_CLIP",
    "build_camera",
    "camera_basis_from_orientation",
    "parse_orientation",
    "projection_matrix",
    "view_matrix",
    "rotate_vector",
    "normalise",
    "vector_length",
]
