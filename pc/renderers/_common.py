"""Shared camera and projection logic for renderers.

This module provides common camera math functions used by both CPU and OpenGL
renderers to ensure consistent FOV, aspect ratio, and orientation across
different rendering backends.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np


# Default near clipping plane distance in world units
DEFAULT_NEAR_CLIP = 0.05


def _vector_length(vec: np.ndarray) -> float:
    """Compute the length of a vector."""
    return float(np.linalg.norm(vec))


def _normalise(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length.
    
    Returns the original vector if length is too small.
    """
    length = _vector_length(vec)
    if length <= 1e-6:
        return vec
    return vec / length


def _rotate_vector(
    vector: np.ndarray, axis: np.ndarray, angle: float
) -> np.ndarray:
    """Rotate a vector around an axis by an angle (Rodrigues' rotation formula).
    
    Parameters
    ----------
    vector : np.ndarray
        The vector to rotate.
    axis : np.ndarray
        The axis of rotation (will be normalized).
    angle : float
        Rotation angle in radians.
    
    Returns
    -------
    np.ndarray
        The rotated vector.
    """
    vec = np.asarray(vector, dtype=np.float32)
    axis_vec = np.asarray(axis, dtype=np.float32)
    if abs(angle) <= 1e-6:
        return vec.copy()

    axis_length = _vector_length(axis_vec)
    if axis_length <= 1e-6:
        return vec.copy()

    axis_norm = axis_vec / axis_length
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cross = np.cross(axis_norm, vec)
    dot = float(np.dot(axis_norm, vec))
    rotated = vec * cos_a + cross * sin_a + axis_norm * dot * (1.0 - cos_a)
    return rotated.astype(np.float32)


def _parse_orientation(orientation: Any) -> Optional[Tuple[float, float, float]]:
    """Parse orientation from dict or array-like to (yaw, pitch, roll) in degrees."""
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


def _camera_basis_from_orientation(
    orientation: Any
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute camera basis vectors (forward, right, up) from orientation.
    
    Parameters
    ----------
    orientation : Any
        Orientation as dict with yaw/pitch/roll or array-like [yaw, pitch, roll].
    
    Returns
    -------
    Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        (forward, right, up) basis vectors or None if invalid.
    """
    parsed = _parse_orientation(orientation)
    if parsed is None:
        return None
    yaw_deg, pitch_deg, roll_deg = parsed

    if not (
        math.isfinite(yaw_deg)
        and math.isfinite(pitch_deg)
        and math.isfinite(roll_deg)
    ):
        return None

    # Clamp pitch to avoid singularities
    pitch_deg = max(-89.9, min(89.9, pitch_deg))

    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)
    roll_rad = math.radians(roll_deg)

    # Start with default basis
    forward = np.array((0.0, 0.0, -1.0), dtype=np.float32)
    up = np.array((0.0, 1.0, 0.0), dtype=np.float32)
    right = np.array((1.0, 0.0, 0.0), dtype=np.float32)

    # Apply yaw rotation around Y axis
    if abs(yaw_rad) > 1e-6:
        yaw_axis = np.array((0.0, 1.0, 0.0), dtype=np.float32)
        forward = _rotate_vector(forward, yaw_axis, yaw_rad)
        right = _rotate_vector(right, yaw_axis, yaw_rad)
        up = _rotate_vector(up, yaw_axis, yaw_rad)

    # Apply pitch rotation around right axis
    if abs(pitch_rad) > 1e-6:
        pitch_axis = right
        forward = _rotate_vector(forward, pitch_axis, pitch_rad)
        up = _rotate_vector(up, pitch_axis, pitch_rad)

    # Apply roll rotation around forward axis
    if abs(roll_rad) > 1e-6:
        roll_axis = forward
        right = _rotate_vector(right, roll_axis, roll_rad)
        up = _rotate_vector(up, roll_axis, roll_rad)

    # Normalize and validate
    forward = _normalise(forward)
    if _vector_length(forward) < 1e-6:
        return None

    up = _normalise(up)
    if _vector_length(up) < 1e-6:
        up = np.array((0.0, 1.0, 0.0), dtype=np.float32)

    # Recompute right and up to ensure orthonormal basis
    right = _normalise(np.cross(up, forward))
    if _vector_length(right) < 1e-6:
        # Fallback if up and forward are parallel
        right = np.cross(forward, np.array((0.0, 1.0, 0.0), dtype=np.float32))
        if _vector_length(right) < 1e-6:
            right = np.cross(forward, np.array((1.0, 0.0, 0.0), dtype=np.float32))
            if _vector_length(right) < 1e-6:
                return None
    right = _normalise(right)
    true_up = _normalise(np.cross(forward, right))
    if _vector_length(true_up) < 1e-6:
        return None

    return forward, right, true_up


def _build_camera(
    camera_state: Dict[str, Any],
    width: int,
    height: int,
    context: Any = None,
) -> Optional[Dict[str, Any]]:
    """Build camera parameters from camera state dict.
    
    Parameters
    ----------
    camera_state : Dict[str, Any]
        Camera state with 'position', optional 'orientation' or 'target', etc.
    width : int
        Viewport width in pixels.
    height : int
        Viewport height in pixels.
    context : Any, optional
        Context object that may provide world_up attribute.
    
    Returns
    -------
    Optional[Dict[str, Any]]
        Camera dict with position, forward, right, up, fov_y, aspect or None.
    """
    try:
        position = np.asarray(camera_state["position"], dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return None

    orientation = camera_state.get("orientation")
    if orientation is not None:
        # Use orientation-based camera
        basis = _camera_basis_from_orientation(orientation)
        if basis is None:
            return None
        forward, right, true_up = basis
    else:
        # Use target-based camera
        try:
            target = np.asarray(camera_state["target"], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            return None

        up = camera_state.get("up")
        if up is None and context is not None:
            up = getattr(context, "world_up", (0.0, 1.0, 0.0))
        if up is None:
            up = (0.0, 1.0, 0.0)
        up_vec = _normalise(np.asarray(up, dtype=np.float32))
        forward = _normalise(target - position)
        if _vector_length(forward) < 1e-6:
            return None

        # Compute right and up vectors
        right = np.cross(up_vec, forward)  # up × forward → +X
        if _vector_length(right) < 1e-6:
            # Fallback axes if up≈forward
            right = np.cross(np.array((0.0, 1.0, 0.0), np.float32), forward)
            if _vector_length(right) < 1e-6:
                right = np.cross(np.array((1.0, 0.0, 0.0), np.float32), forward)
        right = _normalise(right)
        true_up = _normalise(np.cross(forward, right))  # F × R → U

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


def _projection_matrix(
    fov_y_deg: float, aspect: float, near: float, far: float
) -> np.ndarray:
    """Build a perspective projection matrix.
    
    Parameters
    ----------
    fov_y_deg : float
        Vertical field of view in degrees.
    aspect : float
        Aspect ratio (width / height).
    near : float
        Near clipping plane distance.
    far : float
        Far clipping plane distance.
    
    Returns
    -------
    np.ndarray
        4x4 projection matrix (float32).
    """
    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    nf = 1.0 / (near - far)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) * nf
    m[2, 3] = (2.0 * far * near) * nf
    m[3, 2] = -1.0
    return m


def _view_matrix(
    eye: np.ndarray, forward: np.ndarray, up: np.ndarray
) -> np.ndarray:
    """Build a view matrix from camera position and orientation.
    
    Parameters
    ----------
    eye : np.ndarray
        Camera position in world space.
    forward : np.ndarray
        Forward direction vector (should be normalized).
    up : np.ndarray
        Up direction vector (should be normalized).
    
    Returns
    -------
    np.ndarray
        4x4 view matrix (float32).
    """
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


def _near_clip() -> float:
    """Return the default near clipping plane distance."""
    return DEFAULT_NEAR_CLIP


__all__ = [
    "DEFAULT_NEAR_CLIP",
    "_vector_length",
    "_normalise",
    "_rotate_vector",
    "_parse_orientation",
    "_camera_basis_from_orientation",
    "_build_camera",
    "_projection_matrix",
    "_view_matrix",
    "_near_clip",
]
