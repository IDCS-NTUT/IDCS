"""Shared camera math utilities for CPU and OpenGL renderers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "CameraDescription",
    "build_camera_description",
    "project_point",
]


@dataclass(frozen=True)
class CameraDescription:
    """Fully populated camera parameters and derived matrices."""

    image_width: int
    image_height: int
    position: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    fov_y: float
    aspect: float
    near_clip: float
    far_clip: float
    view_matrix: np.ndarray
    projection_matrix: np.ndarray
    view_projection_matrix: np.ndarray
    gl_view_matrix: np.ndarray
    gl_projection_matrix: np.ndarray
    gl_view_projection_matrix: np.ndarray

    def as_mapping(self) -> Mapping[str, Any]:
        """Expose the description as a mapping compatible with legacy callers."""

        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "position": self.position,
            "forward": self.forward,
            "right": self.right,
            "up": self.up,
            "fov_y": self.fov_y,
            "aspect": self.aspect,
            "near_clip": self.near_clip,
            "far_clip": self.far_clip,
            "view_matrix": self.view_matrix,
            "projection_matrix": self.projection_matrix,
            "view_projection_matrix": self.view_projection_matrix,
            "gl_view_matrix": self.gl_view_matrix,
            "gl_projection_matrix": self.gl_projection_matrix,
            "gl_view_projection_matrix": self.gl_view_projection_matrix,
        }


def build_camera_description(
    camera_state: Mapping[str, Any],
    image_width: int,
    image_height: int,
    *,
    default_up: Iterable[float] = (0.0, 1.0, 0.0),
    near_clip: float = 0.05,
    fallback_far_clip: float = 500.0,
) -> Optional[CameraDescription]:
    """Normalise camera parameters and derive view/projection matrices."""

    try:
        position = np.asarray(camera_state["position"], dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return None

    if position.shape != (3,):
        try:
            position = position.reshape(3)
        except Exception:  # pragma: no cover - defensive path
            return None

    orientation = camera_state.get("orientation")
    if orientation is not None:
        basis = _basis_from_orientation(orientation)
        if basis is None:
            return None
        forward, right, true_up = basis
    else:
        try:
            target = np.asarray(camera_state["target"], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            return None

        up_raw = camera_state.get("up", default_up)
        up_vec = _normalise(np.asarray(up_raw, dtype=np.float32))
        forward = _normalise(target - position)
        if _vector_length(forward) < 1e-6:
            return None

        right = np.cross(up_vec, forward)
        if _vector_length(right) < 1e-6:
            right = np.cross(np.array((0.0, 1.0, 0.0), dtype=np.float32), forward)
            if _vector_length(right) < 1e-6:
                right = np.cross(np.array((1.0, 0.0, 0.0), dtype=np.float32), forward)
                if _vector_length(right) < 1e-6:
                    return None
        right = _normalise(right)
        true_up = _normalise(np.cross(forward, right))
        if _vector_length(true_up) < 1e-6:
            return None

    try:
        fov_y = float(camera_state.get("fov_y", 60.0))
    except (TypeError, ValueError):
        fov_y = 60.0

    aspect = float(image_width) / float(image_height)

    try:
        near = float(near_clip)
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        near = 0.05
    far_raw = camera_state.get("far_clip")
    if far_raw is None:
        far = float(fallback_far_clip)
    else:
        try:
            far = float(far_raw)
        except (TypeError, ValueError):
            far = float(fallback_far_clip)

    if not math.isfinite(fov_y) or fov_y <= 1e-3:
        fov_y = 60.0

    forward = _normalise(forward)
    right = _normalise(right)
    true_up = _normalise(true_up)

    view_matrix = _build_view_matrix(position, right, true_up, forward)
    projection_matrix = _build_perspective_matrix(fov_y, aspect, near, far)
    view_projection_matrix = projection_matrix @ view_matrix

    gl_view_matrix = _build_gl_view_matrix(position, right, true_up, forward)
    gl_projection_matrix = _build_gl_perspective_matrix(fov_y, aspect, near, far)
    gl_view_projection_matrix = gl_projection_matrix @ gl_view_matrix

    return CameraDescription(
        image_width=int(image_width),
        image_height=int(image_height),
        position=position.astype(np.float32),
        forward=forward,
        right=right,
        up=true_up,
        fov_y=float(fov_y),
        aspect=float(aspect),
        near_clip=float(near),
        far_clip=float(far),
        view_matrix=view_matrix,
        projection_matrix=projection_matrix,
        view_projection_matrix=view_projection_matrix,
        gl_view_matrix=gl_view_matrix,
        gl_projection_matrix=gl_projection_matrix,
        gl_view_projection_matrix=gl_view_projection_matrix,
    )


def project_point(camera: CameraDescription, point: Sequence[float]) -> Optional[Tuple[float, float]]:
    """Project a world-space point into image coordinates."""

    coords = np.asarray(point, dtype=np.float32)
    if coords.shape != (3,):
        try:
            coords = coords.reshape(3)
        except Exception:  # pragma: no cover - defensive
            return None

    rel = coords - camera.position
    camera_space = np.array(
        [
            float(np.dot(rel, camera.right)),
            float(np.dot(rel, camera.up)),
            float(np.dot(rel, camera.forward)),
        ],
        dtype=np.float32,
    )
    if camera_space[2] < camera.near_clip:
        return None

    f = 1.0 / math.tan(math.radians(camera.fov_y) * 0.5)
    x_ndc = (camera_space[0] / camera_space[2]) * (f / camera.aspect)
    y_ndc = (camera_space[1] / camera_space[2]) * f

    if not math.isfinite(x_ndc) or not math.isfinite(y_ndc):
        return None

    x_px = (x_ndc + 1.0) * 0.5 * (camera.image_width - 1)
    y_px = (1.0 - (y_ndc + 1.0) * 0.5) * (camera.image_height - 1)
    return float(x_px), float(y_px)


def _basis_from_orientation(
    orientation: Any,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    yaw_pitch_roll = _parse_orientation(orientation)
    if yaw_pitch_roll is None:
        return None
    yaw_deg, pitch_deg, roll_deg = yaw_pitch_roll

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
        forward = _rotate_vector(forward, yaw_axis, yaw_rad)
        right = _rotate_vector(right, yaw_axis, yaw_rad)
        up = _rotate_vector(up, yaw_axis, yaw_rad)

    if abs(pitch_rad) > 1e-6:
        pitch_axis = right
        forward = _rotate_vector(forward, pitch_axis, pitch_rad)
        up = _rotate_vector(up, pitch_axis, pitch_rad)

    if abs(roll_rad) > 1e-6:
        roll_axis = forward
        right = _rotate_vector(right, roll_axis, roll_rad)
        up = _rotate_vector(up, roll_axis, roll_rad)

    forward = _normalise(forward)
    if _vector_length(forward) < 1e-6:
        return None

    up = _normalise(up)
    if _vector_length(up) < 1e-6:
        up = np.array((0.0, 1.0, 0.0), dtype=np.float32)

    right = _normalise(np.cross(up, forward))
    if _vector_length(right) < 1e-6:
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


def _parse_orientation(orientation: Any) -> Optional[Tuple[float, float, float]]:
    if isinstance(orientation, Mapping):
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


def _rotate_vector(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
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


def _normalise(vector: np.ndarray) -> np.ndarray:
    length = _vector_length(vector)
    if length <= 1e-6:
        return vector.astype(np.float32)
    return (vector / length).astype(np.float32)


def _vector_length(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector.astype(np.float32)))


def _build_view_matrix(
    position: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
) -> np.ndarray:
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = forward
    translation = np.array(
        [
            -float(np.dot(right, position)),
            -float(np.dot(up, position)),
            -float(np.dot(forward, position)),
        ],
        dtype=np.float32,
    )
    view[:3, 3] = translation
    return view


def _build_gl_view_matrix(
    position: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
) -> np.ndarray:
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = -forward
    translation = np.array(
        [
            -float(np.dot(right, position)),
            -float(np.dot(up, position)),
            float(np.dot(forward, position)),
        ],
        dtype=np.float32,
    )
    view[:3, 3] = translation
    return view


def _build_perspective_matrix(
    fov_y: float,
    aspect: float,
    near: float,
    far: float,
) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y) * 0.5)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (far - near)
    proj[2, 3] = (-2.0 * far * near) / (far - near)
    proj[3, 2] = 1.0
    return proj


def _build_gl_perspective_matrix(
    fov_y: float,
    aspect: float,
    near: float,
    far: float,
) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y) * 0.5)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj
