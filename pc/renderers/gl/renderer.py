"""Core OpenGL renderer scaffolding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .context import ContextConfig, GLContext, GLContextError, create_gl_context


@dataclass(frozen=True)
class CameraMatrices:
    """Container for the camera transform derived from the world description."""

    position: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    view: np.ndarray
    projection: np.ndarray


class GLRenderer:
    """Thin moderngl based renderer that mirrors the SimCamera world contract."""

    def __init__(
        self,
        *,
        context: Any,
        gl_context: GLContext | None = None,
        context_config: ContextConfig | Mapping[str, Any] | None = None,
        clear_color: Sequence[float] | None = None,
        msaa_samples: int = 0,
        near_clip: float = 0.05,
        far_clip: float = 200.0,
        **_: Any,
    ) -> None:
        self._sim_context = context
        try:
            self._width = int(getattr(context, "width"))
            self._height = int(getattr(context, "height"))
        except AttributeError as exc:  # pragma: no cover - defensive
            raise AttributeError("SimCamera context must expose width/height") from exc

        if gl_context is None:
            try:
                gl_context = create_gl_context(context_config)
            except GLContextError as exc:  # pragma: no cover - passthrough
                raise RuntimeError("Unable to create OpenGL context") from exc

        self._gl_context = gl_context
        self._ctx = gl_context.handle

        colour = clear_color or (0.05, 0.07, 0.09, 1.0)
        if len(colour) == 3:
            colour = (*colour, 1.0)
        if len(colour) != 4:
            raise ValueError("clear_color must have 3 or 4 components")
        self._clear_color = tuple(float(c) for c in colour)

        self._near_clip = float(near_clip)
        self._far_clip = float(far_clip)

        samples = int(msaa_samples)
        self._msaa_samples = samples if samples >= 2 else 0

        self._framebuffer = None
        self._resolve_fb = None
        self._colour_tex = None
        self._depth_rb = None
        self._msaa_colour_tex = None
        self._msaa_depth_rb = None

    # ------------------------------------------------------------------ public
    def render(self, frame: np.ndarray, /, *, frame_id: int | None = None) -> None:
        if frame_id is None:
            frame_id = 0

        if frame.shape[0] != self._height or frame.shape[1] != self._width:
            self._resize(frame.shape[1], frame.shape[0])

        self._ensure_framebuffers()

        camera = self._fetch_camera(frame_id)

        self._begin_frame(camera)
        self._draw_scene(camera)
        image = self._read_colour_buffer()
        frame[:, :, :] = image

    # --------------------------------------------------------------- frame ops
    def _resize(self, width: int, height: int) -> None:
        self._width = int(width)
        self._height = int(height)
        self._release_framebuffers()

    def _ensure_framebuffers(self) -> None:
        if self._framebuffer is not None:
            return

        ctx = self._ctx
        size = (self._width, self._height)

        self._colour_tex = ctx.texture(size, components=4, dtype="u1")
        self._depth_rb = ctx.depth_renderbuffer(size)
        self._framebuffer = ctx.framebuffer(
            color_attachments=[self._colour_tex], depth_attachment=self._depth_rb
        )

        if self._msaa_samples > 0:
            self._msaa_colour_tex = ctx.texture(
                size, components=4, dtype="u1", samples=self._msaa_samples
            )
            self._msaa_depth_rb = ctx.depth_renderbuffer(size, samples=self._msaa_samples)
            self._framebuffer = ctx.framebuffer(
                color_attachments=[self._msaa_colour_tex],
                depth_attachment=self._msaa_depth_rb,
            )
            self._resolve_fb = ctx.framebuffer(color_attachments=[self._colour_tex])

        self._framebuffer.clear(*self._clear_color)

    def _release_framebuffers(self) -> None:
        for resource in (
            self._framebuffer,
            self._resolve_fb,
            self._colour_tex,
            self._depth_rb,
            self._msaa_colour_tex,
            self._msaa_depth_rb,
        ):
            if resource is not None:
                release = getattr(resource, "release", None)
                if callable(release):
                    release()

        self._framebuffer = None
        self._resolve_fb = None
        self._colour_tex = None
        self._depth_rb = None
        self._msaa_colour_tex = None
        self._msaa_depth_rb = None

    def _read_colour_buffer(self) -> np.ndarray:
        framebuffer = self._framebuffer
        if framebuffer is None:
            raise RuntimeError("Frame buffer has not been initialised")

        if self._msaa_samples > 0:
            if self._resolve_fb is None:
                raise RuntimeError("Resolve framebuffer missing for MSAA readback")
            self._ctx.copy_framebuffer(self._resolve_fb, framebuffer)
            framebuffer = self._resolve_fb

        data = framebuffer.read(components=3, dtype="u1", alignment=1)
        rgb = np.frombuffer(data, dtype=np.uint8).reshape(self._height, self._width, 3)
        rgb = np.flip(rgb, axis=0)
        bgr = rgb[:, :, ::-1]
        return bgr.copy()

    # --------------------------------------------------------------- world data
    def _fetch_camera(self, frame_id: int) -> CameraMatrices | None:
        describe = getattr(self._sim_context, "describe_world", None)
        if not callable(describe):
            return None

        try:
            world = describe(frame_id)
        except Exception:  # pragma: no cover - defensive fallback
            return None

        if not isinstance(world, Mapping):
            return None

        camera_state = world.get("camera")
        if not isinstance(camera_state, Mapping):
            return None

        aspect = float(self._width) / float(self._height)
        up = getattr(self._sim_context, "world_up", (0.0, 1.0, 0.0))
        return build_camera_matrices(
            camera_state,
            aspect=aspect,
            near=self._near_clip,
            far=self._far_clip,
            world_up=up,
        )

    # --------------------------------------------------------------- rendering
    def _begin_frame(self, camera: CameraMatrices | None) -> None:
        framebuffer = self._framebuffer
        if framebuffer is None:
            raise RuntimeError("Frame buffer has not been initialised")

        framebuffer.use()
        self._ctx.viewport = (0, 0, self._width, self._height)
        self._ctx.clear(*self._clear_color)

    def _draw_scene(self, camera: CameraMatrices | None) -> None:  # pragma: no cover -
        # Rendering logic will be implemented in subsequent steps.  The method is
        # left intentionally blank so that the renderer can be exercised and the
        # framebuffer/readback path validated against the SimCamera contract.
        return None


# --------------------------------------------------------------------------- util
def build_camera_matrices(
    camera_state: Mapping[str, Any],
    *,
    aspect: float,
    near: float,
    far: float,
    world_up: Sequence[float] | None = None,
) -> CameraMatrices | None:
    try:
        position = np.asarray(camera_state["position"], dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return None

    orientation = camera_state.get("orientation")
    if orientation is not None:
        basis = _camera_basis_from_orientation(orientation)
        if basis is None:
            return None
        forward, right, true_up = basis
    else:
        try:
            target = np.asarray(camera_state["target"], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            return None

        up_vec = (
            _normalise_vector(world_up)
            if world_up is not None
            else np.array((0.0, 1.0, 0.0), dtype=np.float32)
        )
        forward = target - position
        forward = _normalise_vector(forward)
        if not np.isfinite(forward).all():
            return None

        right = np.cross(up_vec, forward)
        if np.linalg.norm(right) <= 1e-6:
            right = np.cross(np.array((0.0, 1.0, 0.0), dtype=np.float32), forward)
            if np.linalg.norm(right) <= 1e-6:
                right = np.cross(np.array((1.0, 0.0, 0.0), dtype=np.float32), forward)
        right = _normalise_vector(right)
        true_up = _normalise_vector(np.cross(forward, right))
        if np.linalg.norm(true_up) <= 1e-6:
            return None

    fov_y = float(camera_state.get("fov_y", 60.0))
    if not math.isfinite(fov_y) or fov_y <= 0.0:
        fov_y = 60.0

    view = _build_view_matrix(position, forward, right, true_up)
    projection = _build_projection_matrix(math.radians(fov_y), aspect, near, far)

    return CameraMatrices(
        position=position,
        forward=forward,
        right=right,
        up=true_up,
        view=view,
        projection=projection,
    )


def _camera_basis_from_orientation(orientation: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
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

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    forward = np.array((0.0, 0.0, -1.0), dtype=np.float32)
    up = np.array((0.0, 1.0, 0.0), dtype=np.float32)
    right = np.array((1.0, 0.0, 0.0), dtype=np.float32)

    if abs(yaw) > 1e-6:
        axis = np.array((0.0, 1.0, 0.0), dtype=np.float32)
        forward = _rotate_vector(forward, axis, yaw)
        right = _rotate_vector(right, axis, yaw)
        up = _rotate_vector(up, axis, yaw)

    if abs(pitch) > 1e-6:
        axis = right
        forward = _rotate_vector(forward, axis, pitch)
        up = _rotate_vector(up, axis, pitch)

    if abs(roll) > 1e-6:
        axis = forward
        right = _rotate_vector(right, axis, roll)
        up = _rotate_vector(up, axis, roll)

    forward = _normalise_vector(forward)
    if np.linalg.norm(forward) <= 1e-6:
        return None

    up = _normalise_vector(up)
    if np.linalg.norm(up) <= 1e-6:
        up = np.array((0.0, 1.0, 0.0), dtype=np.float32)

    right = _normalise_vector(np.cross(up, forward))
    if np.linalg.norm(right) <= 1e-6:
        right = _normalise_vector(np.cross(forward, np.array((0.0, 1.0, 0.0), dtype=np.float32)))
        if np.linalg.norm(right) <= 1e-6:
            right = _normalise_vector(np.cross(forward, np.array((1.0, 0.0, 0.0), dtype=np.float32)))
            if np.linalg.norm(right) <= 1e-6:
                return None

    true_up = _normalise_vector(np.cross(forward, right))
    if np.linalg.norm(true_up) <= 1e-6:
        return None

    return forward, right, true_up


def _parse_orientation(orientation: Any) -> tuple[float, float, float] | None:
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

    axis_length = np.linalg.norm(axis_vec)
    if axis_length <= 1e-6:
        return vec.copy()

    axis_norm = axis_vec / axis_length
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cross = np.cross(axis_norm, vec)
    dot = float(np.dot(axis_norm, vec))
    rotated = vec * cos_a + cross * sin_a + axis_norm * dot * (1.0 - cos_a)
    return rotated.astype(np.float32)


def _normalise_vector(vector: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    length = float(np.linalg.norm(arr))
    if length <= 1e-6:
        return arr.astype(np.float32)
    return (arr / length).astype(np.float32)


def _build_view_matrix(
    position: np.ndarray, forward: np.ndarray, right: np.ndarray, up: np.ndarray
) -> np.ndarray:
    pos = np.asarray(position, dtype=np.float32)
    fwd = np.asarray(forward, dtype=np.float32)
    rgt = np.asarray(right, dtype=np.float32)
    up_vec = np.asarray(up, dtype=np.float32)

    z_axis = -fwd
    view = np.array(
        (
            (rgt[0], rgt[1], rgt[2], -np.dot(rgt, pos)),
            (up_vec[0], up_vec[1], up_vec[2], -np.dot(up_vec, pos)),
            (z_axis[0], z_axis[1], z_axis[2], -np.dot(z_axis, pos)),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    return view


def _build_projection_matrix(
    fov_y_rad: float, aspect: float, near: float, far: float
) -> np.ndarray:
    f = 1.0 / math.tan(max(1e-6, fov_y_rad) / 2.0)
    nf = 1.0 / (near - far)
    proj = np.array(
        (
            (f / max(aspect, 1e-6), 0.0, 0.0, 0.0),
            (0.0, f, 0.0, 0.0),
            (0.0, 0.0, (far + near) * nf, (2.0 * far * near) * nf),
            (0.0, 0.0, -1.0, 0.0),
        ),
        dtype=np.float32,
    )
    return proj


__all__ = ["GLRenderer", "CameraMatrices", "build_camera_matrices"]

