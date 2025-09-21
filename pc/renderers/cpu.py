"""Minimal CPU renderer used by :mod:`pc.sim_camera`.

The renderer provides a tiny software rasteriser capable of projecting a
handful of primitives (currently just a spinning cube and a ground grid) so
that the simulation camera once again exposes a notion of 3D world space.  The
implementation intentionally stays lightweight while more fully featured
rendering back-ends are being rebuilt.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import register_renderer

from ._geometry import Point, clip_segment_to_rect


class CPURenderer:
    """Trivial placeholder renderer with a minimal sense of 3D space.

    A soft gradient is still used for the background, but the renderer now
    projects the ``SimCamera`` world description to draw a ground grid and a
    wireframe cube.  When no world data is available the previous crosshair and
    animated dot are rendered instead so existing callers see a familiar
    placeholder output.
    """

    def __init__(self, *, context: Any) -> None:
        try:
            self.width = int(getattr(context, "width"))
            self.height = int(getattr(context, "height"))
        except AttributeError as exc:  # pragma: no cover - defensive only
            raise AttributeError("SimCamera context must expose width/height") from exc

        self._context = context
        self._crosshair_radius = max(2, min(self.width, self.height) // 32)
        self._dot_radius = max(4, min(self.width, self.height) // 24)
        self._near_clip = 0.05

    def render(self, frame: np.ndarray, /, *, frame_id: Optional[int] = None) -> None:
        """Render a single frame into ``frame``.

        Parameters
        ----------
        frame:
            Destination image buffer in BGR format.
        frame_id:
            Optional monotonically increasing identifier used to animate the
            moving dot.  When omitted the renderer falls back to ``0``.
        """

        if frame_id is None:
            frame_id = 0

        frame[:] = (48, 60, 76)

        # Diagonal gradient so the background isn't completely flat.
        grad = np.linspace(0.0, 32.0, self.width, dtype=np.float32)
        channel = frame[:, :, 2].astype(np.float32)
        channel = np.clip(channel + grad, 0.0, 255.0)
        frame[:, :, 2] = channel.astype(np.uint8)

        world = self._fetch_world(frame_id)
        if world is not None:
            if not self._render_world(frame, world):
                self._draw_overlay(frame, frame_id)
        else:
            self._draw_overlay(frame, frame_id)

        # Subtle border to make the frame edges easy to see.
        cv2.rectangle(frame, (0, 0), (self.width - 1, self.height - 1), (90, 110, 130), 1)

    # ---------------------------------------------------------------- helpers
    def _fetch_world(self, frame_id: int) -> Optional[Dict[str, Any]]:
        describe = getattr(self._context, "describe_world", None)
        if not callable(describe):
            return None
        try:
            world = describe(frame_id)
        except Exception:  # pragma: no cover - best effort fall back
            return None
        if not isinstance(world, dict):
            return None
        return world

    def _draw_overlay(self, frame: np.ndarray, frame_id: int) -> None:
        centre = (self.width // 2, self.height // 2)
        crosshair_segments = [
            ((centre[0] - self._crosshair_radius, centre[1]), (centre[0] + self._crosshair_radius, centre[1])),
            ((centre[0], centre[1] - self._crosshair_radius), (centre[0], centre[1] + self._crosshair_radius)),
        ]

        for start, end in crosshair_segments:
            clipped = clip_segment_to_rect(start, end, self.width, self.height)
            if clipped is None:
                continue
            (x0, y0), (x1, y1) = clipped
            cv2.line(
                frame,
                (int(round(x0)), int(round(y0))),
                (int(round(x1)), int(round(y1))),
                (220, 220, 220),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        angle = (frame_id % 360) * math.pi / 180.0
        orbit = min(self.width, self.height) * 0.25
        offset = (int(math.cos(angle) * orbit), int(math.sin(angle) * orbit))
        dot_pos = (centre[0] + offset[0], centre[1] + offset[1])
        cv2.circle(frame, dot_pos, self._dot_radius, (64, 180, 250), -1, cv2.LINE_AA)

    def _render_world(self, frame: np.ndarray, world: Dict[str, Any]) -> bool:
        camera_state = world.get("camera")
        if not isinstance(camera_state, dict):
            return False

        camera = self._build_camera(camera_state)
        if camera is None:
            return False

        self._draw_ground_grid(frame, camera)

        objects = world.get("objects", ())
        if isinstance(objects, dict):
            objects = (objects,)
        if isinstance(objects, Iterable):
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                if obj.get("type") == "cube":
                    self._draw_cube(frame, camera, obj)
                else:
                    self._draw_points(frame, camera, obj)

        return True

    # -------------------------------------------------------------- primitives
    def _build_camera(self, camera_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            position = np.asarray(camera_state["position"], dtype=np.float32)
            target = np.asarray(camera_state["target"], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            return None

        up = camera_state.get("up")
        if up is None:
            up = getattr(self._context, "world_up", (0.0, 1.0, 0.0))
        up_vec = self._normalise(np.asarray(up, dtype=np.float32))
        forward = self._normalise(target - position)
        if self._vector_length(forward) < 1e-6:
            return None

        right = np.cross(forward, up_vec)
        if self._vector_length(right) < 1e-6:
            right = np.cross(forward, np.array((0.0, 1.0, 0.0), dtype=np.float32))
            if self._vector_length(right) < 1e-6:
                right = np.cross(forward, np.array((1.0, 0.0, 0.0), dtype=np.float32))
        right = self._normalise(right)
        true_up = self._normalise(np.cross(right, forward))

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
            "aspect": float(self.width) / float(self.height),
        }

    def _project_point(self, camera: Dict[str, Any], point: Sequence[float]) -> Optional[Tuple[float, float]]:
        rel = np.asarray(point, dtype=np.float32) - camera["position"]
        z = float(np.dot(rel, camera["forward"]))
        if z <= self._near_clip:
            return None

        x = float(np.dot(rel, camera["right"]))
        y = float(np.dot(rel, camera["up"]))

        f = 1.0 / math.tan(math.radians(camera["fov_y"]) * 0.5)
        x_ndc = (x / z) * (f / camera["aspect"])
        y_ndc = (y / z) * f

        if not math.isfinite(x_ndc) or not math.isfinite(y_ndc):
            return None

        x_px = (x_ndc + 1.0) * 0.5 * (self.width - 1)
        y_px = (1.0 - (y_ndc + 1.0) * 0.5) * (self.height - 1)
        return (float(x_px), float(y_px))

    def _draw_world_line(
        self,
        frame: np.ndarray,
        camera: Dict[str, Any],
        start: Sequence[float],
        end: Sequence[float],
        colour: Tuple[int, int, int],
        *,
        thickness: int = 1,
    ) -> None:
        start_px = self._project_point(camera, start)
        end_px = self._project_point(camera, end)
        if start_px is None or end_px is None:
            return
        p0 = (int(round(start_px[0])), int(round(start_px[1])))
        p1 = (int(round(end_px[0])), int(round(end_px[1])))
        cv2.line(frame, p0, p1, colour, thickness, cv2.LINE_AA)

    def _draw_ground_grid(self, frame: np.ndarray, camera: Dict[str, Any]) -> None:
        extent = 8
        step = 1
        base_colour = (70, 85, 110)
        axis_colour = (110, 150, 180)
        for ix in range(-extent, extent + 1, step):
            colour = axis_colour if ix == 0 else base_colour
            self._draw_world_line(
                frame,
                camera,
                (float(ix), 0.0, -float(extent)),
                (float(ix), 0.0, float(extent)),
                colour,
            )

        for iz in range(-extent, extent + 1, step):
            colour = axis_colour if iz == 0 else base_colour
            self._draw_world_line(
                frame,
                camera,
                (-float(extent), 0.0, float(iz)),
                (float(extent), 0.0, float(iz)),
                colour,
            )

    def _draw_cube(self, frame: np.ndarray, camera: Dict[str, Any], cube: Dict[str, Any]) -> None:
        centre = np.asarray(cube.get("centre", (0.0, 0.0, 0.0)), dtype=np.float32)
        half_extents = np.asarray(cube.get("half_extents", (0.5, 0.5, 0.5)), dtype=np.float32)
        rotation = np.asarray(cube.get("rotation", np.eye(3, dtype=np.float32)), dtype=np.float32)
        if rotation.shape != (3, 3):
            rotation = np.eye(3, dtype=np.float32)

        offsets = np.array(
            [
                (-1.0, -1.0, -1.0),
                (1.0, -1.0, -1.0),
                (1.0, 1.0, -1.0),
                (-1.0, 1.0, -1.0),
                (-1.0, -1.0, 1.0),
                (1.0, -1.0, 1.0),
                (1.0, 1.0, 1.0),
                (-1.0, 1.0, 1.0),
            ],
            dtype=np.float32,
        )

        scaled = offsets * half_extents
        rotated = scaled @ rotation.T
        vertices = rotated + centre

        projected = [self._project_point(camera, vertex) for vertex in vertices]

        edges = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )

        colour = cube.get("color", (120, 200, 240))
        colour_bgr = tuple(int(max(0, min(255, c))) for c in colour)

        for start, end in edges:
            p0 = projected[start]
            p1 = projected[end]
            if p0 is None or p1 is None:
                continue
            pt0 = (int(round(p0[0])), int(round(p0[1])))
            pt1 = (int(round(p1[0])), int(round(p1[1])))
            cv2.line(frame, pt0, pt1, colour_bgr, 2, cv2.LINE_AA)

        # Draw a faint shadow on the ground plane to help with depth cues.
        shadow_colour = (30, 40, 50)
        shadow_vertices = vertices.copy()
        shadow_vertices[:, 1] = 0.0
        shadow_proj = [self._project_point(camera, vertex) for vertex in shadow_vertices]
        for start, end in edges[:4]:  # only draw the base face shadow
            p0 = shadow_proj[start]
            p1 = shadow_proj[end]
            if p0 is None or p1 is None:
                continue
            pt0 = (int(round(p0[0])), int(round(p0[1])))
            pt1 = (int(round(p1[0])), int(round(p1[1])))
            cv2.line(frame, pt0, pt1, shadow_colour, 1, cv2.LINE_AA)

    def _draw_points(self, frame: np.ndarray, camera: Dict[str, Any], obj: Dict[str, Any]) -> None:
        points = obj.get("points")
        if points is None:
            return
        colour = obj.get("color", (200, 200, 200))
        colour_bgr = tuple(int(max(0, min(255, c))) for c in colour)
        for point in points:
            projected = self._project_point(camera, point)
            if projected is None:
                continue
            centre = (int(round(projected[0])), int(round(projected[1])))
            cv2.circle(frame, centre, 3, colour_bgr, -1, cv2.LINE_AA)

    @staticmethod
    def _vector_length(vec: np.ndarray) -> float:
        return float(np.linalg.norm(vec))

    def _normalise(self, vec: np.ndarray) -> np.ndarray:
        length = self._vector_length(vec)
        if length <= 1e-6:
            return vec
        return vec / length


register_renderer("cpu", lambda **kwargs: CPURenderer(**kwargs))

__all__ = ["CPURenderer", "clip_segment_to_rect"]
