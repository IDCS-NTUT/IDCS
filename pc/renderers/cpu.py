"""Minimal CPU renderer used by :mod:`pc.sim_camera`.

The renderer provides a tiny software rasteriser capable of projecting a
handful of primitives (currently just a spinning cube and a simple ground
plane) so that the simulation camera once again exposes a notion of 3D world
space.  The implementation intentionally stays lightweight while more fully
featured rendering back-ends are being rebuilt.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import register_renderer


class CPURenderer:
    """Trivial placeholder renderer with a minimal sense of 3D space.

    The renderer uses a solid two-tone background and projects the
    ``SimCamera`` world description to draw a filled ground plane and a
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
        self._background_colour = (44, 58, 76)
        self._ground_colour = (72, 104, 92)

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

        frame[:] = self._background_colour

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
        cv2.drawMarker(
            frame,
            centre,
            (220, 220, 220),
            markerType=cv2.MARKER_CROSS,
            markerSize=self._crosshair_radius * 2,
            thickness=1,
            line_type=cv2.LINE_AA,
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

        self._draw_ground_plane(frame, camera)

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

    def _draw_ground_plane(self, frame: np.ndarray, camera: Dict[str, Any]) -> None:
        extent = 12.0
        samples: list[Tuple[float, float]] = []
        perimeter = np.linspace(-extent, extent, num=25)

        for x in (-extent, extent):
            for z in perimeter:
                projected = self._project_point(camera, (x, 0.0, float(z)))
                if projected is not None:
                    samples.append(projected)

        for z in (-extent, extent):
            for x in perimeter:
                projected = self._project_point(camera, (float(x), 0.0, z))
                if projected is not None:
                    samples.append(projected)

        if len(samples) < 3:
            return

        hull = cv2.convexHull(np.array(samples, dtype=np.float32))
        cv2.fillConvexPoly(frame, hull.astype(np.int32), self._ground_colour)

        axis_colour = tuple(int(min(255, c + 30)) for c in self._ground_colour)
        self._draw_world_line(
            frame,
            camera,
            (0.0, 0.0, -extent),
            (0.0, 0.0, extent),
            axis_colour,
            thickness=2,
        )
        self._draw_world_line(
            frame,
            camera,
            (-extent, 0.0, 0.0),
            (extent, 0.0, 0.0),
            axis_colour,
            thickness=2,
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

__all__ = ["CPURenderer"]
