"""Minimal CPU renderer used by :mod:`pc.sim_camera`.

The renderer provides a tiny software rasteriser capable of projecting a
handful of primitives (ground grid, simple building volumes, and a debug cube)
so that the simulation camera once again exposes a notion of 3D world space.
The implementation intentionally stays lightweight while more fully featured
rendering back-ends are being rebuilt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from . import register_renderer

from ._geometry import Point, clip_segment_to_rect


@dataclass
class Sprite:
    image: np.ndarray
    width: int
    height: int
    aspect: float
    path: str


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
        self._building_light_dir = self._normalise(
            np.array((-0.4, 0.9, 0.3), dtype=np.float32)
        )
        self._sprite_cache: Dict[str, Sprite] = {}
        self._missing_sprites: Set[str] = set()

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

        frame[:] = (200, 200, 200)

        # Diagonal gradient in light grey to keep the background from feeling flat.
        grad_x = np.linspace(0.0, 30.0, self.width, dtype=np.float32)
        grad_y = np.linspace(0.0, 30.0, self.height, dtype=np.float32)
        gradient = grad_y[:, None] + grad_x[None, :]
        blended = frame.astype(np.float32) + gradient[..., None]
        np.clip(blended, 0.0, 255.0, out=blended)
        frame[:] = blended.astype(np.uint8)

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

        billboard_queue: List[Dict[str, Any]] = []
        if isinstance(objects, Iterable):
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                obj_type = obj.get("type")
                if obj_type == "cube":
                    self._draw_cube(frame, camera, obj)
                elif obj_type == "building":
                    self._draw_building(frame, camera, obj)
                elif obj_type == "billboard":
                    billboard_queue.append(obj)
                else:
                    self._draw_points(frame, camera, obj)

        if billboard_queue:
            billboard_queue.sort(
                key=lambda item: self._billboard_depth(camera, item),
                reverse=True,
            )
            for billboard in billboard_queue:
                self._draw_billboard(frame, camera, billboard)

        return True

    # -------------------------------------------------------------- primitives
    def _build_camera(self, camera_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            position = np.asarray(camera_state["position"], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            return None

        orientation = camera_state.get("orientation")
        if orientation is not None:
            basis = self._camera_basis_from_orientation(orientation)
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
                up = getattr(self._context, "world_up", (0.0, 1.0, 0.0))
            up_vec = self._normalise(np.asarray(up, dtype=np.float32))
            forward = self._normalise(target - position)
            if self._vector_length(forward) < 1e-6:
                return None

            right = np.cross(up_vec, forward)                 # up × forward  → +X
            if self._vector_length(right) < 1e-6:
                # fallback axes if up≈forward
                right = np.cross(np.array((0.0, 1.0, 0.0), np.float32), forward)
                if self._vector_length(right) < 1e-6:
                    right = np.cross(np.array((1.0, 0.0, 0.0), np.float32), forward)
            right   = self._normalise(right)
            true_up = self._normalise(np.cross(forward, right))  # F × R → U

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

    def _camera_basis_from_orientation(
        self, orientation: Any
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        parsed = self._parse_orientation(orientation)
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
            forward = self._rotate_vector(forward, yaw_axis, yaw_rad)
            right = self._rotate_vector(right, yaw_axis, yaw_rad)
            up = self._rotate_vector(up, yaw_axis, yaw_rad)

        if abs(pitch_rad) > 1e-6:
            pitch_axis = right
            forward = self._rotate_vector(forward, pitch_axis, pitch_rad)
            up = self._rotate_vector(up, pitch_axis, pitch_rad)

        if abs(roll_rad) > 1e-6:
            roll_axis = forward
            right = self._rotate_vector(right, roll_axis, roll_rad)
            up = self._rotate_vector(up, roll_axis, roll_rad)

        forward = self._normalise(forward)
        if self._vector_length(forward) < 1e-6:
            return None

        up = self._normalise(up)
        if self._vector_length(up) < 1e-6:
            up = np.array((0.0, 1.0, 0.0), dtype=np.float32)

        right   = self._normalise(np.cross(up, forward))    
        if self._vector_length(right) < 1e-6:
            right = np.cross(forward, np.array((0.0, 1.0, 0.0), dtype=np.float32))
            if self._vector_length(right) < 1e-6:
                right = np.cross(forward, np.array((1.0, 0.0, 0.0), dtype=np.float32))
                if self._vector_length(right) < 1e-6:
                    return None
        right = self._normalise(right)
        true_up = self._normalise(np.cross(forward, right)) 
        if self._vector_length(true_up) < 1e-6:
            return None

        return forward, right, true_up

    def _parse_orientation(self, orientation: Any) -> Optional[Tuple[float, float, float]]:
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

    def _rotate_vector(
        self, vector: np.ndarray, axis: np.ndarray, angle: float
    ) -> np.ndarray:
        vec = np.asarray(vector, dtype=np.float32)
        axis_vec = np.asarray(axis, dtype=np.float32)
        if abs(angle) <= 1e-6:
            return vec.copy()

        axis_length = self._vector_length(axis_vec)
        if axis_length <= 1e-6:
            return vec.copy()

        axis_norm = axis_vec / axis_length
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        cross = np.cross(axis_norm, vec)
        dot = float(np.dot(axis_norm, vec))
        rotated = vec * cos_a + cross * sin_a + axis_norm * dot * (1.0 - cos_a)
        return rotated.astype(np.float32)

    def _project_point(self, camera: Dict[str, Any], point: Sequence[float]) -> Optional[Tuple[float, float]]:
        coords = self._to_camera_space(camera, point)
        return self._project_camera_coords(camera, coords)

    def _project_camera_coords(
        self, camera: Dict[str, Any], coords: Tuple[float, float, float]
    ) -> Optional[Tuple[float, float]]:
        x, y, z = coords
        if z < self._near_clip:
            return None

        f = 1.0 / math.tan(math.radians(camera["fov_y"]) * 0.5)
        x_ndc = (x / z) * (f / camera["aspect"])
        y_ndc = (y / z) * f

        if not math.isfinite(x_ndc) or not math.isfinite(y_ndc):
            return None

        x_px = (x_ndc + 1.0) * 0.5 * (self.width - 1)
        y_px = (1.0 - (y_ndc + 1.0) * 0.5) * (self.height - 1)
        return (float(x_px), float(y_px))

    def _clip_project_segment(
        self,
        camera: Dict[str, Any],
        start: Sequence[float],
        end: Sequence[float],
    ) -> Optional[Tuple[Point, Point]]:
        """Project a 3D segment to screen space with clipping."""

        start_rel = np.asarray(start, dtype=np.float32) - camera["position"]
        end_rel = np.asarray(end, dtype=np.float32) - camera["position"]

        start_cam = np.array(
            [
                float(np.dot(start_rel, camera["right"])),
                float(np.dot(start_rel, camera["up"])),
                float(np.dot(start_rel, camera["forward"])),
            ],
            dtype=np.float32,
        )
        end_cam = np.array(
            [
                float(np.dot(end_rel, camera["right"])),
                float(np.dot(end_rel, camera["up"])),
                float(np.dot(end_rel, camera["forward"])),
            ],
            dtype=np.float32,
        )

        near = float(self._near_clip)
        z0 = float(start_cam[2])
        z1 = float(end_cam[2])
        if z0 < near and z1 < near:
            return None

        if z0 < near <= z1:
            t = (near - z0) / (z1 - z0)
            start_cam = start_cam + t * (end_cam - start_cam)
            z0 = near
        elif z1 < near <= z0:
            t = (near - z0) / (z1 - z0)
            end_cam = start_cam + t * (end_cam - start_cam)
            z1 = near

        far = camera.get("far_clip")
        if far is not None:
            far = float(far)
            if z0 > far and z1 > far:
                return None
            if z0 > far >= z1:
                t = (far - z0) / (z1 - z0)
                start_cam = start_cam + t * (end_cam - start_cam)
                z0 = far
            elif z1 > far >= z0:
                t = (far - z0) / (z1 - z0)
                end_cam = start_cam + t * (end_cam - start_cam)
                z1 = far

        if z0 <= 0.0 or z1 <= 0.0:
            return None

        f = 1.0 / math.tan(math.radians(camera["fov_y"]) * 0.5)
        x0_ndc = (start_cam[0] / z0) * (f / camera["aspect"])
        y0_ndc = (start_cam[1] / z0) * f
        x1_ndc = (end_cam[0] / z1) * (f / camera["aspect"])
        y1_ndc = (end_cam[1] / z1) * f

        if not all(math.isfinite(v) for v in (x0_ndc, y0_ndc, x1_ndc, y1_ndc)):
            return None

        clipped_ndc = self._clip_segment_ndc((x0_ndc, y0_ndc), (x1_ndc, y1_ndc))
        if clipped_ndc is None:
            return None
        (sx_ndc, sy_ndc), (ex_ndc, ey_ndc) = clipped_ndc

        start_px = (
            (sx_ndc + 1.0) * 0.5 * (self.width - 1),
            (1.0 - (sy_ndc + 1.0) * 0.5) * (self.height - 1),
        )
        end_px = (
            (ex_ndc + 1.0) * 0.5 * (self.width - 1),
            (1.0 - (ey_ndc + 1.0) * 0.5) * (self.height - 1),
        )

        return (start_px, end_px)

    @staticmethod
    def _clip_segment_ndc(
        start: Point,
        end: Point,
    ) -> Optional[Tuple[Point, Point]]:
        x_min = -1.0
        y_min = -1.0
        x_max = 1.0
        y_max = 1.0

        dx = end[0] - start[0]
        dy = end[1] - start[1]

        p = (-dx, dx, -dy, dy)
        q = (
            start[0] - x_min,
            x_max - start[0],
            start[1] - y_min,
            y_max - start[1],
        )

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
        segment = self._clip_project_segment(camera, start, end)
        if segment is None:
            return
        (x0, y0), (x1, y1) = segment
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)), int(round(y1)))
        cv2.line(frame, p0, p1, colour, thickness, cv2.LINE_AA)

    def _draw_ground_grid(self, frame: np.ndarray, camera: Dict[str, Any]) -> None:
        extent = 500
        self._draw_ground_plane(frame, camera, extent)
        step = 25
        base_colour = (100, 100, 100)
        axis_colour = (100, 100, 100)
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

    def _draw_ground_plane(self, frame: np.ndarray, camera: Dict[str, Any], extent: int) -> None:
        corners = (
            (-float(extent), 0.0, -float(extent)),
            (float(extent), 0.0, -float(extent)),
            (float(extent), 0.0, float(extent)),
            (-float(extent), 0.0, float(extent)),
        )

        camera_space = [self._to_camera_space(camera, corner) for corner in corners]
        clipped = self._clip_polygon_to_near_plane(camera_space)
        if len(clipped) < 3:
            return

        projected: List[Tuple[int, int]] = []
        for vertex in clipped:
            projected_point = self._project_camera_coords(camera, vertex)
            if projected_point is None:
                continue
            px = int(round(projected_point[0]))
            py = int(round(projected_point[1]))
            projected.append((px, py))

        if len(projected) < 3:
            return

        points = np.array(projected, dtype=np.int32)
        cv2.fillConvexPoly(frame, points, (200, 200, 200), lineType=cv2.LINE_AA)

    def _draw_building(
        self, frame: np.ndarray, camera: Dict[str, Any], building: Dict[str, Any]
    ) -> None:
        base_centre = building.get("base_centre")
        footprint = building.get("footprint")
        height = building.get("height")
        if base_centre is None or footprint is None or height is None:
            return

        try:
            base = np.asarray(base_centre, dtype=np.float32).reshape(-1)
            footprint_values = np.asarray(footprint, dtype=np.float32).reshape(-1)
            height_value = float(height)
        except (TypeError, ValueError):
            return

        if base.size < 2 or footprint_values.size < 2:
            return
        if not math.isfinite(height_value) or height_value <= 1e-3:
            return

        half_width = float(abs(footprint_values[0])) * 0.5
        half_depth = float(abs(footprint_values[1])) * 0.5
        if half_width <= 1e-6 or half_depth <= 1e-6:
            return

        cx = float(base[0])
        cz = float(base[1])
        y_bottom = 0.0
        y_top = y_bottom + height_value

        vertices = np.array(
            [
                (cx - half_width, y_bottom, cz - half_depth),
                (cx + half_width, y_bottom, cz - half_depth),
                (cx + half_width, y_top, cz - half_depth),
                (cx - half_width, y_top, cz - half_depth),
                (cx - half_width, y_bottom, cz + half_depth),
                (cx + half_width, y_bottom, cz + half_depth),
                (cx + half_width, y_top, cz + half_depth),
                (cx - half_width, y_top, cz + half_depth),
            ],
            dtype=np.float32,
        )

        faces: Tuple[Tuple[Tuple[int, ...], np.ndarray], ...] = (
            ((4, 5, 6, 7), np.array((0.0, 0.0, 1.0), dtype=np.float32)),
            ((1, 2, 6, 5), np.array((1.0, 0.0, 0.0), dtype=np.float32)),
            ((0, 4, 7, 3), np.array((-1.0, 0.0, 0.0), dtype=np.float32)),
            ((0, 3, 2, 1), np.array((0.0, 0.0, -1.0), dtype=np.float32)),
            ((3, 7, 6, 2), np.array((0.0, 1.0, 0.0), dtype=np.float32)),
        )

        colour_spec = building.get("color", building.get("colour"))
        if colour_spec is None:
            base_colour = np.array((180.0, 180.0, 200.0), dtype=np.float32)
        else:
            try:
                base_colour = np.asarray(colour_spec, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError):
                base_colour = np.array((180.0, 180.0, 200.0), dtype=np.float32)
            else:
                if base_colour.size < 3:
                    base_colour = np.array((180.0, 180.0, 200.0), dtype=np.float32)
                else:
                    base_colour = base_colour[:3].astype(np.float32)

        light_dir = self._building_light_dir
        if self._vector_length(light_dir) <= 1e-6:
            light_dir = self._normalise(np.array((-0.4, 0.9, 0.3), dtype=np.float32))
            self._building_light_dir = light_dir

        face_polygons: List[Tuple[float, List[Tuple[float, float]], np.ndarray]] = []
        for indices, normal_world in faces:
            world_vertices = [vertices[i] for i in indices]
            camera_vertices = [self._to_camera_space(camera, vertex) for vertex in world_vertices]
            if len(camera_vertices) < 3:
                continue

            clipped = self._clip_polygon_to_near_plane(camera_vertices)
            if len(clipped) < 3:
                continue

            orientation = self._camera_polygon_orientation(clipped)
            if orientation >= -1e-6:
                continue

            projected: List[Tuple[float, float]] = []
            skip_face = False
            for vertex in clipped:
                projected_point = self._project_camera_coords(camera, vertex)
                if projected_point is None:
                    skip_face = True
                    break
                projected.append(projected_point)

            if skip_face or len(projected) < 3:
                continue

            depth = float(sum(v[2] for v in clipped) / len(clipped))
            face_polygons.append((depth, projected, normal_world))

        if not face_polygons:
            return

        face_polygons.sort(key=lambda item: item[0], reverse=True)

        ambient = 0.35
        diffuse = 0.65

        for _, projected, normal_world in face_polygons:
            pts = np.array(
                [[int(round(px)), int(round(py))] for px, py in projected],
                dtype=np.int32,
            )
            if pts.shape[0] < 3:
                continue

            polygon = pts.reshape(-1, 1, 2)

            normal_vec = np.asarray(normal_world, dtype=np.float32)
            if self._vector_length(normal_vec) <= 1e-6:
                normal_vec = np.array((0.0, 0.0, 1.0), dtype=np.float32)
            else:
                normal_vec = self._normalise(normal_vec)

            diffuse_term = max(0.0, float(np.dot(normal_vec, light_dir)))
            intensity = ambient + diffuse * diffuse_term
            fill_colour = tuple(
                int(max(0, min(255, round(float(channel) * intensity))))
                for channel in base_colour
            )

            cv2.fillConvexPoly(frame, polygon, fill_colour, lineType=cv2.LINE_AA)

            edge_colour = tuple(
                int(max(0, min(255, round(float(value) * 0.7))))
                for value in fill_colour
            )
            cv2.polylines(frame, [polygon], True, edge_colour, 1, cv2.LINE_AA)

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
            segment = self._clip_project_segment(camera, vertices[start], vertices[end])
            if segment is None:
                continue
            (x0, y0), (x1, y1) = segment
            pt0 = (int(round(x0)), int(round(y0)))
            pt1 = (int(round(x1)), int(round(y1)))
            cv2.line(frame, pt0, pt1, colour_bgr, 2, cv2.LINE_AA)


    def _draw_billboard(
        self, frame: np.ndarray, camera: Dict[str, Any], billboard: Dict[str, Any]
    ) -> None:
        sprite = self._load_sprite(billboard.get("sprite"))
        if sprite is None:
            return

        position_spec = billboard.get("position")
        if position_spec is None:
            return
        try:
            position = np.asarray(position_spec, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return
        if position.size < 3:
            return
        position = position[:3].astype(np.float32)

        coords = self._to_camera_space(camera, position)
        depth = float(coords[2])
        if depth <= self._near_clip:
            return

        centre = self._project_camera_coords(camera, coords)
        if centre is None:
            return
        centre_x, centre_y = centre

        height_spec = billboard.get("height")
        if height_spec is None:
            height_spec = billboard.get("size_m", billboard.get("size"))
        try:
            height_m = float(height_spec) if height_spec is not None else 1.0
        except (TypeError, ValueError):
            height_m = 1.0
        if not math.isfinite(height_m) or height_m <= 1e-3:
            return

        up_spec = billboard.get("up")
        if up_spec is None:
            up_spec = getattr(self._context, "world_up", (0.0, 1.0, 0.0))
        try:
            up_vec = np.asarray(up_spec, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            up_vec = np.array((0.0, 1.0, 0.0), dtype=np.float32)
        if up_vec.size < 3:
            up_vec = np.array((0.0, 1.0, 0.0), dtype=np.float32)
        up_vec = self._normalise(up_vec[:3])
        if self._vector_length(up_vec) <= 1e-6:
            up_vec = np.array((0.0, 1.0, 0.0), dtype=np.float32)

        half_height = height_m * 0.5
        top_world = position + up_vec * half_height
        bottom_world = position - up_vec * half_height
        top_px = self._project_point(camera, top_world)
        bottom_px = self._project_point(camera, bottom_world)

        if top_px is None or bottom_px is None:
            height_px = self._estimate_billboard_height(height_m, depth, camera)
        else:
            height_px = abs(top_px[1] - bottom_px[1])
        if height_px <= 1.0:
            return

        aspect_spec = billboard.get("aspect_ratio")
        if aspect_spec is None:
            aspect = sprite.aspect
        else:
            try:
                aspect = float(aspect_spec)
            except (TypeError, ValueError):
                aspect = sprite.aspect
            else:
                if not math.isfinite(aspect) or aspect <= 1e-6:
                    aspect = sprite.aspect

        width_px = height_px * aspect
        if width_px <= 1.0:
            return

        x0 = centre_x - width_px * 0.5
        x1 = centre_x + width_px * 0.5
        y0 = centre_y - height_px * 0.5
        y1 = centre_y + height_px * 0.5

        ix0 = int(math.floor(x0))
        ix1 = int(math.ceil(x1))
        iy0 = int(math.floor(y0))
        iy1 = int(math.ceil(y1))

        if ix1 <= 0 or iy1 <= 0 or ix0 >= self.width or iy0 >= self.height:
            return

        dest_w = max(1, ix1 - ix0)
        dest_h = max(1, iy1 - iy0)

        sprite_rgba = cv2.resize(
            sprite.image,
            (dest_w, dest_h),
            interpolation=cv2.INTER_LINEAR,
        )
        if sprite_rgba.ndim != 3 or sprite_rgba.shape[2] < 4:
            return

        tint = billboard.get("tint")
        if tint is None:
            tint = billboard.get("color", billboard.get("colour"))
        tint_multiplier: Optional[np.ndarray]
        tint_multiplier = None
        if tint is not None:
            try:
                tint_values = np.asarray(tint, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError):
                tint_multiplier = None
            else:
                if tint_values.size >= 3:
                    tint_multiplier = (tint_values[:3] / 255.0).reshape(1, 1, 3)
        if tint_multiplier is not None:
            sprite_rgba[:, :, :3] = np.clip(
                sprite_rgba[:, :, :3] * tint_multiplier,
                0.0,
                1.0,
            )

        opacity = billboard.get("opacity")
        if opacity is not None:
            try:
                opacity_value = float(opacity)
            except (TypeError, ValueError):
                opacity_value = 1.0
            else:
                if not math.isfinite(opacity_value):
                    opacity_value = 1.0
                opacity_value = max(0.0, min(1.0, opacity_value))
            sprite_rgba[:, :, 3] *= opacity_value

        clip_x0 = max(0, ix0)
        clip_y0 = max(0, iy0)
        clip_x1 = min(self.width, ix1)
        clip_y1 = min(self.height, iy1)
        if clip_x1 <= clip_x0 or clip_y1 <= clip_y0:
            return

        src_x0 = clip_x0 - ix0
        src_y0 = clip_y0 - iy0
        src_x1 = src_x0 + (clip_x1 - clip_x0)
        src_y1 = src_y0 + (clip_y1 - clip_y0)

        sprite_roi = sprite_rgba[src_y0:src_y1, src_x0:src_x1, :]
        if sprite_roi.size == 0:
            return

        colour = sprite_roi[:, :, :3]
        alpha = sprite_roi[:, :, 3:4]
        if np.max(alpha) <= 1e-6:
            return

        frame_roi = frame[clip_y0:clip_y1, clip_x0:clip_x1].astype(np.float32) / 255.0
        blended = colour * alpha + frame_roi * (1.0 - alpha)
        frame[clip_y0:clip_y1, clip_x0:clip_x1] = np.clip(
            blended * 255.0 + 0.5,
            0.0,
            255.0,
        ).astype(np.uint8)


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

    def _billboard_depth(self, camera: Dict[str, Any], billboard: Dict[str, Any]) -> float:
        position_spec = billboard.get("position")
        if position_spec is None:
            return float("inf")
        try:
            position = np.asarray(position_spec, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return float("inf")
        if position.size < 3:
            return float("inf")
        coords = self._to_camera_space(camera, position[:3])
        depth = float(coords[2])
        if not math.isfinite(depth):
            return float("inf")
        return depth

    def _estimate_billboard_height(
        self, height_m: float, depth: float, camera: Dict[str, Any]
    ) -> float:
        if not math.isfinite(height_m) or height_m <= 0.0:
            return 0.0
        if not math.isfinite(depth) or depth <= 1e-6:
            return 0.0
        frame_height = max(1, self.height - 1)
        try:
            f = 1.0 / math.tan(math.radians(camera["fov_y"]) * 0.5)
        except (KeyError, TypeError, ValueError):
            return 0.0
        if not math.isfinite(f):
            return 0.0
        pixels = frame_height * height_m * f / (2.0 * depth)
        return float(abs(pixels))

    def _load_sprite(self, sprite_spec: Any) -> Optional[Sprite]:
        if not sprite_spec:
            return None

        resolved = self._resolve_sprite_path(sprite_spec)
        if resolved is None:
            return None

        try:
            resolved_path = resolved.resolve()
        except Exception:
            resolved_path = resolved

        key = str(resolved_path)
        cached = self._sprite_cache.get(key)
        if cached is not None:
            return cached
        if key in self._missing_sprites:
            return None

        image = cv2.imread(key, cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3:
            self._missing_sprites.add(key)
            return None

        if image.shape[2] == 3:
            alpha = np.full(image.shape[:2] + (1,), 255, dtype=image.dtype)
            image = np.concatenate((image, alpha), axis=2)
        elif image.shape[2] > 4:
            image = image[:, :, :4]

        if image.shape[2] < 4:
            self._missing_sprites.add(key)
            return None

        image_f = image.astype(np.float32) / 255.0
        height = int(image.shape[0])
        width = int(image.shape[1])
        aspect = float(width) / float(max(1, height))
        sprite = Sprite(image=image_f, width=width, height=height, aspect=aspect, path=key)
        self._sprite_cache[key] = sprite
        return sprite

    def _resolve_sprite_path(self, sprite_spec: Any) -> Optional[Path]:
        try:
            initial = Path(str(sprite_spec)).expanduser()
        except Exception:
            return None

        candidates: List[Path] = [initial]

        if not initial.is_absolute():
            context_root = getattr(self._context, "asset_root", None)
            if context_root is not None:
                try:
                    candidates.append(Path(context_root) / initial)
                except Exception:
                    pass
            repo_root = Path(__file__).resolve().parents[2]
            candidates.append(repo_root / initial)
            candidates.append(repo_root / "assets" / initial)

        seen: Set[Path] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve() if candidate.exists() else candidate
            except Exception:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                if resolved.exists():
                    return resolved
            except OSError:
                continue

        try:
            if initial.exists():
                return initial
        except OSError:
            return None
        return None

    @staticmethod
    def _vector_length(vec: np.ndarray) -> float:
        return float(np.linalg.norm(vec))

    def _normalise(self, vec: np.ndarray) -> np.ndarray:
        length = self._vector_length(vec)
        if length <= 1e-6:
            return vec
        return vec / length

    def _to_camera_space(self, camera: Dict[str, Any], point: Sequence[float]) -> Tuple[float, float, float]:
        rel = np.asarray(point, dtype=np.float32) - camera["position"]
        x = float(np.dot(rel, camera["right"]))
        y = float(np.dot(rel, camera["up"]))
        z = float(np.dot(rel, camera["forward"]))
        return (x, y, z)

    @staticmethod
    def _camera_polygon_orientation(
        vertices: Sequence[Tuple[float, float, float]]
    ) -> float:
        if len(vertices) < 3:
            return 0.0

        base_x, base_y, base_z = vertices[0]
        if base_z <= 1e-6:
            return 0.0

        base_px = base_x / base_z
        base_py = base_y / base_z
        orientation = 0.0

        for idx in range(1, len(vertices) - 1):
            x1, y1, z1 = vertices[idx]
            x2, y2, z2 = vertices[idx + 1]
            if z1 <= 1e-6 or z2 <= 1e-6:
                continue

            px1 = x1 / z1
            py1 = y1 / z1
            px2 = x2 / z2
            py2 = y2 / z2
            orientation += (px1 - base_px) * (py2 - base_py) - (py1 - base_py) * (px2 - base_px)

        return float(orientation)

    def _clip_polygon_to_near_plane(
        self, vertices: Sequence[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        if not vertices:
            return []

        clipped: List[Tuple[float, float, float]] = []
        near = self._near_clip

        prev = vertices[-1]
        prev_inside = prev[2] >= near

        for current in vertices:
            curr_inside = current[2] >= near

            if curr_inside:
                if not prev_inside:
                    clipped.append(self._intersect_near_plane(prev, current, near))
                clipped.append(current)
            elif prev_inside:
                clipped.append(self._intersect_near_plane(prev, current, near))

            prev = current
            prev_inside = curr_inside

        return clipped

    @staticmethod
    def _intersect_near_plane(
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        near: float,
    ) -> Tuple[float, float, float]:
        z0 = start[2]
        z1 = end[2]
        denom = z1 - z0
        if abs(denom) <= 1e-6:
            t = 0.0
        else:
            t = (near - z0) / denom
        t = max(0.0, min(1.0, t))
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        return (x, y, near)

register_renderer("cpu", lambda **kwargs: CPURenderer(**kwargs))

__all__ = ["CPURenderer", "clip_segment_to_rect"]
