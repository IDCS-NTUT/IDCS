"""Minimal simulation frame generator.

The previous simulation stack exposed a full world description with multiple
render back-ends.  Those pieces are still being rebuilt, but the simulation
camera once again exposes a tiny 3D world so renderers can reason about a
camera pose and simple geometry.  The public :meth:`SimCamera.next_frame` API
remains unchanged so that the rest of the streaming pipeline keeps working
while new rendering features are prototyped.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .renderers import get_renderer


@dataclass
class BillboardTarget:
    """Description of a simulated billboard target."""

    cls: str
    sprite: str
    size_m: float
    centre: np.ndarray
    mode: str
    path: Dict[str, Any]
    tint: Optional[Tuple[int, int, int]] = None
    opacity: float = 1.0
    phase: float = 0.0

    def describe(self, frame_id: int, world_up: np.ndarray) -> Dict[str, Any]:
        """Return a world object dictionary for this target."""

        position = self._position(frame_id)
        description: Dict[str, Any] = {
            "type": "billboard",
            "class": self.cls,
            "sprite": self.sprite,
            "height": float(self.size_m),
            "position": (
                float(position[0]),
                float(position[1]),
                float(position[2]),
            ),
            "up": (
                float(world_up[0]),
                float(world_up[1]),
                float(world_up[2]),
            ),
        }

        aspect = self._aspect_ratio()
        if aspect is not None:
            description["aspect_ratio"] = aspect

        if self.tint is not None:
            description["tint"] = self.tint

        if self.opacity < 1.0 - 1e-6:
            description["opacity"] = self.opacity

        return description

    def _position(self, frame_id: int) -> np.ndarray:
        params = self.path
        base = self.centre.astype(np.float32).copy()

        mode = self.mode
        if mode == "circle":
            radius = max(0.0, self._param(("radius",), 4.0))
            speed_deg = self._param(
                ("angular_speed_deg", "speed_deg", "speed_deg_per_frame"),
                20.0,
            )
            angle = math.radians(speed_deg) * frame_id + self.phase
            base[0] += math.cos(angle) * radius
            base[2] += math.sin(angle) * radius
        elif mode in {"figure8", "figure_eight"}:
            radius = max(0.0, self._param(("radius",), 3.5))
            speed_deg = self._param(
                ("angular_speed_deg", "speed_deg", "speed_deg_per_frame"),
                30.0,
            )
            angle = math.radians(speed_deg) * frame_id + self.phase
            base[0] += math.sin(angle) * radius
            base[2] += math.sin(angle * 0.5) * radius
        elif mode == "random":
            radius = max(0.0, self._param(("radius",), 3.0))
            speed_deg = self._param(
                ("speed_deg", "angular_speed_deg", "speed_deg_per_frame"),
                17.0,
            )
            angle = math.radians(speed_deg) * frame_id
            phase_x = self.phase
            phase_z = self.phase * 0.73 + 1.11
            base[0] += math.sin(angle + phase_x) * radius
            base[2] += math.cos(angle * 0.79 + phase_z) * radius

        height_offset = self._param(("height_offset", "y_offset"), 0.0)
        base[1] += height_offset

        bob_amplitude = self._param(("bob_amplitude", "bob"), 0.0)
        if bob_amplitude != 0.0:
            bob_speed = self._param(
                ("bob_speed_deg", "bob_speed", "bob_speed_deg_per_frame"),
                45.0,
            )
            base[1] += math.sin(math.radians(bob_speed) * frame_id + self.phase * 1.3) * bob_amplitude

        return base

    def _aspect_ratio(self) -> Optional[float]:
        value = self.path.get("aspect_ratio")
        if value is None:
            return None
        try:
            aspect = float(value)
        except (TypeError, ValueError):
            return None
        if aspect <= 1e-6:
            return None
        return float(aspect)

    def _param(self, keys: Sequence[str], default: float) -> float:
        if isinstance(keys, str):
            keys = (keys,)
        for key in keys:
            value = self.path.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return float(default)


class SimCamera:
    """Tiny frame generator used while the real renderer is rebuilt."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        *,
        renderer_name: str | None = None,
        renderer_opts: Dict[str, Any] | None = None,
        targets: Sequence[Dict[str, Any]] | None = None,
        debug: bool = False,
        **_: Any,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self._frame_id = 0

        opts = renderer_opts or {}
        self._renderer = get_renderer(renderer_name, context=self, **opts)

        self._debug_mode = bool(debug)

        # Basic world state used by the CPU renderer.  Future tasks will grow
        # this into a richer scene description that supports multiple objects
        # and rendering back-ends.  Coordinates are expressed in metres.
        self.world_up = np.array((0.0, 1.0, 0.0), dtype=np.float32)
        self._camera_target = np.array((0.0, 0.75, 0.0), dtype=np.float32)
        self._camera_fov_y = 60.0
        self._camera_orbit_radius = 7.5
        self._camera_orbit_height = 3.2
        self._camera_orbit_speed = math.radians(0.6)
        self._camera_fixed_position = np.array(
            (0.0, 50.0, 0.0),
            dtype=np.float32,
        )
        self._camera_fixed_orientation = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        # Single spinning cube used as a placeholder object in the world.
        self._cube_half_extents = np.array((0.75, 0.75, 0.75), dtype=np.float32)
        self._cube_spin_speed = math.radians(1.5)
        self._cube_colour = (64, 180, 250)
        self._building_specs: Tuple[Dict[str, Any], ...] = (
            {
                "base_centre": (0.0, -100.0),
                "footprint": (18.0, 14.0),
                "height": 36.0,
                "color": (190, 190, 215),
            },
        )

        self._billboard_targets = self._build_targets(targets)

    def next_frame(self) -> Tuple[bool, np.ndarray]:
        """Return the next simulated frame.

        The method maintains a monotonically increasing frame identifier so the
        renderer can animate simple placeholder elements.  A fresh NumPy buffer
        is allocated for each call to keep the implementation straightforward.
        """

        self._frame_id += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._renderer.render(frame, frame_id=self._frame_id)
        return True, frame

    # ------------------------------------------------------------------ world
    def describe_world(self, frame_id: int) -> Dict[str, Any]:
        """Return a minimal world description for ``frame_id``.

        The world exposes a single camera that remains fixed in normal
        operation.  When debug mode is enabled the camera resumes its orbit
        around the origin and a spinning cube is injected so renderers can
        visualise orientation cues.  The contract is intentionally small so
        renderers can consume it without needing a full scene graph or material
        system.
        """

        objects = self._describe_buildings()
        objects.extend(self._describe_billboards(frame_id))

        if self._debug_mode:
            orbit_angle = frame_id * self._camera_orbit_speed
            camera_position = np.array(
                (
                    math.cos(orbit_angle) * self._camera_orbit_radius,
                    self._camera_orbit_height,
                    math.sin(orbit_angle) * self._camera_orbit_radius,
                ),
                dtype=np.float32,
            )
            orientation = self._compute_camera_orientation(
                camera_position, self._camera_target
            )

            cube_spin = frame_id * self._cube_spin_speed
            cube_rotation = self._y_axis_rotation(cube_spin)
            objects.append(
                {
                    "type": "cube",
                    "centre": self._camera_target.copy(),
                    "half_extents": self._cube_half_extents.copy(),
                    "rotation": cube_rotation,
                    "color": self._cube_colour,
                }
            )
        else:
            camera_position = self._camera_fixed_position.copy()
            orientation = self._camera_fixed_orientation.copy()

        camera_info = {
            "position": camera_position,
            "target": self._camera_target.copy(),
            "up": self.world_up.copy(),
            "fov_y": self._camera_fov_y,
        }
        if orientation is not None:
            camera_info["orientation"] = orientation

        return {
            "camera": camera_info,
            "objects": objects,
        }

    def _describe_buildings(self) -> list[Dict[str, Any]]:
        buildings: list[Dict[str, Any]] = []
        for spec in self._building_specs:
            try:
                base = np.asarray(spec["base_centre"], dtype=np.float32).reshape(-1)
                footprint = np.asarray(spec["footprint"], dtype=np.float32).reshape(-1)
                height = float(spec["height"])
            except (KeyError, TypeError, ValueError):
                continue

            if base.size < 2 or footprint.size < 2:
                continue
            if not math.isfinite(height) or height <= 0.0:
                continue

            base_tuple = (float(base[0]), float(base[1]))
            footprint_tuple = (float(abs(footprint[0])), float(abs(footprint[1])))

            colour_spec = spec.get("color", spec.get("colour"))
            if colour_spec is None:
                colour = (180, 180, 200)
            else:
                try:
                    colour_values = np.asarray(colour_spec, dtype=np.float32).reshape(-1)
                except (TypeError, ValueError):
                    colour = (180, 180, 200)
                else:
                    if colour_values.size < 3:
                        colour = (180, 180, 200)
                    else:
                        colour = tuple(
                            int(max(0, min(255, round(float(v)))))
                            for v in colour_values[:3]
                        )

            buildings.append(
                {
                    "type": "building",
                    "base_centre": base_tuple,
                    "footprint": footprint_tuple,
                    "height": height,
                    "color": colour,
                }
            )

        return buildings

    def _describe_billboards(self, frame_id: int) -> list[Dict[str, Any]]:
        billboards: list[Dict[str, Any]] = []
        for target in self._billboard_targets:
            try:
                billboards.append(target.describe(frame_id, self.world_up))
            except Exception:
                continue
        return billboards

    def _build_targets(
        self, targets: Sequence[Dict[str, Any]] | None
    ) -> Tuple[BillboardTarget, ...]:
        if not targets:
            return ()

        parsed: List[BillboardTarget] = []
        for spec in targets:
            target = self._parse_billboard_target(spec)
            if target is None:
                continue
            parsed.append(target)
        return tuple(parsed)

    def _parse_billboard_target(
        self, spec: Any
    ) -> Optional[BillboardTarget]:
        if not isinstance(spec, dict):
            return None

        sprite = spec.get("sprite")
        if not sprite:
            return None
        sprite_path = str(sprite)

        size_spec = spec.get("size_m")
        if size_spec is None:
            size_spec = spec.get("height", spec.get("size"))
        try:
            size_m = float(size_spec) if size_spec is not None else 1.8
        except (TypeError, ValueError):
            size_m = 1.8
        if not math.isfinite(size_m) or size_m <= 1e-3:
            return None

        position_spec = (
            spec.get("position")
            or spec.get("centre")
            or spec.get("center")
        )
        if position_spec is None:
            centre = np.array((0.0, size_m * 0.5, -5.0), dtype=np.float32)
        else:
            try:
                centre_values = np.asarray(position_spec, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError):
                centre = np.array((0.0, size_m * 0.5, -5.0), dtype=np.float32)
            else:
                if centre_values.size < 3:
                    centre = np.array((0.0, size_m * 0.5, -5.0), dtype=np.float32)
                else:
                    centre = centre_values[:3].astype(np.float32)

        path_params_raw = spec.get("path")
        if isinstance(path_params_raw, dict):
            path_params = dict(path_params_raw)
        else:
            path_params = {}

        mode_value = spec.get("mode")
        if mode_value is None:
            mode_value = path_params.get("mode")
        mode = str(mode_value or "static").strip().lower().replace("-", "_")
        if mode == "figure_eight":
            mode = "figure8"
        if mode not in {"static", "circle", "figure8", "random"}:
            mode = "static"

        tint = None
        for key in ("tint", "color", "colour"):
            if key not in spec:
                continue
            value = spec.get(key)
            if value is None:
                continue
            try:
                tint_values = np.asarray(value, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError):
                continue
            if tint_values.size < 3:
                continue
            tint = tuple(
                int(max(0, min(255, round(float(component)))))
                for component in tint_values[:3]
            )
            break

        opacity_value = spec.get("opacity")
        if opacity_value is None:
            opacity = 1.0
        else:
            try:
                opacity = float(opacity_value)
            except (TypeError, ValueError):
                opacity = 1.0
            else:
                if not math.isfinite(opacity):
                    opacity = 1.0
                else:
                    opacity = max(0.0, min(1.0, opacity))

        cls_name = str(spec.get("cls", "target")).strip() or "target"

        phase_override = path_params.pop("phase_deg", None)
        if phase_override is None:
            phase_override = path_params.pop("phase", None)
        try:
            phase_offset = (
                math.radians(float(phase_override))
                if phase_override is not None
                else 0.0
            )
        except (TypeError, ValueError):
            phase_offset = 0.0

        phase = self._billboard_phase_seed(cls_name, sprite_path) + phase_offset

        return BillboardTarget(
            cls=cls_name,
            sprite=sprite_path,
            size_m=float(size_m),
            centre=centre.astype(np.float32),
            mode=mode,
            path=path_params,
            tint=tint,
            opacity=opacity,
            phase=phase,
        )

    def _billboard_phase_seed(self, cls_name: str, sprite_path: str) -> float:
        data = f"{cls_name}|{sprite_path}".encode("utf-8", "ignore")
        digest = hashlib.sha256(data).digest()
        value = int.from_bytes(digest[:8], "big")
        if value == 0:
            value = 1
        return (value / float(1 << 64)) * math.tau

    @staticmethod
    def _y_axis_rotation(angle: float) -> np.ndarray:
        """Return a 3×3 rotation matrix for a rotation around the world Y axis."""

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return np.array(
            (
                (cos_a, 0.0, sin_a),
                (0.0, 1.0, 0.0),
                (-sin_a, 0.0, cos_a),
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _compute_camera_orientation(
        position: np.ndarray, target: np.ndarray
    ) -> Optional[Dict[str, float]]:
        forward = np.asarray(target, dtype=np.float32) - np.asarray(position, dtype=np.float32)
        length = float(np.linalg.norm(forward))
        if length <= 1e-6:
            return None

        forward /= length
        y_component = float(np.clip(forward[1], -1.0, 1.0))
        pitch_rad = math.asin(y_component)
        yaw_rad = math.atan2(float(-forward[0]), float(-forward[2]))
        if not (math.isfinite(pitch_rad) and math.isfinite(yaw_rad)):
            return None

        return {
            "yaw": math.degrees(yaw_rad),
            "pitch": math.degrees(pitch_rad),
            "roll": 0.0,
        }
