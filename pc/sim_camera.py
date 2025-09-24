"""Minimal simulation frame generator.

The previous simulation stack exposed a full world description with multiple
render back-ends.  Those pieces are still being rebuilt, but the simulation
camera once again exposes a tiny 3D world so renderers can reason about a
camera pose and simple geometry.  The public :meth:`SimCamera.next_frame` API
remains unchanged so that the rest of the streaming pipeline keeps working
while new rendering features are prototyped.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import cv2

from .renderers import get_renderer


class SimCamera:
    """Tiny frame generator used while the real renderer is rebuilt."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        *,
        renderer_name: str | None = None,
        renderer_opts: Dict[str, Any] | None = None,
        debug: bool = False,
        targets: Sequence[Dict[str, Any]] | None = None,
        **_: Any,
    ) -> None:
        """Initialise the simulation camera with optional overrides."""
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
        self._billboard_specs = self._load_billboard_targets(targets)

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
        """Return world building objects consumed by the renderer."""

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

    def _describe_billboards(self, frame_id: int) -> List[Dict[str, Any]]:
        """Build a list of billboard objects for the current frame."""

        billboards: List[Dict[str, Any]] = []
        for spec in self._billboard_specs:
            mode = spec.get("mode", "static")
            if mode == "circle" and spec.get("orbit"):
                orbit = spec["orbit"]
                angle = orbit["phase"] + frame_id * orbit["speed"]
                x = orbit["centre"][0] + math.cos(angle) * orbit["radius"]
                z = orbit["centre"][1] + math.sin(angle) * orbit["radius"]
                position = np.array((x, orbit["height"], z), dtype=np.float32)
            else:
                position = spec["position"].copy()
            billboards.append(
                {
                    "type": "billboard",
                    "cls": spec.get("cls"),
                    "position": position,
                    "size_m": spec["size_m"],
                    "sprite": spec["sprite"],
                    "anchor_v": spec["anchor_v"],
                }
            )
        return billboards

    def _load_billboard_targets(
        self, targets: Optional[Sequence[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Resolve billboard specifications from the configuration."""

        if not targets:
            return []

        project_root = Path(__file__).resolve().parents[1]
        resolved: List[Dict[str, Any]] = []
        for spec in targets:
            if not isinstance(spec, dict):
                continue

            sprite_path = spec.get("sprite")
            if not sprite_path:
                continue
            path = Path(str(sprite_path))
            if not path.is_absolute():
                path = project_root / path
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None or image.size == 0:
                continue

            try:
                size_m = float(spec.get("size_m", 1.0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(size_m) or size_m <= 0.0:
                continue

            position_spec = spec.get("position")
            if position_spec is None:
                continue
            try:
                position = np.asarray(position_spec, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError):
                continue
            if position.size < 3:
                continue

            anchor_v = self._parse_anchor(spec.get("anchor"))

            mode = str(spec.get("mode", "static")).lower()
            orbit_params = None
            final_position = position[:3].astype(np.float32)

            if mode == "circle":
                centre_spec = spec.get("centre", spec.get("center", position))
                try:
                    centre_values = np.asarray(centre_spec, dtype=np.float32).reshape(-1)
                except (TypeError, ValueError):
                    centre_values = position
                if centre_values.size < 3:
                    centre_values = np.pad(centre_values, (0, max(0, 3 - centre_values.size)), constant_values=0.0)

                try:
                    radius = abs(float(spec.get("radius", 4.0)))
                except (TypeError, ValueError):
                    radius = 4.0
                try:
                    orbit_height = float(spec.get("height", centre_values[1]))
                except (TypeError, ValueError):
                    orbit_height = float(centre_values[1])
                try:
                    speed_deg = float(spec.get("speed_deg", 12.0))
                except (TypeError, ValueError):
                    speed_deg = 12.0
                try:
                    phase_deg = float(spec.get("phase_deg", 0.0))
                except (TypeError, ValueError):
                    phase_deg = 0.0

                orbit_params = {
                    "centre": np.array((centre_values[0], centre_values[2]), dtype=np.float32),
                    "height": float(orbit_height),
                    "radius": float(radius),
                    "speed": math.radians(speed_deg),
                    "phase": math.radians(phase_deg),
                }

                angle0 = orbit_params["phase"]
                final_position = np.array(
                    (
                        orbit_params["centre"][0] + math.cos(angle0) * orbit_params["radius"],
                        orbit_params["height"],
                        orbit_params["centre"][1] + math.sin(angle0) * orbit_params["radius"],
                    ),
                    dtype=np.float32,
                )

            resolved.append(
                {
                    "mode": mode,
                    "cls": spec.get("cls"),
                    "position": final_position,
                    "size_m": size_m,
                    "sprite": image,
                    "anchor_v": anchor_v,
                    "orbit": orbit_params,
                }
            )

        return resolved

    @staticmethod
    def _parse_anchor(anchor: Any) -> float:
        """Convert an anchor specification to a usable float in [0,1]."""

        if isinstance(anchor, str):
            key = anchor.strip().lower()
            if key == "bottom":
                return 0.0
            if key == "top":
                return 1.0
            return 0.5

        try:
            value = float(anchor)
        except (TypeError, ValueError):
            return 0.5
        return float(max(0.0, min(1.0, value)))

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
        """Convert a look-at pair into Euler angles for the renderer."""
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
