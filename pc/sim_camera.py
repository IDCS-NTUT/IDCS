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
from typing import Any, Dict, Optional, Tuple

import numpy as np

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
            (0.0,40.0, 0.0),
            dtype=np.float32,
        )
        self._camera_fixed_orientation = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        # Single spinning cube used as a placeholder object in the world.
        self._cube_half_extents = np.array((0.75, 0.75, 0.75), dtype=np.float32)
        self._cube_spin_speed = math.radians(1.5)
        self._cube_colour = (64, 180, 250)

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
            objects: list[Dict[str, Any]] = [
                {
                    "type": "cube",
                    "centre": self._camera_target.copy(),
                    "half_extents": self._cube_half_extents.copy(),
                    "rotation": cube_rotation,
                    "color": self._cube_colour,
                }
            ]
        else:
            camera_position = self._camera_fixed_position.copy()
            orientation = self._camera_fixed_orientation.copy()
            objects = []

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
