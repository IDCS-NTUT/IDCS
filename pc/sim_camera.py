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

_TAU = math.tau if hasattr(math, "tau") else (2.0 * math.pi)


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _wrap_angle(angle: float) -> float:
    if not math.isfinite(angle):
        return 0.0
    wrapped = (angle + math.pi) % _TAU
    return wrapped - math.pi

import numpy as np

from .renderers import get_renderer
from ._sprites import get_sprite_aspect_ratio
from .sim_eval import SimEvaluationManager


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
        scene: Dict[str, Any] | None = None,
        evaluation: Dict[str, Any] | None = None,
        threat_eval: Dict[str, Any] | None = None,
        fps_hz: float = 30.0,
        **_: Any,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self._frame_id = 0
        self._frame_buffer: Optional[np.ndarray] = None
        try:
            fps_value = float(fps_hz)
        except (TypeError, ValueError):
            fps_value = 30.0
        if not math.isfinite(fps_value) or fps_value <= 0.0:
            fps_value = 30.0
        self._fps_hz = fps_value

        opts = renderer_opts or {}
        self.renderer_opts = dict(opts)
        self._renderer = get_renderer(renderer_name, context=self)

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
            (0.0, 1.0, 0.0),
            dtype=np.float32,
        )
        self._camera_fixed_orientation = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        # Live pan/tilt pose that can be driven by external control commands.
        self._pan_rad = 0.0
        self._tilt_rad = math.radians(self._camera_fixed_orientation["pitch"])
        self._roll_rad = math.radians(self._camera_fixed_orientation.get("roll", 0.0))
        self._pan_rate = 0.0
        self._tilt_rate = 0.0
        self._tilt_limits = (
            math.radians(-80.0),
            math.radians(80.0),
        )
        self._home_pan_rad = self._pan_rad
        self._home_tilt_rad = self._tilt_rad

        # Single spinning cube used as a placeholder object in the world.
        self._cube_half_extents = np.array((0.75, 0.75, 0.75), dtype=np.float32)
        self._cube_spin_speed = math.radians(1.5)
        self._cube_colour = (64, 180, 250)
        self._billboard_circle_radius = 6.0
        self._billboard_circle_speed = math.radians(0.75)
        default_building_specs: Tuple[Dict[str, Any], ...] = (
            {
                "base_centre": (0.0, -100.0),
                "footprint": (18.0, 14.0),
                "height": 36.0,
                "color": (190, 190, 215),
            },
        )
        default_billboard_specs: Tuple[Dict[str, Any], ...] = (
            {
                "ground": (0.5, -6.0),
                "ground_y": 0.0,
                "height": 1.7,
                "sprite": "person",
                "movement": {
                    "type": "circle",
                    "radius": self._billboard_circle_radius,
                    "speed": self._billboard_circle_speed,
                },
            },
            {
                "ground": (2.0, -10.0),
                "ground_y": 3.0,
                "width": 0.4,
                "sprite": "drone",
                "movement": {
                    "type": "circle",
                    "radius": self._billboard_circle_radius,
                    "speed": self._billboard_circle_speed,
                },
            },
        )

        self._building_specs = default_building_specs
        self._billboard_specs = default_billboard_specs
        self._billboard_motion_states: Dict[Any, Dict[str, Any]] = {}
        self._billboard_path_states = self._billboard_motion_states
        self._cube_specs: Tuple[Dict[str, Any], ...] = ()
        self._use_scene_cubes = False
        self._mesh_specs: Tuple[Dict[str, Any], ...] = ()
        if isinstance(scene, dict):
            if "buildings" in scene:
                self._building_specs = self._coerce_scene_specs(scene.get("buildings"))
            if "targets" in scene:
                self._billboard_specs = self._coerce_scene_specs(scene.get("targets"))
            elif "billboards" in scene:
                self._billboard_specs = self._coerce_scene_specs(scene.get("billboards"))
            if "cubes" in scene:
                self._cube_specs = self._coerce_scene_specs(scene.get("cubes"))
                self._use_scene_cubes = True
            if "meshes" in scene:
                self._mesh_specs = self._coerce_scene_specs(scene.get("meshes"))
        self._evaluation = SimEvaluationManager.from_config(
            evaluation,
            scene=scene,
            threat_eval=threat_eval,
            width=self.width,
            height=self.height,
            fps_hz=self._fps_hz,
            camera_state=self._evaluation_camera_state(),
            context=self,
        )

    def next_frame(self) -> Tuple[bool, np.ndarray]:
        """Return the next simulated frame.

        The method maintains a monotonically increasing frame identifier so the
        renderer can animate simple placeholder elements.  A single NumPy buffer
        is reused between calls, so downstream consumers must copy the frame if
        they need to retain it after calling :meth:`next_frame` again.
        """

        self._frame_id += 1
        if (
            self._frame_buffer is None
            or self._frame_buffer.shape[0] != self.height
            or self._frame_buffer.shape[1] != self.width
        ):
            self._frame_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            self._frame_buffer.fill(0)
        self._renderer.render(self._frame_buffer, frame_id=self._frame_id)
        return True, self._frame_buffer

    # ---------------------------------------------------------------- control
    def apply_control_rates(self, pan_rate: float, tilt_rate: float, dt: float) -> None:
        """Integrate the commanded pan/tilt rates over ``dt`` seconds."""

        if dt <= 0.0 or not math.isfinite(dt):
            return

        self._pan_rate = float(pan_rate)
        self._tilt_rate = float(tilt_rate)

        self._pan_rad = _wrap_angle(self._pan_rad + self._pan_rate * dt)
        next_tilt = self._tilt_rad + self._tilt_rate * dt
        self._tilt_rad = _clamp(next_tilt, self._tilt_limits[0], self._tilt_limits[1])

    def apply_cam_state(
        self,
        *,
        pan: float,
        tilt: float,
        pan_rate: Optional[float] = None,
        tilt_rate: Optional[float] = None,
    ) -> None:
        """Apply an externally measured pan/tilt pose to the simulator camera."""

        self._pan_rad = _wrap_angle(float(pan))
        self._tilt_rad = _clamp(float(tilt), self._tilt_limits[0], self._tilt_limits[1])
        if pan_rate is not None and math.isfinite(float(pan_rate)):
            self._pan_rate = float(pan_rate)
        else:
            self._pan_rate = 0.0
        if tilt_rate is not None and math.isfinite(float(tilt_rate)):
            self._tilt_rate = float(tilt_rate)
        else:
            self._tilt_rate = 0.0

    def get_pose(self) -> Dict[str, float]:
        """Return the current pan/tilt pose in radians and rates."""

        return {
            "pan": self._pan_rad,
            "tilt": self._tilt_rad,
            "pan_rate": self._pan_rate,
            "tilt_rate": self._tilt_rate,
        }

    def get_home_pose(self) -> Dict[str, float]:
        """Return the default/rest pan/tilt pose in radians."""

        return {
            "pan": self._home_pan_rad,
            "tilt": self._home_tilt_rad,
        }

    def apply_evaluation_control(self, cmd: Any) -> None:
        """Feed a live control command into the optional evaluation manager."""

        if self._evaluation is None:
            return
        self._evaluation.apply_control_cmd(
            cmd,
            camera_state=self._evaluation_camera_state(),
            context=self,
        )

    def evaluation_metrics(self) -> Optional[Dict[str, float]]:
        """Return visual evaluation counters when evaluation mode is active."""

        if self._evaluation is None:
            return None
        return self._evaluation.metrics()

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
        if self._evaluation is not None:
            objects.extend(self._evaluation.describe_targets(frame_id))
        else:
            objects.extend(self._describe_billboards(frame_id))
        objects.extend(self._describe_meshes())

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

            if not self._use_scene_cubes:
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
            # inject a sample mesh if not provided
            if not self._mesh_specs:
                sample_path = "assets/meshes/person.obj"
                objects.append(
                    {
                        "type": "target",
                        "asset": sample_path,
                        "sprite": "person",
                        "centre": (0.0, 0.0, 0.0),
                        "size": (1.0, 1.0),
                        "scale": 1.0,
                        "alpha": 1.0,
                    }
                )
        else:
            camera_position = self._camera_fixed_position.copy()
            orientation = {
                "yaw": math.degrees(self._pan_rad),
                "pitch": math.degrees(self._tilt_rad),
                "roll": math.degrees(self._roll_rad),
            }

        # Only include scene-defined cubes when debug mode is active so that
        # cubes are not rendered during normal (non-debug) operation.
        if self._debug_mode and self._use_scene_cubes:
            objects.extend(self._describe_cubes())

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

    def _evaluation_camera_state(self) -> Dict[str, Any]:
        return {
            "position": self._camera_fixed_position.copy(),
            "target": self._camera_target.copy(),
            "up": self.world_up.copy(),
            "fov_y": self._camera_fov_y,
            "orientation": {
                "yaw": math.degrees(self._pan_rad),
                "pitch": math.degrees(self._tilt_rad),
                "roll": math.degrees(self._roll_rad),
            },
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
                    "albedo_map": spec.get("albedo_map"),
                    "normal_map": spec.get("normal_map"),
                    "metallic": spec.get("metallic"),
                    "roughness": spec.get("roughness"),
                    "uv_scale": spec.get("uv_scale"),
                }
            )

        return buildings

    def _describe_billboards(self, frame_id: int) -> list[Dict[str, Any]]:
        # Workflow: normalise billboard specs and format shared target entries for renderers.
        targets: list[Dict[str, Any]] = []
        for target_idx, spec in enumerate(self._billboard_specs):
            sprite_name = spec.get("sprite")
            if sprite_name is None:
                continue

            width_spec = spec.get("width")
            height_spec = spec.get("height")

            width = None
            height = None

            movement = self._normalise_billboard_movement(spec)

            if width_spec is not None:
                try:
                    width = float(width_spec)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(width) or width == 0.0:
                    continue

            if height_spec is not None:
                try:
                    height = float(height_spec)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(height) or height == 0.0:
                    continue

            if width is None and height is None:
                continue

            if width is None or height is None:
                try:
                    aspect_ratio = get_sprite_aspect_ratio(sprite_name)
                except ValueError:
                    continue

                if not math.isfinite(aspect_ratio) or aspect_ratio <= 0.0:
                    continue

                if width is None and height is not None:
                    width = height * aspect_ratio
                elif height is None and width is not None:
                    height = width / aspect_ratio

            if width is None or height is None:
                continue

            width = abs(float(width))
            height = abs(float(height))

            if width <= 1e-6 or height <= 1e-6:
                continue

            centre_override = spec.get("centre")
            if centre_override is not None:
                try:
                    centre_values = (
                        np.asarray(centre_override, dtype=np.float32).reshape(-1)
                    )
                except (TypeError, ValueError):
                    continue

                if centre_values.size < 3:
                    continue

                centre = np.asarray(centre_values[:3], dtype=np.float32)
                if movement is not None:
                    if movement.get("type") == "path":
                        path_centre = self._apply_billboard_path_movement(
                            movement,
                            frame_id,
                            target_idx=target_idx,
                        )
                        if path_centre is not None:
                            centre = path_centre
                    else:
                        start_planar = np.array((centre[0], centre[2]), dtype=np.float32)
                        planar_position = self._apply_billboard_planar_movement(
                            movement,
                            start_planar,
                            frame_id,
                            target_idx=target_idx,
                            y_value=float(centre[1]),
                        )
                        centre[0] = planar_position[0]
                        centre[2] = planar_position[1]
            else:
                base_y = 0.0
                start_planar = None
                if movement is not None and movement.get("type") == "path":
                    path_centre = self._apply_billboard_path_movement(
                        movement,
                        frame_id,
                        target_idx=target_idx,
                    )
                    if path_centre is not None:
                        centre = path_centre
                    else:
                        try:
                            ground = np.asarray(spec["ground"], dtype=np.float32).reshape(-1)
                        except (KeyError, TypeError, ValueError):
                            continue
                        if ground.size < 2:
                            continue
                        try:
                            base_y = float(spec.get("ground_y", 0.0))
                        except (TypeError, ValueError):
                            base_y = 0.0
                        if not math.isfinite(base_y):
                            base_y = 0.0
                        start_planar = np.array(
                            (float(ground[0]), float(ground[1])),
                            dtype=np.float32,
                        )
                        planar_position = self._apply_billboard_planar_movement(
                            None,
                            start_planar,
                            frame_id,
                            target_idx=target_idx,
                            y_value=base_y + abs(height) * 0.5,
                        )
                        base = np.array(
                            (float(planar_position[0]), base_y, float(planar_position[1])),
                            dtype=np.float32,
                        )
                        centre = base + np.array(
                            (0.0, abs(height) * 0.5, 0.0),
                            dtype=np.float32,
                        )
                else:
                    try:
                        ground = np.asarray(spec["ground"], dtype=np.float32).reshape(-1)
                    except (KeyError, TypeError, ValueError):
                        continue
                    if ground.size < 2:
                        continue
                    try:
                        base_y = float(spec.get("ground_y", 0.0))
                    except (TypeError, ValueError):
                        base_y = 0.0
                    if not math.isfinite(base_y):
                        base_y = 0.0
                    start_planar = np.array(
                        (float(ground[0]), float(ground[1])),
                        dtype=np.float32,
                    )
                    planar_position = self._apply_billboard_planar_movement(
                        movement,
                        start_planar,
                        frame_id,
                        target_idx=target_idx,
                        y_value=base_y + abs(height) * 0.5,
                    )
                    base = np.array(
                        (float(planar_position[0]), base_y, float(planar_position[1])),
                        dtype=np.float32,
                    )
                    centre = base + np.array(
                        (0.0, abs(height) * 0.5, 0.0),
                        dtype=np.float32,
                    )

            if not np.all(np.isfinite(centre)):
                continue

            entry: Dict[str, Any] = {
                "type": "target",
                "centre": (
                    float(centre[0]),
                    float(centre[1]),
                    float(centre[2]),
                ),
                "size": (float(abs(width)), float(abs(height))),
                "sprite": sprite_name,
            }
            colour = spec.get("color")
            if colour is None:
                colour = spec.get("colour")
            if colour is not None:
                entry["color"] = colour
            orientation = spec.get("orientation")
            if orientation is None:
                orientation = spec.get("sprite_orientation")
            if orientation is not None:
                entry["orientation"] = orientation
            rotation = spec.get("rotation")
            if rotation is not None:
                entry["rotation"] = rotation
            targets.append(entry)

        return targets

    def _describe_cubes(self) -> list[Dict[str, Any]]:
        cubes: list[Dict[str, Any]] = []
        for spec in self._cube_specs:
            if not isinstance(spec, dict):
                continue
            entry: Dict[str, Any] = {"type": "cube"}
            centre = spec.get("centre")
            if centre is None:
                centre = spec.get("center")
            if centre is not None:
                entry["centre"] = centre
            half_extents = spec.get("half_extents")
            if half_extents is not None:
                entry["half_extents"] = half_extents
            rotation = spec.get("rotation")
            if rotation is not None:
                entry["rotation"] = rotation
            colour = spec.get("color")
            if colour is None:
                colour = spec.get("colour")
            if colour is not None:
                entry["color"] = colour
            for key in ("albedo_map", "normal_map", "metallic", "roughness", "uv_scale"):
                if key in spec:
                    entry[key] = spec[key]
            cubes.append(entry)
        return cubes

    def _describe_meshes(self) -> list[Dict[str, Any]]:
        targets: list[Dict[str, Any]] = []
        for spec in self._mesh_specs:
            if not isinstance(spec, dict):
                continue
            asset = spec.get("asset") or spec.get("path")
            if not asset:
                continue
            sprite = spec.get("sprite")
            if sprite is None:
                asset_name = str(asset).lower()
                if "person" in asset_name:
                    sprite = "person"
                elif "drone" in asset_name:
                    sprite = "drone"

            entry: Dict[str, Any] = {"type": "target", "asset": asset}
            if sprite is not None:
                entry["sprite"] = sprite

            for key in (
                "centre",
                "center",
                "rotation",
                "color",
                "colour",
                "alpha",
                "albedo_map",
                "normal_map",
                "metallic",
                "roughness",
                "uv_scale",
            ):
                if key in spec:
                    canonical = key
                    if key == "center":
                        canonical = "centre"
                    if key == "colour":
                        canonical = "color"
                    entry[canonical] = spec[key]

            size_spec = spec.get("size")
            if size_spec is None:
                size_spec = spec.get("scale")
            if size_spec is not None:
                try:
                    size_values = np.asarray(size_spec, dtype=np.float32).reshape(-1)
                except (TypeError, ValueError):
                    size_values = np.asarray((), dtype=np.float32)
                if size_values.size >= 1:
                    if size_values.size == 1:
                        width = float(size_values[0])
                        height = float(size_values[0])
                    else:
                        width = float(size_values[0])
                        height = float(size_values[1])
                    if math.isfinite(width) and math.isfinite(height):
                        width = abs(width)
                        height = abs(height)
                        if width > 0.0 and height > 0.0:
                            entry["size"] = (width, height)

            if "scale" in spec:
                entry["scale"] = spec["scale"]
            targets.append(entry)
        return targets

    @staticmethod
    def _coerce_scene_specs(value: Any) -> Tuple[Dict[str, Any], ...]:
        if isinstance(value, dict):
            return (value,)
        if isinstance(value, (list, tuple)):
            return tuple(item for item in value if isinstance(item, dict))
        return ()

    def _normalise_billboard_movement(
        self, spec: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        movement_spec = spec.get("movement")
        if movement_spec is None:
            return None

        movement_type: Optional[str]
        params: Dict[str, Any]

        if isinstance(movement_spec, str):
            movement_type = movement_spec.strip().lower()
            params = {}
        elif isinstance(movement_spec, dict):
            raw_type = movement_spec.get("type")
            if raw_type is None:
                raw_type = movement_spec.get("kind")
            if raw_type is None:
                raw_type = movement_spec.get("mode")
            movement_type = str(raw_type).strip().lower() if raw_type is not None else None
            params = movement_spec
        else:
            return None

        if not movement_type:
            return None

        if movement_type == "circle":
            radius_value = params.get("radius")
            if radius_value is None:
                radius_value = spec.get("radius")
            try:
                radius = float(radius_value)
            except (TypeError, ValueError):
                return None
            radius = abs(radius)
            if not math.isfinite(radius) or radius <= 1e-6:
                return None

            speed_value = params.get("speed")
            if speed_value is None:
                speed_value = spec.get("speed")
            if speed_value is None:
                speed = self._billboard_circle_speed
            else:
                try:
                    speed = float(speed_value)
                except (TypeError, ValueError):
                    speed = self._billboard_circle_speed

            if not math.isfinite(speed):
                speed = self._billboard_circle_speed

            speed = abs(speed)

            phase_value = params.get("phase")
            if phase_value is None:
                phase = 0.0
            else:
                try:
                    phase = float(phase_value)
                except (TypeError, ValueError):
                    phase = 0.0

            return {
                "type": "circle",
                "radius": radius,
                "speed": speed,
                "phase": phase,
                "dynamics": self._normalise_movement_dynamics(params),
            }

        if movement_type == "path":
            points_value = params.get("points")
            if points_value is None:
                points_value = spec.get("points")
            if not isinstance(points_value, (list, tuple)):
                return None

            points: list[Tuple[float, float, float]] = []
            for raw_point in points_value:
                try:
                    point_values = np.asarray(raw_point, dtype=np.float32).reshape(-1)
                except (TypeError, ValueError):
                    return None
                if point_values.size < 3:
                    return None
                point = (
                    float(point_values[0]),
                    float(point_values[1]),
                    float(point_values[2]),
                )
                if not all(math.isfinite(v) for v in point):
                    return None
                points.append(point)

            if len(points) < 2:
                return None

            points_np = np.asarray(points, dtype=np.float32)
            next_points_np = np.roll(points_np, shift=-1, axis=0)
            segment_lengths = np.linalg.norm(next_points_np - points_np, axis=1)
            if segment_lengths.size != points_np.shape[0] or not np.all(np.isfinite(segment_lengths)):
                return None

            loop_length = float(np.sum(segment_lengths))
            if not math.isfinite(loop_length) or loop_length <= 1e-6:
                return None

            speed_value = params.get("speed_m_s")
            if speed_value is None:
                speed_value = spec.get("speed_m_s")
            try:
                speed_m_s = float(speed_value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(speed_m_s) or speed_m_s <= 0.0:
                return None

            segment_ends = np.cumsum(segment_lengths, dtype=np.float32)
            dynamics = self._normalise_movement_dynamics(params)
            return {
                "type": "path",
                "points": points_np,
                "segment_lengths": segment_lengths.astype(np.float32, copy=False),
                "segment_ends": segment_ends,
                "loop_length": loop_length,
                "speed_m_s": speed_m_s,
                "dynamics": dynamics,
            }

        return None

    def _normalise_movement_dynamics(
        self, params: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        dynamics_spec = params.get("dynamics")
        if not isinstance(dynamics_spec, dict):
            return None

        if not self._coerce_bool(dynamics_spec.get("enabled", False)):
            return None

        try:
            max_accel = float(dynamics_spec["max_accel_m_s2"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(max_accel) or max_accel <= 0.0:
            return None

        max_decel_raw = dynamics_spec.get("max_decel_m_s2", max_accel)
        try:
            max_decel = float(max_decel_raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(max_decel) or max_decel <= 0.0:
            return None

        arrival_raw = dynamics_spec.get("arrival_radius_m", 0.15)
        try:
            arrival_radius = float(arrival_raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(arrival_radius) or arrival_radius <= 0.0:
            return None

        dynamics = {
            "max_accel_m_s2": max_accel,
            "max_decel_m_s2": max_decel,
            "arrival_radius_m": arrival_radius,
        }
        max_speed_raw = dynamics_spec.get("max_speed_m_s")
        if max_speed_raw is not None:
            try:
                max_speed = float(max_speed_raw)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(max_speed) or max_speed <= 0.0:
                return None
            dynamics["max_speed_m_s"] = max_speed
        return dynamics

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return False

    def _apply_billboard_planar_movement(
        self,
        movement: Optional[Dict[str, Any]],
        start_planar: np.ndarray,
        frame_id: int,
        *,
        target_idx: int,
        y_value: float,
    ) -> np.ndarray:
        if movement is None:
            return start_planar

        movement_type = movement.get("type")
        if movement_type == "circle":
            radius = float(movement.get("radius", 0.0))
            speed = float(movement.get("speed", 0.0))
            if radius <= 1e-6 or speed <= 0.0:
                return start_planar

            phase = float(movement.get("phase", 0.0))
            angle = phase + frame_id * speed
            base_cos = math.cos(phase)
            base_sin = math.sin(phase)
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)
            offset_x = (cos_angle - base_cos) * radius
            offset_z = (sin_angle - base_sin) * radius
            exact_planar = start_planar + np.array((offset_x, offset_z), dtype=np.float32)
            dynamics = movement.get("dynamics")
            if not isinstance(dynamics, dict):
                return exact_planar

            centre = self._apply_billboard_dynamic_reference_movement(
                movement,
                frame_id,
                target_idx=target_idx,
                initial_position=np.array(
                    (float(start_planar[0]), float(y_value), float(start_planar[1])),
                    dtype=np.float32,
                ),
            )
            if centre is None:
                return exact_planar
            return np.array((float(centre[0]), float(centre[2])), dtype=np.float32)

        return start_planar

    def _apply_billboard_path_movement(
        self,
        movement: Dict[str, Any],
        frame_id: int,
        *,
        target_idx: int,
    ) -> Optional[np.ndarray]:
        points = movement.get("points")
        segment_lengths = movement.get("segment_lengths")
        segment_ends = movement.get("segment_ends")
        if not isinstance(points, np.ndarray):
            return None
        if not isinstance(segment_lengths, np.ndarray):
            return None
        if not isinstance(segment_ends, np.ndarray):
            return None
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 3:
            return None
        if segment_lengths.ndim != 1 or segment_ends.ndim != 1:
            return None
        if segment_lengths.shape[0] != points.shape[0] or segment_ends.shape[0] != points.shape[0]:
            return None

        try:
            speed_m_s = float(movement.get("speed_m_s", 0.0))
            loop_length = float(movement.get("loop_length", 0.0))
        except (TypeError, ValueError):
            return None

        if not math.isfinite(speed_m_s) or speed_m_s <= 0.0:
            return None
        if not math.isfinite(loop_length) or loop_length <= 1e-6:
            return None

        dynamics = movement.get("dynamics")
        if isinstance(dynamics, dict):
            return self._apply_billboard_dynamic_path_movement(
                movement,
                frame_id,
                target_idx=target_idx,
            )

        step_distance = speed_m_s / self._fps_hz
        if not math.isfinite(step_distance) or step_distance <= 0.0:
            return None

        travelled = max(int(frame_id) - 1, 0) * step_distance
        wrapped_distance = math.fmod(travelled, loop_length)
        if wrapped_distance < 0.0:
            wrapped_distance += loop_length

        segment_start = 0.0
        last_idx = int(segment_ends.shape[0] - 1)
        for idx in range(segment_ends.shape[0]):
            segment_end = float(segment_ends[idx])
            segment_length = float(segment_lengths[idx])
            if idx < last_idx and wrapped_distance > segment_end:
                segment_start = segment_end
                continue
            if segment_length <= 1e-6:
                segment_start = segment_end
                continue

            ratio = (wrapped_distance - segment_start) / segment_length
            ratio = _clamp(float(ratio), 0.0, 1.0)
            start_point = np.asarray(points[idx, :3], dtype=np.float32)
            end_point = np.asarray(points[(idx + 1) % points.shape[0], :3], dtype=np.float32)
            centre = start_point + (end_point - start_point) * ratio
            if not np.all(np.isfinite(centre)):
                return None
            return centre

        fallback = np.asarray(points[0, :3], dtype=np.float32)
        if not np.all(np.isfinite(fallback)):
            return None
        return fallback

    def _apply_billboard_dynamic_path_movement(
        self,
        movement: Dict[str, Any],
        frame_id: int,
        *,
        target_idx: int,
    ) -> Optional[np.ndarray]:
        points = movement.get("points")
        dynamics = movement.get("dynamics")
        if not isinstance(points, np.ndarray):
            return None
        if not isinstance(dynamics, dict):
            return None
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 3:
            return None

        try:
            speed_m_s = float(movement.get("speed_m_s", 0.0))
        except (TypeError, ValueError):
            return None
        values = self._movement_dynamics_values(dynamics, default_max_speed=speed_m_s)
        if values is None:
            return None
        max_speed, max_accel, max_decel, arrival_radius = values

        dt = 1.0 / self._fps_hz
        if not math.isfinite(dt) or dt <= 0.0:
            return None

        def _step(state: Dict[str, Any], _: int) -> None:
            self._integrate_dynamic_path_step(
                state,
                points,
                dt=dt,
                speed_m_s=max_speed,
                max_accel=max_accel,
                max_decel=max_decel,
                arrival_radius=arrival_radius,
            )

        return self._apply_billboard_dynamic_motion(
            target_idx=target_idx,
            frame_id=frame_id,
            signature=self._movement_dynamics_signature("path", movement),
            initial_position=np.asarray(points[0, :3], dtype=np.float32),
            step_fn=_step,
            initial_extra={"waypoint_idx": 1},
        )

    def _apply_billboard_dynamic_reference_movement(
        self,
        movement: Dict[str, Any],
        frame_id: int,
        *,
        target_idx: int,
        initial_position: np.ndarray,
    ) -> Optional[np.ndarray]:
        dynamics = movement.get("dynamics")
        if not isinstance(dynamics, dict):
            return None
        try:
            radius = float(movement.get("radius", 0.0))
            speed = float(movement.get("speed", 0.0))
        except (TypeError, ValueError):
            return None
        default_max_speed = abs(radius * speed * self._fps_hz)
        values = self._movement_dynamics_values(
            dynamics,
            default_max_speed=default_max_speed,
        )
        if values is None:
            return None
        max_speed, max_accel, max_decel, _arrival_radius = values

        dt = 1.0 / self._fps_hz
        if not math.isfinite(dt) or dt <= 0.0:
            return None

        def _step(state: Dict[str, Any], next_frame_id: int) -> None:
            reference = self._circle_reference_position(
                movement,
                initial_position,
                next_frame_id,
            )
            self._integrate_dynamic_reference_step(
                state,
                reference,
                dt=dt,
                max_speed_m_s=max_speed,
                max_accel=max_accel,
                max_decel=max_decel,
            )

        return self._apply_billboard_dynamic_motion(
            target_idx=target_idx,
            frame_id=frame_id,
            signature=self._movement_dynamics_signature(
                "circle",
                movement,
                initial_position=initial_position,
            ),
            initial_position=initial_position,
            step_fn=_step,
        )

    def _apply_billboard_dynamic_motion(
        self,
        *,
        target_idx: int,
        frame_id: int,
        signature: Tuple[Any, ...],
        initial_position: np.ndarray,
        step_fn: Any,
        initial_extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[np.ndarray]:
        if not np.all(np.isfinite(initial_position)):
            return None

        state = self._billboard_motion_states.get(target_idx)
        requested_frame = max(int(frame_id), 1)
        if (
            state is None
            or state.get("signature") != signature
            or int(state.get("last_frame_id", 0)) >= requested_frame
        ):
            state = {
                "position": np.asarray(initial_position[:3], dtype=np.float32).copy(),
                "velocity": np.zeros(3, dtype=np.float32),
                "last_frame_id": 1,
                "signature": signature,
            }
            if initial_extra:
                state.update(initial_extra)
            self._billboard_motion_states[target_idx] = state

        while int(state["last_frame_id"]) < requested_frame:
            next_frame = int(state["last_frame_id"]) + 1
            step_fn(state, next_frame)
            state["last_frame_id"] = next_frame

        position = np.asarray(state["position"], dtype=np.float32)
        if not np.all(np.isfinite(position)):
            return None
        return position.copy()

    def _integrate_dynamic_path_step(
        self,
        state: Dict[str, Any],
        points: np.ndarray,
        *,
        dt: float,
        speed_m_s: float,
        max_accel: float,
        max_decel: float,
        arrival_radius: float,
    ) -> None:
        position = np.asarray(state["position"], dtype=np.float32)
        velocity = np.asarray(state["velocity"], dtype=np.float32)
        waypoint_idx = int(state.get("waypoint_idx", 1)) % int(points.shape[0])

        target = np.asarray(points[waypoint_idx, :3], dtype=np.float32)
        delta = target - position
        distance = float(np.linalg.norm(delta))
        if distance <= arrival_radius:
            waypoint_idx = (waypoint_idx + 1) % int(points.shape[0])
            target = np.asarray(points[waypoint_idx, :3], dtype=np.float32)
            delta = target - position
            distance = float(np.linalg.norm(delta))

        if distance <= 1e-6 or not math.isfinite(distance):
            desired_velocity = np.zeros(3, dtype=np.float32)
        else:
            direction = delta / distance
            current_speed = float(np.linalg.norm(velocity))
            braking_distance = (current_speed * current_speed) / (2.0 * max_decel)
            if distance <= max(arrival_radius, braking_distance):
                stopping_distance = max(0.0, distance - arrival_radius)
                desired_speed = min(
                    speed_m_s,
                    math.sqrt(max(0.0, 2.0 * max_decel * stopping_distance)),
                )
            else:
                desired_speed = speed_m_s
            desired_velocity = direction * float(desired_speed)

        velocity = self._apply_velocity_limit(
            velocity,
            desired_velocity,
            dt=dt,
            max_accel=max_accel,
            max_decel=max_decel,
        )

        previous_delta = target - position
        position = position + velocity * dt

        next_delta = target - position
        next_distance = float(np.linalg.norm(next_delta))
        crossed_waypoint = float(np.dot(previous_delta, next_delta)) <= 0.0
        if math.isfinite(next_distance) and (
            next_distance <= arrival_radius or crossed_waypoint
        ):
            waypoint_idx = (waypoint_idx + 1) % int(points.shape[0])

        state["position"] = position.astype(np.float32, copy=False)
        state["velocity"] = velocity.astype(np.float32, copy=False)
        state["waypoint_idx"] = waypoint_idx

    def _integrate_dynamic_reference_step(
        self,
        state: Dict[str, Any],
        reference_position: np.ndarray,
        *,
        dt: float,
        max_speed_m_s: float,
        max_accel: float,
        max_decel: float,
    ) -> None:
        position = np.asarray(state["position"], dtype=np.float32)
        velocity = np.asarray(state["velocity"], dtype=np.float32)
        reference = np.asarray(reference_position[:3], dtype=np.float32)
        delta = reference - position
        if not np.all(np.isfinite(delta)):
            return

        desired_velocity = delta / max(dt, 1e-6)
        desired_speed = float(np.linalg.norm(desired_velocity))
        if desired_speed > max_speed_m_s and desired_speed > 1e-9:
            desired_velocity = desired_velocity * (max_speed_m_s / desired_speed)

        velocity = self._apply_velocity_limit(
            velocity,
            desired_velocity,
            dt=dt,
            max_accel=max_accel,
            max_decel=max_decel,
        )
        position = position + velocity * dt

        state["position"] = position.astype(np.float32, copy=False)
        state["velocity"] = velocity.astype(np.float32, copy=False)

    @staticmethod
    def _apply_velocity_limit(
        velocity: np.ndarray,
        desired_velocity: np.ndarray,
        *,
        dt: float,
        max_accel: float,
        max_decel: float,
    ) -> np.ndarray:
        delta_v = desired_velocity - velocity
        delta_v_norm = float(np.linalg.norm(delta_v))
        if delta_v_norm <= 1e-9 or not math.isfinite(delta_v_norm):
            return velocity

        current_speed = float(np.linalg.norm(velocity))
        desired_speed = float(np.linalg.norm(desired_velocity))
        limit = max_decel if desired_speed < current_speed else max_accel
        max_delta = limit * dt
        if delta_v_norm > max_delta:
            return velocity + (delta_v / delta_v_norm) * max_delta
        return desired_velocity.astype(np.float32, copy=False)

    @staticmethod
    def _movement_dynamics_values(
        dynamics: Dict[str, Any],
        *,
        default_max_speed: float,
    ) -> Optional[Tuple[float, float, float, float]]:
        try:
            max_speed = float(dynamics.get("max_speed_m_s", default_max_speed))
            max_accel = float(dynamics.get("max_accel_m_s2", 0.0))
            max_decel = float(dynamics.get("max_decel_m_s2", max_accel))
            arrival_radius = float(dynamics.get("arrival_radius_m", 0.15))
        except (TypeError, ValueError):
            return None
        if not all(
            math.isfinite(v) and v > 0.0
            for v in (max_speed, max_accel, max_decel, arrival_radius)
        ):
            return None
        return (max_speed, max_accel, max_decel, arrival_radius)

    @staticmethod
    def _circle_reference_position(
        movement: Dict[str, Any],
        initial_position: np.ndarray,
        frame_id: int,
    ) -> np.ndarray:
        radius = float(movement.get("radius", 0.0))
        speed = float(movement.get("speed", 0.0))
        phase = float(movement.get("phase", 0.0))
        angle = phase + max(int(frame_id), 0) * speed
        base_cos = math.cos(phase)
        base_sin = math.sin(phase)
        offset_x = (math.cos(angle) - base_cos) * radius
        offset_z = (math.sin(angle) - base_sin) * radius
        return np.array(
            (
                float(initial_position[0]) + offset_x,
                float(initial_position[1]),
                float(initial_position[2]) + offset_z,
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _movement_dynamics_signature(
        movement_type: str,
        movement: Dict[str, Any],
        *,
        initial_position: Optional[np.ndarray] = None,
    ) -> Tuple[Any, ...]:
        points = movement.get("points")
        dynamics = movement.get("dynamics")
        if isinstance(points, np.ndarray):
            points_sig = tuple(
                tuple(float(v) for v in row[:3])
                for row in points.astype(np.float32, copy=False)
            )
        else:
            points_sig = ()
        initial_sig: Tuple[float, ...]
        if initial_position is None:
            initial_sig = ()
        else:
            initial_sig = tuple(float(v) for v in initial_position[:3])
        if isinstance(dynamics, dict):
            dynamics_sig = (
                float(dynamics.get("max_accel_m_s2", 0.0)),
                float(dynamics.get("max_decel_m_s2", 0.0)),
                float(dynamics.get("arrival_radius_m", 0.0)),
                float(dynamics.get("max_speed_m_s", 0.0)),
            )
        else:
            dynamics_sig = ()
        return (
            movement_type,
            float(movement.get("speed_m_s", 0.0)),
            float(movement.get("radius", 0.0)),
            float(movement.get("speed", 0.0)),
            float(movement.get("phase", 0.0)),
            points_sig,
            initial_sig,
            dynamics_sig,
        )

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
