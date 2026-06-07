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
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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


def _positive_finite(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def _slew_rate(
    current_rate: float,
    desired_rate: float,
    accel_limit_rad_s2: Optional[float],
    dt: float,
) -> float:
    accel = _positive_finite(accel_limit_rad_s2)
    if accel is None:
        return float(desired_rate)
    current = float(current_rate) if math.isfinite(float(current_rate)) else 0.0
    desired = float(desired_rate) if math.isfinite(float(desired_rate)) else current
    max_delta = max(0.0, accel * dt)
    return current + _clamp(desired - current, -max_delta, max_delta)

import numpy as np

from .renderers import get_renderer
from .renderers._common import NEAR_CLIP, build_camera
from ._sprites import get_sprite_aspect_ratio


@dataclass
class _PlannerEvalTarget:
    target_id: int
    spawn_position: np.ndarray
    velocity: np.ndarray
    position: np.ndarray
    spawn_time_s: float
    sprite: str
    width: float
    height: float
    color: Tuple[int, int, int]
    aim_dwell_s: float = 0.0
    last_aim_frame_id: Optional[int] = None


class _PlannerEvalScenario:
    """Spawn and score live synthetic threats for planner evaluation."""

    def __init__(
        self,
        scene: Mapping[str, Any],
        *,
        threat_eval: Any = None,
        fps_hz: float,
    ) -> None:
        cfg_raw = scene.get("planner_eval", {})
        cfg = cfg_raw if isinstance(cfg_raw, Mapping) else {}

        self._fps_hz = max(float(fps_hz), 1e-6)
        self._rng = np.random.default_rng(self._coerce_int(cfg.get("seed"), 7))

        self.max_active_targets = max(1, self._coerce_int(cfg.get("max_active_targets"), 6))
        # Spawn placement is intentionally internal: planner-eval targets are
        # sampled from the live camera frustum so they begin visible.
        self.spawn_interval_s = self._coerce_range(
            cfg.get("spawn_interval_s"),
            (1.0, 3.0),
            min_value=1e-3,
        )
        self.spawn_distance_m = self._coerce_range(
            cfg.get("spawn_distance_m"),
            (18.0, 35.0),
            min_value=1e-3,
        )
        self.spawn_arc_deg = self._coerce_range(
            cfg.get("spawn_arc_deg"),
            (-45.0, 45.0),
        )
        self.altitude_m = self._coerce_range(
            cfg.get("altitude_m"),
            (1.5, 4.0),
            min_value=0.0,
        )
        self.speed_m_s = self._coerce_range(
            cfg.get("speed_m_s"),
            (1.0, 4.0),
            min_value=1e-3,
        )
        self.engage_dwell_s = max(
            self._coerce_float(cfg.get("engage_dwell_s"), 0.35),
            1.0 / self._fps_hz,
        )
        self.match_radius_px = max(
            self._coerce_float(cfg.get("match_radius_px"), 40.0),
            1.0,
        )
        self.breach_zone = str(cfg.get("breach_zone", "critical") or "critical").strip()
        if not self.breach_zone:
            self.breach_zone = "critical"

        self.sprite = str(cfg.get("sprite", "drone") or "drone")
        self.width = max(self._coerce_float(cfg.get("width"), 0.4), 1e-3)
        height_default = self.width
        try:
            aspect = get_sprite_aspect_ratio(self.sprite)
        except ValueError:
            aspect = 1.0
        if math.isfinite(aspect) and aspect > 1e-6:
            height_default = self.width / aspect
        self.height = max(self._coerce_float(cfg.get("height"), height_default), 1e-3)
        self.color = self._coerce_color(cfg.get("color", cfg.get("colour")), (32, 32, 32))

        self.asset_position, self.zone_radii = self._resolve_protected_area(scene, threat_eval)
        self.asset_xz = (
            float(self.asset_position[0]),
            float(self.asset_position[2]),
        )
        self.breach_radius_m = self._resolve_breach_radius()
        min_spawn_distance = self.breach_radius_m + 0.5
        if self.spawn_distance_m[0] < min_spawn_distance:
            self.spawn_distance_m = (
                min_spawn_distance,
                max(self.spawn_distance_m[1], min_spawn_distance),
            )

        self.active: list[_PlannerEvalTarget] = []
        self.total_spawned = 0
        self.eliminated_count = 0
        self.breach_count = 0
        self._next_target_id = 1
        self._next_spawn_time_s = 0.0
        self._last_frame_id = 0

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "active": len(self.active),
            "spawned": self.total_spawned,
            "eliminated": self.eliminated_count,
            "breached": self.breach_count,
            "breach_radius_m": self.breach_radius_m,
        }

    def describe_targets(
        self,
        frame_id: int,
        *,
        spawn_camera: Optional[Dict[str, Any]] = None,
    ) -> list[Dict[str, Any]]:
        self.update(frame_id, spawn_camera=spawn_camera)
        targets: list[Dict[str, Any]] = []
        for target in self.active:
            targets.append(
                {
                    "type": "target",
                    "target_id": target.target_id,
                    "centre": (
                        float(target.position[0]),
                        float(target.position[1]),
                        float(target.position[2]),
                    ),
                    "size": (float(target.width), float(target.height)),
                    "sprite": target.sprite,
                    "color": target.color,
                }
            )
        return targets

    def update(
        self,
        frame_id: int,
        *,
        spawn_camera: Optional[Dict[str, Any]] = None,
    ) -> None:
        requested_frame = max(int(frame_id), 1)
        if requested_frame < self._last_frame_id:
            return

        time_s = self._time_for_frame(requested_frame)
        while time_s + 1e-9 >= self._next_spawn_time_s:
            if len(self.active) < self.max_active_targets:
                self._spawn_target(
                    self._next_spawn_time_s,
                    camera=spawn_camera,
                )
            self._schedule_next_spawn(self._next_spawn_time_s)

        survivors: list[_PlannerEvalTarget] = []
        for target in self.active:
            elapsed_s = max(0.0, time_s - float(target.spawn_time_s))
            previous_position = np.asarray(target.position, dtype=np.float32).copy()
            next_position = (
                target.spawn_position + target.velocity * elapsed_s
            ).astype(np.float32, copy=False)
            target.position = next_position
            if (
                self._planar_distance_to_asset(target.position) <= self.breach_radius_m
                or self._segment_planar_distance_to_asset(
                    previous_position,
                    target.position,
                )
                <= self.breach_radius_m
            ):
                self.breach_count += 1
                continue
            survivors.append(target)
        self.active = survivors
        self._last_frame_id = max(self._last_frame_id, requested_frame)

    def ingest_aim_feedback(
        self,
        *,
        target_id: Optional[int],
        frame_id: int,
        aimed: bool,
    ) -> None:
        if target_id is None:
            return
        target = next((item for item in self.active if item.target_id == target_id), None)
        if target is None:
            return

        if not aimed:
            target.aim_dwell_s = 0.0
            target.last_aim_frame_id = None
            return

        frame_id = max(int(frame_id), 1)
        if target.last_aim_frame_id is None or frame_id <= target.last_aim_frame_id:
            dwell_step_s = 1.0 / self._fps_hz
        else:
            dwell_step_s = (frame_id - target.last_aim_frame_id) / self._fps_hz
        target.aim_dwell_s += max(dwell_step_s, 0.0)
        target.last_aim_frame_id = frame_id

        if target.aim_dwell_s + 1e-9 >= self.engage_dwell_s:
            self.active = [item for item in self.active if item.target_id != target_id]
            self.eliminated_count += 1

    def nearest_projected_target(
        self,
        uv: Tuple[float, float],
        projected_targets: Sequence[Tuple[int, Tuple[float, float]]],
    ) -> Optional[int]:
        best_id: Optional[int] = None
        best_dist = self.match_radius_px
        u, v = float(uv[0]), float(uv[1])
        for target_id, projected_uv in projected_targets:
            du = float(projected_uv[0]) - u
            dv = float(projected_uv[1]) - v
            dist = math.hypot(du, dv)
            if dist <= best_dist:
                best_dist = dist
                best_id = int(target_id)
        return best_id

    def _spawn_target(
        self,
        spawn_time_s: float,
        *,
        camera: Optional[Dict[str, Any]] = None,
    ) -> None:
        distance = self._rng.uniform(self.spawn_distance_m[0], self.spawn_distance_m[1])
        speed = self._rng.uniform(self.speed_m_s[0], self.speed_m_s[1])

        asset_x, asset_z = self.asset_xz
        spawn_position = self._sample_camera_visible_position(camera, float(distance))
        if spawn_position is None:
            angle_deg = self._rng.uniform(self.spawn_arc_deg[0], self.spawn_arc_deg[1])
            angle_rad = math.radians(float(angle_deg))
            altitude = self._rng.uniform(self.altitude_m[0], self.altitude_m[1])
            spawn_x = asset_x + math.sin(angle_rad) * float(distance)
            spawn_z = asset_z - math.cos(angle_rad) * float(distance)
            spawn_position = np.array((spawn_x, altitude, spawn_z), dtype=np.float32)

        travel_delta = self.asset_position - np.asarray(spawn_position, dtype=np.float32)
        travel_norm = float(np.linalg.norm(travel_delta))
        if travel_norm <= 1e-6 or not math.isfinite(travel_norm):
            travel_direction = np.array((0.0, 0.0, 1.0), dtype=np.float32)
        else:
            travel_direction = travel_delta / travel_norm
        velocity = (travel_direction * float(speed)).astype(np.float32, copy=False)

        target = _PlannerEvalTarget(
            target_id=self._next_target_id,
            spawn_position=spawn_position,
            velocity=velocity,
            position=spawn_position.copy(),
            spawn_time_s=float(spawn_time_s),
            sprite=self.sprite,
            width=self.width,
            height=self.height,
            color=self.color,
        )
        self._next_target_id += 1
        self.total_spawned += 1
        self.active.append(target)

    def _sample_camera_visible_position(
        self,
        camera: Optional[Dict[str, Any]],
        distance_m: float,
    ) -> Optional[np.ndarray]:
        if camera is None:
            return None
        try:
            position = np.asarray(camera["position"], dtype=np.float32)
            right = np.asarray(camera["right"], dtype=np.float32)
            up = np.asarray(camera["up"], dtype=np.float32)
            forward = np.asarray(camera["forward"], dtype=np.float32)
            fov_y = float(camera["fov_y"])
            aspect = float(camera["aspect"])
        except (KeyError, TypeError, ValueError):
            return None

        if not (
            np.all(np.isfinite(position))
            and np.all(np.isfinite(right))
            and np.all(np.isfinite(up))
            and np.all(np.isfinite(forward))
            and math.isfinite(fov_y)
            and math.isfinite(aspect)
            and math.isfinite(distance_m)
        ):
            return None
        if distance_m <= NEAR_CLIP or aspect <= 0.0:
            return None

        tan_half_y = math.tan(math.radians(fov_y) * 0.5)
        if not math.isfinite(tan_half_y) or tan_half_y <= 0.0:
            return None

        min_asset_clearance = max(self.breach_radius_m + 0.5, 0.5)
        for _ in range(16):
            x_ndc = float(self._rng.uniform(-0.72, 0.72))
            y_ndc = float(self._rng.uniform(0.02, 0.42))
            x_cam = x_ndc * distance_m * tan_half_y * aspect
            y_cam = y_ndc * distance_m * tan_half_y
            candidate = position + right * x_cam + up * y_cam + forward * distance_m
            if not np.all(np.isfinite(candidate)):
                continue
            if float(candidate[1]) < 0.35:
                continue
            if self._planar_distance_to_asset(candidate) <= min_asset_clearance:
                continue
            return candidate.astype(np.float32, copy=False)

        return None

    def _schedule_next_spawn(self, from_time_s: float) -> None:
        interval = self._rng.uniform(self.spawn_interval_s[0], self.spawn_interval_s[1])
        self._next_spawn_time_s = float(from_time_s) + max(float(interval), 1e-3)

    def _planar_distance_to_asset(self, position: np.ndarray) -> float:
        dx = float(position[0]) - float(self.asset_xz[0])
        dz = float(position[2]) - float(self.asset_xz[1])
        return math.hypot(dx, dz)

    def _segment_planar_distance_to_asset(
        self,
        start: np.ndarray,
        end: np.ndarray,
    ) -> float:
        asset = np.array(self.asset_xz, dtype=np.float32)
        start_xz = np.array((float(start[0]), float(start[2])), dtype=np.float32)
        end_xz = np.array((float(end[0]), float(end[2])), dtype=np.float32)
        segment = end_xz - start_xz
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-12 or not math.isfinite(length_sq):
            return self._planar_distance_to_asset(end)
        t = float(np.dot(asset - start_xz, segment) / length_sq)
        t = _clamp(t, 0.0, 1.0)
        closest = start_xz + segment * t
        delta = closest - asset
        return float(np.linalg.norm(delta))

    def _resolve_breach_radius(self) -> float:
        radii = {
            str(name).strip().lower(): float(radius)
            for name, radius in self.zone_radii.items()
            if math.isfinite(float(radius)) and float(radius) > 0.0
        }
        requested = self.breach_zone.strip().lower()
        if requested in radii:
            return radii[requested]
        if radii:
            return min(radii.values())
        return 2.0

    def _time_for_frame(self, frame_id: int) -> float:
        return max(int(frame_id) - 1, 0) / self._fps_hz

    @classmethod
    def _resolve_protected_area(
        cls,
        scene: Mapping[str, Any],
        threat_eval: Any,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if threat_eval is not None:
            enabled = bool(getattr(threat_eval, "enabled", False))
            zone_radii = getattr(threat_eval, "zone_radii", {}) or {}
            if enabled and zone_radii:
                asset_world = getattr(threat_eval, "asset_world", None)
                if asset_world is None:
                    asset_xy = getattr(threat_eval, "asset_xy", None)
                    asset_world = cls._world_point_from_planar(asset_xy)
                asset = cls._coerce_world_point(asset_world, default=(0.0, 0.0, 0.0))
                zones = cls._coerce_zone_radii(zone_radii)
                if zones:
                    return asset, zones

        asset = np.array((0.0, 0.0, 0.0), dtype=np.float32)
        asset_spec = scene.get("defended_asset")
        if isinstance(asset_spec, Mapping):
            asset = cls._coerce_world_point(
                asset_spec.get("position_world"),
                default=asset,
            )

        zones: Dict[str, float] = {}
        legacy_zones = scene.get("threat_eval_zones")
        if isinstance(legacy_zones, Mapping) and cls._coerce_bool(
            legacy_zones.get("enabled", True)
        ):
            zone_specs = legacy_zones.get("zones", {})
            if isinstance(zone_specs, Mapping):
                zones = cls._coerce_zone_specs(zone_specs)
        return asset, zones

    @staticmethod
    def _coerce_world_point(
        value: Any,
        *,
        default: Tuple[float, float, float] | np.ndarray,
    ) -> np.ndarray:
        default_arr = np.asarray(default, dtype=np.float32).reshape(3)
        try:
            values = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return default_arr.copy()
        if values.size >= 3:
            x = float(values[0])
            y = float(values[1])
            z = float(values[2])
        elif values.size >= 2:
            x = float(values[0])
            y = 0.0
            z = float(values[1])
        else:
            return default_arr.copy()
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return default_arr.copy()
        return np.array((x, y, z), dtype=np.float32)

    @classmethod
    def _world_point_from_planar(cls, value: Any) -> np.ndarray:
        return cls._coerce_world_point(value, default=(0.0, 0.0, 0.0))

    @staticmethod
    def _coerce_planar_point(
        value: Any,
        *,
        default: Tuple[float, float],
        prefer_third: bool = False,
    ) -> Tuple[float, float]:
        try:
            values = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return default
        if values.size < 2:
            return default
        second_idx = 2 if prefer_third and values.size >= 3 else 1
        x = float(values[0])
        z = float(values[second_idx])
        if not (math.isfinite(x) and math.isfinite(z)):
            return default
        return (x, z)

    @classmethod
    def _coerce_zone_specs(cls, zone_specs: Mapping[str, Any]) -> Dict[str, float]:
        radii: Dict[str, float] = {}
        for name, spec in zone_specs.items():
            if isinstance(spec, Mapping):
                radius = spec.get("radius_m")
            else:
                radius = spec
            value = cls._coerce_float(radius, math.nan)
            if math.isfinite(value) and value >= 0.0:
                radii[str(name)] = value
        return radii

    @classmethod
    def _coerce_zone_radii(cls, zone_radii: Mapping[str, Any]) -> Dict[str, float]:
        radii: Dict[str, float] = {}
        for name, radius in zone_radii.items():
            value = cls._coerce_float(radius, math.nan)
            if math.isfinite(value) and value >= 0.0:
                radii[str(name)] = value
        return radii

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(parsed):
            return float(default)
        return parsed

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return False

    @classmethod
    def _coerce_range(
        cls,
        value: Any,
        default: Tuple[float, float],
        *,
        min_value: Optional[float] = None,
    ) -> Tuple[float, float]:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            lo = cls._coerce_float(value[0], default[0])
            hi = cls._coerce_float(value[1], default[1])
        elif value is not None:
            lo = hi = cls._coerce_float(value, default[0])
        else:
            lo, hi = default
        if min_value is not None:
            lo = max(lo, min_value)
            hi = max(hi, min_value)
        if hi < lo:
            lo, hi = hi, lo
        return (float(lo), float(hi))

    @staticmethod
    def _coerce_color(value: Any, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
        try:
            values = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return default
        if values.size < 3:
            return default
        return tuple(int(max(0, min(255, round(float(v))))) for v in values[:3])


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
        fps_hz: float = 30.0,
        threat_eval: Any = None,
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
        self._planner_eval: Optional[_PlannerEvalScenario] = None
        if isinstance(scene, dict):
            scene_mode = str(scene.get("mode", "") or "").strip().lower()
            if scene_mode == "planner_eval":
                self._planner_eval = _PlannerEvalScenario(
                    scene,
                    threat_eval=threat_eval,
                    fps_hz=self._fps_hz,
                )
                self._billboard_specs = ()
            if "buildings" in scene:
                self._building_specs = self._coerce_scene_specs(scene.get("buildings"))
            if self._planner_eval is None and "targets" in scene:
                self._billboard_specs = self._coerce_scene_specs(scene.get("targets"))
            elif self._planner_eval is None and "billboards" in scene:
                self._billboard_specs = self._coerce_scene_specs(scene.get("billboards"))
            if "cubes" in scene:
                self._cube_specs = self._coerce_scene_specs(scene.get("cubes"))
                self._use_scene_cubes = True
            if "meshes" in scene:
                self._mesh_specs = self._coerce_scene_specs(scene.get("meshes"))

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
    def apply_control_rates(
        self,
        pan_rate: float,
        tilt_rate: float,
        dt: float,
        *,
        pan_accel_rad_s2: Optional[float] = None,
        tilt_accel_rad_s2: Optional[float] = None,
    ) -> None:
        """Integrate commanded pan/tilt rates, optionally slew-limited by acceleration."""

        if dt <= 0.0 or not math.isfinite(dt):
            return

        prev_pan_rate = self._pan_rate
        prev_tilt_rate = self._tilt_rate
        next_pan_rate = _slew_rate(prev_pan_rate, pan_rate, pan_accel_rad_s2, dt)
        next_tilt_rate = _slew_rate(prev_tilt_rate, tilt_rate, tilt_accel_rad_s2, dt)

        pan_step_rate = (
            0.5 * (prev_pan_rate + next_pan_rate)
            if _positive_finite(pan_accel_rad_s2) is not None
            else next_pan_rate
        )
        tilt_step_rate = (
            0.5 * (prev_tilt_rate + next_tilt_rate)
            if _positive_finite(tilt_accel_rad_s2) is not None
            else next_tilt_rate
        )

        self._pan_rate = next_pan_rate
        self._tilt_rate = next_tilt_rate

        self._pan_rad = _wrap_angle(self._pan_rad + pan_step_rate * dt)
        next_tilt = self._tilt_rad + tilt_step_rate * dt
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

    def planner_eval_enabled(self) -> bool:
        """Return whether the live planner-evaluation scenario is active."""

        return self._planner_eval is not None

    def get_planner_eval_stats(self) -> Optional[Dict[str, Any]]:
        """Return live planner-eval counters when the mode is active."""

        if self._planner_eval is None:
            return None
        return dict(self._planner_eval.stats)

    def apply_detection_feedback(self, msg: Any) -> None:
        """Apply selected-target feedback from a detection message.

        The simulator intentionally consumes the existing DetectionMsg shape
        without importing schema details here: it needs only target_idx,
        laser_on_target, image size, and boxes.
        """

        if self._planner_eval is None:
            return

        target_idx = getattr(msg, "target_idx", None)
        boxes = getattr(msg, "boxes", None)
        if target_idx is None or boxes is None:
            return
        try:
            target_idx_int = int(target_idx)
        except (TypeError, ValueError):
            return
        if target_idx_int < 0 or target_idx_int >= len(boxes):
            return

        try:
            img_w = float(getattr(msg, "img_w"))
            img_h = float(getattr(msg, "img_h"))
        except (TypeError, ValueError):
            return
        if not (math.isfinite(img_w) and math.isfinite(img_h)) or img_w <= 0.0 or img_h <= 0.0:
            return

        box = boxes[target_idx_int]
        try:
            target_u = (float(box.x) + float(box.w) * 0.5) * img_w
            target_v = (float(box.y) + float(box.h) * 0.5) * img_h
        except (AttributeError, TypeError, ValueError):
            return
        if not (math.isfinite(target_u) and math.isfinite(target_v)):
            return

        frame_id = int(getattr(msg, "frame_id", self._frame_id) or self._frame_id or 1)
        projected = self._project_planner_eval_targets(frame_id)
        matched_id = self._planner_eval.nearest_projected_target(
            (target_u, target_v),
            projected,
        )
        if matched_id is None:
            return

        laser_on_target = getattr(msg, "laser_on_target", None)
        if laser_on_target is True:
            aimed = True
        elif laser_on_target is False:
            aimed = False
        else:
            aimed = math.hypot(target_u - img_w * 0.5, target_v - img_h * 0.5) <= (
                self._planner_eval.match_radius_px
            )

        self._planner_eval.ingest_aim_feedback(
            target_id=matched_id,
            frame_id=frame_id,
            aimed=aimed,
        )

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
        objects.extend(self._describe_meshes())

        camera_position, orientation = self._camera_pose_for_frame(frame_id)

        # Only include scene-defined cubes when debug mode is active so that
        # cubes are not rendered during normal (non-debug) operation.
        if self._debug_mode:
            if self._use_scene_cubes:
                objects.extend(self._describe_cubes())

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

    def _camera_pose_for_frame(
        self,
        frame_id: int,
    ) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
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
                camera_position,
                self._camera_target,
            )
            return camera_position, orientation

        return (
            self._camera_fixed_position.copy(),
            {
                "yaw": math.degrees(self._pan_rad),
                "pitch": math.degrees(self._tilt_rad),
                "roll": math.degrees(self._roll_rad),
            },
        )

    def _camera_info_for_frame(self, frame_id: int) -> Dict[str, Any]:
        camera_position, orientation = self._camera_pose_for_frame(frame_id)
        camera_info: Dict[str, Any] = {
            "position": camera_position,
            "target": self._camera_target.copy(),
            "up": self.world_up.copy(),
            "fov_y": self._camera_fov_y,
        }
        if orientation is not None:
            camera_info["orientation"] = orientation
        return camera_info

    def _project_planner_eval_targets(
        self,
        frame_id: int,
    ) -> list[Tuple[int, Tuple[float, float]]]:
        if self._planner_eval is None:
            return []
        camera = build_camera(
            self._camera_info_for_frame(frame_id),
            context=self,
            width=self.width,
            height=self.height,
        )
        if camera is None:
            return []
        self._planner_eval.update(frame_id, spawn_camera=camera)

        projected: list[Tuple[int, Tuple[float, float]]] = []
        for target in self._planner_eval.active:
            uv = self._project_world_point(camera, target.position)
            if uv is not None:
                projected.append((int(target.target_id), uv))
        return projected

    def _project_world_point(
        self,
        camera: Dict[str, Any],
        point: Sequence[float],
    ) -> Optional[Tuple[float, float]]:
        rel = np.asarray(point, dtype=np.float32) - np.asarray(
            camera["position"],
            dtype=np.float32,
        )
        x = float(np.dot(rel, camera["right"]))
        y = float(np.dot(rel, camera["up"]))
        z = float(np.dot(rel, camera["forward"]))
        if z < NEAR_CLIP:
            return None

        f = 1.0 / math.tan(math.radians(float(camera["fov_y"])) * 0.5)
        x_ndc = (x / z) * (f / float(camera["aspect"]))
        y_ndc = (y / z) * f
        if not (math.isfinite(x_ndc) and math.isfinite(y_ndc)):
            return None

        x_px = (x_ndc + 1.0) * 0.5 * (self.width - 1)
        y_px = (1.0 - (y_ndc + 1.0) * 0.5) * (self.height - 1)
        return (float(x_px), float(y_px))

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
        if self._planner_eval is not None:
            spawn_camera = build_camera(
                self._camera_info_for_frame(frame_id),
                context=self,
                width=self.width,
                height=self.height,
            )
            return self._planner_eval.describe_targets(
                frame_id,
                spawn_camera=spawn_camera,
            )

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
