"""Live evaluation target manager for :mod:`pc.sim_camera`.

The manager owns a small deterministic scenario loop: spawn targets in the
initial camera view, move them toward a defended asset, and remove them when the
control loop keeps them on target long enough.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._sprites import get_sprite_aspect_ratio
from .renderers._common import build_camera


def _coerce_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _coerce_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _coerce_xy(value: Any, default: Tuple[float, float]) -> Tuple[float, float]:
    try:
        values = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return default
    if values.size < 2:
        return default
    x = float(values[0])
    y = float(values[1])
    if not math.isfinite(x) or not math.isfinite(y):
        return default
    return (x, y)


def _coerce_size(value: Any, default: Tuple[Optional[float], Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    if value is None:
        return default
    try:
        values = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return default
    if values.size == 0:
        return default
    if values.size == 1:
        size = float(values[0])
        if math.isfinite(size) and size > 0.0:
            return (size, size)
        return default
    width = float(values[0])
    height = float(values[1])
    if math.isfinite(width) and math.isfinite(height) and width > 0.0 and height > 0.0:
        return (width, height)
    return default


@dataclass(frozen=True)
class EvaluationClassSpec:
    name: str
    sprite: str
    width: Optional[float]
    height: Optional[float]
    damage_weight: float
    ground_y: float
    color: Optional[Any]
    orientation: Optional[Any]


@dataclass
class EvaluationTarget:
    target_id: int
    class_spec: EvaluationClassSpec
    position: np.ndarray
    velocity: np.ndarray
    lock_s: float = 0.0


class SimEvaluationManager:
    """Manage visual-only evaluation targets for the simulator."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        scene: Optional[Mapping[str, Any]],
        threat_eval: Optional[Mapping[str, Any]],
        width: int,
        height: int,
        fps_hz: float,
        camera_state: Mapping[str, Any],
        context: Any,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps_hz = max(float(fps_hz), 1e-6)
        self._rng = np.random.default_rng(_coerce_int(config.get("seed", 1), 1))
        self._max_active_targets = max(1, _coerce_int(config.get("max_active_targets", 4), 4))
        self._spawn_interval_s = max(0.0, _coerce_float(config.get("spawn_interval_s", 1.5), 1.5))
        self._spawn_radius_m = max(1.0, _coerce_float(config.get("spawn_radius_m", 18.0), 18.0))
        self._target_speed_m_s = max(0.01, _coerce_float(config.get("target_speed_m_s", 3.0), 3.0))
        self.lock_dwell_s = max(0.0, _coerce_float(config.get("lock_dwell_s", 0.35), 0.35))
        self.lock_tolerance_px = max(0.0, _coerce_float(config.get("lock_tolerance_px", 24.0), 24.0))
        self._asset_xy = self._resolve_asset_xy(scene, threat_eval)
        self._critical_radius_m = self._resolve_critical_radius(threat_eval, scene)
        self._class_specs = self._parse_classes(config.get("classes"))
        self._targets: Dict[int, EvaluationTarget] = {}
        self._next_target_id = 1
        self._last_frame_id = 0
        self._next_spawn_time_s = 0.0
        self._last_lock_target_id: Optional[int] = None
        self._last_lock_ok = False
        self.spawned_count = 0
        self.neutralized_count = 0
        self.breakthrough_count = 0

        self._initial_camera = build_camera(
            dict(camera_state),
            context=context,
            width=self.width,
            height=self.height,
        )
        if self._initial_camera is None:
            raise ValueError("evaluation mode requires a valid initial camera")

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        scene: Optional[Mapping[str, Any]],
        threat_eval: Optional[Mapping[str, Any]],
        width: int,
        height: int,
        fps_hz: float,
        camera_state: Mapping[str, Any],
        context: Any,
    ) -> Optional["SimEvaluationManager"]:
        if not isinstance(config, Mapping):
            return None
        if not cls.enabled(config):
            return None
        return cls(
            config,
            scene=scene,
            threat_eval=threat_eval,
            width=width,
            height=height,
            fps_hz=fps_hz,
            camera_state=camera_state,
            context=context,
        )

    @staticmethod
    def enabled(config: Any) -> bool:
        if not isinstance(config, Mapping):
            return False
        value = config.get("enabled", False)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return False

    def update(self, frame_id: int) -> None:
        requested_frame = max(int(frame_id), 1)
        if requested_frame <= self._last_frame_id:
            return
        while self._last_frame_id < requested_frame:
            self._last_frame_id += 1
            now_s = (self._last_frame_id - 1) / self.fps_hz
            self._spawn_due_targets(now_s)
            self._advance_targets(1.0 / self.fps_hz)

    def describe_targets(self, frame_id: int) -> list[Dict[str, Any]]:
        self.update(frame_id)
        entries: list[Dict[str, Any]] = []
        for target in sorted(self._targets.values(), key=lambda item: item.target_id):
            spec = target.class_spec
            entry: Dict[str, Any] = {
                "type": "target",
                "target_id": int(target.target_id),
                "centre": (
                    float(target.position[0]),
                    float(target.position[1]),
                    float(target.position[2]),
                ),
                "sprite": spec.sprite,
                "damage_weight": float(spec.damage_weight),
            }
            entry["size"] = self._resolved_target_size(spec)
            if spec.color is not None:
                entry["color"] = spec.color
            if spec.orientation is not None:
                entry["orientation"] = spec.orientation
            entries.append(entry)
        return entries

    @staticmethod
    def _resolved_target_size(spec: EvaluationClassSpec) -> Tuple[float, float]:
        width = spec.width
        height = spec.height
        if width is not None and height is not None:
            return (float(width), float(height))
        try:
            aspect_ratio = get_sprite_aspect_ratio(spec.sprite)
        except ValueError:
            aspect_ratio = 1.0
        if not math.isfinite(aspect_ratio) or aspect_ratio <= 0.0:
            aspect_ratio = 1.0
        if width is not None:
            return (float(width), float(width) / aspect_ratio)
        if height is not None:
            return (float(height) * aspect_ratio, float(height))
        return (0.4, 0.4)

    def active_targets(self) -> Tuple[EvaluationTarget, ...]:
        return tuple(sorted(self._targets.values(), key=lambda item: item.target_id))

    def apply_control_cmd(
        self,
        cmd: Any,
        *,
        camera_state: Mapping[str, Any],
        context: Any,
    ) -> None:
        target_ok = bool(getattr(cmd, "target_ok", False))
        lock_ok = False
        if target_ok:
            laser_on_target = getattr(cmd, "laser_on_target", None)
            if laser_on_target is not None:
                lock_ok = bool(laser_on_target)
            else:
                err_uv = getattr(cmd, "err_uv", None)
                try:
                    err_x = float(err_uv[0])
                    err_y = float(err_uv[1])
                except (TypeError, ValueError, IndexError):
                    lock_ok = False
                else:
                    lock_ok = math.hypot(err_x, err_y) <= self.lock_tolerance_px

        target_id = None
        if lock_ok:
            target_uv = getattr(cmd, "target_uv", None)
            target_id = self._nearest_target_id(target_uv, camera_state, context)

        self._last_lock_target_id = target_id
        self._last_lock_ok = lock_ok and target_id is not None

    def metrics(self) -> Dict[str, float]:
        return {
            "active": float(len(self._targets)),
            "spawned": float(self.spawned_count),
            "neutralized": float(self.neutralized_count),
            "breakthrough": float(self.breakthrough_count),
        }

    def _spawn_due_targets(self, now_s: float) -> None:
        if self._spawn_interval_s <= 0.0:
            while len(self._targets) < self._max_active_targets:
                target = self._make_target()
                self._targets[target.target_id] = target
                self.spawned_count += 1
            return

        if now_s + 1e-9 < self._next_spawn_time_s:
            return

        if len(self._targets) < self._max_active_targets:
            target = self._make_target()
            self._targets[target.target_id] = target
            self.spawned_count += 1
            self._next_spawn_time_s = now_s + self._spawn_interval_s

    def _advance_targets(self, dt_s: float) -> None:
        remove_ids: list[int] = []
        for target_id, target in self._targets.items():
            target.position = target.position + target.velocity * float(dt_s)
            distance = math.hypot(
                float(target.position[0]) - self._asset_xy[0],
                float(target.position[2]) - self._asset_xy[1],
            )
            if distance <= self._critical_radius_m:
                remove_ids.append(target_id)
                self.breakthrough_count += 1
                continue

            if self._last_lock_ok and target_id == self._last_lock_target_id:
                target.lock_s += dt_s
                if target.lock_s + 1e-9 >= self.lock_dwell_s:
                    remove_ids.append(target_id)
                    self.neutralized_count += 1
            else:
                target.lock_s = 0.0

        for target_id in remove_ids:
            self._targets.pop(target_id, None)
            if self._last_lock_target_id == target_id:
                self._last_lock_target_id = None
                self._last_lock_ok = False

    def _make_target(self) -> EvaluationTarget:
        spec = self._class_specs[int(self._rng.integers(0, len(self._class_specs)))]
        position = self._sample_spawn_position(spec)
        target_xy = np.array((self._asset_xy[0], self._asset_xy[1]), dtype=np.float32)
        pos_xy = np.array((float(position[0]), float(position[2])), dtype=np.float32)
        direction_xy = target_xy - pos_xy
        distance = float(np.linalg.norm(direction_xy))
        if distance <= 1e-6 or not math.isfinite(distance):
            direction_xy = np.array((0.0, 1.0), dtype=np.float32)
        else:
            direction_xy = direction_xy / distance

        velocity = np.array(
            (
                float(direction_xy[0]) * self._target_speed_m_s,
                0.0,
                float(direction_xy[1]) * self._target_speed_m_s,
            ),
            dtype=np.float32,
        )
        target_id = self._next_target_id
        self._next_target_id += 1
        return EvaluationTarget(
            target_id=target_id,
            class_spec=spec,
            position=position,
            velocity=velocity,
        )

    def _sample_spawn_position(self, spec: EvaluationClassSpec) -> np.ndarray:
        radius = self._spawn_radius_m
        asset_x, asset_z = self._asset_xy
        height = spec.height if spec.height is not None else 0.4
        centre_y = spec.ground_y + float(height) * 0.5

        for _ in range(64):
            # Bias to the far side of the defended asset for the default camera
            # while sampling only the visible horizontal cone.
            z_offset = -float(self._rng.uniform(radius * 0.75, radius * 1.05))
            z_depth = max(0.1, -(asset_z + z_offset))
            half_width = z_depth * math.tan(math.radians(60.0) * 0.5) * (self.width / self.height)
            half_width *= 0.78
            x_limit = min(radius * 0.5, max(0.2, half_width))
            x_offset = float(self._rng.uniform(-x_limit, x_limit))
            position = np.array((asset_x + x_offset, centre_y, asset_z + z_offset), dtype=np.float32)
            if self._is_visible_initial(position):
                return position

        return np.array((asset_x, centre_y, asset_z - radius), dtype=np.float32)

    def _is_visible_initial(self, position: Sequence[float]) -> bool:
        projected = self._project_with_camera(self._initial_camera, position)
        if projected is None:
            return False
        x_px, y_px = projected
        margin = 2.0
        return (
            margin <= x_px <= float(self.width - 1) - margin
            and margin <= y_px <= float(self.height - 1) - margin
        )

    def _nearest_target_id(
        self,
        target_uv: Any,
        camera_state: Mapping[str, Any],
        context: Any,
    ) -> Optional[int]:
        if not self._targets:
            return None
        try:
            uv = np.asarray(target_uv, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            uv = np.asarray((), dtype=np.float32)
        if uv.size < 2 or not np.all(np.isfinite(uv[:2])):
            return min(self._targets)

        camera = build_camera(dict(camera_state), context=context, width=self.width, height=self.height)
        if camera is None:
            camera = self._initial_camera
        best_id: Optional[int] = None
        best_dist = math.inf
        for target in self._targets.values():
            projected = self._project_with_camera(camera, target.position)
            if projected is None:
                continue
            dist = math.hypot(float(projected[0]) - float(uv[0]), float(projected[1]) - float(uv[1]))
            if dist < best_dist:
                best_dist = dist
                best_id = target.target_id
        if best_id is not None:
            return best_id
        return min(self._targets)

    def _project_with_camera(
        self,
        camera: Optional[Mapping[str, Any]],
        point: Sequence[float],
    ) -> Optional[Tuple[float, float]]:
        if camera is None:
            return None
        rel = np.asarray(point, dtype=np.float32) - np.asarray(camera["position"], dtype=np.float32)
        x = float(np.dot(rel, camera["right"]))
        y = float(np.dot(rel, camera["up"]))
        z = float(np.dot(rel, camera["forward"]))
        if z <= 0.05:
            return None
        f = 1.0 / math.tan(math.radians(float(camera["fov_y"])) * 0.5)
        x_ndc = (x / z) * (f / float(camera["aspect"]))
        y_ndc = (y / z) * f
        if not math.isfinite(x_ndc) or not math.isfinite(y_ndc):
            return None
        return (
            float((x_ndc + 1.0) * 0.5 * (self.width - 1)),
            float((1.0 - (y_ndc + 1.0) * 0.5) * (self.height - 1)),
        )

    def _parse_classes(self, raw: Any) -> Tuple[EvaluationClassSpec, ...]:
        items: Iterable[Any]
        if isinstance(raw, Mapping):
            items = raw.values()
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            items = raw
        else:
            items = (
                {
                    "name": "drone",
                    "sprite": "drone",
                    "width": 0.4,
                    "damage_weight": 2.0,
                    "ground_y": 2.0,
                    "color": [32, 32, 32],
                },
            )

        specs: list[EvaluationClassSpec] = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name", item.get("class", f"target_{index}")))
            sprite = str(item.get("sprite", name or "drone"))
            width = item.get("width")
            height = item.get("height")
            width_value: Optional[float] = None
            height_value: Optional[float] = None
            if width is not None:
                parsed_width = _coerce_float(width, 0.0)
                if parsed_width > 0.0:
                    width_value = parsed_width
            if height is not None:
                parsed_height = _coerce_float(height, 0.0)
                if parsed_height > 0.0:
                    height_value = parsed_height
            size_width, size_height = _coerce_size(item.get("size"), (width_value, height_value))
            width_value = size_width
            height_value = size_height
            if width_value is None and height_value is None:
                height_value = 0.4
            specs.append(
                EvaluationClassSpec(
                    name=name,
                    sprite=sprite,
                    width=width_value,
                    height=height_value,
                    damage_weight=_coerce_float(item.get("damage_weight"), 1.0),
                    ground_y=_coerce_float(item.get("ground_y"), 0.0),
                    color=item.get("color", item.get("colour")),
                    orientation=item.get("orientation", item.get("sprite_orientation")),
                )
            )
        if specs:
            return tuple(specs)
        return (
            EvaluationClassSpec(
                name="drone",
                sprite="drone",
                width=0.4,
                height=None,
                damage_weight=2.0,
                ground_y=2.0,
                color=(32, 32, 32),
                orientation=None,
            ),
        )

    @staticmethod
    def _resolve_asset_xy(
        scene: Optional[Mapping[str, Any]],
        threat_eval: Optional[Mapping[str, Any]],
    ) -> Tuple[float, float]:
        if isinstance(threat_eval, Mapping):
            defended = threat_eval.get("defended_asset")
            if isinstance(defended, Mapping):
                xy = _coerce_xy(defended.get("position_world"), (math.nan, math.nan))
                if math.isfinite(xy[0]) and math.isfinite(xy[1]):
                    return xy
        if isinstance(scene, Mapping):
            defended = scene.get("defended_asset")
            if isinstance(defended, Mapping):
                xy = _coerce_xy(defended.get("position_world"), (math.nan, math.nan))
                if math.isfinite(xy[0]) and math.isfinite(xy[1]):
                    return xy
        return (0.0, 0.0)

    @staticmethod
    def _resolve_critical_radius(
        threat_eval: Optional[Mapping[str, Any]],
        scene: Optional[Mapping[str, Any]],
    ) -> float:
        for section in (threat_eval, scene.get("threat_eval_zones") if isinstance(scene, Mapping) else None):
            if not isinstance(section, Mapping):
                continue
            zones = section.get("zones")
            if not isinstance(zones, Mapping):
                continue
            critical = zones.get("critical")
            if isinstance(critical, Mapping):
                radius = _coerce_float(critical.get("radius_m"), 0.0)
                if radius > 0.0:
                    return radius
        return 1.0
