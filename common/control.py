"""Shared helpers for loading the control-loop configuration.

This module centralizes parsing of the `control` section in the runtime
configuration so both Jetson and PC components can share consistent defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

from common.camera import CameraIntrinsics, CameraIntrinsicsConfigError
from common.threat_calc import parse_zone_config, validate_asset_position


class ControlConfigError(ValueError):
    """Raised when the control configuration is invalid or incomplete."""


class LaserConfigError(ValueError):
    """Raised when the laser configuration is invalid or incomplete."""


@dataclass(frozen=True)
class AxisPair:
    """Convenience container for paired yaw/pitch values.

    The container is intentionally unit-agnostic so it can represent pixels,
    radians, rates, etc. Helper functions document the expected unit for the
    values they return.
    """

    yaw: float
    pitch: float

    def as_tuple(self) -> Tuple[float, float]:
        return (self.yaw, self.pitch)


@dataclass(frozen=True)
class Vector3:
    """Simple immutable 3D vector."""

    x: float
    y: float
    z: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class LaserRenderConfig:
    """Rendering preferences for the laser overlay."""

    beam_length_m: float
    colour_bgr: Tuple[int, int, int]
    thickness_px: int
    hit_tolerance_px: float


@dataclass(frozen=True)
class LaserMountConfig:
    """Physical mounting parameters for a laser emitter.

    Configuration values are expressed in a right-handed frame with ``+X`` to
    the right, ``+Y`` up, and ``+Z`` forward to match typical rig calibration
    workflows. Internally we continue to operate in the computer-vision frame
    (``+Y`` down), so the loader flips the vertical component when parsing the
    configuration.
    """

    offset_m: Vector3
    dir_cam: Vector3
    render: LaserRenderConfig

    @classmethod
    def from_raw_config(cls, cfg: Mapping[str, Any]) -> "LaserMountConfig":
        section = cfg.get("laser", {}) or {}
        if not isinstance(section, Mapping):
            raise LaserConfigError("config 'laser' section must be a mapping")

        raw_offset = _parse_vector3(section.get("offset_m"), default=(0.0, 0.0, 0.0))
        # Convert config frame (+Y up) to the internal CV frame (+Y down).
        offset = (raw_offset[0], -raw_offset[1], raw_offset[2])

        raw_direction = _parse_vector3(section.get("dir_cam"), default=(0.0, 0.0, 1.0))
        direction = (raw_direction[0], -raw_direction[1], raw_direction[2])
        dir_norm = math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
        if dir_norm <= 0.0:
            raise LaserConfigError("laser.dir_cam must not be the zero vector")
        dir_unit = (direction[0] / dir_norm, direction[1] / dir_norm, direction[2] / dir_norm)

        render_section = section.get("render", {}) or {}
        if not isinstance(render_section, Mapping):
            raise LaserConfigError("laser.render must be a mapping when provided")
        render = _parse_render_config(render_section)

        return cls(
            offset_m=Vector3(*offset),
            dir_cam=Vector3(*dir_unit),
            render=render,
        )


@dataclass(frozen=True)
class LaserAimingControlConfig:
    """Controller-specific parameters for laser-based aiming."""

    tolerance_px: float
    use_range: str
    default_distance_m: float


@dataclass(frozen=True)
class SwarmTimingConfig:
    """Deterministic timing knobs for swarm engagement planning."""

    base_track_lock_s: float = 0.15
    search_track_lock_s: float = 0.25
    recover_track_lock_s: float = 0.40
    low_conf_threshold: float = 0.60
    low_conf_penalty_s: float = 0.20
    min_track_observations: int = 3
    low_continuity_penalty_s: float = 0.08
    missing_range_penalty_s: float = 0.10
    predictive_penalty_s: float = 0.20
    effect_time_s: float = 0.25
    effect_distance_scale_s_per_m: float = 0.01
    confirm_time_s: float = 0.10
    confirm_distance_scale_s_per_m: float = 0.004
    settle_margin_s: float = 0.05


@dataclass(frozen=True)
class SwarmFeatureNormalizationConfig:
    """Feature scaling ranges shared by swarm dataset, training, and runtime."""

    max_distance_m: float = 80.0
    max_closing_speed_m_s: float = 12.0
    max_breakthrough_time_s: float = 40.0
    max_time_to_engage_s: float = 8.0
    max_damage_weight: float = 10.0
    max_angle_rad: float = 0.8
    max_track_observations: float = 10.0
    max_track_age_s: float = 5.0


@dataclass(frozen=True)
class ThreatEvalConfig:
    """Source-agnostic defended-asset and zone config for threat evaluation."""

    enabled: bool = False
    asset_xy: Tuple[float, float] = (0.0, 0.0)
    asset_world: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    zone_radii: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def disabled(cls) -> "ThreatEvalConfig":
        return cls()


@dataclass(frozen=True)
class SwarmLearnedModelConfig:
    """Optional learned-policy settings layered on top of the planner."""

    enabled: bool = False
    backend: str = "auto"
    model_path: Optional[str] = None
    fallback_to_planner: bool = True
    async_worker: bool = True
    max_update_rate_hz: Optional[float] = None
    max_result_age_ms: float = 150.0
    max_targets_tensor: int = 8
    normalization: SwarmFeatureNormalizationConfig = field(
        default_factory=SwarmFeatureNormalizationConfig
    )


@dataclass(frozen=True)
class SwarmEvalConfig:
    """Configuration for live swarm threat evaluation and scheduling."""

    enabled: bool = False
    hostile_levels: Tuple[str, ...] = ("suspicious", "threatening")
    max_engage_distance_m: Optional[float] = None
    exact_search_limit: int = 6
    beam_width: int = 8
    switch_absolute_damage_gain: float = 0.25
    switch_relative_improvement: float = 0.10
    default_damage_weight: float = 1.0
    excluded_target_classes: Tuple[str, ...] = tuple()
    damage_by_class: Dict[str, float] = field(default_factory=dict)
    timing: SwarmTimingConfig = field(default_factory=SwarmTimingConfig)
    learned_model: SwarmLearnedModelConfig = field(
        default_factory=SwarmLearnedModelConfig
    )

    @classmethod
    def disabled(cls) -> "SwarmEvalConfig":
        return cls()


@dataclass(frozen=True)
class ControlConfig:
    """Aiming geometry and the limits shared with simulation and planning."""

    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    aim_mode: str
    rate_limits: AxisPair
    accel_limits: AxisPair
    yaw_sign: float
    pitch_sign: float
    frame_size: Tuple[int, int]
    fov_deg: Optional[Tuple[float, float]]
    laser: LaserAimingControlConfig
    swarm_eval: SwarmEvalConfig = field(default_factory=SwarmEvalConfig.disabled)
    threat_eval: ThreatEvalConfig = field(default_factory=ThreatEvalConfig.disabled)

    @property
    def width(self) -> int:
        return self.frame_size[0]

    @property
    def height(self) -> int:
        return self.frame_size[1]

    def for_frame_size(self, frame_size: Tuple[int, int]) -> "ControlConfig":
        """Scale aiming geometry and laser tolerance to the observed frame."""
        width, height = (int(frame_size[0]), int(frame_size[1]))
        if width <= 0 or height <= 0:
            raise ControlConfigError("frame dimensions must be positive")
        if (width, height) == self.frame_size:
            return self
        scale_x = width / float(self.width)
        scale_y = height / float(self.height)
        return replace(
            self,
            fx_px=self.fx_px * scale_x,
            fy_px=self.fy_px * scale_y,
            cx_px=self.cx_px * scale_x,
            cy_px=self.cy_px * scale_y,
            frame_size=(width, height),
            laser=replace(self.laser, tolerance_px=self.laser.tolerance_px * min(scale_x, scale_y)),
        )

    @classmethod
    def from_raw_config(
        cls, cfg: Mapping[str, Any], frame_size: Tuple[int, int],
        *, camera_intrinsics: Optional[CameraIntrinsics] = None,
    ) -> "ControlConfig":
        control_section: MutableMapping[str, Any] = dict(cfg.get("control", {}))
        if not control_section:
            raise ControlConfigError("config is missing 'control' section")

        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ControlConfigError("frame dimensions must be positive")

        try:
            intrinsics = (camera_intrinsics if camera_intrinsics is not None
                          else CameraIntrinsics.from_raw_config(cfg, frame_size))
        except CameraIntrinsicsConfigError as exc:
            raise ControlConfigError(str(exc)) from exc

        # The live controller owns these caps. Consumers retain the paired
        # interface for simulator integration and planner slew estimates.
        controller = cfg.get("controller", {})
        rate = abs(float(controller.get("rate_limit_rad_s", 0.8)))
        accel = abs(float(controller.get("accel_limit_rad_s2", 3.5)))
        rate_limits = AxisPair(rate, rate)
        accel_limits = AxisPair(accel, accel)

        yaw_sign, pitch_sign = _parse_signs(control_section)

        aim_mode = str(control_section.get("aim_mode", "laser_point")).strip().lower()
        valid_aim_modes = {"camera_center", "laser_point"}
        if aim_mode not in valid_aim_modes:
            raise ControlConfigError(
                f"control.aim_mode must be one of {sorted(valid_aim_modes)}, got {aim_mode!r}"
            )

        raw_laser_section = control_section.get("laser", {}) or {}
        if not isinstance(raw_laser_section, Mapping):
            raise ControlConfigError("control.laser must be a mapping when provided")
        try:
            tolerance_px = float(raw_laser_section.get("tolerance_px", 0.0))
        except (TypeError, ValueError) as exc:
            raise ControlConfigError("control.laser.tolerance_px must be numeric") from exc
        if tolerance_px < 0.0:
            raise ControlConfigError("control.laser.tolerance_px cannot be negative")

        use_range = str(raw_laser_section.get("use_range", "known_size")).strip().lower()
        valid_use_range = {"known_size", "ground_plane", "auto", "infinite"}
        if use_range not in valid_use_range:
            raise ControlConfigError(
                "control.laser.use_range must be one of "
                f"{sorted(valid_use_range)}, got {use_range!r}"
            )

        try:
            default_distance_m = float(raw_laser_section.get("default_distance_m", 25.0))
        except (TypeError, ValueError) as exc:
            raise ControlConfigError(
                "control.laser.default_distance_m must be a positive number"
            ) from exc
        if default_distance_m <= 0.0:
            raise ControlConfigError("control.laser.default_distance_m must be positive")

        return cls(
            fx_px=intrinsics.fx_px,
            fy_px=intrinsics.fy_px,
            cx_px=intrinsics.cx_px,
            cy_px=intrinsics.cy_px,
            aim_mode=aim_mode,
            rate_limits=rate_limits,
            accel_limits=accel_limits,
            yaw_sign=yaw_sign,
            pitch_sign=pitch_sign,
            frame_size=(width, height),
            fov_deg=intrinsics.fov_deg,
            laser=LaserAimingControlConfig(
                tolerance_px=tolerance_px,
                use_range=use_range,
                default_distance_m=default_distance_m,
            ),
            swarm_eval=_parse_swarm_eval_config(cfg),
            threat_eval=_parse_threat_eval_config(cfg),
        )


def _parse_vector3(
    raw: Any,
    *,
    default: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    if raw is None:
        return default
    if isinstance(raw, Mapping):
        try:
            x = float(raw.get("x", default[0]))
            y = float(raw.get("y", default[1]))
            z = float(raw.get("z", default[2]))
        except (TypeError, ValueError) as exc:
            raise LaserConfigError("laser vectors must contain numeric x/y/z") from exc
        return (x, y, z)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if len(raw) != 3:
            raise LaserConfigError("laser vectors must have exactly 3 elements")
        try:
            x, y, z = (float(raw[0]), float(raw[1]), float(raw[2]))
        except (TypeError, ValueError) as exc:
            raise LaserConfigError("laser vectors must contain numeric values") from exc
        return (x, y, z)
    raise LaserConfigError("laser vectors must be provided as a mapping or length-3 sequence")


def _parse_render_config(section: Mapping[str, Any]) -> LaserRenderConfig:
    try:
        beam_length_m = float(section.get("beam_length_m", 5.0))
    except (TypeError, ValueError) as exc:
        raise LaserConfigError("laser.render.beam_length_m must be numeric") from exc
    if beam_length_m <= 0.0:
        raise LaserConfigError("laser.render.beam_length_m must be positive")

    colour = _parse_colour(section.get("color_bgr"))

    try:
        thickness_px = int(section.get("thickness_px", 2))
    except (TypeError, ValueError) as exc:
        raise LaserConfigError("laser.render.thickness_px must be an integer") from exc
    if thickness_px <= 0:
        raise LaserConfigError("laser.render.thickness_px must be positive")

    try:
        from common.config_constants import LASER_RENDER
        hit_tolerance_px = float(section.get("hit_tolerance_px", LASER_RENDER["hit_tolerance_px"]))
    except (TypeError, ValueError) as exc:
        raise LaserConfigError("laser.render.hit_tolerance_px must be numeric") from exc
    if hit_tolerance_px < 0.0:
        raise LaserConfigError("laser.render.hit_tolerance_px cannot be negative")

    return LaserRenderConfig(
        beam_length_m=beam_length_m,
        colour_bgr=colour,
        thickness_px=thickness_px,
        hit_tolerance_px=hit_tolerance_px,
    )


def _parse_colour(raw: Any) -> Tuple[int, int, int]:
    if raw is None:
        return (0, 0, 255)
    if isinstance(raw, Mapping):
        try:
            b = int(raw.get("b"))
            g = int(raw.get("g"))
            r = int(raw.get("r"))
        except (TypeError, ValueError) as exc:
            raise LaserConfigError("laser.render.color_bgr must contain integer r/g/b") from exc
        colour = (b, g, r)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if len(raw) != 3:
            raise LaserConfigError("laser.render.color_bgr must have exactly 3 entries")
        try:
            colour = (int(raw[0]), int(raw[1]), int(raw[2]))
        except (TypeError, ValueError) as exc:
            raise LaserConfigError("laser.render.color_bgr entries must be integers") from exc
    else:
        raise LaserConfigError("laser.render.color_bgr must be a mapping or length-3 sequence")

    clamped = tuple(max(0, min(255, c)) for c in colour)
    return (int(clamped[0]), int(clamped[1]), int(clamped[2]))


def _extract_optional_axis_pair(section: Mapping[str, Any], key: str) -> Optional[AxisPair]:
    raw = section.get(key)
    if raw is None:
        return None
    try:
        yaw = float(raw["yaw"])
        pitch = float(raw["pitch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ControlConfigError(f"control.{key} must contain yaw/pitch floats") from exc
    return AxisPair(yaw=yaw, pitch=pitch)


def _parse_signs(section: Mapping[str, Any]) -> Tuple[float, float]:
    signs = section.get("sign_convention", {}) or {}
    yaw_positive = str(signs.get("yaw_positive", "right")).lower()
    pitch_positive = str(signs.get("pitch_positive", "down")).lower()

    yaw_sign = _sign_from_alias(yaw_positive, {"right": 1.0, "left": -1.0})
    pitch_sign = _sign_from_alias(pitch_positive, {"down": 1.0, "up": -1.0})
    return yaw_sign, pitch_sign


def _sign_from_alias(value: str, mapping: Mapping[str, float]) -> float:
    if value not in mapping:
        valid = ", ".join(sorted(mapping.keys()))
        raise ControlConfigError(f"invalid sign convention '{value}', expected one of: {valid}")
    return mapping[value]


def _parse_swarm_eval_config(cfg: Mapping[str, Any]) -> SwarmEvalConfig:
    raw = cfg.get("swarm_eval", {}) or {}
    if not isinstance(raw, Mapping):
        raise ControlConfigError("swarm_eval must be a mapping when provided")

    enabled = bool(raw.get("enabled", False))
    raw_hostile_levels = raw.get("hostile_levels", ("suspicious", "threatening"))
    if isinstance(raw_hostile_levels, str):
        hostile_levels = (raw_hostile_levels.strip().lower(),)
    elif isinstance(raw_hostile_levels, Sequence):
        hostile_levels = tuple(str(level).strip().lower() for level in raw_hostile_levels)
    else:
        raise ControlConfigError("swarm_eval.hostile_levels must be a string or sequence")
    if not hostile_levels:
        raise ControlConfigError("swarm_eval.hostile_levels must not be empty")

    max_engage_distance_raw = raw.get("max_engage_distance_m")
    max_engage_distance_m: Optional[float] = None
    if max_engage_distance_raw is not None:
        try:
            max_engage_distance_m = float(max_engage_distance_raw)
        except (TypeError, ValueError) as exc:
            raise ControlConfigError("swarm_eval.max_engage_distance_m must be numeric") from exc
        if max_engage_distance_m <= 0.0:
            raise ControlConfigError("swarm_eval.max_engage_distance_m must be > 0")

    exact_search_limit = _parse_optional_int(raw, "exact_search_limit", 6)
    if exact_search_limit < 1:
        raise ControlConfigError("swarm_eval.exact_search_limit must be >= 1")

    beam_width = _parse_optional_int(raw, "beam_width", 8)
    if beam_width < 1:
        raise ControlConfigError("swarm_eval.beam_width must be >= 1")

    switch_absolute_damage_gain = _parse_optional_non_negative_float(
        raw, "switch_absolute_damage_gain", 0.25
    )
    switch_relative_improvement = _parse_optional_non_negative_float(
        raw, "switch_relative_improvement", 0.10
    )
    default_damage_weight = _parse_optional_non_negative_float(
        raw, "default_damage_weight", 1.0
    )
    raw_excluded_classes = raw.get("excluded_target_classes", ()) or ()
    if isinstance(raw_excluded_classes, str):
        excluded_target_classes = (raw_excluded_classes.strip(),)
    elif isinstance(raw_excluded_classes, Sequence):
        excluded_target_classes = tuple(
            str(cls_name).strip() for cls_name in raw_excluded_classes if str(cls_name).strip()
        )
    else:
        raise ControlConfigError(
            "swarm_eval.excluded_target_classes must be a string or sequence"
        )

    raw_damage_by_class = raw.get("damage_by_class", {}) or {}
    if not isinstance(raw_damage_by_class, Mapping):
        raise ControlConfigError("swarm_eval.damage_by_class must be a mapping")
    damage_by_class: Dict[str, float] = {}
    for cls_name, value in raw_damage_by_class.items():
        try:
            damage_weight = float(value)
        except (TypeError, ValueError) as exc:
            raise ControlConfigError(
                "swarm_eval.damage_by_class values must be numeric"
            ) from exc
        if damage_weight < 0.0:
            raise ControlConfigError("swarm_eval.damage_by_class values must be >= 0")
        damage_by_class[str(cls_name)] = damage_weight

    timing_raw = raw.get("timing", {}) or {}
    if not isinstance(timing_raw, Mapping):
        raise ControlConfigError("swarm_eval.timing must be a mapping")
    timing = SwarmTimingConfig(
        base_track_lock_s=_parse_optional_non_negative_float(timing_raw, "base_track_lock_s", 0.15),
        search_track_lock_s=_parse_optional_non_negative_float(timing_raw, "search_track_lock_s", 0.25),
        recover_track_lock_s=_parse_optional_non_negative_float(timing_raw, "recover_track_lock_s", 0.40),
        low_conf_threshold=_parse_optional_bounded_float(timing_raw, "low_conf_threshold", 0.60, 0.0, 1.0),
        low_conf_penalty_s=_parse_optional_non_negative_float(timing_raw, "low_conf_penalty_s", 0.20),
        min_track_observations=_parse_optional_int(timing_raw, "min_track_observations", 3),
        low_continuity_penalty_s=_parse_optional_non_negative_float(
            timing_raw, "low_continuity_penalty_s", 0.08
        ),
        missing_range_penalty_s=_parse_optional_non_negative_float(
            timing_raw, "missing_range_penalty_s", 0.10
        ),
        predictive_penalty_s=_parse_optional_non_negative_float(
            timing_raw, "predictive_penalty_s", 0.20
        ),
        effect_time_s=_parse_optional_non_negative_float(timing_raw, "effect_time_s", 0.25),
        effect_distance_scale_s_per_m=_parse_optional_non_negative_float(
            timing_raw, "effect_distance_scale_s_per_m", 0.01
        ),
        confirm_time_s=_parse_optional_non_negative_float(timing_raw, "confirm_time_s", 0.10),
        confirm_distance_scale_s_per_m=_parse_optional_non_negative_float(
            timing_raw, "confirm_distance_scale_s_per_m", 0.004
        ),
        settle_margin_s=_parse_optional_non_negative_float(timing_raw, "settle_margin_s", 0.05),
    )

    if timing.min_track_observations < 1:
        raise ControlConfigError("swarm_eval.timing.min_track_observations must be >= 1")

    learned_raw = raw.get("learned_model", {}) or {}
    if not isinstance(learned_raw, Mapping):
        raise ControlConfigError("swarm_eval.learned_model must be a mapping")

    learned_enabled = bool(learned_raw.get("enabled", False))
    learned_backend = str(learned_raw.get("backend", "auto")).strip().lower() or "auto"
    if learned_backend not in {"auto", "torch", "tensorrt"}:
        raise ControlConfigError(
            "swarm_eval.learned_model.backend must be one of: auto, torch, tensorrt"
        )
    learned_model_path_raw = learned_raw.get("model_path")
    learned_model_path = None
    if learned_model_path_raw is not None:
        learned_model_path = str(learned_model_path_raw).strip() or None
    learned_fallback_to_planner = bool(learned_raw.get("fallback_to_planner", True))
    learned_async_worker = bool(learned_raw.get("async_worker", True))
    learned_max_update_rate_hz_raw = learned_raw.get("max_update_rate_hz")
    learned_max_update_rate_hz: Optional[float]
    if learned_max_update_rate_hz_raw is None:
        learned_max_update_rate_hz = None
    else:
        try:
            learned_max_update_rate_hz = float(learned_max_update_rate_hz_raw)
        except (TypeError, ValueError) as exc:
            raise ControlConfigError(
                "swarm_eval.learned_model.max_update_rate_hz must be numeric when provided"
            ) from exc
        if learned_max_update_rate_hz <= 0.0:
            raise ControlConfigError(
                "swarm_eval.learned_model.max_update_rate_hz must be > 0 when provided"
            )
    learned_max_result_age_ms = _parse_optional_non_negative_float(
        learned_raw, "max_result_age_ms", 150.0
    )
    learned_max_targets_tensor = _parse_optional_int(
        learned_raw, "max_targets_tensor", 8
    )
    if learned_max_targets_tensor < 1:
        raise ControlConfigError(
            "swarm_eval.learned_model.max_targets_tensor must be >= 1"
        )

    normalization_raw = learned_raw.get("normalization", {}) or {}
    if not isinstance(normalization_raw, Mapping):
        raise ControlConfigError(
            "swarm_eval.learned_model.normalization must be a mapping"
        )
    normalization = SwarmFeatureNormalizationConfig(
        max_distance_m=_parse_optional_non_negative_float(
            normalization_raw, "max_distance_m", 80.0
        ),
        max_closing_speed_m_s=_parse_optional_non_negative_float(
            normalization_raw, "max_closing_speed_m_s", 12.0
        ),
        max_breakthrough_time_s=_parse_optional_non_negative_float(
            normalization_raw, "max_breakthrough_time_s", 40.0
        ),
        max_time_to_engage_s=_parse_optional_non_negative_float(
            normalization_raw, "max_time_to_engage_s", 8.0
        ),
        max_damage_weight=_parse_optional_non_negative_float(
            normalization_raw, "max_damage_weight", 10.0
        ),
        max_angle_rad=_parse_optional_non_negative_float(
            normalization_raw, "max_angle_rad", 0.8
        ),
        max_track_observations=_parse_optional_non_negative_float(
            normalization_raw, "max_track_observations", 10.0
        ),
        max_track_age_s=_parse_optional_non_negative_float(
            normalization_raw, "max_track_age_s", 5.0
        ),
    )

    return SwarmEvalConfig(
        enabled=enabled,
        hostile_levels=hostile_levels,
        max_engage_distance_m=max_engage_distance_m,
        exact_search_limit=exact_search_limit,
        beam_width=beam_width,
        switch_absolute_damage_gain=switch_absolute_damage_gain,
        switch_relative_improvement=switch_relative_improvement,
        default_damage_weight=default_damage_weight,
        excluded_target_classes=excluded_target_classes,
        damage_by_class=damage_by_class,
        timing=timing,
        learned_model=SwarmLearnedModelConfig(
            enabled=learned_enabled,
            backend=learned_backend,
            model_path=learned_model_path,
            fallback_to_planner=learned_fallback_to_planner,
            async_worker=learned_async_worker,
            max_update_rate_hz=learned_max_update_rate_hz,
            max_result_age_ms=learned_max_result_age_ms,
            max_targets_tensor=learned_max_targets_tensor,
            normalization=normalization,
        ),
    )


def _parse_threat_eval_config(cfg: Mapping[str, Any]) -> ThreatEvalConfig:
    raw = cfg.get("threat_eval")
    if raw is None:
        sim_section = cfg.get("sim", {}) or {}
        if isinstance(sim_section, Mapping):
            scene = sim_section.get("scene", {}) or {}
            if isinstance(scene, Mapping):
                legacy_zones = scene.get("threat_eval_zones")
                legacy_asset = scene.get("defended_asset")
                if legacy_zones is not None or legacy_asset is not None:
                    raw = {
                        "enabled": bool(
                            legacy_zones.get("enabled", True)
                            if isinstance(legacy_zones, Mapping)
                            else True
                        ),
                        "defended_asset": legacy_asset,
                        "zones": legacy_zones.get("zones", {})
                        if isinstance(legacy_zones, Mapping)
                        else {},
                    }
    raw = raw or {}
    if not isinstance(raw, Mapping):
        raise ControlConfigError("threat_eval must be a mapping when provided")

    enabled = bool(raw.get("enabled", False))
    asset_xy = (0.0, 0.0)
    asset_world = (0.0, 0.0, 0.0)
    defended_asset = raw.get("defended_asset", {}) or {}
    if defended_asset:
        if not isinstance(defended_asset, Mapping):
            raise ControlConfigError("threat_eval.defended_asset must be a mapping")
        position_world = defended_asset.get("position_world", (0.0, 0.0, 0.0))
        if not isinstance(position_world, Sequence) or len(position_world) < 2:
            raise ControlConfigError(
                "threat_eval.defended_asset.position_world must have at least 2 values"
            )
        if len(position_world) >= 3:
            asset_world = (
                float(position_world[0]),
                float(position_world[1]),
                float(position_world[2]),
            )
            asset_xy = (asset_world[0], asset_world[2])
        else:
            asset_xy = (float(position_world[0]), float(position_world[1]))
            asset_world = (asset_xy[0], 0.0, asset_xy[1])
        if not all(math.isfinite(value) for value in asset_world):
            raise ControlConfigError(
                "threat_eval.defended_asset.position_world values must be finite"
            )
        validate_asset_position(asset_xy)

    zones_raw = raw.get("zones", {}) or {}
    if not isinstance(zones_raw, Mapping):
        raise ControlConfigError("threat_eval.zones must be a mapping")
    try:
        zone_radii = parse_zone_config(dict(zones_raw))
    except ValueError as exc:
        raise ControlConfigError(f"invalid threat_eval.zones config: {exc}") from exc

    return ThreatEvalConfig(
        enabled=enabled,
        asset_xy=asset_xy,
        asset_world=asset_world,
        zone_radii=zone_radii,
    )


def _parse_optional_int(section: Mapping[str, Any], key: str, default: int) -> int:
    raw_value = section.get(key, default)
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ControlConfigError(f"swarm_eval.{key} must be an integer") from exc


def _parse_optional_non_negative_float(
    section: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    raw_value = section.get(key, default)
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ControlConfigError(f"swarm_eval.{key} must be numeric") from exc
    if value < 0.0:
        raise ControlConfigError(f"swarm_eval.{key} must be >= 0")
    return value


def _parse_optional_bounded_float(
    section: Mapping[str, Any],
    key: str,
    default: float,
    lo: float,
    hi: float,
) -> float:
    value = _parse_optional_non_negative_float(section, key, default)
    if value < lo or value > hi:
        raise ControlConfigError(f"swarm_eval.{key} must be between {lo} and {hi}")
    return value


def pixel_delta(
    u_px: float,
    v_px: float,
    ref_u_px: float,
    ref_v_px: float,
    config: ControlConfig,
) -> AxisPair:
    """Return signed pixel deltas between two image coordinates.

    Parameters
    ----------
    u_px, v_px:
        Pixel coordinates of the measured point (e.g. target centroid).
    ref_u_px, ref_v_px:
        Reference pixel coordinates (e.g. image centre or predicted laser dot).
    config:
        Parsed :class:`ControlConfig` providing axis sign conventions.
    """

    du = config.yaw_sign * (u_px - ref_u_px)
    dv = config.pitch_sign * (v_px - ref_v_px)


    return AxisPair(yaw=du, pitch=dv)


def pixel_error(
    u_px: float,
    v_px: float,
    config: ControlConfig,
) -> AxisPair:
    """Backward-compatible wrapper returning deltas to the image centre."""

    return pixel_delta(
        u_px,
        v_px,
        config.cx_px,
        config.cy_px,
        config,
    )


def angular_error_from_pixels(
    u_px: float,
    v_px: float,
    config: ControlConfig,
    *,
    linearize: bool = False,
) -> AxisPair:
    """Convert a pixel coordinate into yaw/pitch angular errors in radians.

    Parameters
    ----------
    u_px, v_px:
        Pixel coordinates of the target centroid.
    config:
        Parsed :class:`ControlConfig` containing focal lengths and signs.
    linearize:
        If ``True`` use the small-angle approximation ``err ≈ Δpx / f``. When
        ``False`` the full ``atan`` relation is applied.
    """

    px_err = pixel_error(u_px, v_px, config)

    return angular_error_from_pixel_delta(px_err, config, linearize=linearize)


def angular_error_from_pixel_delta(
    px_err: AxisPair,
    config: ControlConfig,
    *,
    linearize: bool = False,
) -> AxisPair:
    """Convert pixel deltas into yaw/pitch angular errors."""

    if linearize:
        yaw_err = px_err.yaw / config.fx_px
        pitch_err = px_err.pitch / config.fy_px
    else:
        yaw_err = math.atan(px_err.yaw / config.fx_px)
        pitch_err = math.atan(px_err.pitch / config.fy_px)

    return AxisPair(yaw=yaw_err, pitch=pitch_err)
