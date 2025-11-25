"""Shared helpers for loading the control-loop configuration.

This module centralizes parsing of the `control` section in the runtime
configuration so both Jetson and PC components can share consistent defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, MutableMapping, Optional, Sequence, Tuple

from common.camera import CameraIntrinsicsConfigError, focal_lengths_from_fov


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
class MpcHorizonConfig:
    """Timing and horizon parameters for the MPC controller."""

    prediction_horizon: int
    control_horizon: int
    sample_time_s: float
    gamma: float
    move_blocking: bool


@dataclass(frozen=True)
class MpcPlantConfig:
    """Simple 3-state gimbal plant parameters."""

    a_u: float
    a_f: float


@dataclass(frozen=True)
class MpcEstimatorConfig:
    """Noise covariances for the [theta, omega, d] Kalman filter."""

    q_theta: float
    q_omega: float
    q_d: float
    r_theta: float


@dataclass(frozen=True)
class MpcAxisTrackingCost:
    """Weights for angular tracking error."""

    q_theta: float
    l_theta: float
    q_omega: float
    l_omega: float


@dataclass(frozen=True)
class MpcAxisApproachCost:
    """Weights for changes in tracking error (approach shape)."""

    q_dtheta: float
    l_dtheta: float


@dataclass(frozen=True)
class MpcAxisSmoothnessCost:
    """Weights for control effort and delta controls."""

    r: float
    s: float
    l_du: float


@dataclass(frozen=True)
class MpcAxisCostConfig:
    """Per-axis mixed cost coefficients."""

    tracking: MpcAxisTrackingCost
    approach: MpcAxisApproachCost
    smoothness: MpcAxisSmoothnessCost


@dataclass(frozen=True)
class MpcCostConfig:
    """Mixed quadratic/linear penalty weights and slack penalty."""

    yaw: MpcAxisCostConfig
    pitch: MpcAxisCostConfig
    terminal: Optional[float]
    rho: float
    theta_unit_scale_rad: float = 1.0
    omega_unit_scale_rad_s: float = 1.0
    effort_unit_scale: float = 1.0
    slew_unit_scale: float = 1.0


@dataclass(frozen=True)
class MpcMetaKnobConfig:
    """High-level tuning knobs that derive per-axis cost defaults."""

    tracking_aggressiveness: float
    approach_bias_strength: float
    stability_vs_response: float


@dataclass(frozen=True)
class MpcConstraintConfig:
    """Input, rate, and optional state limits."""

    u_min: float
    u_max: float
    du_max: float
    theta_min: Optional[float]
    theta_max: Optional[float]
    omega_min: Optional[float]
    omega_max: Optional[float]


@dataclass(frozen=True)
class ControlDebugOverlayConfig:
    """Rendering preferences for MPC diagnostics on the return feed."""

    enabled: bool
    history_window_s: float
    opacity: float
    bar_height_px: int
    show_terms: Tuple[str, ...]

    DEFAULT_TERMS: ClassVar[Tuple[str, ...]] = (
        "theta",
        "omega",
        "approach",
        "effort",
        "slew",
        "slack",
    )

    @classmethod
    def disabled(cls) -> "ControlDebugOverlayConfig":
        return cls(
            enabled=False,
            history_window_s=1.5,
            opacity=0.85,
            bar_height_px=48,
            show_terms=cls.DEFAULT_TERMS,
        )


@dataclass(frozen=True)
class MpcConfig:
    """Top-level MPC configuration bundle."""

    horizon: MpcHorizonConfig
    plant: MpcPlantConfig
    estimator: MpcEstimatorConfig
    costs: MpcCostConfig
    constraints: MpcConstraintConfig
    meta_knobs: Optional[MpcMetaKnobConfig] = None


@dataclass(frozen=True)
class ControlConfig:
    """Typed view over the `control` section of ``configs/dev.yaml``.

    The class normalizes units, derives focal lengths when requested, and
    exposes helpful pre-computed quantities (e.g. image center and axis signs).
    """

    mode: str
    loop_hz: Optional[float]
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    aim_mode: str
    kp: AxisPair
    kd: AxisPair
    ki: AxisPair
    rate_limits: AxisPair
    accel_limits: AxisPair
    deadband_px: float
    smooth_px_alpha: float
    lost_target_timeout_ms: int
    reinit_on_lost: bool
    target_selector: str
    yaw_sign: float
    pitch_sign: float
    frame_size: Tuple[int, int]
    fov_deg: Optional[Tuple[float, float]]
    laser: LaserAimingControlConfig
    controller: str = "pid"
    mpc: Optional[MpcConfig] = None
    debug_overlay: ControlDebugOverlayConfig = field(
        default_factory=ControlDebugOverlayConfig.disabled
    )

    @property
    def width(self) -> int:
        return self.frame_size[0]

    @property
    def height(self) -> int:
        return self.frame_size[1]

    @property
    def loop_dt(self) -> Optional[float]:
        return None if self.loop_hz in (None, 0) else 1.0 / float(self.loop_hz)

    @classmethod
    def from_raw_config(
        cls, cfg: Mapping[str, Any], frame_size: Tuple[int, int]
    ) -> "ControlConfig":
        control_section: MutableMapping[str, Any] = dict(cfg.get("control", {}))
        if not control_section:
            raise ControlConfigError("config is missing 'control' section")

        mode = control_section.get("mode", "rate")
        if mode not in {"rate", "position"}:
            raise ControlConfigError(f"unsupported control mode: {mode}")

        controller_type = str(control_section.get("controller", "pid")).strip().lower()
        if controller_type not in {"pid", "mpc"}:
            raise ControlConfigError(
                "control.controller must be either 'pid' or 'mpc' when provided"
            )

        loop_hz = control_section.get("loop_hz")
        if loop_hz is not None:
            loop_hz = float(loop_hz)
            if loop_hz <= 0:
                raise ControlConfigError("loop_hz must be > 0 or null")

        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ControlConfigError("frame dimensions must be positive")

        fx_px, fy_px, fov_deg = _derive_focal_lengths(control_section, width, height)

        kp = _extract_axis_pair(control_section, "kp")
        kd = _extract_axis_pair(control_section, "kd")
        ki = _extract_axis_pair(
            control_section, "ki", allow_missing=True, default=AxisPair(0.0, 0.0)
        )
        rate_limits = _extract_axis_pair(control_section, "rate_limits")
        accel_limits = _extract_axis_pair(control_section, "accel_limits")

        deadband_px = float(control_section.get("deadband_px", 0.0))
        if deadband_px < 0:
            raise ControlConfigError("deadband_px cannot be negative")

        smooth_px_alpha = float(control_section.get("smooth_px_alpha", 0.0))
        if not 0.0 <= smooth_px_alpha <= 1.0:
            raise ControlConfigError("smooth_px_alpha must be between 0 and 1")

        lost_target_timeout_ms = int(control_section.get("lost_target_timeout_ms", 0))
        if lost_target_timeout_ms < 0:
            raise ControlConfigError("lost_target_timeout_ms cannot be negative")

        reinit_on_lost = bool(control_section.get("reinit_on_lost", True))
        target_selector = str(control_section.get("target_selector", "max_conf"))

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
            tolerance_px = float(raw_laser_section.get("tolerance_px", deadband_px))
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

        cx_px = width / 2.0
        cy_px = height / 2.0

        return cls(
            mode=mode,
            loop_hz=loop_hz,
            fx_px=fx_px,
            fy_px=fy_px,
            cx_px=cx_px,
            cy_px=cy_px,
            aim_mode=aim_mode,
            kp=kp,
            kd=kd,
            ki=ki,
            rate_limits=rate_limits,
            accel_limits=accel_limits,
            deadband_px=deadband_px,
            smooth_px_alpha=smooth_px_alpha,
            lost_target_timeout_ms=lost_target_timeout_ms,
            reinit_on_lost=reinit_on_lost,
            target_selector=target_selector,
            yaw_sign=yaw_sign,
            pitch_sign=pitch_sign,
            frame_size=(width, height),
            fov_deg=fov_deg,
            laser=LaserAimingControlConfig(
                tolerance_px=tolerance_px,
                use_range=use_range,
                default_distance_m=default_distance_m,
            ),
            controller=controller_type,
            mpc=_parse_mpc_config(control_section, controller_type),
            debug_overlay=_parse_debug_overlay_config(control_section),
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


def _parse_debug_overlay_config(
    control_section: Mapping[str, Any]
) -> ControlDebugOverlayConfig:
    raw = control_section.get("debug_overlay")
    if raw is None:
        return ControlDebugOverlayConfig.disabled()
    if not isinstance(raw, Mapping):
        raise ControlConfigError("control.debug_overlay must be a mapping when provided")

    enabled = bool(raw.get("enabled", False))
    history_window_s = _coerce_float(
        raw.get("history_window_s", 1.5), "control.debug_overlay.history_window_s"
    )
    if history_window_s <= 0.0:
        raise ControlConfigError("control.debug_overlay.history_window_s must be positive")

    opacity = _coerce_float(raw.get("opacity", 0.85), "control.debug_overlay.opacity")
    if not 0.0 <= opacity <= 1.0:
        raise ControlConfigError("control.debug_overlay.opacity must be within [0, 1]")

    try:
        bar_height_px = int(raw.get("bar_height_px", 48))
    except (TypeError, ValueError) as exc:
        raise ControlConfigError("control.debug_overlay.bar_height_px must be an integer") from exc
    if bar_height_px <= 0:
        raise ControlConfigError("control.debug_overlay.bar_height_px must be positive")

    show_terms_raw = raw.get("show_terms")
    if show_terms_raw is None:
        show_terms = ControlDebugOverlayConfig.DEFAULT_TERMS
    else:
        if not isinstance(show_terms_raw, Sequence) or isinstance(show_terms_raw, (str, bytes)):
            raise ControlConfigError("control.debug_overlay.show_terms must be a list of strings")
        allowed = set(ControlDebugOverlayConfig.DEFAULT_TERMS)
        normalized = []
        for term in show_terms_raw:
            name = str(term).strip().lower()
            if not name:
                raise ControlConfigError("control.debug_overlay.show_terms entries must be non-empty")
            if name not in allowed:
                valid = ", ".join(sorted(allowed))
                raise ControlConfigError(
                    f"control.debug_overlay.show_terms entries must be in: {valid}"
                )
            if name not in normalized:
                normalized.append(name)
        if not normalized:
            raise ControlConfigError("control.debug_overlay.show_terms cannot be empty")
        show_terms = tuple(normalized)

    return ControlDebugOverlayConfig(
        enabled=enabled,
        history_window_s=history_window_s,
        opacity=opacity,
        bar_height_px=bar_height_px,
        show_terms=show_terms,
    )


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
        hit_tolerance_px = float(section.get("hit_tolerance_px", 3.0))
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


def _extract_axis_pair(
    section: Mapping[str, Any],
    key: str,
    *,
    allow_missing: bool = False,
    default: Optional[AxisPair] = None,
) -> AxisPair:
    raw = section.get(key)
    if raw is None:
        if allow_missing:
            if default is not None:
                return default
            return AxisPair(0.0, 0.0)
        raise ControlConfigError(f"control.{key} is required")

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


def _derive_focal_lengths(
    section: Mapping[str, Any], width: int, height: int
) -> Tuple[float, float, Optional[Tuple[float, float]]]:
    if section.get("fx_fy_from_fov", False):
        fov = section.get("fov_deg")
        if not isinstance(fov, Mapping):
            raise ControlConfigError("control.fov_deg must be a mapping with 'h' and 'v'")
        try:
            hfov = float(fov["h"])
            vfov = float(fov["v"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ControlConfigError("control.fov_deg must include numeric h and v") from exc
        if not (0 < hfov < 180 and 0 < vfov < 180):
            raise ControlConfigError("control.fov_deg values must be between 0 and 180 degrees")
        try:
            fx, fy = focal_lengths_from_fov(width, height, hfov, vfov)
        except CameraIntrinsicsConfigError as exc:
            raise ControlConfigError(str(exc)) from exc
        return fx, fy, (hfov, vfov)

    try:
        fx = float(section["fx_px"])
        fy = float(section["fy_px"])
    except KeyError as exc:
        raise ControlConfigError(
            "control.fx_px and control.fy_px are required when fx_fy_from_fov is false"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ControlConfigError("control.fx_px and control.fy_px must be numeric") from exc

    if fx <= 0 or fy <= 0:
        raise ControlConfigError("focal lengths must be positive")
    return fx, fy, None


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _lerp(low: float, high: float, t: float) -> float:
    return low + (high - low) * t


def _parse_meta_knobs(raw_mpc_section: Mapping[str, Any]) -> Optional[MpcMetaKnobConfig]:
    raw_meta = raw_mpc_section.get("meta_knobs")
    if raw_meta is None:
        return None
    if not isinstance(raw_meta, Mapping):
        raise ControlConfigError("control.mpc.meta_knobs must be a mapping when provided")
    try:
        tracking = float(raw_meta.get("tracking_aggressiveness", 0.6))
        approach = float(raw_meta.get("approach_bias_strength", 0.4))
        stability = float(raw_meta.get("stability_vs_response", 0.6))
    except (TypeError, ValueError) as exc:
        raise ControlConfigError("control.mpc.meta_knobs values must be numeric") from exc
    return MpcMetaKnobConfig(
        tracking_aggressiveness=_clamp01(tracking),
        approach_bias_strength=_clamp01(approach),
        stability_vs_response=_clamp01(stability),
    )


def _derive_meta_cost_defaults(
    meta: Optional[MpcMetaKnobConfig],
    u_min: Optional[float],
    u_max: Optional[float],
    du_max: Optional[float],
) -> Tuple[Mapping[str, Mapping[str, float]], Optional[float], float, Mapping[str, float]]:
    cost_defaults: Mapping[str, Mapping[str, float]] = {
        "tracking": {"q_theta": 1.0, "l_theta": 0.0, "q_omega": 0.5, "l_omega": 0.0},
        "approach": {"q_dtheta": 0.0, "l_dtheta": 0.0},
        "smoothness": {"r": 0.05, "s": 0.05, "l_du": 0.0},
    }
    terminal_default: Optional[float] = None
    rho_default = 0.0

    if meta is not None:
        t = meta.tracking_aggressiveness
        a = meta.approach_bias_strength
        s = meta.stability_vs_response
        cost_defaults = {
            "tracking": {
                "q_theta": _lerp(0.5, 3.5, t),
                "l_theta": 0.0,
                "q_omega": _lerp(0.25, 2.5, t),
                "l_omega": 0.0,
            },
            "approach": {
                "q_dtheta": _lerp(0.0, 1.0, a),
                "l_dtheta": _lerp(0.0, 0.5, a),
            },
            "smoothness": {
                "r": _lerp(0.02, 0.25, s),
                "s": _lerp(0.01, 0.12, s),
                "l_du": _lerp(0.0, 0.2, s),
            },
        }
        terminal_default = _lerp(1.0, 12.0, t)
        rho_default = _lerp(0.0, 10000.0, s)

    effort_scale = float(
        1.0 if (u_min is None and u_max is None) else max(abs(u_min or 0.0), abs(u_max or 0.0), 1.0)
    )
    slew_scale = float(max(1e-6, abs(du_max) if du_max is not None else 1.0))
    scale_defaults = {
        "theta_unit_scale_rad": 0.03,
        "omega_unit_scale_rad_s": 1.0,
        "effort_unit_scale": effort_scale,
        "slew_unit_scale": slew_scale,
    }
    return cost_defaults, terminal_default, rho_default, scale_defaults


def _parse_mpc_config(
    control_section: Mapping[str, Any], controller_type: str
) -> Optional[MpcConfig]:
    raw = control_section.get("mpc")
    if raw is None:
        if controller_type == "mpc":
            raise ControlConfigError("control.mpc section is required when controller='mpc'")
        return None
    if not isinstance(raw, Mapping):
        raise ControlConfigError("control.mpc must be a mapping when provided")
    if not raw:
        if controller_type == "mpc":
            raise ControlConfigError("control.mpc cannot be empty when controller='mpc'")
        return None

    meta_knobs = _parse_meta_knobs(raw)

    horizons = _require_mapping(raw, "horizons", path="control.mpc.horizons")
    prediction = _parse_int_field(
        horizons,
        key="prediction",
        path="control.mpc.horizons.prediction",
        aliases=("Np",),
        positive=True,
    )
    control = _parse_int_field(
        horizons,
        key="control",
        path="control.mpc.horizons.control",
        aliases=("Nc",),
        positive=True,
    )
    if control > prediction:
        raise ControlConfigError("control.mpc.horizons.control cannot exceed prediction horizon")
    sample_time = _parse_float_field(
        horizons,
        key="sample_time_s",
        path="control.mpc.horizons.sample_time_s",
        aliases=("ts", "Ts"),
        positive=True,
    )
    gamma = _parse_float_field(
        horizons,
        key="gamma",
        path="control.mpc.horizons.gamma",
        positive=True,
        default=1.0,
    )
    move_blocking = bool(horizons.get("move_blocking", False))

    plant_section = _require_mapping(raw, "plant", path="control.mpc.plant")
    a_u = _parse_float_field(
        plant_section,
        key="a_u",
        path="control.mpc.plant.a_u",
    )
    a_f = _parse_float_field(
        plant_section,
        key="a_f",
        path="control.mpc.plant.a_f",
        non_negative=True,
    )

    estimator_section = _require_mapping(raw, "estimator", path="control.mpc.estimator")
    q_theta = _parse_float_field(
        estimator_section,
        key="q_theta",
        path="control.mpc.estimator.q_theta",
        non_negative=True,
    )
    q_omega = _parse_float_field(
        estimator_section,
        key="q_omega",
        path="control.mpc.estimator.q_omega",
        non_negative=True,
    )
    q_d = _parse_float_field(
        estimator_section,
        key="q_d",
        path="control.mpc.estimator.q_d",
        non_negative=True,
    )
    r_theta = _parse_float_field(
        estimator_section,
        key="r_theta",
        path="control.mpc.estimator.r_theta",
        positive=True,
    )

    constraints_section = _require_mapping(raw, "constraints", path="control.mpc.constraints")
    u_min = _parse_optional_float_field(
        constraints_section,
        key="u_min",
        path="control.mpc.constraints.u_min",
    )
    u_max = _parse_optional_float_field(
        constraints_section,
        key="u_max",
        path="control.mpc.constraints.u_max",
    )
    du_max = _parse_optional_float_field(
        constraints_section,
        key="du_max",
        path="control.mpc.constraints.du_max",
        positive=True,
        default=1.0 if u_min is None and u_max is None else None,
    )
    if du_max is not None and du_max <= 0.0:
        raise ControlConfigError("control.mpc.constraints.du_max must be positive when provided")
    if u_min is not None and u_max is not None and u_min >= u_max:
        raise ControlConfigError("control.mpc.constraints.u_min must be < u_max")
    if u_min is None and u_max is None and du_max is None:
        raise ControlConfigError("control.mpc.constraints.du_max is required when u_min/u_max are missing")

    theta_min = _parse_optional_float_field(
        constraints_section,
        key="theta_min",
        path="control.mpc.constraints.theta_min",
    )
    theta_max = _parse_optional_float_field(
        constraints_section,
        key="theta_max",
        path="control.mpc.constraints.theta_max",
    )
    if theta_min is not None and theta_max is not None and theta_min >= theta_max:
        raise ControlConfigError("control.mpc.constraints.theta_min must be < theta_max")
    omega_min = _parse_optional_float_field(
        constraints_section,
        key="omega_min",
        path="control.mpc.constraints.omega_min",
    )
    omega_max = _parse_optional_float_field(
        constraints_section,
        key="omega_max",
        path="control.mpc.constraints.omega_max",
    )
    if omega_min is not None and omega_max is not None and omega_min >= omega_max:
        raise ControlConfigError("control.mpc.constraints.omega_min must be < omega_max")

    cost_defaults, terminal_default, rho_default, scale_defaults = _derive_meta_cost_defaults(
        meta_knobs, u_min, u_max, du_max
    )
    costs_section = _require_mapping(raw, "costs", path="control.mpc.costs")
    yaw_cost = _parse_axis_costs(costs_section, "yaw", cost_defaults)
    pitch_cost = _parse_axis_costs(costs_section, "pitch", cost_defaults)

    terminal = _parse_optional_float_field(
        costs_section,
        key="terminal",
        path="control.mpc.costs.terminal",
        aliases=("Q_T", "terminal_weight"),
        non_negative=True,
        default=terminal_default,
    )
    rho = _parse_float_field(
        costs_section,
        key="rho",
        path="control.mpc.costs.rho",
        non_negative=True,
        default=rho_default,
    )
    theta_unit_scale_rad = _parse_float_field(
        costs_section,
        key="theta_unit_scale_rad",
        path="control.mpc.costs.theta_unit_scale_rad",
        positive=True,
        default=scale_defaults.get("theta_unit_scale_rad", 0.03),
    )
    omega_unit_scale_rad_s = _parse_float_field(
        costs_section,
        key="omega_unit_scale_rad_s",
        path="control.mpc.costs.omega_unit_scale_rad_s",
        positive=True,
        default=scale_defaults.get("omega_unit_scale_rad_s", 1.0),
    )
    effort_unit_scale = _parse_float_field(
        costs_section,
        key="effort_unit_scale",
        path="control.mpc.costs.effort_unit_scale",
        positive=True,
        default=scale_defaults.get("effort_unit_scale"),
    )
    slew_unit_scale = _parse_float_field(
        costs_section,
        key="slew_unit_scale",
        path="control.mpc.costs.slew_unit_scale",
        positive=True,
        default=scale_defaults.get("slew_unit_scale"),
    )

    return MpcConfig(
        horizon=MpcHorizonConfig(
            prediction_horizon=prediction,
            control_horizon=control,
            sample_time_s=sample_time,
            gamma=gamma,
            move_blocking=move_blocking,
        ),
        plant=MpcPlantConfig(a_u=a_u, a_f=a_f),
        estimator=MpcEstimatorConfig(q_theta=q_theta, q_omega=q_omega, q_d=q_d, r_theta=r_theta),
        costs=MpcCostConfig(
            yaw=yaw_cost,
            pitch=pitch_cost,
            terminal=terminal,
            rho=rho,
            theta_unit_scale_rad=theta_unit_scale_rad,
            omega_unit_scale_rad_s=omega_unit_scale_rad_s,
            effort_unit_scale=effort_unit_scale,
            slew_unit_scale=slew_unit_scale,
        ),
        constraints=MpcConstraintConfig(
            u_min=u_min,
            u_max=u_max,
            du_max=du_max,
            theta_min=theta_min,
            theta_max=theta_max,
            omega_min=omega_min,
            omega_max=omega_max,
        ),
        meta_knobs=meta_knobs,
    )


def _parse_axis_costs(
    costs_section: Mapping[str, Any], axis: str, defaults: Mapping[str, Mapping[str, float]]
) -> MpcAxisCostConfig:
    axis_section = _require_mapping(costs_section, axis, path=f"control.mpc.costs.{axis}")
    tracking = MpcAxisTrackingCost(
        q_theta=_parse_float_field(
            axis_section,
            key="q_theta",
            path=f"control.mpc.costs.{axis}.q_theta",
            non_negative=True,
            default=defaults["tracking"]["q_theta"],
        ),
        l_theta=_parse_float_field(
            axis_section,
            key="l_theta",
            path=f"control.mpc.costs.{axis}.l_theta",
            default=defaults["tracking"]["l_theta"],
        ),
        q_omega=_parse_float_field(
            axis_section,
            key="q_omega",
            path=f"control.mpc.costs.{axis}.q_omega",
            non_negative=True,
            default=defaults["tracking"]["q_omega"],
        ),
        l_omega=_parse_float_field(
            axis_section,
            key="l_omega",
            path=f"control.mpc.costs.{axis}.l_omega",
            default=defaults["tracking"]["l_omega"],
        ),
    )
    approach = MpcAxisApproachCost(
        q_dtheta=_parse_float_field(
            axis_section,
            key="q_dtheta",
            path=f"control.mpc.costs.{axis}.q_dtheta",
            non_negative=True,
            default=defaults["approach"]["q_dtheta"],
        ),
        l_dtheta=_parse_float_field(
            axis_section,
            key="l_dtheta",
            path=f"control.mpc.costs.{axis}.l_dtheta",
            default=defaults["approach"]["l_dtheta"],
        ),
    )
    smoothness = MpcAxisSmoothnessCost(
        r=_parse_float_field(
            axis_section,
            key="r",
            path=f"control.mpc.costs.{axis}.r",
            non_negative=True,
            default=defaults["smoothness"]["r"],
        ),
        s=_parse_float_field(
            axis_section,
            key="s",
            path=f"control.mpc.costs.{axis}.s",
            non_negative=True,
            default=defaults["smoothness"]["s"],
        ),
        l_du=_parse_float_field(
            axis_section,
            key="l_du",
            path=f"control.mpc.costs.{axis}.l_du",
            default=defaults["smoothness"]["l_du"],
        ),
    )
    return MpcAxisCostConfig(tracking=tracking, approach=approach, smoothness=smoothness)


def _require_mapping(section: Mapping[str, Any], key: str, *, path: str) -> Mapping[str, Any]:
    raw = section.get(key)
    if raw is None:
        raise ControlConfigError(f"{path} section is required")
    if not isinstance(raw, Mapping):
        raise ControlConfigError(f"{path} must be a mapping")
    return raw


def _parse_int_field(
    section: Mapping[str, Any],
    *,
    key: str,
    path: str,
    aliases: Sequence[str] = (),
    positive: bool = False,
) -> int:
    raw_value = _get_with_alias(section, key, *aliases)
    if raw_value is None:
        raise ControlConfigError(f"{path} is required")
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ControlConfigError(f"{path} must be an integer") from exc
    if positive and value <= 0:
        raise ControlConfigError(f"{path} must be positive")
    return value


def _parse_float_field(
    section: Mapping[str, Any],
    *,
    key: str,
    path: str,
    aliases: Sequence[str] = (),
    positive: bool = False,
    non_negative: bool = False,
    default: Optional[float] = None,
) -> float:
    raw_value = _get_with_alias(section, key, *aliases)
    if raw_value is None:
        if default is not None:
            return float(default)
        raise ControlConfigError(f"{path} is required")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ControlConfigError(f"{path} must be numeric") from exc
    if positive and value <= 0.0:
        raise ControlConfigError(f"{path} must be positive")
    if non_negative and value < 0.0:
        raise ControlConfigError(f"{path} must be non-negative")
    return value


def _parse_optional_float_field(
    section: Mapping[str, Any],
    *,
    key: str,
    path: str,
    aliases: Sequence[str] = (),
    positive: bool = False,
    non_negative: bool = False,
    default: Optional[float] = None,
) -> Optional[float]:
    raw_value = _get_with_alias(section, key, *aliases)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ControlConfigError(f"{path} must be numeric") from exc
    if positive and value <= 0.0:
        raise ControlConfigError(f"{path} must be positive")
    if non_negative and value < 0.0:
        raise ControlConfigError(f"{path} must be non-negative")
    return value


def _get_with_alias(section: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in section:
            return section[key]
    return None


def _coerce_float(value: Any, path: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ControlConfigError(f"{path} must be numeric") from exc
def pixel_delta(
    u_px: float,
    v_px: float,
    ref_u_px: float,
    ref_v_px: float,
    config: ControlConfig,
    *,
    apply_deadband: bool = True,
) -> AxisPair:
    """Return signed pixel deltas between two image coordinates.

    Parameters
    ----------
    u_px, v_px:
        Pixel coordinates of the measured point (e.g. target centroid).
    ref_u_px, ref_v_px:
        Reference pixel coordinates (e.g. image centre or predicted laser dot).
    config:
        Parsed :class:`ControlConfig` providing axis sign conventions and
        deadband settings.
    apply_deadband:
        When ``True`` the configured deadband is applied to the resulting
        offsets.
    """

    du = config.yaw_sign * (u_px - ref_u_px)
    dv = config.pitch_sign * (v_px - ref_v_px)

    if apply_deadband and config.deadband_px > 0.0:
        if abs(du) <= config.deadband_px:
            du = 0.0
        if abs(dv) <= config.deadband_px:
            dv = 0.0

    return AxisPair(yaw=du, pitch=dv)


def pixel_error(
    u_px: float,
    v_px: float,
    config: ControlConfig,
    *,
    apply_deadband: bool = True,
) -> AxisPair:
    """Backward-compatible wrapper returning deltas to the image centre."""

    return pixel_delta(
        u_px,
        v_px,
        config.cx_px,
        config.cy_px,
        config,
        apply_deadband=apply_deadband,
    )


def angular_error_from_pixels(
    u_px: float,
    v_px: float,
    config: ControlConfig,
    *,
    linearize: bool = False,
    apply_deadband: bool = True,
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
    apply_deadband:
        When enabled the configured ``deadband_px`` is applied before the
        angular conversion.
    """

    px_err = pixel_error(u_px, v_px, config, apply_deadband=apply_deadband)

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
