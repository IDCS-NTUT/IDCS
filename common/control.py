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


MPC_THETA_UNIT_SCALE_RAD = 0.03
MPC_OMEGA_UNIT_SCALE_RAD_S = 1.0
MPC_EFFORT_UNIT_SCALE = 8.0
MPC_SLEW_UNIT_SCALE = 50.0


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
class TrackerConfig:
    """Configuration for lightweight multi-target tracking."""

    enabled: bool = False
    min_hits: int = 2
    max_missed: int = 5
    iou_gate: float = 0.1
    center_dist_gate_px: float = 160.0
    use_hungarian: bool = False


@dataclass(frozen=True)
class MpcHorizonConfig:
    """Timing and horizon parameters for the MPC controller."""

    prediction_horizon: int
    control_horizon: int
    sample_time_s: float
    gamma: float
    move_blocking: bool
    effect_delay_mode: str = "fixed"
    effect_delay_s: float = 0.0
    projectile_speed_m_s: Optional[float] = None
    impact_delay_bias_s: float = 0.0
    predictor_enabled: bool = False
    predictor_alpha: float = 0.85
    predictor_beta: float = 0.05
    adaptive_effect_delay_enabled: bool = False
    adaptive_effect_delay_min_s: float = 0.0
    adaptive_effect_delay_max_s: float = 0.25
    adaptive_effect_delay_alpha: float = 0.1
    adaptive_effect_delay_gain: float = 0.2
    adaptive_effect_delay_rate_eps: float = 1e-3


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
    use_rate_measurement: bool = True


@dataclass(frozen=True)
class MpcCostConfig:
    """Static quadratic and linear MPC cost weights (shared across axes).

    These weights shape the objective described in the MPC controller docstring:

    ``J = tracking_cost + approach_cost + smoothness_cost``

    - ``q_theta``: quadratic positional tracking penalty (symmetric); increase to
      punish large angular error more aggressively.
    - ``l_theta``: signed positional tracking bias; direction follows the sign of
      the angular error after target-velocity scaling (set to 0 to disable).
    - ``q_omega``: quadratic rate-tracking penalty to keep velocities aligned.
    - ``q_dtheta``: quadratic penalty on the change in error across steps to
      temper oscillations and overshoot.
    - ``l_dtheta``: signed bias on error change; direction follows the sign of
      delta error after target-velocity scaling (set to 0 to disable).
    - ``r``: quadratic effort penalty on absolute control magnitude ``|u|``.
    - ``s``: quadratic smoothness penalty on ``Δu`` to discourage abrupt jumps.
        - ``l_du``: signed bias on ``Δu`` that steers the controller toward or away
            from positive command changes (set to 0 to disable).
    - ``terminal``: optional terminal state cost multiplier applied to the final
      theta error term when provided.
    - ``rho``: penalty on constraint slack when soft limits activate.

    Unit scales normalize raw values so tuning can be performed in intuitive
    physical units rather than solver magnitudes. These scales are fixed in
    code to keep behaviour consistent across deployments.
    """

    q_theta: float
    l_theta: float
    q_omega: float
    q_dtheta: float
    l_dtheta: float
    r: float
    s: float
    l_du: float
    terminal: Optional[float] = None
    rho: float = 0.0
    theta_unit_scale_rad: float = 1.0
    omega_unit_scale_rad_s: float = 1.0
    effort_unit_scale: float = 1.0
    slew_unit_scale: float = 1.0


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
class MpcOuterTunerConfig:
    """Configuration for periodic outer-loop MPC cost auto-tuning."""

    enabled: bool
    update_interval_s: float
    history_window_s: float
    min_samples: int
    target_abs_err_rad: float
    target_abs_cmd_rad_s: float
    step_up: float
    step_down: float
    min_scale: float
    max_scale: float
    parameter_group: Optional[str]
    weights: Tuple[str, ...]
    state_path: Optional[str] = None
    load_on_start: bool = True
    save_on_update: bool = True


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
        "theta_linear",
        "omega",
        "dtheta",
        "dtheta_linear",
        "effort",
        "slew",
        "slew_linear",
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
    outer_tuner: Optional[MpcOuterTunerConfig] = None


@dataclass(frozen=True)
class ControlConfig:
    """Typed view over the `control` section of ``configs/dev_extra.yaml``.

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
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    motion_vel_alpha: float = 0.2
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

        motion_vel_alpha = float(control_section.get("motion_vel_alpha", 0.2))
        if not 0.0 <= motion_vel_alpha <= 1.0:
            raise ControlConfigError("motion_vel_alpha must be between 0 and 1")

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

        if "tracker" in control_section:
            raise ControlConfigError(
                "control.tracker is no longer supported; use control.tracking.multi_target"
            )

        raw_tracking_section = control_section.get("tracking", {}) or {}
        if not isinstance(raw_tracking_section, Mapping):
            raise ControlConfigError("control.tracking must be a mapping when provided")

        raw_tracker_section = raw_tracking_section.get("multi_target", {}) or {}
        if not isinstance(raw_tracker_section, Mapping):
            raise ControlConfigError(
                "control.tracking.multi_target must be a mapping when provided"
            )

        tracker_enabled = bool(raw_tracker_section.get("enabled", False))
        try:
            tracker_min_hits = int(raw_tracker_section.get("min_hits", 2))
        except (TypeError, ValueError) as exc:
            raise ControlConfigError(
                "control.tracking.multi_target.min_hits must be an integer"
            ) from exc
        if tracker_min_hits < 1:
            raise ControlConfigError("control.tracking.multi_target.min_hits must be >= 1")

        try:
            tracker_max_missed = int(raw_tracker_section.get("max_missed", 5))
        except (TypeError, ValueError) as exc:
            raise ControlConfigError(
                "control.tracking.multi_target.max_missed must be an integer"
            ) from exc
        if tracker_max_missed < 1:
            raise ControlConfigError(
                "control.tracking.multi_target.max_missed must be >= 1"
            )

        try:
            tracker_iou_gate = float(raw_tracker_section.get("iou_gate", 0.1))
        except (TypeError, ValueError) as exc:
            raise ControlConfigError(
                "control.tracking.multi_target.iou_gate must be numeric"
            ) from exc
        if not 0.0 <= tracker_iou_gate <= 1.0:
            raise ControlConfigError(
                "control.tracking.multi_target.iou_gate must be within [0, 1]"
            )

        try:
            tracker_center_dist_gate_px = float(
                raw_tracker_section.get("center_dist_gate_px", 160.0)
            )
        except (TypeError, ValueError) as exc:
            raise ControlConfigError(
                "control.tracking.multi_target.center_dist_gate_px must be numeric"
            ) from exc
        if tracker_center_dist_gate_px <= 0.0:
            raise ControlConfigError(
                "control.tracking.multi_target.center_dist_gate_px must be > 0"
            )

        tracker_use_hungarian = bool(raw_tracker_section.get("use_hungarian", False))

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
            motion_vel_alpha=motion_vel_alpha,
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
            tracker=TrackerConfig(
                enabled=tracker_enabled,
                min_hits=tracker_min_hits,
                max_missed=tracker_max_missed,
                iou_gate=tracker_iou_gate,
                center_dist_gate_px=tracker_center_dist_gate_px,
                use_hungarian=tracker_use_hungarian,
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
    effect_delay = _parse_float_field(
        horizons,
        key="effect_delay_s",
        path="control.mpc.horizons.effect_delay_s",
        non_negative=True,
        default=0.0,
    )
    effect_delay_mode = str(horizons.get("effect_delay_mode", "fixed")).strip().lower()
    if effect_delay_mode not in {"fixed", "time_to_impact"}:
        raise ControlConfigError(
            "control.mpc.horizons.effect_delay_mode must be either 'fixed' or 'time_to_impact'"
        )
    projectile_speed = _parse_optional_float_field(
        horizons,
        key="projectile_speed_m_s",
        path="control.mpc.horizons.projectile_speed_m_s",
    )
    if projectile_speed is not None and projectile_speed <= 0.0:
        raise ControlConfigError("control.mpc.horizons.projectile_speed_m_s must be positive")
    if effect_delay_mode == "time_to_impact" and projectile_speed is None:
        raise ControlConfigError(
            "control.mpc.horizons.projectile_speed_m_s is required when effect_delay_mode='time_to_impact'"
        )
    impact_delay_bias = _parse_float_field(
        horizons,
        key="impact_delay_bias_s",
        path="control.mpc.horizons.impact_delay_bias_s",
        default=0.0,
    )
    gamma = _parse_float_field(
        horizons,
        key="gamma",
        path="control.mpc.horizons.gamma",
        positive=True,
        default=1.0,
    )
    move_blocking = _parse_bool_field(
        horizons,
        key="move_blocking",
        path="control.mpc.horizons.move_blocking",
        default=False,
    )
    predictor_enabled = _parse_bool_field(
        horizons,
        key="predictor_enabled",
        path="control.mpc.horizons.predictor_enabled",
        default=False,
    )
    predictor_alpha = _parse_float_field(
        horizons,
        key="predictor_alpha",
        path="control.mpc.horizons.predictor_alpha",
        non_negative=True,
        default=0.85,
    )
    predictor_beta = _parse_float_field(
        horizons,
        key="predictor_beta",
        path="control.mpc.horizons.predictor_beta",
        non_negative=True,
        default=0.05,
    )
    adaptive_effect_delay_enabled = _parse_bool_field(
        horizons,
        key="adaptive_effect_delay_enabled",
        path="control.mpc.horizons.adaptive_effect_delay_enabled",
        default=False,
    )
    adaptive_effect_delay_min_s = _parse_float_field(
        horizons,
        key="adaptive_effect_delay_min_s",
        path="control.mpc.horizons.adaptive_effect_delay_min_s",
        non_negative=True,
        default=0.0,
    )
    adaptive_effect_delay_max_s = _parse_float_field(
        horizons,
        key="adaptive_effect_delay_max_s",
        path="control.mpc.horizons.adaptive_effect_delay_max_s",
        non_negative=True,
        default=0.25,
    )
    if adaptive_effect_delay_max_s < adaptive_effect_delay_min_s:
        raise ControlConfigError(
            "control.mpc.horizons.adaptive_effect_delay_max_s cannot be less than adaptive_effect_delay_min_s"
        )
    adaptive_effect_delay_alpha = _parse_float_field(
        horizons,
        key="adaptive_effect_delay_alpha",
        path="control.mpc.horizons.adaptive_effect_delay_alpha",
        non_negative=True,
        default=0.1,
    )
    if adaptive_effect_delay_alpha > 1.0:
        raise ControlConfigError(
            "control.mpc.horizons.adaptive_effect_delay_alpha must be within [0, 1]"
        )
    adaptive_effect_delay_gain = _parse_float_field(
        horizons,
        key="adaptive_effect_delay_gain",
        path="control.mpc.horizons.adaptive_effect_delay_gain",
        non_negative=True,
        default=0.2,
    )
    adaptive_effect_delay_rate_eps = _parse_float_field(
        horizons,
        key="adaptive_effect_delay_rate_eps",
        path="control.mpc.horizons.adaptive_effect_delay_rate_eps",
        positive=True,
        default=1e-3,
    )

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
    use_rate_measurement = _parse_bool_field(
        estimator_section,
        key="use_rate_measurement",
        path="control.mpc.estimator.use_rate_measurement",
        default=True,
    )

    constraints_section = _require_mapping(raw, "constraints", path="control.mpc.constraints")
    costs_section = _require_mapping(raw, "costs", path="control.mpc.costs")
    q_theta_weight = _parse_float_field(
        costs_section,
        key="q_theta",
        path="control.mpc.costs.q_theta",
        non_negative=True,
    )
    l_theta_weight = _parse_float_field(
        costs_section,
        key="l_theta",
        path="control.mpc.costs.l_theta",
        default=0.0,
    )
    q_omega_weight = _parse_float_field(
        costs_section,
        key="q_omega",
        path="control.mpc.costs.q_omega",
        non_negative=True,
    )
    q_dtheta_weight = _parse_float_field(
        costs_section,
        key="q_dtheta",
        path="control.mpc.costs.q_dtheta",
        non_negative=True,
        default=0.0,
    )
    l_dtheta_weight = _parse_float_field(
        costs_section,
        key="l_dtheta",
        path="control.mpc.costs.l_dtheta",
        default=0.0,
    )
    r_weight = _parse_float_field(
        costs_section,
        key="r",
        path="control.mpc.costs.r",
        non_negative=True,
        default=0.0,
    )
    s_weight = _parse_float_field(
        costs_section,
        key="s",
        path="control.mpc.costs.s",
        non_negative=True,
        default=0.0,
    )
    l_du_weight = _parse_float_field(
        costs_section,
        key="l_du",
        path="control.mpc.costs.l_du",
        default=0.0,
    )
    terminal_weights = _parse_optional_float_field(
        costs_section,
        key="terminal",
        path="control.mpc.costs.terminal",
        non_negative=True,
    )
    rho = _parse_float_field(
        costs_section,
        key="rho",
        path="control.mpc.costs.rho",
        non_negative=True,
        default=0.0,
    )

    theta_unit_scale_rad = MPC_THETA_UNIT_SCALE_RAD
    omega_unit_scale_rad_s = MPC_OMEGA_UNIT_SCALE_RAD_S
    effort_unit_scale = MPC_EFFORT_UNIT_SCALE
    slew_unit_scale = MPC_SLEW_UNIT_SCALE

    constraints_section = _require_mapping(raw, "constraints", path="control.mpc.constraints")
    u_min = _parse_float_field(
        constraints_section,
        key="u_min",
        path="control.mpc.constraints.u_min",
    )
    u_max = _parse_float_field(
        constraints_section,
        key="u_max",
        path="control.mpc.constraints.u_max",
    )
    if u_min >= u_max:
        raise ControlConfigError("control.mpc.constraints.u_min must be less than u_max")
    du_max = _parse_float_field(
        constraints_section,
        key="du_max",
        path="control.mpc.constraints.du_max",
        positive=True,
    )
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
    if theta_min is not None and theta_max is not None and theta_min > theta_max:
        raise ControlConfigError("control.mpc.constraints.theta_min cannot exceed theta_max")
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
    if omega_min is not None and omega_max is not None and omega_min > omega_max:
        raise ControlConfigError("control.mpc.constraints.omega_min cannot exceed omega_max")

    outer_tuner = _parse_mpc_outer_tuner_config(raw)

    return MpcConfig(
        horizon=MpcHorizonConfig(
            prediction_horizon=prediction,
            control_horizon=control,
            sample_time_s=sample_time,
            effect_delay_mode=effect_delay_mode,
            effect_delay_s=effect_delay,
            projectile_speed_m_s=projectile_speed,
            impact_delay_bias_s=impact_delay_bias,
            gamma=gamma,
            move_blocking=move_blocking,
            predictor_enabled=predictor_enabled,
            predictor_alpha=predictor_alpha,
            predictor_beta=predictor_beta,
            adaptive_effect_delay_enabled=adaptive_effect_delay_enabled,
            adaptive_effect_delay_min_s=adaptive_effect_delay_min_s,
            adaptive_effect_delay_max_s=adaptive_effect_delay_max_s,
            adaptive_effect_delay_alpha=adaptive_effect_delay_alpha,
            adaptive_effect_delay_gain=adaptive_effect_delay_gain,
            adaptive_effect_delay_rate_eps=adaptive_effect_delay_rate_eps,
        ),
        plant=MpcPlantConfig(a_u=a_u, a_f=a_f),
        estimator=MpcEstimatorConfig(
            q_theta=q_theta,
            q_omega=q_omega,
            q_d=q_d,
            r_theta=r_theta,
            use_rate_measurement=use_rate_measurement,
        ),
        costs=MpcCostConfig(
            q_theta=q_theta_weight,
            l_theta=l_theta_weight,
            q_omega=q_omega_weight,
            q_dtheta=q_dtheta_weight,
            l_dtheta=l_dtheta_weight,
            r=r_weight,
            s=s_weight,
            l_du=l_du_weight,
            terminal=terminal_weights,
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
        outer_tuner=outer_tuner,
    )


def _parse_mpc_outer_tuner_config(raw_mpc: Mapping[str, Any]) -> Optional[MpcOuterTunerConfig]:
    raw = raw_mpc.get("outer_tuner")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ControlConfigError("control.mpc.outer_tuner must be a mapping when provided")

    enabled = _parse_bool_field(
        raw,
        key="enabled",
        path="control.mpc.outer_tuner.enabled",
        default=False,
    )
    update_interval_s = _parse_float_field(
        raw,
        key="update_interval_s",
        path="control.mpc.outer_tuner.update_interval_s",
        positive=True,
        default=3.0,
    )
    history_window_s = _parse_float_field(
        raw,
        key="history_window_s",
        path="control.mpc.outer_tuner.history_window_s",
        positive=True,
        default=8.0,
    )
    raw_min_samples = raw.get("min_samples", 40)
    try:
        min_samples = int(raw_min_samples)
    except (TypeError, ValueError) as exc:
        raise ControlConfigError("control.mpc.outer_tuner.min_samples must be an integer") from exc
    if min_samples <= 0:
        raise ControlConfigError("control.mpc.outer_tuner.min_samples must be positive")
    target_abs_err_rad = _parse_float_field(
        raw,
        key="target_abs_err_rad",
        path="control.mpc.outer_tuner.target_abs_err_rad",
        positive=True,
        default=0.02,
    )
    target_abs_cmd_rad_s = _parse_float_field(
        raw,
        key="target_abs_cmd_rad_s",
        path="control.mpc.outer_tuner.target_abs_cmd_rad_s",
        positive=True,
        default=0.05,
    )
    step_up = _parse_float_field(
        raw,
        key="step_up",
        path="control.mpc.outer_tuner.step_up",
        positive=True,
        default=0.1,
    )
    step_down = _parse_float_field(
        raw,
        key="step_down",
        path="control.mpc.outer_tuner.step_down",
        positive=True,
        default=0.05,
    )
    min_scale = _parse_float_field(
        raw,
        key="min_scale",
        path="control.mpc.outer_tuner.min_scale",
        positive=True,
        default=0.4,
    )
    max_scale = _parse_float_field(
        raw,
        key="max_scale",
        path="control.mpc.outer_tuner.max_scale",
        positive=True,
        default=2.5,
    )
    if min_scale > max_scale:
        raise ControlConfigError(
            "control.mpc.outer_tuner.min_scale cannot exceed max_scale"
        )

    allowed = {"q_theta", "q_omega", "q_dtheta", "r", "s", "terminal", "rho"}
    allowed_groups = {
        "costs_tracking",
        "costs_effort",
        "estimator",
        "predictor",
        "adaptive_delay",
    }
    group_raw = raw.get("parameter_group")
    parameter_group: Optional[str]
    if group_raw is None:
        parameter_group = None
    else:
        parameter_group = str(group_raw).strip().lower()
        if not parameter_group:
            parameter_group = None
        elif parameter_group not in allowed_groups:
            valid = ", ".join(sorted(allowed_groups))
            raise ControlConfigError(
                f"control.mpc.outer_tuner.parameter_group must be in: {valid}"
            )
    weights_raw = raw.get("weights")
    if weights_raw is None:
        weights = ("q_theta", "q_dtheta", "r", "s")
    else:
        if not isinstance(weights_raw, Sequence) or isinstance(weights_raw, (str, bytes)):
            raise ControlConfigError("control.mpc.outer_tuner.weights must be a list of strings")
        normalized = []
        for value in weights_raw:
            name = str(value).strip().lower()
            if not name:
                raise ControlConfigError("control.mpc.outer_tuner.weights entries must be non-empty")
            if name not in allowed:
                valid = ", ".join(sorted(allowed))
                raise ControlConfigError(
                    f"control.mpc.outer_tuner.weights entries must be in: {valid}"
                )
            if name not in normalized:
                normalized.append(name)
        if not normalized:
            raise ControlConfigError("control.mpc.outer_tuner.weights cannot be empty")
        weights = tuple(normalized)

    state_path_raw = raw.get("state_path")
    state_path: Optional[str]
    if state_path_raw is None:
        state_path = None
    else:
        state_path = str(state_path_raw).strip()
        if not state_path:
            state_path = None

    load_on_start = _parse_bool_field(
        raw,
        key="load_on_start",
        path="control.mpc.outer_tuner.load_on_start",
        default=True,
    )
    save_on_update = _parse_bool_field(
        raw,
        key="save_on_update",
        path="control.mpc.outer_tuner.save_on_update",
        default=True,
    )

    return MpcOuterTunerConfig(
        enabled=enabled,
        update_interval_s=update_interval_s,
        history_window_s=history_window_s,
        min_samples=min_samples,
        target_abs_err_rad=target_abs_err_rad,
        target_abs_cmd_rad_s=target_abs_cmd_rad_s,
        step_up=step_up,
        step_down=step_down,
        min_scale=min_scale,
        max_scale=max_scale,
        parameter_group=parameter_group,
        weights=weights,
        state_path=state_path,
        load_on_start=load_on_start,
        save_on_update=save_on_update,
    )


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


def _parse_bool_field(
    section: Mapping[str, Any],
    *,
    key: str,
    path: str,
    aliases: Sequence[str] = (),
    default: Optional[bool] = None,
) -> bool:
    raw_value = _get_with_alias(section, key, *aliases)
    if raw_value is None:
        if default is not None:
            return default
        raise ControlConfigError(f"{path} is required")

    if isinstance(raw_value, bool):
        return raw_value

    if isinstance(raw_value, (int, float)):
        if raw_value in (0, 0.0):
            return False
        if raw_value in (1, 1.0):
            return True
        raise ControlConfigError(f"{path} must be a boolean")

    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
        raise ControlConfigError(f"{path} must be a boolean")

    raise ControlConfigError(f"{path} must be a boolean")


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
    if not math.isfinite(value):
        raise ControlConfigError(f"{path} must be finite")
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
    if not math.isfinite(value):
        raise ControlConfigError(f"{path} must be finite")
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
