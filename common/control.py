"""Shared helpers for loading the control-loop configuration.

This module centralizes parsing of the `control` section in the runtime
configuration so both Jetson and PC components can share consistent defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

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
class MPCHorizonConfig:
    """Prediction horizon parameters for the MPC controller."""

    steps: int
    step_dt_s: float


@dataclass(frozen=True)
class MPCCostWeights:
    """Quadratic cost weights applied within the MPC optimisation."""

    error: AxisPair
    rate: AxisPair
    accel: AxisPair


@dataclass(frozen=True)
class MPCActuatorLimits:
    """Rate and acceleration limits enforced by the MPC solver."""

    rate: AxisPair
    accel: AxisPair


@dataclass(frozen=True)
class MPCStateConstraints:
    """Optional bounds applied to the tracked state during optimisation."""

    error: Optional[AxisPair]
    rate: Optional[AxisPair]
    accel: Optional[AxisPair]


@dataclass(frozen=True)
class MPCConfig:
    """Aggregate configuration for the MPC controller."""

    horizon: MPCHorizonConfig
    cost: MPCCostWeights
    actuator_limits: MPCActuatorLimits
    state_constraints: MPCStateConstraints


@dataclass(frozen=True)
class ControlConfig:
    """Typed view over the `control` section of ``configs/dev.yaml``.

    The class normalizes units, derives focal lengths when requested, and
    exposes helpful pre-computed quantities (e.g. image center and axis signs).
    """

    mode: str
    controller_type: str
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
    mpc: Optional[MPCConfig]

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

        controller_type = str(control_section.get("controller_type", "pid")).strip().lower()
        if controller_type not in {"pid", "mpc"}:
            raise ControlConfigError(
                "control.controller_type must be either 'pid' or 'mpc'"
            )

        loop_hz = control_section.get("loop_hz")
        if loop_hz is not None:
            loop_hz = float(loop_hz)
            if loop_hz <= 0:
                raise ControlConfigError("loop_hz must be > 0 or null")

        loop_dt_default = None if loop_hz in (None, 0) else 1.0 / float(loop_hz)

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

        raw_mpc_section = control_section.get("mpc")
        if raw_mpc_section is None:
            if controller_type == "mpc":
                raise ControlConfigError(
                    "control.mpc section is required when controller_type is 'mpc'"
                )
            mpc_config = None
        else:
            if not isinstance(raw_mpc_section, Mapping):
                raise ControlConfigError("control.mpc must be a mapping when provided")
            mpc_config = _parse_mpc_config(
                raw_mpc_section,
                loop_dt_default=loop_dt_default,
                default_rate_limits=rate_limits,
                default_accel_limits=accel_limits,
            )

        cx_px = width / 2.0
        cy_px = height / 2.0

        return cls(
            mode=mode,
            controller_type=controller_type,
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
            mpc=mpc_config,
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


def _parse_mpc_config(
    section: Mapping[str, Any],
    *,
    loop_dt_default: Optional[float],
    default_rate_limits: AxisPair,
    default_accel_limits: AxisPair,
) -> MPCConfig:
    horizon_section = section.get("horizon", {}) or {}
    if not isinstance(horizon_section, Mapping):
        raise ControlConfigError("control.mpc.horizon must be a mapping when provided")

    try:
        steps = int(horizon_section.get("steps", 10))
    except (TypeError, ValueError) as exc:
        raise ControlConfigError("control.mpc.horizon.steps must be an integer") from exc
    if steps <= 0:
        raise ControlConfigError("control.mpc.horizon.steps must be positive")

    if "dt_s" in horizon_section:
        step_dt_raw = horizon_section["dt_s"]
    else:
        step_dt_raw = horizon_section.get("dt", loop_dt_default if loop_dt_default else 0.05)
    try:
        step_dt_s = float(step_dt_raw)
    except (TypeError, ValueError) as exc:
        raise ControlConfigError("control.mpc.horizon.dt_s must be numeric") from exc
    if step_dt_s <= 0.0:
        raise ControlConfigError("control.mpc.horizon.dt_s must be positive")

    cost_section = section.get("cost", {}) or {}
    if not isinstance(cost_section, Mapping):
        raise ControlConfigError("control.mpc.cost must be a mapping when provided")
    error_weights = _extract_axis_pair(
        cost_section,
        "error",
        allow_missing=True,
        default=AxisPair(1.0, 1.0),
    )
    rate_weights = _extract_axis_pair(
        cost_section,
        "rate",
        allow_missing=True,
        default=AxisPair(0.1, 0.1),
    )
    accel_weights = _extract_axis_pair(
        cost_section,
        "accel",
        allow_missing=True,
        default=AxisPair(0.01, 0.01),
    )

    actuator_section = section.get("actuator_limits", {}) or {}
    if not isinstance(actuator_section, Mapping):
        raise ControlConfigError(
            "control.mpc.actuator_limits must be a mapping when provided"
        )
    rate_limits = _extract_axis_pair(
        actuator_section,
        "rate",
        allow_missing=True,
        default=default_rate_limits,
    )
    accel_limits = _extract_axis_pair(
        actuator_section,
        "accel",
        allow_missing=True,
        default=default_accel_limits,
    )

    state_section = section.get("state_constraints", {}) or {}
    if not isinstance(state_section, Mapping):
        raise ControlConfigError(
            "control.mpc.state_constraints must be a mapping when provided"
        )
    error_bounds = _extract_optional_axis_pair(state_section, "error")
    rate_bounds = _extract_optional_axis_pair(state_section, "rate")
    accel_bounds = _extract_optional_axis_pair(state_section, "accel")

    return MPCConfig(
        horizon=MPCHorizonConfig(steps=steps, step_dt_s=step_dt_s),
        cost=MPCCostWeights(error=error_weights, rate=rate_weights, accel=accel_weights),
        actuator_limits=MPCActuatorLimits(rate=rate_limits, accel=accel_limits),
        state_constraints=MPCStateConstraints(
            error=error_bounds,
            rate=rate_bounds,
            accel=accel_bounds,
        ),
    )


def _extract_optional_axis_pair(section: Mapping[str, Any], key: str) -> Optional[AxisPair]:
    if key not in section:
        return None
    raw = section.get(key)
    if raw is None:
        return None
    try:
        yaw = float(raw["yaw"])
        pitch = float(raw["pitch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ControlConfigError(f"control.mpc.state_constraints.{key} must contain yaw/pitch floats") from exc
    return AxisPair(yaw=yaw, pitch=pitch)
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
