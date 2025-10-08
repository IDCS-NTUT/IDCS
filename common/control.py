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
class MotionModelDerotationConfig:
    """Configuration for removing/adding camera-induced motion."""

    enabled: bool
    rate_scale: float


@dataclass(frozen=True)
class MotionModelNoiseConfig:
    """Tuning knobs for the motion-model filter noise terms."""

    process_px: AxisPair
    measurement_px: AxisPair
    auto_inflate_with_rates: bool
    max_inflate_scale: float


@dataclass(frozen=True)
class MotionModelWorldFrameConfig:
    """Configuration placeholders for world-frame motion modelling."""

    camera_height_m: float
    default_target_height_m: Optional[float]
    assume_level_ground: bool


@dataclass(frozen=True)
class MotionModelConfig:
    """High-level configuration for target-motion prediction."""

    mode: str
    latency_ms_source: str
    prediction_horizon_ms: Optional[float]
    derotation: MotionModelDerotationConfig
    noise: MotionModelNoiseConfig
    apply_to_control: bool = True
    min_prediction_horizon_ms: Optional[float] = 0.0
    max_prediction_horizon_ms: Optional[float] = 250.0
    world_frame: MotionModelWorldFrameConfig = MotionModelWorldFrameConfig(
        camera_height_m=0.0,
        default_target_height_m=None,
        assume_level_ground=True,
    )


def _parse_motion_model_config(section: Mapping[str, Any]) -> MotionModelConfig:
    raw = section.get("motion_model", {}) or {}
    if not isinstance(raw, Mapping):
        raise ControlConfigError("control.motion_model must be a mapping when provided")

    mode = str(raw.get("mode", "camera_frame")).strip().lower()
    valid_modes = {"camera_frame", "world_frame"}
    if mode not in valid_modes:
        raise ControlConfigError(
            "control.motion_model.mode must be one of "
            f"{sorted(valid_modes)}, got {mode!r}"
        )

    latency_ms_source = str(raw.get("latency_ms_source", "auto")).strip()
    if not latency_ms_source:
        latency_ms_source = "auto"

    raw_horizon = raw.get("prediction_horizon_ms", "auto")
    prediction_horizon_ms: Optional[float]
    if raw_horizon is None:
        prediction_horizon_ms = None
    elif isinstance(raw_horizon, str) and raw_horizon.strip().lower() == "auto":
        prediction_horizon_ms = None
    else:
        try:
            prediction_horizon_ms = float(raw_horizon)
        except (TypeError, ValueError) as exc:
            raise ControlConfigError(
                "control.motion_model.prediction_horizon_ms must be a positive number or 'auto'"
            ) from exc
        if prediction_horizon_ms <= 0.0:
            raise ControlConfigError(
                "control.motion_model.prediction_horizon_ms must be positive when specified"
            )

    def _parse_optional_bound(name: str, default: Optional[float]) -> Optional[float]:
        value = raw.get(name, default)
        if value is None:
            return None
        try:
            bound = float(value)
        except (TypeError, ValueError) as exc:
            raise ControlConfigError(
                f"control.motion_model.{name} must be numeric or null"
            ) from exc
        if bound < 0.0:
            raise ControlConfigError(
                f"control.motion_model.{name} must be non-negative when specified"
            )
        return bound

    min_horizon_ms = _parse_optional_bound("prediction_horizon_min_ms", 0.0)
    max_horizon_ms = _parse_optional_bound("prediction_horizon_max_ms", 250.0)
    if (
        min_horizon_ms is not None
        and max_horizon_ms is not None
        and min_horizon_ms > max_horizon_ms
    ):
        raise ControlConfigError(
            "control.motion_model.prediction_horizon_min_ms must be <= prediction_horizon_max_ms"
        )

    apply_raw = raw.get("apply_to_control", True)
    if isinstance(apply_raw, bool):
        apply_to_control = apply_raw
    elif isinstance(apply_raw, str):
        lowered = apply_raw.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            apply_to_control = True
        elif lowered in {"false", "0", "no", "off"}:
            apply_to_control = False
        else:
            raise ControlConfigError(
                "control.motion_model.apply_to_control must be a boolean or boolean-like string"
            )
    else:
        raise ControlConfigError(
            "control.motion_model.apply_to_control must be a boolean"
        )

    derotation_section = raw.get("derotation", {}) or {}
    if not isinstance(derotation_section, Mapping):
        raise ControlConfigError(
            "control.motion_model.derotation must be a mapping when provided"
        )
    derotation_enabled = bool(derotation_section.get("enabled", True))
    try:
        rate_scale = float(derotation_section.get("rate_scale", 1.0))
    except (TypeError, ValueError) as exc:
        raise ControlConfigError(
            "control.motion_model.derotation.rate_scale must be numeric"
        ) from exc
    if rate_scale <= 0.0:
        raise ControlConfigError(
            "control.motion_model.derotation.rate_scale must be positive"
        )

    noise_section = raw.get("noise", {}) or {}
    if not isinstance(noise_section, Mapping):
        raise ControlConfigError("control.motion_model.noise must be a mapping when provided")

    process_default = AxisPair(0.0, 0.0)
    measurement_default = AxisPair(0.0, 0.0)
    process_px = _extract_axis_pair(
        noise_section,
        "process_px",
        allow_missing=True,
        default=process_default,
    )
    measurement_px = _extract_axis_pair(
        noise_section,
        "measurement_px",
        allow_missing=True,
        default=measurement_default,
    )

    auto_inflate_with_rates = bool(noise_section.get("auto_inflate_with_rates", False))
    try:
        max_inflate_scale = float(noise_section.get("max_inflate_scale", 3.0))
    except (TypeError, ValueError) as exc:
        raise ControlConfigError(
            "control.motion_model.noise.max_inflate_scale must be numeric"
        ) from exc
    if max_inflate_scale < 1.0:
        raise ControlConfigError(
            "control.motion_model.noise.max_inflate_scale must be >= 1"
        )

    world_section = raw.get("world_frame", {}) or {}
    if not isinstance(world_section, Mapping):
        raise ControlConfigError(
            "control.motion_model.world_frame must be a mapping when provided"
        )

    try:
        camera_height_m = float(world_section.get("camera_height_m", 0.0))
    except (TypeError, ValueError) as exc:
        raise ControlConfigError(
            "control.motion_model.world_frame.camera_height_m must be numeric"
        ) from exc

    default_target_height_raw = world_section.get("default_target_height_m")
    if default_target_height_raw is None:
        default_target_height_m: Optional[float] = None
    else:
        try:
            default_target_height_m = float(default_target_height_raw)
        except (TypeError, ValueError) as exc:
            raise ControlConfigError(
                "control.motion_model.world_frame.default_target_height_m must be numeric or null"
            ) from exc

    assume_level_ground = bool(world_section.get("assume_level_ground", True))

    return MotionModelConfig(
        mode=mode,
        latency_ms_source=latency_ms_source,
        prediction_horizon_ms=prediction_horizon_ms,
        derotation=MotionModelDerotationConfig(
            enabled=derotation_enabled,
            rate_scale=rate_scale,
        ),
        noise=MotionModelNoiseConfig(
            process_px=process_px,
            measurement_px=measurement_px,
            auto_inflate_with_rates=auto_inflate_with_rates,
            max_inflate_scale=max_inflate_scale,
        ),
        apply_to_control=apply_to_control,
        min_prediction_horizon_ms=min_horizon_ms,
        max_prediction_horizon_ms=max_horizon_ms,
        world_frame=MotionModelWorldFrameConfig(
            camera_height_m=camera_height_m,
            default_target_height_m=default_target_height_m,
            assume_level_ground=assume_level_ground,
        ),
    )


@dataclass(frozen=True)
class LaserAimingControlConfig:
    """Controller-specific parameters for laser-based aiming."""

    tolerance_px: float
    use_range: str
    default_distance_m: float


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
    motion_model: MotionModelConfig

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
            motion_model=_parse_motion_model_config(control_section),
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
