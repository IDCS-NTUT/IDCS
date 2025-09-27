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
class LaserRenderConfig:
    """Styling options for visualizing the laser beam and hit indicator."""

    beam_length_m: float
    color_bgr: Tuple[int, int, int]
    thickness_px: int
    hit_tolerance_px: float


@dataclass(frozen=True)
class LaserControlConfig:
    """Geometry and behaviour for the laser-aware aim mode."""

    offset_m: Tuple[float, float, float]
    dir_cam: Tuple[float, float, float]
    tolerance: float
    use_range: str
    default_distance_m: Optional[float]
    render: LaserRenderConfig


@dataclass(frozen=True)
class ControlConfig:
    """Typed view over the `control` section of ``configs/dev.yaml``.

    The class normalizes units, derives focal lengths when requested, and
    exposes helpful pre-computed quantities (e.g. image center and axis signs).
    """

    mode: str
    aim_mode: str
    loop_hz: Optional[float]
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
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
    laser: LaserControlConfig
    frame_size: Tuple[int, int]
    fov_deg: Optional[Tuple[float, float]]

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

        aim_mode = str(control_section.get("aim_mode", "camera_center")).strip().lower()
        valid_aim_modes = {"camera_center", "laser_point"}
        if aim_mode not in valid_aim_modes:
            raise ControlConfigError(
                f"unsupported control aim_mode: {aim_mode!r}; expected one of {sorted(valid_aim_modes)}"
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
        ki = _extract_axis_pair(control_section, "ki", allow_missing=True, default=AxisPair(0.0, 0.0))
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
        laser_cfg = _parse_laser_config(control_section)

        cx_px = width / 2.0
        cy_px = height / 2.0

        return cls(
            mode=mode,
            aim_mode=aim_mode,
            loop_hz=loop_hz,
            fx_px=fx_px,
            fy_px=fy_px,
            cx_px=cx_px,
            cy_px=cy_px,
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
            laser=laser_cfg,
            frame_size=(width, height),
            fov_deg=fov_deg,
        )


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


def _parse_laser_config(section: Mapping[str, Any]) -> LaserControlConfig:
    raw = section.get("laser", {}) or {}
    if not isinstance(raw, Mapping):
        raise ControlConfigError("control.laser must be a mapping when provided")

    offset = _extract_vec3(raw, "offset_m", default=(0.0, 0.0, 0.0))
    dir_cam = _extract_vec3(raw, "dir_cam", default=(0.0, 0.0, 1.0))
    dir_norm = _normalize_vec3(dir_cam, "control.laser.dir_cam")

    tolerance = float(raw.get("tolerance", 4.0))
    if tolerance < 0.0:
        raise ControlConfigError("control.laser.tolerance must be non-negative")

    use_range = str(raw.get("use_range", "auto")).strip().lower()
    valid_range_modes = {"auto", "known_size", "ground_plane", "infinite"}
    if use_range not in valid_range_modes:
        raise ControlConfigError(
            "control.laser.use_range must be one of {}".format(sorted(valid_range_modes))
        )

    default_distance_raw = raw.get("default_distance_m")
    default_distance = None
    if default_distance_raw is not None:
        try:
            default_distance = float(default_distance_raw)
        except (TypeError, ValueError) as exc:
            raise ControlConfigError("control.laser.default_distance_m must be numeric") from exc
        if default_distance <= 0.0:
            raise ControlConfigError("control.laser.default_distance_m must be positive")

    render = _parse_laser_render(raw.get("render"), tolerance)

    return LaserControlConfig(
        offset_m=offset,
        dir_cam=dir_norm,
        tolerance=tolerance,
        use_range=use_range,
        default_distance_m=default_distance,
        render=render,
    )


def _parse_laser_render(render_raw: Any, tolerance_default: float) -> LaserRenderConfig:
    if render_raw is None:
        return LaserRenderConfig(
            beam_length_m=10.0,
            color_bgr=(0, 0, 255),
            thickness_px=2,
            hit_tolerance_px=tolerance_default,
        )

    if not isinstance(render_raw, Mapping):
        raise ControlConfigError("control.laser.render must be a mapping when provided")

    try:
        beam_length = float(render_raw.get("beam_length_m", 10.0))
    except (TypeError, ValueError) as exc:
        raise ControlConfigError("control.laser.render.beam_length_m must be numeric") from exc
    if beam_length <= 0.0:
        raise ControlConfigError("control.laser.render.beam_length_m must be positive")

    color = _extract_color(render_raw, "color_bgr", default=(0, 0, 255))

    try:
        thickness = int(render_raw.get("thickness_px", 2))
    except (TypeError, ValueError) as exc:
        raise ControlConfigError("control.laser.render.thickness_px must be an integer") from exc
    if thickness <= 0:
        raise ControlConfigError("control.laser.render.thickness_px must be positive")

    try:
        hit_tol = float(render_raw.get("hit_tolerance_px", tolerance_default))
    except (TypeError, ValueError) as exc:
        raise ControlConfigError("control.laser.render.hit_tolerance_px must be numeric") from exc
    if hit_tol < 0.0:
        raise ControlConfigError("control.laser.render.hit_tolerance_px must be non-negative")

    return LaserRenderConfig(
        beam_length_m=beam_length,
        color_bgr=color,
        thickness_px=thickness,
        hit_tolerance_px=hit_tol,
    )


def _extract_vec3(
    section: Mapping[str, Any],
    key: str,
    *,
    default: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    raw = section.get(key)
    if raw is None:
        return default

    if isinstance(raw, Mapping):
        try:
            x = float(raw["x"])
            y = float(raw["y"])
            z = float(raw["z"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ControlConfigError(
                f"{key} mapping must contain numeric x, y, z entries"
            ) from exc
        return (x, y, z)

    if isinstance(raw, Sequence) and len(raw) == 3:
        try:
            return (float(raw[0]), float(raw[1]), float(raw[2]))
        except (TypeError, ValueError) as exc:
            raise ControlConfigError(f"{key} sequence must contain numeric values") from exc

    raise ControlConfigError(f"{key} must be a mapping or sequence of three numbers")


def _normalize_vec3(vec: Tuple[float, float, float], key: str) -> Tuple[float, float, float]:
    norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
    if norm <= 0.0:
        raise ControlConfigError(f"{key} must be a non-zero vector")
    return (vec[0] / norm, vec[1] / norm, vec[2] / norm)


def _extract_color(
    section: Mapping[str, Any],
    key: str,
    *,
    default: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    raw = section.get(key)
    if raw is None:
        return default

    if isinstance(raw, Mapping):
        # Accept either b/g/r keys or r/g/b for flexibility.
        lower_keys = {str(k).lower(): v for k, v in raw.items()}
        if {"b", "g", "r"}.issubset(lower_keys.keys()):
            ordered = (lower_keys["b"], lower_keys["g"], lower_keys["r"])
        elif {"r", "g", "b"}.issubset(lower_keys.keys()):
            ordered = (lower_keys["b"], lower_keys["g"], lower_keys["r"])
        else:
            raise ControlConfigError(
                f"{key} mapping must contain r/g/b or b/g/r components"
            )
    elif isinstance(raw, Sequence) and len(raw) == 3:
        ordered = raw
    else:
        raise ControlConfigError(f"{key} must be a sequence of three colour components")

    try:
        bgr = tuple(int(round(float(c))) for c in ordered)
    except (TypeError, ValueError) as exc:
        raise ControlConfigError(f"{key} components must be numeric") from exc

    for component in bgr:
        if not 0 <= component <= 255:
            raise ControlConfigError(f"{key} components must be in the range [0, 255]")

    return bgr  # type: ignore[return-value]
def pixel_error(
    u_px: float,
    v_px: float,
    config: ControlConfig,
    *,
    apply_deadband: bool = True,
) -> AxisPair:
    """Return signed pixel deltas from the image center using config signs.

    The yaw component corresponds to the horizontal error (u-axis) and the
    pitch component corresponds to the vertical error (v-axis). Positive signs
    follow the configured convention.
    """

    du = config.yaw_sign * (u_px - config.cx_px)
    dv = config.pitch_sign * (v_px - config.cy_px)

    if apply_deadband and config.deadband_px > 0.0:
        if abs(du) <= config.deadband_px:
            du = 0.0
        if abs(dv) <= config.deadband_px:
            dv = 0.0

    return AxisPair(yaw=du, pitch=dv)


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

    if linearize:
        yaw_err = px_err.yaw / config.fx_px
        pitch_err = px_err.pitch / config.fy_px
    else:
        yaw_err = math.atan(px_err.yaw / config.fx_px)
        pitch_err = math.atan(px_err.pitch / config.fy_px)

    return AxisPair(yaw=yaw_err, pitch=pitch_err)
