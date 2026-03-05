"""Closed-loop controller that turns detections into pan/tilt commands."""

from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import zmq

from common.control import (
    AxisPair,
    ControlConfig,
    MpcConfig,
    MpcOuterTunerConfig,
    LaserMountConfig,
    angular_error_from_pixel_delta,
    pixel_delta,
)
from common.geometry import laser_ray_to_pixel, project_point_to_pixel
from common.schemas import Box, CamState, ControlCmd, DetectionMsg
try:
    from jetson.mpc import MpcAxisController, MpcAxisDiagnostics, MpcSolverError
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    MpcAxisController = None  # type: ignore[assignment]
    MpcAxisDiagnostics = None  # type: ignore[assignment]
    MpcSolverError = RuntimeError  # type: ignore[assignment]

try:
    from jetson.mpc_refs import MpcReferenceBuilder
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    MpcReferenceBuilder = None  # type: ignore[assignment]


_LOG = logging.getLogger("jetson.control")


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class _DetectionState:
    frame_id: int
    src_ts_ms: int
    timestamp: float
    target_uv: Optional[Tuple[float, float]]
    target_distance_m: Optional[float]
    resolved_range_m: Optional[float]
    range_source: Optional[str]
    range_active: bool
    target_velocity_px_s: Optional[Tuple[float, float]]


@dataclass
class _LaserOverlay:
    origin_px: Optional[Tuple[float, float]]
    dot_px: Optional[Tuple[float, float]]
    on_target: Optional[bool]
    active: bool
    range_m: Optional[float]
    range_source: Optional[str]


@dataclass
class _MotionState:
    timestamp: float
    measurement_timestamp: float
    yaw_angle: float
    pitch_angle: float
    yaw_rate: float
    pitch_rate: float


@dataclass
class _MpcTuneSample:
    timestamp: float
    abs_err_rad: float
    abs_cmd_rad_s: float


class ControlLoop:
    """Runs the configured controller (PID or MPC) for yaw/pitch tracking."""

    _MIN_DT = 1e-3
    _MAX_DT = 0.2

    def __init__(
        self,
        config: ControlConfig,
        pub: zmq.Socket,
        *,
        laser_mount: Optional[LaserMountConfig] = None,
        distance_alpha: Optional[float] = None,
        cli_json_logs: bool = False,
        mpc_axis_factory: Optional[
            Callable[[str, ControlConfig, MpcConfig], MpcAxisController]
        ] = None,
    ) -> None:
        self._cfg = config
        self._pub = pub
        self._laser_mount = laser_mount

        self._lost_timeout_s = config.lost_target_timeout_ms / 1000.0
        self._default_dt = (
            config.loop_dt if config.loop_dt is not None else 1.0 / max(1.0, config.loop_hz or 30.0)
        )

        self._latest_detection: Optional[_DetectionState] = None
        self._latest_target_idx: Optional[int] = None
        self._last_frame_id: int = 0
        self._last_src_ts_ms: int = 0
        self._last_detection_ts: Optional[float] = None

        self._smoothed_uv: Optional[Tuple[float, float]] = None
        self._prev_err: Optional[AxisPair] = None
        self._integ = AxisPair(0.0, 0.0)
        self._prev_rate = AxisPair(0.0, 0.0)
        self._last_cmd_time: Optional[float] = None
        self._tracking_active = False

        self._distance_alpha = None if distance_alpha is None else _clamp(distance_alpha, 0.0, 1.0)
        self._distance_ema: Optional[float] = None
        self._resolved_range: Optional[float] = None
        self._warned_ground_plane = False

        self._cam_state: Optional[CamState] = None
        self._home_pan: Optional[float] = None
        self._home_tilt: Optional[float] = None
        self._home_deadband = math.radians(0.5)
        self._last_home_yaw_err: Optional[float] = None

        self._motion_state: Optional[_MotionState] = None
        self._motion_target_idx: Optional[int] = None
        self._vel_ema: Optional[AxisPair] = None
        self._vel_alpha = _clamp(self._cfg.motion_vel_alpha, 0.0, 1.0)
        self._lead_time_s = max(self._default_dt, 1e-3)
        self._lead_latency_alpha = 0.2
        self._lead_latency_ready = False
        self._last_motion_rates: Optional[AxisPair] = None
        self._last_known_target_uv: Optional[Tuple[float, float]] = None
        self._predictive_end_time: Optional[float] = None
        self._predictive_rates: Optional[AxisPair] = None
        self._last_target_box_size_px: Optional[Tuple[float, float]] = None

        self._mpc_enabled = config.controller == "mpc"
        self._mpc_builder: Optional[MpcReferenceBuilder] = None
        self._mpc_axes: Dict[str, MpcAxisController] = {}
        self._mpc_axis_names: Tuple[str, ...] = tuple()
        self._mpc_last_applied: Dict[str, float] = {}
        self._mpc_theta_estimates: Dict[str, float] = {}
        self._mpc_omega_estimates: Dict[str, float] = {}
        self._mpc_last_diag: Dict[str, Optional[MpcAxisDiagnostics]] = {}
        self._mpc_missing_measurement_last_log: float = 0.0
        self._mpc_outer_tuner_cfg: Optional[MpcOuterTunerConfig] = None
        self._mpc_outer_tuner_history = deque()
        self._mpc_outer_tuner_scales: Dict[str, float] = {}
        self._mpc_outer_tuner_group: Optional[str] = None
        self._mpc_outer_tuner_next_time: float = 0.0
        self._mpc_outer_tuner_state_path: Optional[Path] = None
        self._mpc_outer_tuner_save_on_update: bool = True

        if self._mpc_enabled:
            if config.mpc is None:
                raise ValueError("MPC configuration is required when controller='mpc'")
            if MpcReferenceBuilder is None:
                raise RuntimeError(
                    "MPC reference utilities are not available; cannot enable MPC"
                )
            if mpc_axis_factory is None and MpcAxisController is None:
                raise RuntimeError(
                    "MPC solver dependencies are not installed; cannot enable MPC"
                )
            axis_factory = mpc_axis_factory or (
                lambda axis_name, ctrl_cfg, mpc_cfg: MpcAxisController(
                    axis_name, ctrl_cfg, mpc_cfg
                )
            )
            self._mpc_builder = MpcReferenceBuilder(
                control_cfg=config,
                horizon_cfg=config.mpc.horizon,
                axes=("yaw", "pitch"),
            )
            self._mpc_axis_names = self._mpc_builder.axes
            for axis in self._mpc_axis_names:
                controller = axis_factory(axis, config, config.mpc)
                self._mpc_axes[axis] = controller
                self._mpc_last_applied[axis] = 0.0
                self._mpc_theta_estimates[axis] = 0.0
                self._mpc_omega_estimates[axis] = 0.0
                self._mpc_last_diag[axis] = None

            if config.mpc.outer_tuner is not None and config.mpc.outer_tuner.enabled:
                self._mpc_outer_tuner_cfg = config.mpc.outer_tuner
                self._mpc_outer_tuner_group = config.mpc.outer_tuner.parameter_group
                if self._mpc_outer_tuner_group:
                    self._mpc_outer_tuner_scales = {
                        name: 1.0
                        for name in self._outer_tuner_group_parameters(
                            self._mpc_outer_tuner_group
                        )
                    }
                else:
                    self._mpc_outer_tuner_scales = {
                        name: 1.0 for name in config.mpc.outer_tuner.weights
                    }
                self._mpc_outer_tuner_save_on_update = bool(
                    config.mpc.outer_tuner.save_on_update
                )

                if config.mpc.outer_tuner.state_path:
                    candidate = Path(config.mpc.outer_tuner.state_path).expanduser()
                    if not candidate.is_absolute():
                        candidate = Path.cwd() / candidate
                    self._mpc_outer_tuner_state_path = candidate

                if config.mpc.outer_tuner.load_on_start:
                    self._restore_mpc_outer_tuner_state()

                self._apply_mpc_outer_tuner_overrides()

        self._log_interval_s = 0.5
        self._last_log_time = 0.0
        self._last_log_target_ok: Optional[bool] = None
        self._log_float_precision = 4
        self._log_json = cli_json_logs
        self._laser_overlay: Optional[_LaserOverlay] = None

        selector = (config.target_selector or "max_conf").strip().lower()
        self._selector_strategy = "max_conf"
        self._class_filter: Optional[str] = None
        if selector.startswith("class:"):
            self._class_filter = selector.split(":", 1)[1].strip()
        elif selector == "largest_area":
            self._selector_strategy = "largest_area"

        # keep logs terse JSON; if nothing configured ensure we emit info-level lines
        if not _LOG.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            _LOG.addHandler(handler)
            _LOG.propagate = False
        _LOG.setLevel(logging.INFO)

    @staticmethod
    def _outer_tuner_group_parameters(group: str) -> Tuple[str, ...]:
        mapping = {
            "costs_tracking": ("q_theta", "q_omega", "q_dtheta", "terminal"),
            "costs_effort": ("r", "s", "rho"),
            "estimator": ("est_q_theta", "est_q_omega", "est_q_d", "est_r_theta"),
            "predictor": ("predictor_alpha", "predictor_beta"),
            "adaptive_delay": (
                "adaptive_effect_delay_gain",
                "adaptive_effect_delay_alpha",
                "adaptive_effect_delay_rate_eps",
            ),
        }
        return mapping.get(group, tuple())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def log_interval_s(self) -> float:
        """Return the minimum interval between consecutive info logs."""

        return self._log_interval_s

    def update_detection(self, msg: DetectionMsg) -> None:
        """Consume the newest detection message."""

        now = time.monotonic()
        prev_had_target = (
            self._latest_detection is not None
            and self._latest_detection.target_uv is not None
        )
        self._update_latency_estimate(msg, now)
        target_uv = self._select_target(msg)

        if target_uv is None:
            self._distance_ema = None
            msg.target_distance_smoothed_m = None
            msg.laser_origin_px = None
            msg.laser_dot_px = None
            msg.laser_on_target = None
            msg.laser_range_m = None
            msg.laser_range_source = None
            msg.parallax_compensation_active = False
            self._laser_overlay = None
            self._resolved_range = None
            msg.target_velocity_px_s = None
            msg.target_lead_uv = None
            msg.target_lead_time_s = None

        if target_uv is not None:
            self._last_known_target_uv = (float(target_uv[0]), float(target_uv[1]))

        self._latest_detection = _DetectionState(
            frame_id=msg.frame_id,
            src_ts_ms=msg.src_ts_ms,
            timestamp=now,
            target_uv=target_uv,
            target_distance_m=msg.target_distance_smoothed_m,
            resolved_range_m=None,
            range_source=None,
            range_active=False,
            target_velocity_px_s=msg.target_velocity_px_s,
        )
        self._last_frame_id = msg.frame_id
        self._last_src_ts_ms = msg.src_ts_ms

        if target_uv is not None:
            self._last_detection_ts = now
            if self._smoothed_uv is None or not self._tracking_active:
                self._smoothed_uv = target_uv
            else:
                self._smoothed_uv = self._smooth_uv(target_uv)

            if (
                self._latest_target_idx is not None
                and 0 <= self._latest_target_idx < len(msg.boxes)
            ):
                selected_box = msg.boxes[self._latest_target_idx]
                self._last_target_box_size_px = (
                    float(selected_box.w * msg.img_w),
                    float(selected_box.h * msg.img_h),
                )

            range_m, range_source, parallax_active = self._resolve_laser_range(
                msg.target_distance_smoothed_m
            )
            msg.laser_range_m = range_m
            msg.laser_range_source = range_source

            self._latest_detection.resolved_range_m = range_m
            self._latest_detection.range_source = range_source
            self._latest_detection.range_active = parallax_active
            measurement_timestamp = self._measurement_timestamp_from_msg(msg, fallback=now)
            self._update_motion_state(
                msg,
                target_uv=target_uv,
                timestamp=now,
                measurement_timestamp=measurement_timestamp,
                target_idx=self._latest_target_idx,
            )
            self._clear_predictive_mode()
        else:
            msg.laser_range_m = None
            msg.laser_range_source = None
            self._resolved_range = None
            parallax_active = False
            msg.target_velocity_px_s = None
            msg.target_lead_uv = None
            msg.target_lead_time_s = None

            if prev_had_target:
                self._start_predictive_mode(now)

            if not self._is_predictive_active(now):
                self._motion_state = None
                self._vel_ema = None
            self._motion_target_idx = None

        if target_uv is None:
            range_m = None
            range_source = None
            parallax_active = False
        else:
            range_m = self._latest_detection.resolved_range_m
            range_source = self._latest_detection.range_source

        self._update_laser_overlay(
            msg,
            raw_target_uv=target_uv,
            range_m=range_m,
            range_source=range_source,
            parallax_active=parallax_active,
        )
        self._populate_predictive_overlay(msg, now)

    def update_cam_state(self, state: CamState) -> None:
        self._cam_state = state
        if state.home_pan is not None:
            self._home_pan = float(state.home_pan)
        if state.home_tilt is not None:
            self._home_tilt = float(state.home_tilt)

    def tick(self, now: Optional[float] = None) -> None:
        """Advance the controller and publish a command if due."""

        if now is None:
            now = time.monotonic()

        if self._cfg.loop_dt is not None and self._last_cmd_time is not None:
            if (now - self._last_cmd_time) < (self._cfg.loop_dt - 1e-6):
                return

        dt = self._compute_dt(now)

        if self._mpc_enabled:
            self._update_mpc_estimates()

        detection = self._latest_detection
        target_recent = self._is_target_recent(now)

        if detection and detection.target_uv is not None and target_recent:
            if self._mpc_enabled:
                cmd = self._build_mpc_tracking_cmd(detection, dt, now)
            else:
                cmd = self._build_tracking_cmd(detection, dt, now)
            self._tracking_active = True
            self._last_home_yaw_err = None
        elif self._is_predictive_active(now):
            cmd = self._build_predictive_cmd(dt, now)
            self._tracking_active = False
            self._last_home_yaw_err = None
        else:
            if self._cfg.reinit_on_lost:
                self._prev_err = None
                self._integ = AxisPair(0.0, 0.0)
                self._smoothed_uv = None
                self._prev_rate = AxisPair(0.0, 0.0)
                self._clear_predictive_mode()
            cmd = self._build_hold_cmd(now)
            self._tracking_active = False

        self._last_cmd_time = now
        self._send_cmd(cmd)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _select_target(self, msg: DetectionMsg) -> Optional[Tuple[float, float]]:
        boxes: Sequence[Box] = msg.boxes
        prev_idx = self._latest_target_idx
        self._latest_target_idx = None
        msg.target_idx = None
        msg.target_distance_smoothed_m = None

        if not boxes:
            self._distance_ema = None
            return None

        enumerated: Sequence[Tuple[int, Box]] = list(enumerate(boxes))

        # Enforce tracking only targets labelled as 'drone'. This overrides
        # any configured class selector so the controller ignores non-drone
        # detections entirely.
        enumerated = [pair for pair in enumerated if pair[1].cls == "drone"]
        if not enumerated:
            self._distance_ema = None
            return None

        if self._class_filter:
            enumerated = [pair for pair in enumerated if pair[1].cls == self._class_filter]
            if not enumerated:
                self._distance_ema = None
                return None

        if self._selector_strategy == "largest_area":
            best_idx, best = max(enumerated, key=lambda item: item[1].w * item[1].h)
        else:
            best_idx, best = max(enumerated, key=lambda item: item[1].conf)

        self._latest_target_idx = best_idx
        msg.target_idx = best_idx
        self._update_target_distance(msg, best, previous_idx=prev_idx)

        u = (best.x + (best.w / 2.0)) * msg.img_w
        v = (best.y + (best.h / 2.0)) * msg.img_h
        return (u, v)

    def _update_target_distance(
        self,
        msg: DetectionMsg,
        box: Box,
        *,
        previous_idx: Optional[int],
    ) -> None:
        measurement = box.distance_m
        if measurement is None or not math.isfinite(measurement):
            msg.target_distance_smoothed_m = self._distance_ema
            return

        alpha = self._distance_alpha
        if previous_idx != self._latest_target_idx:
            self._distance_ema = None

        if alpha is None:
            self._distance_ema = measurement
        else:
            if self._distance_ema is None:
                self._distance_ema = measurement
            else:
                if alpha <= 0.0 or alpha >= 1.0:
                    self._distance_ema = measurement
                else:
                    self._distance_ema = (
                        alpha * measurement + (1.0 - alpha) * self._distance_ema
                    )

        msg.target_distance_smoothed_m = self._distance_ema

    def _smooth_uv(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        alpha = _clamp(self._cfg.smooth_px_alpha, 0.0, 1.0)
        if self._smoothed_uv is None or alpha <= 0.0:
            return measurement
        if alpha >= 1.0:
            return measurement
        prev_u, prev_v = self._smoothed_uv
        meas_u, meas_v = measurement
        new_u = alpha * meas_u + (1.0 - alpha) * prev_u
        new_v = alpha * meas_v + (1.0 - alpha) * prev_v
        return (new_u, new_v)

    def _resolve_laser_range(
        self, smoothed_distance: Optional[float]
    ) -> Tuple[Optional[float], Optional[str], bool]:
        if self._cfg.aim_mode != "laser_point" or self._laser_mount is None:
            self._resolved_range = None
            return None, None, False

        measurement: Optional[float] = None
        if smoothed_distance is not None:
            measurement = float(smoothed_distance)
            if not math.isfinite(measurement) or measurement <= 0.0:
                measurement = None

        policy = self._cfg.laser.use_range
        default_distance = float(self._cfg.laser.default_distance_m)
        resolved: Optional[float]
        source: Optional[str]

        if policy in {"known_size", "auto"}:
            if measurement is not None:
                resolved = measurement
                source = "known_size"
            else:
                resolved = self._blend_range_towards(default_distance)
                source = "default"
        elif policy == "ground_plane":
            if not self._warned_ground_plane:
                _LOG.warning(
                    "control.laser.use_range=ground_plane is not implemented; "
                    "falling back to known_size/default distances",
                )
                self._warned_ground_plane = True
            if measurement is not None:
                resolved = measurement
                source = "known_size"
            else:
                resolved = self._blend_range_towards(default_distance)
                source = "default"
        elif policy == "infinite":
            resolved = self._blend_range_towards(default_distance)
            source = "infinite"
        else:
            resolved = self._blend_range_towards(default_distance)
            source = policy or "default"

        if resolved is None or not math.isfinite(resolved) or resolved <= 0.0:
            self._resolved_range = None
            return None, None, False

        self._resolved_range = resolved
        return resolved, source, True

    def _blend_range_towards(self, target: float) -> float:
        if not math.isfinite(target) or target <= 0.0:
            return target
        prev = self._resolved_range
        alpha = self._distance_alpha
        if prev is None or alpha is None or alpha <= 0.0 or alpha >= 1.0:
            return target
        return alpha * target + (1.0 - alpha) * prev

    def _update_laser_overlay(
        self,
        msg: DetectionMsg,
        *,
        raw_target_uv: Optional[Tuple[float, float]],
        range_m: Optional[float],
        range_source: Optional[str],
        parallax_active: bool,
    ) -> None:
        if self._laser_mount is None:
            self._laser_overlay = None
            msg.laser_origin_px = None
            msg.laser_dot_px = None
            msg.laser_on_target = None
            msg.laser_range_m = None
            msg.laser_range_source = None
            msg.parallax_compensation_active = False
            return

        overlay = _LaserOverlay(
            origin_px=None,
            dot_px=None,
            on_target=None,
            active=parallax_active and range_m is not None,
            range_m=range_m,
            range_source=range_source,
        )

        if overlay.active and range_m is not None:
            offset = self._laser_mount.offset_m.as_tuple()
            direction = self._laser_mount.dir_cam.as_tuple()
            try:
                hit_px = laser_ray_to_pixel(
                    offset,
                    direction,
                    fx_px=self._cfg.fx_px,
                    fy_px=self._cfg.fy_px,
                    cx_px=self._cfg.cx_px,
                    cy_px=self._cfg.cy_px,
                    depth_m=float(range_m),
                )
            except ValueError:
                hit_px = None

            if hit_px is not None:
                overlay.dot_px = (float(hit_px[0]), float(hit_px[1]))

            origin_px = None
            try:
                projected_origin = project_point_to_pixel(
                    offset,
                    fx_px=self._cfg.fx_px,
                    fy_px=self._cfg.fy_px,
                    cx_px=self._cfg.cx_px,
                    cy_px=self._cfg.cy_px,
                )
            except ValueError:
                projected_origin = None

            if projected_origin is not None:
                origin_px = projected_origin
            else:
                near_depth = max(float(offset[2]) + 1e-3, 1e-3)
                try:
                    origin_px = laser_ray_to_pixel(
                        offset,
                        direction,
                        fx_px=self._cfg.fx_px,
                        fy_px=self._cfg.fy_px,
                        cx_px=self._cfg.cx_px,
                        cy_px=self._cfg.cy_px,
                        depth_m=near_depth,
                    )
                except ValueError:
                    origin_px = None

            if origin_px is not None:
                overlay.origin_px = (float(origin_px[0]), float(origin_px[1]))

            target_for_error = self._smoothed_uv if self._smoothed_uv is not None else raw_target_uv
            if overlay.dot_px is not None and target_for_error is not None:
                err_u = overlay.dot_px[0] - float(target_for_error[0])
                err_v = overlay.dot_px[1] - float(target_for_error[1])
                overlay.on_target = math.hypot(err_u, err_v) <= self._cfg.laser.tolerance_px

        self._laser_overlay = overlay
        msg.laser_origin_px = overlay.origin_px
        msg.laser_dot_px = overlay.dot_px
        msg.laser_on_target = overlay.on_target
        msg.laser_range_m = overlay.range_m
        msg.laser_range_source = overlay.range_source
        msg.parallax_compensation_active = overlay.active

    def _update_latency_estimate(self, msg: DetectionMsg, now: float) -> None:
        now_ms = now * 1000.0

        def _candidate(ts_ms: Optional[int]) -> Optional[float]:
            if ts_ms is None:
                return None
            delta = now_ms - float(ts_ms)
            if not math.isfinite(delta):
                return None
            if delta <= 0.0 or delta > 5000.0:
                return None
            return delta

        candidates = []
        for ts in (msg.src_ts_ms, msg.rx_ts_ms, msg.infer_ts_ms):
            candidate = _candidate(ts)
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            return

        latency_s = max(candidates) / 1000.0
        latency_s = max(latency_s, 1e-3)

        if self._lead_latency_ready:
            alpha = self._lead_latency_alpha
            self._lead_time_s = max(
                1e-3, alpha * latency_s + (1.0 - alpha) * self._lead_time_s
            )
        else:
            self._lead_time_s = max(latency_s, 1e-3)
            self._lead_latency_ready = True

    def _update_motion_state(
        self,
        msg: DetectionMsg,
        *,
        target_uv: Tuple[float, float],
        timestamp: float,
        measurement_timestamp: float,
        target_idx: Optional[int],
    ) -> None:
        if target_idx is None:
            self._motion_state = None
            self._motion_target_idx = None
            self._vel_ema = None
            msg.target_velocity_px_s = None
            msg.target_lead_uv = None
            msg.target_lead_time_s = None
            return

        prev_state = self._motion_state if self._motion_target_idx == target_idx else None
        yaw_angle = math.atan((target_uv[0] - self._cfg.cx_px) / self._cfg.fx_px)
        pitch_angle = math.atan((target_uv[1] - self._cfg.cy_px) / self._cfg.fy_px)

        velocity_px = (0.0, 0.0)
        lead_uv = target_uv
        lead_time = self._overlay_lead_horizon_s()
        motion_rates: Optional[AxisPair] = None
        max_rate_yaw = self._rate_measurement_bound("yaw")
        max_rate_pitch = self._rate_measurement_bound("pitch")

        if prev_state is not None:
            dt = measurement_timestamp - prev_state.measurement_timestamp
            if dt < 5e-3 or not math.isfinite(dt) or dt > 1.0:
                prev_state = None
                self._vel_ema = None
            else:
                raw_yaw_vel = (yaw_angle - prev_state.yaw_angle) / dt
                raw_pitch_vel = (pitch_angle - prev_state.pitch_angle) / dt
                raw_yaw_vel = _clamp(raw_yaw_vel, -max_rate_yaw, max_rate_yaw)
                raw_pitch_vel = _clamp(raw_pitch_vel, -max_rate_pitch, max_rate_pitch)
                cam_pan_rate = 0.0
                cam_tilt_rate = 0.0
                if self._cam_state is not None:
                    if self._cam_state.pan_rate is not None and math.isfinite(self._cam_state.pan_rate):
                        candidate = float(self._cam_state.pan_rate)
                        if abs(candidate) <= max_rate_yaw:
                            cam_pan_rate = candidate
                    if self._cam_state.tilt_rate is not None and math.isfinite(self._cam_state.tilt_rate):
                        candidate = float(self._cam_state.tilt_rate)
                        if abs(candidate) <= max_rate_pitch:
                            cam_tilt_rate = candidate

                yaw_vel = _clamp(raw_yaw_vel + cam_pan_rate, -max_rate_yaw, max_rate_yaw)
                pitch_vel = _clamp(raw_pitch_vel + cam_tilt_rate, -max_rate_pitch, max_rate_pitch)

                raw_motion_rates = AxisPair(yaw=yaw_vel, pitch=pitch_vel)
                alpha = _clamp(self._vel_alpha, 0.0, 1.0)
                if self._vel_ema is None or alpha >= 1.0:
                    motion_rates = raw_motion_rates
                elif alpha <= 0.0:
                    motion_rates = self._vel_ema
                else:
                    motion_rates = AxisPair(
                        yaw=alpha * raw_motion_rates.yaw + (1.0 - alpha) * self._vel_ema.yaw,
                        pitch=alpha * raw_motion_rates.pitch + (1.0 - alpha) * self._vel_ema.pitch,
                    )
                self._vel_ema = motion_rates

                yaw_angle_lead = yaw_angle + motion_rates.yaw * lead_time
                pitch_angle_lead = pitch_angle + motion_rates.pitch * lead_time

                lead_u = self._cfg.cx_px + self._cfg.fx_px * math.tan(yaw_angle_lead)
                lead_v = self._cfg.cy_px + self._cfg.fy_px * math.tan(pitch_angle_lead)

                lead_u = _clamp(lead_u, 0.0, self._cfg.width - 1.0)
                lead_v = _clamp(lead_v, 0.0, self._cfg.height - 1.0)

                lead_uv = (lead_u, lead_v)
                if lead_time > 0.0:
                    velocity_px = (
                        (lead_u - target_uv[0]) / lead_time,
                        (lead_v - target_uv[1]) / lead_time,
                    )
        else:
            self._vel_ema = None

        msg.target_velocity_px_s = (float(velocity_px[0]), float(velocity_px[1]))
        msg.target_lead_uv = (float(lead_uv[0]), float(lead_uv[1]))
        msg.target_lead_time_s = float(lead_time)

        self._motion_state = _MotionState(
            timestamp=timestamp,
            measurement_timestamp=measurement_timestamp,
            yaw_angle=yaw_angle,
            pitch_angle=pitch_angle,
            yaw_rate=motion_rates.yaw if motion_rates else 0.0,
            pitch_rate=motion_rates.pitch if motion_rates else 0.0,
        )
        self._motion_target_idx = target_idx
        if motion_rates is not None:
            self._last_motion_rates = motion_rates

    def _overlay_lead_horizon_s(self) -> float:
        lead_time = max(self._lead_time_s, 1e-3)
        if not self._mpc_enabled:
            return lead_time

        horizon_cfg = self._cfg.mpc.horizon if self._cfg.mpc is not None else None
        if horizon_cfg is None:
            return lead_time

        prediction_span = max(0, int(horizon_cfg.prediction_horizon) - 1)
        horizon_time = float(horizon_cfg.sample_time_s) * float(prediction_span)

        effect_delay = max(0.0, float(horizon_cfg.effect_delay_s))
        if self._mpc_builder is not None:
            effect_delay = max(
                effect_delay,
                float(self._mpc_builder.effect_delay_for_axis("yaw")),
                float(self._mpc_builder.effect_delay_for_axis("pitch")),
            )

        return max(1e-3, lead_time + effect_delay + horizon_time)

    def _measurement_timestamp_from_msg(self, msg: DetectionMsg, *, fallback: float) -> float:
        ts_s = float(msg.infer_ts_ms) / 1000.0
        if not math.isfinite(ts_s) or ts_s <= 0.0:
            return fallback
        return ts_s

    def _populate_predictive_overlay(self, msg: DetectionMsg, now: float) -> None:
        if self._is_predictive_active(now):
            msg.predictive_active = True
            predicted_uv = self._compute_predictive_target_uv(now)
            if predicted_uv is not None:
                msg.predictive_target_uv = (
                    float(predicted_uv[0]),
                    float(predicted_uv[1]),
                )
                box_px = self._predictive_box_from_uv(predicted_uv)
                msg.predictive_box_px = tuple(box_px) if box_px is not None else None
            else:
                msg.predictive_target_uv = None
                msg.predictive_box_px = None
        else:
            msg.predictive_active = None
            msg.predictive_target_uv = None
            msg.predictive_box_px = None

    def _compute_predictive_target_uv(self, now: float) -> Optional[Tuple[float, float]]:
        if self._last_known_target_uv is None:
            return None
        if self._predictive_rates is None or self._motion_state is None:
            return self._last_known_target_uv

        dt = max(0.0, now - self._motion_state.timestamp)
        yaw_angle = self._motion_state.yaw_angle + self._predictive_rates.yaw * dt
        pitch_angle = self._motion_state.pitch_angle + self._predictive_rates.pitch * dt

        try:
            u = self._cfg.cx_px + self._cfg.fx_px * math.tan(yaw_angle)
            v = self._cfg.cy_px + self._cfg.fy_px * math.tan(pitch_angle)
        except (OverflowError, ValueError):
            return self._last_known_target_uv

        if not math.isfinite(u) or not math.isfinite(v):
            return self._last_known_target_uv

        u = _clamp(u, 0.0, self._cfg.width - 1.0)
        v = _clamp(v, 0.0, self._cfg.height - 1.0)
        return (u, v)

    def _predictive_box_from_uv(
        self, uv: Tuple[float, float]
    ) -> Optional[Tuple[float, float, float, float]]:
        if self._last_target_box_size_px is None:
            return None
        width_px, height_px = self._last_target_box_size_px
        if width_px <= 0.0 or height_px <= 0.0:
            return None

        half_w = width_px / 2.0
        half_h = height_px / 2.0
        x1 = uv[0] - half_w
        y1 = uv[1] - half_h
        x2 = uv[0] + half_w
        y2 = uv[1] + half_h

        x1 = _clamp(x1, 0.0, self._cfg.width - 1.0)
        x2 = _clamp(x2, 0.0, self._cfg.width - 1.0)
        y1 = _clamp(y1, 0.0, self._cfg.height - 1.0)
        y2 = _clamp(y2, 0.0, self._cfg.height - 1.0)

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)

    def _is_target_recent(self, now: float) -> bool:
        if self._last_detection_ts is None:
            return False
        if self._lost_timeout_s <= 0.0:
            return True if self._latest_detection and self._latest_detection.target_uv else False
        return (now - self._last_detection_ts) <= self._lost_timeout_s

    def _is_predictive_active(self, now: float) -> bool:
        if self._predictive_end_time is None or self._predictive_rates is None:
            return False
        if now > self._predictive_end_time:
            self._clear_predictive_mode()
            return False
        return True

    def _start_predictive_mode(self, now: float) -> None:
        if self._lost_timeout_s <= 0.0:
            return
        if self._predictive_end_time is not None and self._predictive_rates is not None:
            return
        if self._last_motion_rates is None:
            return
        self._predictive_end_time = now + self._lost_timeout_s
        self._predictive_rates = self._last_motion_rates

    def _clear_predictive_mode(self) -> None:
        self._predictive_end_time = None
        self._predictive_rates = None

    def _compute_dt(self, now: float) -> float:
        if self._last_cmd_time is None:
            dt = self._default_dt
        else:
            dt = now - self._last_cmd_time
        if not (dt > 0.0):
            dt = self._default_dt
        return _clamp(dt, self._MIN_DT, self._MAX_DT)

    def _build_mpc_tracking_cmd(
        self, detection: _DetectionState, dt: float, now: float
    ) -> ControlCmd:
        if not self._mpc_builder:
            raise RuntimeError("MPC components are not initialized")
        assert detection.target_uv is not None
        target_uv = self._smoothed_uv or detection.target_uv
        aim_uv = self._aim_reference_uv(detection)

        raw_px_err = pixel_delta(
            target_uv[0],
            target_uv[1],
            aim_uv[0],
            aim_uv[1],
            self._cfg,
            apply_deadband=False,
        )
        ctrl_px_err = pixel_delta(
            target_uv[0],
            target_uv[1],
            aim_uv[0],
            aim_uv[1],
            self._cfg,
            apply_deadband=True,
        )
        err_rad = angular_error_from_pixel_delta(ctrl_px_err, self._cfg)

        references = self._mpc_builder.build(
            target_uv=(float(target_uv[0]), float(target_uv[1])),
            aim_uv=(float(aim_uv[0]), float(aim_uv[1])),
            timestamp=now,
            cam_state=self._cam_state,
            theta_estimates=self._mpc_theta_estimates,
            omega_estimates=self._mpc_omega_estimates,
            distance_m=detection.target_distance_m,
            target_velocity_px_s=detection.target_velocity_px_s,
        )

        axis_cmds: Dict[str, float] = {}
        diag_map: Dict[str, Optional[MpcAxisDiagnostics]] = {}
        ref_debug: Dict[str, Dict[str, float]] = {}
        for axis, controller in self._mpc_axes.items():
            seq = references.get(axis)
            if seq is None:
                continue
            ref_entry: Dict[str, float] = {}
            if seq.theta:
                theta_ref0 = float(seq.theta[0])
                if math.isfinite(theta_ref0):
                    ref_entry["theta_ref0"] = theta_ref0
            if seq.omega:
                omega_ref0 = float(seq.omega[0])
                if math.isfinite(omega_ref0):
                    ref_entry["omega_ref0"] = omega_ref0
            if self._mpc_builder is not None:
                effect_delay = float(self._mpc_builder.effect_delay_for_axis(axis))
                if math.isfinite(effect_delay):
                    ref_entry["effect_delay_s"] = effect_delay
            if ref_entry:
                ref_debug[axis] = ref_entry
            try:
                command, diagnostics = controller.compute_control(
                    theta_ref_seq=seq.theta,
                    omega_ref_seq=seq.omega,
                    distance_seq=seq.distance,
                    lateral_seq=seq.lateral,
                    radial_seq=seq.radial,
                )
            except MpcSolverError as exc:
                _LOG.error("mpc_solver_error axis=%s error=%s", axis, exc)
                command = self._mpc_last_applied.get(axis, 0.0)
                diagnostics = None
            axis_cmds[axis] = command
            diag_map[axis] = diagnostics
            self._mpc_last_diag[axis] = diagnostics

        yaw_rate = axis_cmds.get("yaw", self._prev_rate.yaw)
        pitch_rate = axis_cmds.get("pitch", self._prev_rate.pitch)
        self._prev_rate = AxisPair(yaw_rate, pitch_rate)
        self._prev_err = err_rad
        self._record_mpc_command(yaw_rate, pitch_rate)
        self._update_mpc_outer_tuner(err_rad, yaw_rate, pitch_rate, now)

        pan_abs, tilt_abs = self._position_setpoints(yaw_rate, pitch_rate, dt)

        diag_summary = self._summarize_mpc_diagnostics(diag_map, ref_debug)
        cmd = ControlCmd(
            frame_id=detection.frame_id,
            src_ts_ms=detection.src_ts_ms,
            cmd_ts_ms=int(time.monotonic_ns() / 1e6),
            target_ok=True,
            target_uv=(float(target_uv[0]), float(target_uv[1])),
            err_uv=(ctrl_px_err.yaw, ctrl_px_err.pitch),
            err_rad=(err_rad.yaw, err_rad.pitch),
            pan_rate_cmd=yaw_rate,
            tilt_rate_cmd=pitch_rate,
            pan_abs_cmd=pan_abs,
            tilt_abs_cmd=tilt_abs,
            laser_origin_px=self._laser_overlay.origin_px if self._laser_overlay else None,
            laser_dot_px=self._laser_overlay.dot_px if self._laser_overlay else None,
            laser_on_target=self._laser_overlay.on_target if self._laser_overlay else None,
            laser_range_m=self._laser_overlay.range_m if self._laser_overlay else None,
            laser_range_source=self._laser_overlay.range_source if self._laser_overlay else None,
            parallax_compensation_active=(
                self._laser_overlay.active if self._laser_overlay else False
            ),
            controller_mode="mpc",
            mpc=diag_summary,
        )

        payload = {
            "frame_id": detection.frame_id,
            "target_ok": True,
            "dt": dt,
            "uv": [float(target_uv[0]), float(target_uv[1])],
            "err_px": [raw_px_err.yaw, raw_px_err.pitch],
            "err_rad": [err_rad.yaw, err_rad.pitch],
            "cmd_rate": [yaw_rate, pitch_rate],
        }
        if diag_summary:
            payload["mpc"] = diag_summary
        self._log_control_state(payload, target_ok=True, now=now)

        return cmd

    def _update_mpc_outer_tuner(
        self,
        err_rad: AxisPair,
        yaw_rate: float,
        pitch_rate: float,
        now: float,
    ) -> None:
        cfg = self._mpc_outer_tuner_cfg
        if cfg is None or not self._mpc_axes:
            return

        abs_err = 0.5 * (abs(float(err_rad.yaw)) + abs(float(err_rad.pitch)))
        abs_cmd = 0.5 * (abs(float(yaw_rate)) + abs(float(pitch_rate)))
        self._mpc_outer_tuner_history.append(
            _MpcTuneSample(timestamp=now, abs_err_rad=abs_err, abs_cmd_rad_s=abs_cmd)
        )

        cutoff = now - float(cfg.history_window_s)
        while self._mpc_outer_tuner_history and self._mpc_outer_tuner_history[0].timestamp < cutoff:
            self._mpc_outer_tuner_history.popleft()

        if now < self._mpc_outer_tuner_next_time:
            return
        if len(self._mpc_outer_tuner_history) < int(cfg.min_samples):
            return

        self._mpc_outer_tuner_next_time = now + float(cfg.update_interval_s)
        history = list(self._mpc_outer_tuner_history)
        mean_err = sum(sample.abs_err_rad for sample in history) / float(len(history))
        mean_cmd = sum(sample.abs_cmd_rad_s for sample in history) / float(len(history))

        target_err = max(1e-9, float(cfg.target_abs_err_rad))
        target_cmd = max(1e-9, float(cfg.target_abs_cmd_rad_s))
        high_err = mean_err > (1.1 * target_err)
        low_err = mean_err < (0.6 * target_err)
        high_cmd = mean_cmd > (1.1 * target_cmd)
        low_cmd = mean_cmd < (0.6 * target_cmd)

        step_up = max(1e-6, float(cfg.step_up))
        step_down = max(1e-6, float(cfg.step_down))
        min_scale = float(cfg.min_scale)
        max_scale = float(cfg.max_scale)

        def adjust(scale_key: str, factor: float) -> None:
            if scale_key not in self._mpc_outer_tuner_scales:
                return
            current = float(self._mpc_outer_tuner_scales[scale_key])
            updated = current * factor
            self._mpc_outer_tuner_scales[scale_key] = max(
                min_scale, min(max_scale, updated)
            )

        def relax_all() -> None:
            relax = max(0.0, 1.0 - (0.25 * step_down))
            for key in tuple(self._mpc_outer_tuner_scales.keys()):
                current = float(self._mpc_outer_tuner_scales[key])
                updated = current + (1.0 - current) * (1.0 - relax)
                self._mpc_outer_tuner_scales[key] = max(
                    min_scale, min(max_scale, updated)
                )

        if self._mpc_outer_tuner_group:
            group = self._mpc_outer_tuner_group

            if group == "costs_tracking":
                if high_err and not high_cmd:
                    adjust("q_theta", 1.0 + step_up)
                    adjust("q_omega", 1.0 + (0.5 * step_up))
                    adjust("q_dtheta", 1.0 + (0.5 * step_up))
                    adjust("terminal", 1.0 + (0.5 * step_up))
                elif high_cmd and not high_err:
                    adjust("q_theta", max(0.1, 1.0 - step_down))
                    adjust("q_omega", max(0.1, 1.0 - (0.5 * step_down)))
                    adjust("q_dtheta", max(0.1, 1.0 - (0.5 * step_down)))
                    adjust("terminal", max(0.1, 1.0 - (0.5 * step_down)))
            elif group == "costs_effort":
                if high_cmd and not high_err:
                    adjust("r", 1.0 + step_up)
                    adjust("s", 1.0 + step_up)
                    adjust("rho", 1.0 + (0.5 * step_up))
                elif high_err and not high_cmd:
                    adjust("r", max(0.1, 1.0 - step_down))
                    adjust("s", max(0.1, 1.0 - step_down))
                    adjust("rho", max(0.1, 1.0 - (0.5 * step_down)))
            elif group == "estimator":
                if high_err and not high_cmd:
                    adjust("est_q_theta", 1.0 + step_up)
                    adjust("est_q_omega", 1.0 + (0.5 * step_up))
                    adjust("est_q_d", 1.0 + (0.5 * step_up))
                    adjust("est_r_theta", max(0.1, 1.0 - (0.5 * step_down)))
                elif high_cmd and not high_err:
                    adjust("est_q_theta", max(0.1, 1.0 - step_down))
                    adjust("est_q_omega", max(0.1, 1.0 - (0.5 * step_down)))
                    adjust("est_q_d", max(0.1, 1.0 - (0.5 * step_down)))
                    adjust("est_r_theta", 1.0 + (0.5 * step_up))
            elif group == "predictor":
                if high_err and not high_cmd:
                    adjust("predictor_alpha", 1.0 + (0.5 * step_up))
                    adjust("predictor_beta", 1.0 + step_up)
                elif high_cmd and not high_err:
                    adjust("predictor_alpha", max(0.1, 1.0 - (0.5 * step_down)))
                    adjust("predictor_beta", max(0.1, 1.0 - step_down))
            elif group == "adaptive_delay":
                if high_err and high_cmd:
                    adjust("adaptive_effect_delay_gain", 1.0 + step_up)
                    adjust("adaptive_effect_delay_alpha", 1.0 + (0.5 * step_up))
                elif high_cmd and not high_err:
                    adjust("adaptive_effect_delay_gain", max(0.1, 1.0 - step_down))
                    adjust(
                        "adaptive_effect_delay_alpha",
                        max(0.1, 1.0 - (0.5 * step_down)),
                    )
                    adjust("adaptive_effect_delay_rate_eps", 1.0 + (0.5 * step_up))
                elif high_err and not high_cmd:
                    adjust("adaptive_effect_delay_rate_eps", max(0.1, 1.0 - step_down))
            if low_err and low_cmd:
                relax_all()

            self._apply_mpc_outer_tuner_overrides()
            if self._mpc_outer_tuner_save_on_update:
                self._persist_mpc_outer_tuner_state(now)
            _LOG.info(
                "mpc_outer_tuner group=%s err=%.5f cmd=%.5f scales=%s",
                group,
                mean_err,
                mean_cmd,
                {k: round(v, 4) for k, v in self._mpc_outer_tuner_scales.items()},
            )
            return

        if high_cmd and not high_err:
            adjust("r", 1.0 + step_up)
            adjust("s", 1.0 + step_up)
            adjust("q_dtheta", 1.0 + (0.5 * step_up))
            adjust("q_theta", max(0.1, 1.0 - (0.5 * step_down)))
            adjust("q_omega", max(0.1, 1.0 - (0.25 * step_down)))
            adjust("terminal", max(0.1, 1.0 - (0.5 * step_down)))
        elif high_err and not high_cmd:
            adjust("q_theta", 1.0 + step_up)
            adjust("q_omega", 1.0 + (0.5 * step_up))
            adjust("terminal", 1.0 + (0.5 * step_up))
            adjust("r", max(0.1, 1.0 - (0.5 * step_down)))
            adjust("s", max(0.1, 1.0 - (0.5 * step_down)))
        elif high_err and high_cmd:
            adjust("q_theta", 1.0 + (0.5 * step_up))
            adjust("q_dtheta", 1.0 + (0.5 * step_up))
            adjust("s", 1.0 + (0.5 * step_up))
            adjust("rho", 1.0 + (0.5 * step_up))
        elif low_err and low_cmd:
            relax_all()

        self._apply_mpc_outer_tuner_overrides()
        if self._mpc_outer_tuner_save_on_update:
            self._persist_mpc_outer_tuner_state(now)

        _LOG.info(
            "mpc_outer_tuner err=%.5f cmd=%.5f scales=%s",
            mean_err,
            mean_cmd,
            {k: round(v, 4) for k, v in self._mpc_outer_tuner_scales.items()},
        )

    def _apply_mpc_outer_tuner_overrides(self) -> None:
        if self._cfg.mpc is None:
            return
        if self._mpc_outer_tuner_group:
            self._apply_mpc_outer_tuner_group_overrides(self._mpc_outer_tuner_group)
            return
        base_costs = {
            "q_theta": float(self._cfg.mpc.costs.q_theta),
            "q_omega": float(self._cfg.mpc.costs.q_omega),
            "q_dtheta": float(self._cfg.mpc.costs.q_dtheta),
            "r": float(self._cfg.mpc.costs.r),
            "s": float(self._cfg.mpc.costs.s),
            "terminal": float(self._cfg.mpc.costs.terminal or 0.0),
            "rho": float(self._cfg.mpc.costs.rho),
        }
        overrides = {
            key: base_costs[key] * float(scale)
            for key, scale in self._mpc_outer_tuner_scales.items()
            if key in base_costs
        }
        for axis in self._mpc_axes.values():
            if hasattr(axis, "set_cost_overrides"):
                axis.set_cost_overrides(overrides)

    def _apply_mpc_outer_tuner_group_overrides(self, group: str) -> None:
        cfg = self._cfg.mpc
        if cfg is None:
            return
        def _scale(name: str) -> float:
            if name not in self._mpc_outer_tuner_scales:
                return 1.0
            return float(self._mpc_outer_tuner_scales[name])

        if group == "costs_tracking":
            cost_overrides = {
                "q_theta": float(cfg.costs.q_theta) * _scale("q_theta"),
                "q_omega": float(cfg.costs.q_omega) * _scale("q_omega"),
                "q_dtheta": float(cfg.costs.q_dtheta) * _scale("q_dtheta"),
                "terminal": float(cfg.costs.terminal or 0.0) * _scale("terminal"),
            }
            for axis in self._mpc_axes.values():
                if hasattr(axis, "set_cost_overrides"):
                    axis.set_cost_overrides(cost_overrides)
            return

        if group == "costs_effort":
            cost_overrides = {
                "r": float(cfg.costs.r) * _scale("r"),
                "s": float(cfg.costs.s) * _scale("s"),
                "rho": float(cfg.costs.rho) * _scale("rho"),
            }
            for axis in self._mpc_axes.values():
                if hasattr(axis, "set_cost_overrides"):
                    axis.set_cost_overrides(cost_overrides)
            return

        if group == "estimator":
            est_overrides = {
                "q_theta": float(cfg.estimator.q_theta) * _scale("est_q_theta"),
                "q_omega": float(cfg.estimator.q_omega) * _scale("est_q_omega"),
                "q_d": float(cfg.estimator.q_d) * _scale("est_q_d"),
                "r_theta": float(cfg.estimator.r_theta) * _scale("est_r_theta"),
            }
            for axis in self._mpc_axes.values():
                if hasattr(axis, "set_estimator_overrides"):
                    axis.set_estimator_overrides(est_overrides)
            return

        if group == "predictor":
            if self._mpc_builder is not None:
                self._mpc_builder.set_tuning_overrides(
                    {
                        "predictor_alpha": min(
                            1.0,
                            float(cfg.horizon.predictor_alpha) * _scale("predictor_alpha"),
                        ),
                        "predictor_beta": min(
                            1.0,
                            float(cfg.horizon.predictor_beta) * _scale("predictor_beta"),
                        ),
                    }
                )
            return

        if group == "adaptive_delay":
            if self._mpc_builder is not None:
                self._mpc_builder.set_tuning_overrides(
                    {
                        "adaptive_effect_delay_gain": max(
                            0.0,
                            float(cfg.horizon.adaptive_effect_delay_gain)
                            * _scale("adaptive_effect_delay_gain"),
                        ),
                        "adaptive_effect_delay_alpha": min(
                            1.0,
                            max(
                                0.0,
                                float(cfg.horizon.adaptive_effect_delay_alpha)
                                * _scale("adaptive_effect_delay_alpha"),
                            ),
                        ),
                        "adaptive_effect_delay_rate_eps": max(
                            1e-9,
                            float(cfg.horizon.adaptive_effect_delay_rate_eps)
                            * _scale("adaptive_effect_delay_rate_eps"),
                        ),
                    }
                )
            return

    def _persist_mpc_outer_tuner_state(self, now: float) -> None:
        path = self._mpc_outer_tuner_state_path
        if path is None:
            return
        payload = {
            "version": 1,
            "updated_ts": float(now),
            "parameter_group": self._mpc_outer_tuner_group,
            "scales": {
                key: float(value)
                for key, value in self._mpc_outer_tuner_scales.items()
                if math.isfinite(float(value))
            },
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as exc:
            _LOG.warning("mpc_outer_tuner state save failed path=%s error=%s", path, exc)

    def _restore_mpc_outer_tuner_state(self) -> None:
        path = self._mpc_outer_tuner_state_path
        if path is None or not path.exists():
            return
        cfg = self._mpc_outer_tuner_cfg
        min_scale = float(cfg.min_scale) if cfg is not None else 0.0
        max_scale = float(cfg.max_scale) if cfg is not None else float("inf")
        if max_scale < min_scale:
            max_scale = min_scale
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            _LOG.warning("mpc_outer_tuner state load failed path=%s error=%s", path, exc)
            return
        if not isinstance(payload, Mapping):
            return
        scales_raw = payload.get("scales")
        if not isinstance(scales_raw, Mapping):
            return
        restored = 0
        for key, value in scales_raw.items():
            name = str(key)
            if name not in self._mpc_outer_tuner_scales:
                continue
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(val):
                continue
            self._mpc_outer_tuner_scales[name] = max(min_scale, min(max_scale, val))
            restored += 1
        if self._mpc_outer_tuner_group:
            raw_group_scale = payload.get("group_scale")
            if isinstance(raw_group_scale, (int, float)) and math.isfinite(float(raw_group_scale)):
                group_val = max(min_scale, min(max_scale, float(raw_group_scale)))
                if restored == 0:
                    for key in tuple(self._mpc_outer_tuner_scales.keys()):
                        self._mpc_outer_tuner_scales[key] = group_val
        if restored > 0:
            _LOG.info("mpc_outer_tuner restored scales from %s", path)

    def _build_tracking_cmd(
        self, detection: _DetectionState, dt: float, now: float
    ) -> ControlCmd:
        assert detection.target_uv is not None
        target_uv = self._smoothed_uv or detection.target_uv

        aim_uv = self._aim_reference_uv(detection)

        raw_px_err = pixel_delta(
            target_uv[0],
            target_uv[1],
            aim_uv[0],
            aim_uv[1],
            self._cfg,
            apply_deadband=False,
        )
        ctrl_px_err = pixel_delta(
            target_uv[0],
            target_uv[1],
            aim_uv[0],
            aim_uv[1],
            self._cfg,
            apply_deadband=True,
        )
        err_rad = angular_error_from_pixel_delta(ctrl_px_err, self._cfg)

        if not self._tracking_active or self._prev_err is None:
            derr = AxisPair(0.0, 0.0)
        else:
            derr = AxisPair(
                yaw=(err_rad.yaw - self._prev_err.yaw) / dt,
                pitch=(err_rad.pitch - self._prev_err.pitch) / dt,
            )

        integ_yaw = self._integrate(self._integ.yaw, err_rad.yaw, dt, self._cfg.ki.yaw, self._cfg.rate_limits.yaw)
        integ_pitch = self._integrate(
            self._integ.pitch, err_rad.pitch, dt, self._cfg.ki.pitch, self._cfg.rate_limits.pitch
        )
        self._integ = AxisPair(integ_yaw, integ_pitch)

        yaw_rate = (
            self._cfg.kp.yaw * err_rad.yaw
            + self._cfg.kd.yaw * derr.yaw
            + self._cfg.ki.yaw * integ_yaw
        )
        pitch_rate = (
            self._cfg.kp.pitch * err_rad.pitch
            + self._cfg.kd.pitch * derr.pitch
            + self._cfg.ki.pitch * integ_pitch
        )

        yaw_rate = _clamp(yaw_rate, -self._cfg.rate_limits.yaw, self._cfg.rate_limits.yaw)
        pitch_rate = _clamp(pitch_rate, -self._cfg.rate_limits.pitch, self._cfg.rate_limits.pitch)

        yaw_rate = self._slew_axis(self._prev_rate.yaw, yaw_rate, self._cfg.accel_limits.yaw, dt)
        pitch_rate = self._slew_axis(self._prev_rate.pitch, pitch_rate, self._cfg.accel_limits.pitch, dt)
        self._prev_rate = AxisPair(yaw_rate, pitch_rate)

        self._prev_err = err_rad

        pan_abs, tilt_abs = self._position_setpoints(yaw_rate, pitch_rate, dt)

        cmd = ControlCmd(
            frame_id=detection.frame_id,
            src_ts_ms=detection.src_ts_ms,
            cmd_ts_ms=int(time.monotonic_ns() / 1e6),
            target_ok=True,
            target_uv=(float(target_uv[0]), float(target_uv[1])),
            err_uv=(ctrl_px_err.yaw, ctrl_px_err.pitch),
            err_rad=(err_rad.yaw, err_rad.pitch),
            pan_rate_cmd=yaw_rate,
            tilt_rate_cmd=pitch_rate,
            pan_abs_cmd=pan_abs,
            tilt_abs_cmd=tilt_abs,
            laser_origin_px=self._laser_overlay.origin_px if self._laser_overlay else None,
            laser_dot_px=self._laser_overlay.dot_px if self._laser_overlay else None,
            laser_on_target=self._laser_overlay.on_target if self._laser_overlay else None,
            laser_range_m=self._laser_overlay.range_m if self._laser_overlay else None,
            laser_range_source=self._laser_overlay.range_source if self._laser_overlay else None,
            parallax_compensation_active=(
                self._laser_overlay.active if self._laser_overlay else False
            ),
            controller_mode=self._cfg.controller,
        )

        self._log_control_state(
            {
                "frame_id": detection.frame_id,
                "target_ok": True,
                "dt": round(dt, 6),
                "uv": [float(target_uv[0]), float(target_uv[1])],
                "aim_uv": [float(aim_uv[0]), float(aim_uv[1])],
                "err_px": [raw_px_err.yaw, raw_px_err.pitch],
                "err_rad": [err_rad.yaw, err_rad.pitch],
                "cmd_rate": [yaw_rate, pitch_rate],
                "range_m": detection.resolved_range_m,
                "range_src": detection.range_source,
                "parallax_active": detection.range_active,
            },
            target_ok=True,
            now=now,
        )

        return cmd

    def _aim_reference_uv(self, detection: _DetectionState) -> Tuple[float, float]:
        """Return the pixel location the controller should align to."""

        if self._cfg.aim_mode != "laser_point" or self._laser_mount is None:
            return (self._cfg.cx_px, self._cfg.cy_px)

        if not detection.range_active:
            return (self._cfg.cx_px, self._cfg.cy_px)

        distance = detection.resolved_range_m
        if distance is None or not math.isfinite(distance) or float(distance) <= 0.0:
            return (self._cfg.cx_px, self._cfg.cy_px)

        offset = self._laser_mount.offset_m.as_tuple()
        direction = self._laser_mount.dir_cam.as_tuple()
        try:
            dot = laser_ray_to_pixel(
                offset,
                direction,
                fx_px=self._cfg.fx_px,
                fy_px=self._cfg.fy_px,
                cx_px=self._cfg.cx_px,
                cy_px=self._cfg.cy_px,
                depth_m=float(distance),
            )
        except ValueError:
            dot = None

        if dot is None:
            return (self._cfg.cx_px, self._cfg.cy_px)

        return (float(dot[0]), float(dot[1]))

    def _build_hold_cmd(self, now: float) -> ControlCmd:
        home_rates, home_err = self._homeward_rates(self._default_dt)
        self._prev_rate = home_rates

        pan_abs, tilt_abs = self._position_setpoints(home_rates.yaw, home_rates.pitch, self._default_dt)
        uv = self._smoothed_uv or (self._cfg.cx_px, self._cfg.cy_px)

        if pan_abs is None and self._home_pan is not None:
            pan_abs = self._home_pan
        if tilt_abs is None and self._home_tilt is not None:
            tilt_abs = self._home_tilt

        cmd = ControlCmd(
            frame_id=self._last_frame_id,
            src_ts_ms=self._last_src_ts_ms,
            cmd_ts_ms=int(time.monotonic_ns() / 1e6),
            target_ok=False,
            target_uv=(float(uv[0]), float(uv[1])),
            err_uv=(0.0, 0.0),
            err_rad=(home_err.yaw, home_err.pitch),
            pan_rate_cmd=home_rates.yaw,
            tilt_rate_cmd=home_rates.pitch,
            pan_abs_cmd=pan_abs,
            tilt_abs_cmd=tilt_abs,
            laser_origin_px=None,
            laser_dot_px=None,
            laser_on_target=None,
            parallax_compensation_active=False,
            controller_mode=self._cfg.controller,
        )

        self._record_mpc_command(home_rates.yaw, home_rates.pitch)

        self._log_control_state(
            {
                "frame_id": self._last_frame_id,
                "target_ok": False,
                "dt": None,
                "uv": [float(uv[0]), float(uv[1])],
                "err_px": [0.0, 0.0],
                "err_rad": [home_err.yaw, home_err.pitch],
                "cmd_rate": [home_rates.yaw, home_rates.pitch],
                "home": True,
            },
            target_ok=False,
            now=now,
        )

        return cmd

    def _build_predictive_cmd(self, dt: float, now: float) -> ControlCmd:
        assert self._predictive_rates is not None

        yaw_rate = _clamp(
            self._predictive_rates.yaw, -self._cfg.rate_limits.yaw, self._cfg.rate_limits.yaw
        )
        pitch_rate = _clamp(
            self._predictive_rates.pitch,
            -self._cfg.rate_limits.pitch,
            self._cfg.rate_limits.pitch,
        )

        yaw_rate = self._slew_axis(self._prev_rate.yaw, yaw_rate, self._cfg.accel_limits.yaw, dt)
        pitch_rate = self._slew_axis(
            self._prev_rate.pitch, pitch_rate, self._cfg.accel_limits.pitch, dt
        )
        self._prev_rate = AxisPair(yaw_rate, pitch_rate)

        pan_abs, tilt_abs = self._position_setpoints(yaw_rate, pitch_rate, dt)

        uv = self._last_known_target_uv or (self._cfg.cx_px, self._cfg.cy_px)

        cmd = ControlCmd(
            frame_id=self._last_frame_id,
            src_ts_ms=self._last_src_ts_ms,
            cmd_ts_ms=int(time.monotonic_ns() / 1e6),
            target_ok=False,
            target_uv=(float(uv[0]), float(uv[1])),
            err_uv=(0.0, 0.0),
            err_rad=(0.0, 0.0),
            pan_rate_cmd=yaw_rate,
            tilt_rate_cmd=pitch_rate,
            pan_abs_cmd=pan_abs,
            tilt_abs_cmd=tilt_abs,
            laser_origin_px=None,
            laser_dot_px=None,
            laser_on_target=None,
            parallax_compensation_active=False,
            controller_mode=self._cfg.controller,
        )

        self._record_mpc_command(yaw_rate, pitch_rate)

        self._log_control_state(
            {
                "frame_id": self._last_frame_id,
                "target_ok": False,
                "dt": dt,
                "uv": [float(uv[0]), float(uv[1])],
                "err_px": [0.0, 0.0],
                "err_rad": [0.0, 0.0],
                "cmd_rate": [yaw_rate, pitch_rate],
                "predictive": True,
            },
            target_ok=False,
            now=now,
        )

        return cmd

    def _update_mpc_estimates(self) -> None:
        if not self._mpc_enabled or not self._mpc_axes:
            return
        cam_state = self._cam_state
        for axis, controller in self._mpc_axes.items():
            measurement: Optional[float] = None
            omega_measurement: Optional[float] = None
            omega_bound = self._rate_measurement_bound(axis)
            theta_bound = self._theta_estimate_bound(axis)
            if cam_state is not None:
                raw = cam_state.pan if axis == "yaw" else cam_state.tilt
                if raw is not None and math.isfinite(raw):
                    measurement = float(raw)
                raw_rate = cam_state.pan_rate if axis == "yaw" else cam_state.tilt_rate
                if raw_rate is not None and math.isfinite(raw_rate):
                    candidate_rate = float(raw_rate)
                    if abs(candidate_rate) <= omega_bound:
                        omega_measurement = candidate_rate
            if measurement is None:
                now = time.monotonic()
                if (now - self._mpc_missing_measurement_last_log) >= 2.0:
                    _LOG.warning(
                        "mpc estimator skipping update for axis=%s due to missing/invalid CamState angle measurement",
                        axis,
                    )
                    self._mpc_missing_measurement_last_log = now
                continue
            state = controller.step_estimator(
                self._mpc_last_applied.get(axis, 0.0), measurement, omega_measurement
            )
            if len(state) >= 2:
                self._mpc_theta_estimates[axis] = _clamp(float(state[0]), -theta_bound, theta_bound)
                self._mpc_omega_estimates[axis] = _clamp(float(state[1]), -omega_bound, omega_bound)

    def _theta_estimate_bound(self, axis: str) -> float:
        axis_name = str(axis).lower()
        if self._cfg.mpc is None:
            return math.pi
        constraints = self._cfg.mpc.constraints
        if axis_name == "yaw":
            vals = [
                abs(float(v))
                for v in (constraints.theta_min, constraints.theta_max)
                if v is not None and math.isfinite(float(v))
            ]
        else:
            vals = [
                abs(float(v))
                for v in (constraints.theta_min, constraints.theta_max)
                if v is not None and math.isfinite(float(v))
            ]
        base = max(vals) if vals else math.pi
        return max(math.pi / 4.0, 1.5 * base)

    def _rate_measurement_bound(self, axis: str) -> float:
        axis_name = str(axis).lower()
        base_candidates = []
        if axis_name == "yaw":
            base_candidates.append(abs(float(self._cfg.rate_limits.yaw)))
        else:
            base_candidates.append(abs(float(self._cfg.rate_limits.pitch)))

        if self._cfg.mpc is not None:
            constraints = self._cfg.mpc.constraints
            if axis_name == "yaw":
                if constraints.omega_min is not None:
                    base_candidates.append(abs(float(constraints.omega_min)))
                if constraints.omega_max is not None:
                    base_candidates.append(abs(float(constraints.omega_max)))
            else:
                if constraints.omega_min is not None:
                    base_candidates.append(abs(float(constraints.omega_min)))
                if constraints.omega_max is not None:
                    base_candidates.append(abs(float(constraints.omega_max)))

        base = max(base_candidates) if base_candidates else 0.0
        return max(1.0, 1.5 * base)

    def _record_mpc_command(self, yaw_rate: float, pitch_rate: float) -> None:
        if not self._mpc_enabled:
            return
        if "yaw" in self._mpc_last_applied:
            self._mpc_last_applied["yaw"] = float(yaw_rate)
        if "pitch" in self._mpc_last_applied:
            self._mpc_last_applied["pitch"] = float(pitch_rate)

    def _summarize_mpc_diagnostics(
        self,
        diagnostics: Dict[str, Optional[MpcAxisDiagnostics]],
        ref_debug: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> Optional[dict]:
        if not diagnostics:
            return None
        summary: Dict[str, Any] = {}
        for axis, diag in diagnostics.items():
            if diag is None:
                continue
            entry: Dict[str, Any] = {"status": diag.status}
            if diag.cost is not None and math.isfinite(diag.cost):
                entry["cost"] = float(diag.cost)
            seq = getattr(diag, "u_sequence", None)
            if seq is not None:
                try:
                    if len(seq):
                        entry["u0"] = float(seq[0])
                except TypeError:
                    pass
            slack = getattr(diag, "slack", None)
            if isinstance(slack, dict) and slack:
                slack_summary = {
                    key: float(value)
                    for key, value in slack.items()
                    if isinstance(value, (int, float)) and math.isfinite(float(value))
                }
                if slack_summary:
                    entry["slack"] = slack_summary
            solver_info = getattr(diag, "solver_info", None)
            if isinstance(solver_info, Mapping):
                info_summary = {}
                for key, value in solver_info.items():
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        info_summary[key] = float(value)
                if info_summary:
                    entry["solver"] = info_summary
            terms = getattr(diag, "cost_terms", None)
            if isinstance(terms, Mapping):
                term_summary = {}
                for key, value in terms.items():
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        term_summary[key] = float(value)
                if term_summary:
                    entry["terms"] = term_summary
            term_directions = getattr(diag, "cost_term_directions", None)
            if isinstance(term_directions, Mapping):
                direction_summary = {}
                for key, value in term_directions.items():
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        direction_summary[key] = float(value)
                if direction_summary:
                    entry["term_directions"] = direction_summary
            preds_summary: Dict[str, float] = {}
            theta_pred = getattr(diag, "theta_pred", None)
            if theta_pred is not None:
                try:
                    if len(theta_pred):
                        theta_pred0 = float(theta_pred[0])
                        if math.isfinite(theta_pred0):
                            preds_summary["theta_pred0"] = theta_pred0
                except TypeError:
                    pass
            omega_pred = getattr(diag, "omega_pred", None)
            if omega_pred is not None:
                try:
                    if len(omega_pred):
                        omega_pred0 = float(omega_pred[0])
                        if math.isfinite(omega_pred0):
                            preds_summary["omega_pred0"] = omega_pred0
                except TypeError:
                    pass
            if preds_summary:
                entry["pred"] = preds_summary
            if ref_debug is not None:
                refs = ref_debug.get(axis)
                if isinstance(refs, Mapping):
                    refs_summary = {}
                    for key, value in refs.items():
                        if isinstance(value, (int, float)) and math.isfinite(float(value)):
                            refs_summary[key] = float(value)
                    if refs_summary:
                        entry["refs"] = refs_summary
            summary[axis] = entry
        return summary or None

    def _log_control_state(self, payload: dict, *, target_ok: bool, now: float) -> None:
        if not _LOG.isEnabledFor(logging.INFO):
            return

        should_emit = False
        if self._last_log_target_ok is None or self._last_log_target_ok != target_ok:
            should_emit = True
        elif (now - self._last_log_time) >= self._log_interval_s:
            should_emit = True

        if not should_emit:
            return

        rounded = self._round_for_log(payload)
        if self._log_json:
            _LOG.info(json.dumps(rounded))
        else:
            _LOG.info(self._format_control_cli(rounded))
        self._last_log_time = now
        self._last_log_target_ok = target_ok

    def _homeward_rates(self, dt: float) -> Tuple[AxisPair, AxisPair]:
        cam_state = self._cam_state
        if cam_state is None:
            self._last_home_yaw_err = None
            return AxisPair(0.0, 0.0), AxisPair(0.0, 0.0)

        home_pan = self._home_pan
        home_tilt = self._home_tilt

        yaw_err = 0.0
        pitch_err = 0.0

        if home_pan is not None:
            wrapped_yaw_err = _wrap_angle(home_pan - cam_state.pan)
            if self._last_home_yaw_err is not None:
                two_pi = 2.0 * math.pi
                candidates = (
                    wrapped_yaw_err,
                    wrapped_yaw_err + two_pi,
                    wrapped_yaw_err - two_pi,
                )
                yaw_err = min(
                    candidates,
                    key=lambda candidate: abs(candidate - self._last_home_yaw_err),
                )
            else:
                yaw_err = wrapped_yaw_err
            if abs(yaw_err) <= self._home_deadband:
                yaw_err = 0.0
            self._last_home_yaw_err = yaw_err
        if home_tilt is not None:
            pitch_err = home_tilt - cam_state.tilt
            if abs(pitch_err) <= self._home_deadband:
                pitch_err = 0.0

        yaw_rate = 0.0
        pitch_rate = 0.0

        if yaw_err != 0.0:
            desired_yaw_rate = self._cfg.kp.yaw * yaw_err
            desired_yaw_rate = _clamp(
                desired_yaw_rate, -self._cfg.rate_limits.yaw, self._cfg.rate_limits.yaw
            )
            yaw_rate = self._slew_axis(
                self._prev_rate.yaw, desired_yaw_rate, self._cfg.accel_limits.yaw, dt
            )

        if pitch_err != 0.0:
            desired_pitch_rate = self._cfg.kp.pitch * pitch_err
            desired_pitch_rate = _clamp(
                desired_pitch_rate, -self._cfg.rate_limits.pitch, self._cfg.rate_limits.pitch
            )
            pitch_rate = self._slew_axis(
                self._prev_rate.pitch, desired_pitch_rate, self._cfg.accel_limits.pitch, dt
            )

        return AxisPair(yaw_rate, pitch_rate), AxisPair(yaw_err, pitch_err)

    def _integrate(
        self,
        accum: float,
        err: float,
        dt: float,
        ki: float,
        rate_limit: float,
    ) -> float:
        if ki <= 0.0:
            return 0.0
        accum += err * dt
        if ki > 0:
            max_accum = rate_limit / max(ki, 1e-6)
            accum = _clamp(accum, -max_accum, max_accum)
        return accum

    def _slew_axis(self, prev: float, desired: float, accel_limit: float, dt: float) -> float:
        if accel_limit <= 0.0:
            return desired
        max_delta = accel_limit * dt
        return _clamp(desired, prev - max_delta, prev + max_delta)

    def _position_setpoints(
        self, yaw_rate: float, pitch_rate: float, dt: float
    ) -> Tuple[Optional[float], Optional[float]]:
        if self._cfg.mode != "position":
            return None, None
        pan = self._cam_state.pan if self._cam_state is not None else 0.0
        tilt = self._cam_state.tilt if self._cam_state is not None else 0.0
        return pan + yaw_rate * dt, tilt + pitch_rate * dt

    def _send_cmd(self, cmd: ControlCmd) -> None:
        payload = cmd.model_dump_json()
        try:
            self._pub.send_string(payload, flags=zmq.NOBLOCK)
        except zmq.Again:
            _LOG.warning("control_pub_backpressure")

    def _round_for_log(self, value: Any) -> Any:
        if isinstance(value, float):
            return round(value, self._log_float_precision)
        if isinstance(value, (list, tuple)):
            return [self._round_for_log(v) for v in value]
        if isinstance(value, dict):
            return {k: self._round_for_log(v) for k, v in value.items()}
        return value

    def _format_control_cli(self, payload: dict) -> str:
        parts = []

        frame_id = payload.get("frame_id")
        if frame_id is not None:
            parts.append(f"frame={frame_id}")

        target_ok = payload.get("target_ok")
        if target_ok is not None:
            parts.append("target=track" if target_ok else "target=hold")

        dt = payload.get("dt")
        if dt is not None:
            parts.append(f"dt={self._format_float(dt)}s")

        uv = payload.get("uv")
        if isinstance(uv, (list, tuple)) and len(uv) == 2:
            parts.append(f"uv={self._format_pair(uv)}")

        err_px = payload.get("err_px")
        if isinstance(err_px, (list, tuple)) and len(err_px) == 2:
            parts.append(f"err_px={self._format_pair(err_px)}")

        err_rad = payload.get("err_rad")
        if isinstance(err_rad, (list, tuple)) and len(err_rad) == 2:
            parts.append(f"err_rad={self._format_pair(err_rad)}")

        cmd_rate = payload.get("cmd_rate")
        if isinstance(cmd_rate, (list, tuple)) and len(cmd_rate) == 2:
            parts.append(f"cmd={self._format_pair(cmd_rate)}")

        if payload.get("home"):
            parts.append("home")

        return " | ".join(parts)

    def _format_float(self, value: Any) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):.{self._log_float_precision}f}"
        return str(value)

    def _format_pair(self, values: Sequence[Any]) -> str:
        return f"({self._format_float(values[0])}, {self._format_float(values[1])})"

