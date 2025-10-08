"""Closed-loop controller that turns detections into pan/tilt commands."""

from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional, Sequence, Tuple

import zmq

from common.control import (
    AxisPair,
    ControlConfig,
    LaserMountConfig,
    angular_error_from_pixel_delta,
    pixel_delta,
)
from common.geometry import laser_ray_to_pixel, project_point_to_pixel
from common.schemas import Box, CamState, ControlCmd, DetectionMsg


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


@dataclass
class _LaserOverlay:
    origin_px: Optional[Tuple[float, float]]
    dot_px: Optional[Tuple[float, float]]
    on_target: Optional[bool]
    active: bool
    range_m: Optional[float]
    range_source: Optional[str]


class ControlLoop:
    """Runs a rate-mode PID loop for yaw/pitch using the latest detection."""

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

        self._prev_detection_uv: Optional[Tuple[float, float]] = None
        self._prev_detection_time: Optional[float] = None
        self._pixel_velocity: Optional[Tuple[float, float]] = None
        self._prediction_ready = False
        self._stability_window: Deque[float] = deque(
            maxlen=max(1, int(self._cfg.prediction.stabilize_frames))
        )

        self._distance_alpha = None if distance_alpha is None else _clamp(distance_alpha, 0.0, 1.0)
        self._distance_ema: Optional[float] = None
        self._resolved_range: Optional[float] = None
        self._warned_ground_plane = False

        self._cam_state: Optional[CamState] = None
        self._home_pan: Optional[float] = None
        self._home_tilt: Optional[float] = None
        self._home_deadband = math.radians(0.5)

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
            self._reset_prediction_state()

        self._latest_detection = _DetectionState(
            frame_id=msg.frame_id,
            src_ts_ms=msg.src_ts_ms,
            timestamp=now,
            target_uv=target_uv,
            target_distance_m=msg.target_distance_smoothed_m,
            resolved_range_m=None,
            range_source=None,
            range_active=False,
        )
        self._last_frame_id = msg.frame_id
        self._last_src_ts_ms = msg.src_ts_ms

        if target_uv is not None:
            self._last_detection_ts = now
            if self._smoothed_uv is None or not self._tracking_active:
                self._smoothed_uv = target_uv
            else:
                self._smoothed_uv = self._smooth_uv(target_uv)

            self._update_prediction_velocity(target_uv, now)

            range_m, range_source, parallax_active = self._resolve_laser_range(
                msg.target_distance_smoothed_m
            )
            msg.laser_range_m = range_m
            msg.laser_range_source = range_source

            self._latest_detection.resolved_range_m = range_m
            self._latest_detection.range_source = range_source
            self._latest_detection.range_active = parallax_active
        else:
            msg.laser_range_m = None
            msg.laser_range_source = None
            self._resolved_range = None
            parallax_active = False
            self._reset_prediction_state()

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

        detection = self._latest_detection
        target_recent = self._is_target_recent(now)

        if detection and detection.target_uv is not None and target_recent:
            cmd = self._build_tracking_cmd(detection, dt, now)
            self._tracking_active = True
        else:
            if self._cfg.reinit_on_lost:
                self._prev_err = None
                self._integ = AxisPair(0.0, 0.0)
                self._smoothed_uv = None
                self._prev_rate = AxisPair(0.0, 0.0)
                self._reset_prediction_state()
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

    def _reset_prediction_state(self) -> None:
        self._prev_detection_uv = None
        self._prev_detection_time = None
        self._pixel_velocity = None
        self._prediction_ready = False
        self._stability_window.clear()

    def _update_prediction_velocity(
        self, measurement: Tuple[float, float], timestamp: float
    ) -> None:
        cfg = self._cfg.prediction
        if not cfg.enabled:
            self._prev_detection_uv = measurement
            self._prev_detection_time = timestamp
            self._pixel_velocity = None
            return

        prev_uv = self._prev_detection_uv
        prev_time = self._prev_detection_time
        if prev_uv is not None and prev_time is not None:
            dt = timestamp - prev_time
            if dt > 1e-6:
                du = measurement[0] - prev_uv[0]
                dv = measurement[1] - prev_uv[1]
                vx = du / dt
                vy = dv / dt
                alpha = _clamp(cfg.velocity_alpha, 0.0, 1.0)
                if self._pixel_velocity is None or alpha <= 0.0 or alpha >= 1.0:
                    filtered = (vx, vy)
                else:
                    prev_vx, prev_vy = self._pixel_velocity
                    filtered = (
                        alpha * vx + (1.0 - alpha) * prev_vx,
                        alpha * vy + (1.0 - alpha) * prev_vy,
                    )
                if cfg.max_px_per_s is not None:
                    speed = math.hypot(filtered[0], filtered[1])
                    if speed > cfg.max_px_per_s:
                        scale = cfg.max_px_per_s / speed
                        filtered = (filtered[0] * scale, filtered[1] * scale)
                self._pixel_velocity = filtered

        self._prev_detection_uv = measurement
        self._prev_detection_time = timestamp

    def _predict_target_uv(
        self, measurement: Tuple[float, float], detection: _DetectionState, now: float
    ) -> Tuple[float, float]:
        cfg = self._cfg.prediction
        if not cfg.enabled or self._pixel_velocity is None or not self._prediction_ready:
            return measurement

        horizon = max(0.0, (now - detection.timestamp) + cfg.lookahead_s)
        if horizon <= 0.0:
            return measurement

        cam_u = 0.0
        cam_v = 0.0
        if self._cam_state is not None:
            if self._cam_state.pan_rate is not None:
                cam_u = -float(self._cam_state.pan_rate) * self._cfg.fx_px
            if self._cam_state.tilt_rate is not None:
                cam_v = -float(self._cam_state.tilt_rate) * self._cfg.fy_px

        vx = self._pixel_velocity[0] - cam_u
        vy = self._pixel_velocity[1] - cam_v
        predicted_u = measurement[0] + vx * horizon
        predicted_v = measurement[1] + vy * horizon
        predicted_u = _clamp(predicted_u, 0.0, float(self._cfg.width))
        predicted_v = _clamp(predicted_v, 0.0, float(self._cfg.height))
        return (predicted_u, predicted_v)

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

    def _is_target_recent(self, now: float) -> bool:
        if self._last_detection_ts is None:
            return False
        if self._lost_timeout_s <= 0.0:
            return True if self._latest_detection and self._latest_detection.target_uv else False
        return (now - self._last_detection_ts) <= self._lost_timeout_s

    def _compute_dt(self, now: float) -> float:
        if self._last_cmd_time is None:
            dt = self._default_dt
        else:
            dt = now - self._last_cmd_time
        if not (dt > 0.0):
            dt = self._default_dt
        return _clamp(dt, self._MIN_DT, self._MAX_DT)

    def _build_tracking_cmd(
        self, detection: _DetectionState, dt: float, now: float
    ) -> ControlCmd:
        assert detection.target_uv is not None
        base_uv = self._smoothed_uv or detection.target_uv
        aim_uv = self._aim_reference_uv(detection)

        measured_px_err = pixel_delta(
            base_uv[0],
            base_uv[1],
            aim_uv[0],
            aim_uv[1],
            self._cfg,
            apply_deadband=False,
        )
        self._update_prediction_stability(measured_px_err)

        target_uv = self._predict_target_uv(base_uv, detection, now)

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

    def _update_prediction_stability(self, raw_px_err: AxisPair) -> None:
        cfg = self._cfg.prediction
        if not cfg.enabled:
            self._prediction_ready = False
            self._stability_window.clear()
            return

        window_size = max(1, int(cfg.stabilize_frames))
        if self._stability_window.maxlen != window_size:
            self._stability_window = deque(self._stability_window, maxlen=window_size)

        magnitude = math.hypot(raw_px_err.yaw, raw_px_err.pitch)
        self._stability_window.append(magnitude)

        if len(self._stability_window) < window_size:
            self._prediction_ready = False
            return

        mean = sum(self._stability_window) / window_size
        variance = sum((value - mean) ** 2 for value in self._stability_window) / window_size

        threshold = max(0.0, float(cfg.stabilize_err_var_px2))
        self._prediction_ready = variance <= threshold

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
        )

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
            return AxisPair(0.0, 0.0), AxisPair(0.0, 0.0)

        home_pan = self._home_pan
        home_tilt = self._home_tilt

        yaw_err = 0.0
        pitch_err = 0.0

        if home_pan is not None:
            yaw_err = _wrap_angle(home_pan - cam_state.pan)
            if abs(yaw_err) <= self._home_deadband:
                yaw_err = 0.0
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

