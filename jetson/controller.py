"""Closed-loop controller that turns detections into pan/tilt commands."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

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


@dataclass
class _TrackerState:
    timestamp: float
    uv: Tuple[float, float]
    velocity_uv: Tuple[float, float]
    distance_m: Optional[float]
    resolved_range_m: Optional[float]
    range_source: Optional[str]
    range_active: bool


@dataclass
class _TargetEstimate:
    timestamp: float
    uv: Tuple[float, float]
    velocity_uv: Tuple[float, float]
    distance_m: Optional[float]
    resolved_range_m: Optional[float]
    range_source: Optional[str]
    range_active: bool


class _TargetTracker:
    """Simple constant-velocity tracker for pixel-space targets."""

    _MIN_DT = 1e-3

    def __init__(self, config: ControlConfig, *, alpha: float, beta: float) -> None:
        self._cfg = config
        self._alpha = _clamp(alpha, 0.0, 1.0)
        self._beta = _clamp(beta, 0.0, 1.0)
        self._state: Optional[_TrackerState] = None

    def reset(self) -> None:
        self._state = None

    def has_state(self) -> bool:
        return self._state is not None

    def age(self, now: float) -> Optional[float]:
        if self._state is None:
            return None
        return max(0.0, now - self._state.timestamp)

    def last(self) -> Optional[_TargetEstimate]:
        if self._state is None:
            return None
        state = self._state
        return _TargetEstimate(
            timestamp=state.timestamp,
            uv=state.uv,
            velocity_uv=state.velocity_uv,
            distance_m=state.distance_m,
            resolved_range_m=state.resolved_range_m,
            range_source=state.range_source,
            range_active=state.range_active,
        )

    def update(
        self,
        measurement_uv: Tuple[float, float],
        timestamp: float,
        *,
        distance_m: Optional[float],
        resolved_range_m: Optional[float],
        range_source: Optional[str],
        range_active: bool,
        camera_rates: Optional[AxisPair] = None,
    ) -> _TargetEstimate:
        if self._state is None:
            velocity = (0.0, 0.0)
            self._state = _TrackerState(
                timestamp=timestamp,
                uv=(float(measurement_uv[0]), float(measurement_uv[1])),
                velocity_uv=velocity,
                distance_m=distance_m,
                resolved_range_m=resolved_range_m,
                range_source=range_source,
                range_active=range_active,
            )
            return self.last()  # type: ignore[return-value]

        prev = self._state
        dt = max(timestamp - prev.timestamp, self._MIN_DT)
        cam_shift = self._camera_pixel_shift(camera_rates, dt)
        pred_u = prev.uv[0] + prev.velocity_uv[0] * dt + cam_shift[0]
        pred_v = prev.uv[1] + prev.velocity_uv[1] * dt + cam_shift[1]

        resid_u = float(measurement_uv[0]) - pred_u
        resid_v = float(measurement_uv[1]) - pred_v

        alpha = self._alpha
        beta = self._beta

        new_u = pred_u + alpha * resid_u
        new_v = pred_v + alpha * resid_v

        if dt <= 0.0:
            new_vu = 0.0
            new_vv = 0.0
        else:
            new_vu = prev.velocity_uv[0] + (beta * resid_u) / dt
            new_vv = prev.velocity_uv[1] + (beta * resid_v) / dt

        self._state = _TrackerState(
            timestamp=timestamp,
            uv=(new_u, new_v),
            velocity_uv=(new_vu, new_vv),
            distance_m=distance_m if distance_m is not None else prev.distance_m,
            resolved_range_m=(
                resolved_range_m if resolved_range_m is not None else prev.resolved_range_m
            ),
            range_source=range_source if range_source is not None else prev.range_source,
            range_active=range_active,
        )

        return self.last()  # type: ignore[return-value]

    def predict(
        self,
        timestamp: float,
        *,
        camera_rates: Optional[AxisPair] = None,
        max_dt: Optional[float] = None,
    ) -> Optional[_TargetEstimate]:
        if self._state is None:
            return None

        prev = self._state
        dt = timestamp - prev.timestamp
        if max_dt is not None:
            dt = min(dt, max_dt)
        if dt < 0.0:
            dt = 0.0
        cam_shift = self._camera_pixel_shift(camera_rates, dt)
        pred_u = prev.uv[0] + prev.velocity_uv[0] * dt + cam_shift[0]
        pred_v = prev.uv[1] + prev.velocity_uv[1] * dt + cam_shift[1]

        return _TargetEstimate(
            timestamp=prev.timestamp + dt,
            uv=(pred_u, pred_v),
            velocity_uv=prev.velocity_uv,
            distance_m=prev.distance_m,
            resolved_range_m=prev.resolved_range_m,
            range_source=prev.range_source,
            range_active=prev.range_active,
        )

    def _camera_pixel_shift(self, rates: Optional[AxisPair], dt: float) -> Tuple[float, float]:
        if rates is None or dt <= 0.0:
            return (0.0, 0.0)
        yaw_sign = self._cfg.yaw_sign if self._cfg.yaw_sign != 0 else 1.0
        pitch_sign = self._cfg.pitch_sign if self._cfg.pitch_sign != 0 else 1.0
        du = -(self._cfg.fx_px / yaw_sign) * rates.yaw * dt
        dv = -(self._cfg.fy_px / pitch_sign) * rates.pitch * dt
        return (du, dv)

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
        self._lock = threading.RLock()
        self._cfg = config
        self._pub = pub
        self._laser_mount = laser_mount

        self._lost_timeout_s = config.lost_target_timeout_ms / 1000.0
        self._default_dt = (
            config.loop_dt if config.loop_dt is not None else 1.0 / max(1.0, config.loop_hz or 30.0)
        )

        self._tracker = _TargetTracker(
            config,
            alpha=config.tracker.alpha,
            beta=config.tracker.beta,
        )
        self._max_prediction_s = max(0.0, config.tracker.max_prediction_ms / 1000.0)
        self._control_horizon_s = max(0.0, config.lead.default_latency_ms / 1000.0)
        self._horizon_alpha = _clamp(config.lead.ema_alpha, 0.0, 1.0)
        self._lead_min_s = max(0.0, config.lead.min_latency_ms / 1000.0)
        self._lead_max_s = max(self._lead_min_s, config.lead.max_latency_ms / 1000.0)
        self._last_cam_state_time: Optional[float] = None

        self._latest_detection: Optional[_DetectionState] = None
        self._latest_target_idx: Optional[int] = None
        self._last_frame_id: int = 0
        self._last_src_ts_ms: int = 0
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

        with self._lock:
            target_uv = self._select_target(msg)

            if target_uv is None:
                msg.target_distance_smoothed_m = None
                msg.laser_origin_px = None
                msg.laser_dot_px = None
                msg.laser_on_target = None
                msg.laser_range_m = None
                msg.laser_range_source = None
                msg.parallax_compensation_active = False
                self._laser_overlay = None
                self._resolved_range = None

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

            tracker_estimate: Optional[_TargetEstimate] = None
            if target_uv is not None:
                range_m, range_source, parallax_active = self._resolve_laser_range(
                    msg.target_distance_smoothed_m
                )
                msg.laser_range_m = range_m
                msg.laser_range_source = range_source

                self._latest_detection.resolved_range_m = range_m
                self._latest_detection.range_source = range_source
                self._latest_detection.range_active = parallax_active

                tracker_estimate = self._tracker.update(
                    target_uv,
                    now,
                    distance_m=msg.target_distance_smoothed_m,
                    resolved_range_m=range_m,
                    range_source=range_source,
                    range_active=parallax_active,
                    camera_rates=self._current_cam_rates(),
                )
            else:
                msg.laser_range_m = None
                msg.laser_range_source = None
                self._resolved_range = None
                parallax_active = False
                tracker_estimate = self._tracker.predict(
                    now, camera_rates=self._current_cam_rates(), max_dt=self._max_prediction_s
                )

            if tracker_estimate is not None:
                self._latest_detection.target_uv = tracker_estimate.uv
                self._latest_detection.target_distance_m = tracker_estimate.distance_m
                self._latest_detection.resolved_range_m = tracker_estimate.resolved_range_m
                self._latest_detection.range_source = tracker_estimate.range_source
                self._latest_detection.range_active = tracker_estimate.range_active
                msg.target_distance_smoothed_m = tracker_estimate.distance_m

            if tracker_estimate is not None:
                range_m = tracker_estimate.resolved_range_m
                range_source = tracker_estimate.range_source
                parallax_active = tracker_estimate.range_active
                target_for_overlay = tracker_estimate.uv
            else:
                range_m = None
                range_source = None
                parallax_active = False
                target_for_overlay = target_uv

            self._update_laser_overlay(
                msg,
                raw_target_uv=target_uv,
                predicted_target_uv=target_for_overlay,
                range_m=range_m,
                range_source=range_source,
                parallax_active=parallax_active,
            )

    def update_cam_state(self, state: CamState) -> None:
        now = time.monotonic()
        with self._lock:
            self._cam_state = state
            if state.home_pan is not None:
                self._home_pan = float(state.home_pan)
            if state.home_tilt is not None:
                self._home_tilt = float(state.home_tilt)
            self._last_cam_state_time = now

            if self._last_cmd_time is not None:
                latency = max(0.0, now - self._last_cmd_time)
                if latency > 0.0:
                    self._blend_control_horizon(latency)

    def tick(self, now: Optional[float] = None) -> None:
        """Advance the controller and publish a command if due."""

        if now is None:
            now = time.monotonic()

        with self._lock:
            if self._cfg.loop_dt is not None and self._last_cmd_time is not None:
                if (now - self._last_cmd_time) < (self._cfg.loop_dt - 1e-6):
                    return

            dt = self._compute_dt(now)

            detection = self._latest_detection
            target_recent = self._is_target_recent(now)

            camera_rates = self._current_cam_rates()
            tracker_now = self._tracker.predict(
                now,
                camera_rates=camera_rates,
                max_dt=self._max_prediction_s if self._max_prediction_s > 0 else None,
            )

            horizon_s = self._current_horizon_s()
            future_time = now + horizon_s
            tracker_future = self._tracker.predict(
                future_time,
                camera_rates=camera_rates,
                max_dt=self._max_prediction_s if self._max_prediction_s > 0 else None,
            )

            if detection and tracker_future is not None and target_recent:
                tracker_age = self._tracker.age(now)
                cmd = self._build_tracking_cmd(
                    detection,
                    dt=dt,
                    now=now,
                    horizon_s=horizon_s,
                    estimate_now=tracker_now or tracker_future,
                    estimate_future=tracker_future,
                    tracker_age_s=tracker_age,
                )
                self._tracking_active = True
            else:
                if self._cfg.reinit_on_lost and not target_recent:
                    self._prev_err = None
                    self._integ = AxisPair(0.0, 0.0)
                    self._prev_rate = AxisPair(0.0, 0.0)
                    self._tracker.reset()
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
        predicted_target_uv: Optional[Tuple[float, float]],
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

            target_for_error = predicted_target_uv if predicted_target_uv is not None else raw_target_uv
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
        estimate = self._tracker.last()
        if estimate is None:
            return False
        if self._lost_timeout_s <= 0.0:
            return True
        return (now - estimate.timestamp) <= self._lost_timeout_s

    def _compute_dt(self, now: float) -> float:
        if self._last_cmd_time is None:
            dt = self._default_dt
        else:
            dt = now - self._last_cmd_time
        if not (dt > 0.0):
            dt = self._default_dt
        return _clamp(dt, self._MIN_DT, self._MAX_DT)

    def _current_cam_rates(self) -> Optional[AxisPair]:
        state = self._cam_state
        if state is None:
            return None
        pan_rate = float(state.pan_rate) if state.pan_rate is not None else 0.0
        tilt_rate = float(state.tilt_rate) if state.tilt_rate is not None else 0.0
        return AxisPair(pan_rate, tilt_rate)

    def _current_horizon_s(self) -> float:
        return _clamp(self._control_horizon_s, self._lead_min_s, self._lead_max_s)

    def _blend_control_horizon(self, measurement: float) -> None:
        if not math.isfinite(measurement) or measurement <= 0.0:
            return
        measurement = _clamp(measurement, self._lead_min_s, self._lead_max_s)
        alpha = self._horizon_alpha
        if alpha <= 0.0:
            self._control_horizon_s = measurement
            return
        if alpha >= 1.0 or not math.isfinite(self._control_horizon_s):
            self._control_horizon_s = measurement
            return
        self._control_horizon_s = alpha * measurement + (1.0 - alpha) * self._control_horizon_s

    def _build_tracking_cmd(
        self,
        detection: _DetectionState,
        *,
        dt: float,
        now: float,
        horizon_s: float,
        estimate_now: _TargetEstimate,
        estimate_future: _TargetEstimate,
        tracker_age_s: Optional[float],
    ) -> ControlCmd:
        target_uv = estimate_future.uv
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

        horizon_for_ff = max(horizon_s, 1e-3)
        ff_yaw = raw_px_err.yaw / (self._cfg.fx_px * horizon_for_ff)
        ff_pitch = raw_px_err.pitch / (self._cfg.fy_px * horizon_for_ff)

        yaw_rate += ff_yaw
        pitch_rate += ff_pitch

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
            lead_horizon_s=horizon_s,
            tracker_age_s=tracker_age_s,
            target_velocity_uv=(
                (float(estimate_now.velocity_uv[0]), float(estimate_now.velocity_uv[1]))
                if estimate_now is not None
                else None
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
                "ff_rate": [ff_yaw, ff_pitch],
                "lead_s": horizon_s,
                "tracker_age_s": tracker_age_s,
                "target_vel_uv": (
                    [
                        float(estimate_now.velocity_uv[0]),
                        float(estimate_now.velocity_uv[1]),
                    ]
                    if estimate_now is not None
                    else None
                ),
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
        last_estimate = self._tracker.last()
        if last_estimate is not None:
            uv = last_estimate.uv
        else:
            uv = (self._cfg.cx_px, self._cfg.cy_px)

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

