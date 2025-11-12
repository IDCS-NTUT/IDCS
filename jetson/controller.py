"""Closed-loop controller that turns detections into pan/tilt commands."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

import zmq

from common.control import (
    AxisPair,
    ControlConfig,
    MPCConfig,
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
class _MotionState:
    timestamp: float
    yaw_angle: float
    pitch_angle: float
    yaw_rate: float
    pitch_rate: float


class BaseControlLoop:
    """Shared logic for closed-loop controllers operating on detections."""

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

        self._motion_state: Optional[_MotionState] = None
        self._motion_target_idx: Optional[int] = None
        self._lead_time_s = max(self._default_dt, 1e-3)
        self._lead_latency_alpha = 0.2
        self._lead_latency_ready = False
        self._last_motion_rates: Optional[AxisPair] = None
        self._last_known_target_uv: Optional[Tuple[float, float]] = None
        self._predictive_end_time: Optional[float] = None
        self._predictive_rates: Optional[AxisPair] = None
        self._last_target_box_size_px: Optional[Tuple[float, float]] = None

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

        self._reset_controller_state()

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
            self._update_motion_state(
                msg,
                target_uv=target_uv,
                timestamp=now,
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

        detection = self._latest_detection
        target_recent = self._is_target_recent(now)

        if detection and detection.target_uv is not None and target_recent:
            cmd = self._build_tracking_cmd(detection, dt, now)
            self._tracking_active = True
        elif self._is_predictive_active(now):
            cmd = self._build_predictive_cmd(dt, now)
            self._tracking_active = False
        else:
            if self._cfg.reinit_on_lost:
                self._reset_controller_state()
                self._smoothed_uv = None
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
        target_idx: Optional[int],
    ) -> None:
        if target_idx is None:
            self._motion_state = None
            self._motion_target_idx = None
            msg.target_velocity_px_s = None
            msg.target_lead_uv = None
            msg.target_lead_time_s = None
            return

        prev_state = self._motion_state if self._motion_target_idx == target_idx else None
        yaw_angle = math.atan((target_uv[0] - self._cfg.cx_px) / self._cfg.fx_px)
        pitch_angle = math.atan((target_uv[1] - self._cfg.cy_px) / self._cfg.fy_px)

        velocity_px = (0.0, 0.0)
        lead_uv = target_uv
        lead_time = max(self._lead_time_s, 1e-3)
        motion_rates: Optional[AxisPair] = None

        if prev_state is not None:
            dt = timestamp - prev_state.timestamp
            if dt < 1e-3 or not math.isfinite(dt) or dt > 1.0:
                prev_state = None
            else:
                raw_yaw_vel = (yaw_angle - prev_state.yaw_angle) / dt
                raw_pitch_vel = (pitch_angle - prev_state.pitch_angle) / dt
                cam_pan_rate = 0.0
                cam_tilt_rate = 0.0
                if self._cam_state is not None:
                    if self._cam_state.pan_rate is not None and math.isfinite(self._cam_state.pan_rate):
                        cam_pan_rate = float(self._cam_state.pan_rate)
                    if self._cam_state.tilt_rate is not None and math.isfinite(self._cam_state.tilt_rate):
                        cam_tilt_rate = float(self._cam_state.tilt_rate)

                yaw_vel = raw_yaw_vel + cam_pan_rate
                pitch_vel = raw_pitch_vel + cam_tilt_rate

                motion_rates = AxisPair(yaw=yaw_vel, pitch=pitch_vel)

                yaw_angle_lead = yaw_angle + yaw_vel * lead_time
                pitch_angle_lead = pitch_angle + pitch_vel * lead_time

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

        msg.target_velocity_px_s = (float(velocity_px[0]), float(velocity_px[1]))
        msg.target_lead_uv = (float(lead_uv[0]), float(lead_uv[1]))
        msg.target_lead_time_s = float(lead_time)

        self._motion_state = _MotionState(
            timestamp=timestamp,
            yaw_angle=yaw_angle,
            pitch_angle=pitch_angle,
            yaw_rate=motion_rates.yaw if motion_rates else 0.0,
            pitch_rate=motion_rates.pitch if motion_rates else 0.0,
        )
        self._motion_target_idx = target_idx
        if motion_rates is not None:
            self._last_motion_rates = motion_rates

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

        rate_cmd, debug = self._controller_step(
            detection=detection,
            target_uv=target_uv,
            aim_uv=aim_uv,
            raw_err_px=raw_px_err,
            ctrl_err_px=ctrl_px_err,
            err_rad=err_rad,
            prev_err_rad=self._prev_err,
            dt=dt,
            now=now,
            tracking_active=self._tracking_active,
        )

        pan_abs, tilt_abs = self._position_setpoints(rate_cmd.yaw, rate_cmd.pitch, dt)

        cmd = ControlCmd(
            frame_id=detection.frame_id,
            src_ts_ms=detection.src_ts_ms,
            cmd_ts_ms=int(time.monotonic_ns() / 1e6),
            target_ok=True,
            target_uv=(float(target_uv[0]), float(target_uv[1])),
            err_uv=(ctrl_px_err.yaw, ctrl_px_err.pitch),
            err_rad=(err_rad.yaw, err_rad.pitch),
            pan_rate_cmd=rate_cmd.yaw,
            tilt_rate_cmd=rate_cmd.pitch,
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

        log_payload = {
            "frame_id": detection.frame_id,
            "target_ok": True,
            "dt": round(dt, 6),
            "uv": [float(target_uv[0]), float(target_uv[1])],
            "aim_uv": [float(aim_uv[0]), float(aim_uv[1])],
            "err_px": [raw_px_err.yaw, raw_px_err.pitch],
            "err_rad": [err_rad.yaw, err_rad.pitch],
            "cmd_rate": [rate_cmd.yaw, rate_cmd.pitch],
            "range_m": detection.resolved_range_m,
            "range_src": detection.range_source,
            "parallax_active": detection.range_active,
        }
        if debug:
            log_payload.update(debug)

        self._log_control_state(log_payload, target_ok=True, now=now)

        return cmd

    def _controller_step(
        self,
        *,
        detection: _DetectionState,
        target_uv: Tuple[float, float],
        aim_uv: Tuple[float, float],
        raw_err_px: AxisPair,
        ctrl_err_px: AxisPair,
        err_rad: AxisPair,
        prev_err_rad: Optional[AxisPair],
        dt: float,
        now: float,
        tracking_active: bool,
    ) -> Tuple[AxisPair, Optional[dict]]:
        raise NotImplementedError

    def _reset_controller_state(self) -> None:
        self._prev_err = None
        self._prev_rate = AxisPair(0.0, 0.0)

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
        )

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
        return AxisPair(0.0, 0.0), AxisPair(0.0, 0.0)

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


class PIDControlLoop(BaseControlLoop):
    """Compatibility wrapper for the legacy PID-based controller."""

    def _reset_controller_state(self) -> None:
        super()._reset_controller_state()
        self._integ = AxisPair(0.0, 0.0)

    def _controller_step(
        self,
        *,
        detection: _DetectionState,
        target_uv: Tuple[float, float],
        aim_uv: Tuple[float, float],
        raw_err_px: AxisPair,
        ctrl_err_px: AxisPair,
        err_rad: AxisPair,
        prev_err_rad: Optional[AxisPair],
        dt: float,
        now: float,
        tracking_active: bool,
    ) -> Tuple[AxisPair, Optional[dict]]:
        if not tracking_active or prev_err_rad is None:
            derr = AxisPair(0.0, 0.0)
        else:
            derr = AxisPair(
                yaw=(err_rad.yaw - prev_err_rad.yaw) / dt,
                pitch=(err_rad.pitch - prev_err_rad.pitch) / dt,
            )

        integ_yaw = self._integrate(self._integ.yaw, err_rad.yaw, dt, self._cfg.ki.yaw, self._cfg.rate_limits.yaw)
        integ_pitch = self._integrate(
            self._integ.pitch,
            err_rad.pitch,
            dt,
            self._cfg.ki.pitch,
            self._cfg.rate_limits.pitch,
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
        pitch_rate = self._slew_axis(
            self._prev_rate.pitch, pitch_rate, self._cfg.accel_limits.pitch, dt
        )

        self._prev_rate = AxisPair(yaw_rate, pitch_rate)
        self._prev_err = err_rad

        return AxisPair(yaw_rate, pitch_rate), None

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


class MPCControlLoop(BaseControlLoop):
    """Model predictive controller leveraging the structured MPC config."""

    class _AxisController:
        _STATE_DIM = 3

        def __init__(
            self,
            axis: str,
            *,
            config: ControlConfig,
            mpc_cfg: MPCConfig,
        ) -> None:
            self._axis = axis
            self._cfg = config
            self._mpc_cfg = mpc_cfg
            self._horizon_steps = max(1, mpc_cfg.horizon.steps)
            self._control_steps = max(1, mpc_cfg.horizon.control_horizon_steps)
            self._step_dt = mpc_cfg.horizon.step_dt_s

            self._rate_limit = getattr(mpc_cfg.actuator_limits.rate, axis)
            self._accel_limit = getattr(mpc_cfg.actuator_limits.accel, axis)
            self._input_weight = getattr(mpc_cfg.cost.input, axis)
            self._delta_weight = getattr(mpc_cfg.cost.delta, axis)

            self._a_u = getattr(mpc_cfg.plant.a_u, axis)
            self._a_f = getattr(mpc_cfg.plant.a_f, axis)

            self._q_theta_proc = getattr(mpc_cfg.estimator.q_theta, axis)
            self._q_omega_proc = getattr(mpc_cfg.estimator.q_omega, axis)
            self._q_disturbance_proc = getattr(mpc_cfg.estimator.q_disturbance, axis)
            self._r_theta_meas = max(getattr(mpc_cfg.estimator.r_theta, axis), 1e-9)

            self._q_theta_base = getattr(mpc_cfg.adaptive.q_theta_base, axis)
            self._q_omega_base = getattr(mpc_cfg.adaptive.q_omega_base, axis)
            self._alpha_distance = mpc_cfg.adaptive.alpha_distance
            self._alpha_lateral_velocity = mpc_cfg.adaptive.alpha_lateral_velocity
            self._alpha_time = mpc_cfg.adaptive.alpha_time
            self._weight_exponent = max(mpc_cfg.adaptive.exponent, 1e-6)
            self._epsilon = mpc_cfg.adaptive.epsilon
            self._weight_min = mpc_cfg.adaptive.w_min
            self._weight_max = mpc_cfg.adaptive.w_max

            self._theta_bound = self._axis_bound(mpc_cfg.state_constraints.error)
            self._rate_bound = self._axis_bound(mpc_cfg.state_constraints.rate)

            self._xhat = np.zeros(self._STATE_DIM, dtype=float)
            self._P = np.eye(self._STATE_DIM, dtype=float)
            self._last_u = 0.0

            self._Sx, self._Su = self._build_prediction_matrices()
            self._theta_selector, self._omega_selector = self._build_output_selectors()
            self._theta_sx = self._theta_selector @ self._Sx
            self._theta_su = self._theta_selector @ self._Su
            self._omega_sx = self._omega_selector @ self._Sx
            self._omega_su = self._omega_selector @ self._Su
            self._identity = np.eye(self._control_steps, dtype=float)
            self._delta_matrix = self._build_delta_matrix()
            self._C = np.array([[1.0, 0.0, 0.0]], dtype=float)

        def reset(self) -> None:
            self._xhat[:] = 0.0
            self._P = np.eye(self._STATE_DIM, dtype=float)
            self._last_u = 0.0

        def solve(
            self,
            theta_measurement: float,
            *,
            dt_actual: float,
            distance: float,
            lateral_velocity: float,
            radial_velocity: float,
        ) -> Tuple[float, Optional[float], Sequence[dict[str, float]]]:
            dt = max(1e-3, dt_actual)
            self._kalman_update(theta_measurement, dt)

            weights = self._compute_weights(distance, lateral_velocity, radial_velocity)
            q_theta_vec = self._q_theta_base * weights
            q_omega_vec = self._q_omega_base * weights

            theta_free = self._theta_sx @ self._xhat
            omega_free = self._omega_sx @ self._xhat

            H = np.zeros((self._control_steps, self._control_steps), dtype=float)
            f = np.zeros(self._control_steps, dtype=float)

            # Tracking cost for theta
            theta_mat = self._theta_su
            H += 2.0 * theta_mat.T @ (theta_mat * q_theta_vec[:, None])
            f += 2.0 * theta_mat.T @ (q_theta_vec * theta_free)

            # Tracking cost for omega
            omega_mat = self._omega_su
            H += 2.0 * omega_mat.T @ (omega_mat * q_omega_vec[:, None])
            f += 2.0 * omega_mat.T @ (q_omega_vec * omega_free)

            # Control effort
            if self._input_weight > 0.0:
                H += 2.0 * self._input_weight * self._identity

            # Smoothness (delta U)
            if self._delta_weight > 0.0:
                H += 2.0 * self._delta_weight * (self._delta_matrix.T @ self._delta_matrix)
                delta_offset = np.zeros(self._control_steps, dtype=float)
                delta_offset[0] = self._last_u
                f += -2.0 * self._delta_weight * (self._delta_matrix.T @ delta_offset)

            # Regularization to ensure positive definiteness
            H += 1e-6 * self._identity

            control_sequence = self._solve_qp(H, f, theta_free, omega_free)

            plan = self._build_plan(control_sequence)

            u0 = float(control_sequence[0]) if control_sequence.size else 0.0
            self._last_u = self._clamp_input(u0, reference=None)

            A_dt, B_dt = self._discrete_matrices(dt)
            current_rate = float(self._xhat[1])
            x_next = A_dt @ self._xhat + B_dt.flatten() * self._last_u
            omega_cmd = float(np.clip(x_next[1], -self._rate_limit, self._rate_limit))
            accel = (omega_cmd - current_rate) / dt if dt > 0.0 else None

            return omega_cmd, accel, plan

        def _axis_bound(self, bounds: Optional[AxisPair]) -> Optional[float]:
            if bounds is None:
                return None
            value = getattr(bounds, self._axis)
            if value is None or value <= 0.0:
                return None
            return float(value)

        def _compute_weights(
            self,
            distance: float,
            lateral_velocity: float,
            radial_velocity: float,
        ) -> np.ndarray:
            dist = max(distance, self._epsilon)
            tau = dist / max(self._epsilon, abs(radial_velocity))
            weight = (
                self._alpha_distance * (1.0 / (dist + self._epsilon)) ** self._weight_exponent
                + self._alpha_lateral_velocity * (abs(lateral_velocity) / (dist + self._epsilon))
                + self._alpha_time * (1.0 / (tau + self._epsilon))
            )
            weight = float(np.clip(weight, self._weight_min, self._weight_max))
            return np.full(self._horizon_steps, weight, dtype=float)

        def _kalman_update(self, measurement: float, dt: float) -> None:
            A_dt, B_dt = self._discrete_matrices(dt)
            x_pred = A_dt @ self._xhat + B_dt.flatten() * self._last_u
            Q = np.diag(
                [
                    max(self._q_theta_proc, 0.0),
                    max(self._q_omega_proc, 0.0),
                    max(self._q_disturbance_proc, 0.0),
                ]
            ) * dt
            P_pred = A_dt @ self._P @ A_dt.T + Q

            innovation = measurement - float(self._C @ x_pred)
            S = float(self._C @ P_pred @ self._C.T) + self._r_theta_meas
            if S <= 0.0:
                S = self._r_theta_meas
            K = (P_pred @ self._C.T) / S
            x_upd = x_pred + (K.flatten() * innovation)
            P_upd = (np.eye(self._STATE_DIM) - K @ self._C) @ P_pred

            # Ensure symmetry and numerical stability
            self._xhat = x_upd
            self._P = 0.5 * (P_upd + P_upd.T)

        def _solve_qp(
            self,
            H: np.ndarray,
            f: np.ndarray,
            theta_free: np.ndarray,
            omega_free: np.ndarray,
        ) -> np.ndarray:
            try:
                solution = np.linalg.solve(H, -f)
            except np.linalg.LinAlgError:
                solution = np.linalg.lstsq(H, -f, rcond=None)[0]

            control = np.array(solution, dtype=float)
            control = self._apply_input_limits(control)
            control = self._enforce_state_bounds(control, theta_free, omega_free)
            return control

        def _apply_input_limits(self, control: np.ndarray) -> np.ndarray:
            rate_limit = self._rate_limit
            accel_limit = self._accel_limit
            delta_limit = None
            if accel_limit is not None and accel_limit > 0.0:
                delta_limit = accel_limit * self._step_dt

            prev = self._last_u
            for idx in range(self._control_steps):
                lo = -rate_limit
                hi = rate_limit
                if delta_limit is not None:
                    lo = max(lo, prev - delta_limit)
                    hi = min(hi, prev + delta_limit)
                control[idx] = self._clamp_input(control[idx], reference=(lo, hi))
                prev = control[idx]

            if delta_limit is not None:
                # Backward pass to ensure difference constraints remain satisfied
                for idx in range(self._control_steps - 2, -1, -1):
                    ref = self._last_u if idx == 0 else control[idx - 1]
                    lo = max(-rate_limit, ref - delta_limit)
                    hi = min(rate_limit, ref + delta_limit)
                    control[idx] = self._clamp_input(control[idx], reference=(lo, hi))

            return control

        def _enforce_state_bounds(
            self,
            control: np.ndarray,
            theta_free: np.ndarray,
            omega_free: np.ndarray,
        ) -> np.ndarray:
            if self._theta_bound is None and self._rate_bound is None:
                return control

            for _ in range(4):
                adjusted = False
                if self._theta_bound is not None:
                    theta_pred = theta_free + self._theta_su @ control
                    adjusted = self._adjust_for_bound(
                        control,
                        theta_pred,
                        self._theta_bound,
                        self._theta_su,
                        adjusted,
                    )
                if self._rate_bound is not None:
                    omega_pred = omega_free + self._omega_su @ control
                    adjusted = self._adjust_for_bound(
                        control,
                        omega_pred,
                        self._rate_bound,
                        self._omega_su,
                        adjusted,
                    )
                if not adjusted:
                    break
                control = self._apply_input_limits(control)
            return control

        def _adjust_for_bound(
            self,
            control: np.ndarray,
            prediction: np.ndarray,
            bound: float,
            matrix: np.ndarray,
            already_adjusted: bool,
        ) -> bool:
            adjusted = already_adjusted
            for idx, value in enumerate(prediction):
                if value > bound + 1e-6 or value < -bound - 1e-6:
                    limit = bound if value > 0 else -bound
                    delta = limit - value
                    row = matrix[idx]
                    denom = float(row @ row) + 1e-9
                    if denom <= 0.0:
                        continue
                    correction = (delta / denom) * row
                    control += correction
                    adjusted = True
            return adjusted

        def _build_plan(self, control: np.ndarray) -> Sequence[dict[str, float]]:
            states = (self._Sx @ self._xhat + self._Su @ control).reshape(
                self._horizon_steps, self._STATE_DIM
            )
            plan: list[dict[str, float]] = []
            for idx in range(self._horizon_steps):
                u_idx = control[min(idx, self._control_steps - 1)] if control.size else 0.0
                plan.append(
                    {
                        "theta": float(states[idx, 0]),
                        "omega": float(states[idx, 1]),
                        "u": float(u_idx),
                    }
                )
            return plan

        def _build_prediction_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
            nx = self._STATE_DIM
            steps = self._horizon_steps
            control_steps = self._control_steps
            A_step, B_step = self._discrete_matrices(self._step_dt)

            A_powers = [np.eye(nx, dtype=float)]
            for _ in range(steps):
                A_powers.append(A_step @ A_powers[-1])

            Sx = np.zeros((steps * nx, nx), dtype=float)
            Su = np.zeros((steps * nx, control_steps), dtype=float)

            for k in range(steps):
                Sx[k * nx : (k + 1) * nx, :] = A_powers[k + 1]
                for j in range(min(k + 1, control_steps)):
                    influence = A_powers[k - j] @ B_step
                    Su[k * nx : (k + 1) * nx, j] += influence[:, 0]
                if control_steps < k + 1:
                    # Additional contribution from held final control input
                    extra = np.zeros(nx, dtype=float)
                    for offset in range(control_steps, k + 1):
                        extra += (A_powers[k - offset] @ B_step)[:, 0]
                    Su[k * nx : (k + 1) * nx, control_steps - 1] += extra
            return Sx, Su

        def _build_output_selectors(self) -> Tuple[np.ndarray, np.ndarray]:
            nx = self._STATE_DIM
            steps = self._horizon_steps
            theta_selector = np.zeros((steps, steps * nx), dtype=float)
            omega_selector = np.zeros((steps, steps * nx), dtype=float)
            for idx in range(steps):
                theta_selector[idx, idx * nx] = 1.0
                omega_selector[idx, idx * nx + 1] = 1.0
            return theta_selector, omega_selector

        def _build_delta_matrix(self) -> np.ndarray:
            D = np.zeros((self._control_steps, self._control_steps), dtype=float)
            for idx in range(self._control_steps):
                D[idx, idx] = 1.0
                if idx > 0:
                    D[idx, idx - 1] = -1.0
            return D

        def _discrete_matrices(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
            A = np.array(
                [
                    [1.0, dt, 0.0],
                    [0.0, 1.0 - dt * self._a_f, dt],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            B = np.array([[0.0], [dt * self._a_u], [0.0]], dtype=float)
            return A, B

        def _clamp_input(self, value: float, reference: Optional[Tuple[float, float]]) -> float:
            if reference is None:
                lo, hi = -self._rate_limit, self._rate_limit
            else:
                lo, hi = reference
            if value < lo:
                return lo
            if value > hi:
                return hi
            return value

    def __init__(
        self,
        config: ControlConfig,
        pub: zmq.Socket,
        *,
        laser_mount: Optional[LaserMountConfig] = None,
        distance_alpha: Optional[float] = None,
        cli_json_logs: bool = False,
    ) -> None:
        if config.mpc is None:
            raise ValueError("MPCControlLoop requires ControlConfig.mpc to be configured")
        self._mpc_cfg = config.mpc
        self._mpc_last_plan: Optional[dict[str, Sequence[dict[str, float]]]] = None
        self._axis_controllers = {
            "yaw": self._AxisController("yaw", config=config, mpc_cfg=self._mpc_cfg),
            "pitch": self._AxisController("pitch", config=config, mpc_cfg=self._mpc_cfg),
        }
        self._last_distance: Optional[float] = None
        self._last_distance_time: Optional[float] = None
        self._last_radial_velocity = 0.0

        super().__init__(
            config,
            pub,
            laser_mount=laser_mount,
            distance_alpha=distance_alpha,
            cli_json_logs=cli_json_logs,
        )

    def _reset_controller_state(self) -> None:
        super()._reset_controller_state()
        for controller in self._axis_controllers.values():
            controller.reset()
        self._mpc_last_plan = None
        self._last_distance = None
        self._last_distance_time = None
        self._last_radial_velocity = 0.0

    def _controller_step(
        self,
        *,
        detection: _DetectionState,
        target_uv: Tuple[float, float],
        aim_uv: Tuple[float, float],
        raw_err_px: AxisPair,
        ctrl_err_px: AxisPair,
        err_rad: AxisPair,
        prev_err_rad: Optional[AxisPair],
        dt: float,
        now: float,
        tracking_active: bool,
    ) -> Tuple[AxisPair, Optional[dict]]:
        distance = self._resolve_distance(detection)
        radial_velocity = self._estimate_radial_velocity(detection, distance)

        motion_state = self._motion_state
        yaw_lateral = 0.0
        pitch_lateral = 0.0
        if motion_state is not None:
            yaw_lateral = distance * motion_state.yaw_rate
            pitch_lateral = distance * motion_state.pitch_rate

        yaw_rate, yaw_accel, yaw_plan = self._axis_controllers["yaw"].solve(
            err_rad.yaw,
            dt_actual=dt,
            distance=distance,
            lateral_velocity=yaw_lateral,
            radial_velocity=radial_velocity,
        )
        pitch_rate, pitch_accel, pitch_plan = self._axis_controllers["pitch"].solve(
            err_rad.pitch,
            dt_actual=dt,
            distance=distance,
            lateral_velocity=pitch_lateral,
            radial_velocity=radial_velocity,
        )

        self._prev_rate = AxisPair(yaw_rate, pitch_rate)
        self._prev_err = err_rad
        self._mpc_last_plan = {"yaw": yaw_plan, "pitch": pitch_plan}

        debug: dict[str, Any] = {}
        if yaw_plan and len(yaw_plan) > 1:
            debug["mpc_theta_next"] = [yaw_plan[1]["theta"], pitch_plan[1]["theta"] if pitch_plan else None]
        if yaw_accel is not None or pitch_accel is not None:
            debug["mpc_first_accel"] = [yaw_accel or 0.0, pitch_accel or 0.0]
        if not debug:
            debug = None

        return AxisPair(yaw_rate, pitch_rate), debug

    def _resolve_distance(self, detection: _DetectionState) -> float:
        distance = detection.target_distance_m
        if distance is None:
            distance = self._distance_ema
        if distance is None:
            distance = self._cfg.laser.default_distance_m
        return max(distance if distance is not None else 1.0, 1e-3)

    def _estimate_radial_velocity(self, detection: _DetectionState, distance: float) -> float:
        timestamp = detection.timestamp
        radial_velocity = self._last_radial_velocity
        if detection.target_distance_m is not None and timestamp is not None:
            if self._last_distance is not None and self._last_distance_time is not None:
                dt = timestamp - self._last_distance_time
                if dt >= 1e-3:
                    delta = detection.target_distance_m - self._last_distance
                    radial_velocity = delta / dt
            self._last_distance = detection.target_distance_m
            self._last_distance_time = timestamp
        self._last_radial_velocity = radial_velocity
        return radial_velocity

    def _homeward_rates(self, dt: float) -> Tuple[AxisPair, AxisPair]:
        cam_state = self._cam_state
        if cam_state is None:
            return AxisPair(0.0, 0.0), AxisPair(0.0, 0.0)

        yaw_err = 0.0
        pitch_err = 0.0
        if self._home_pan is not None:
            yaw_err = _wrap_angle(self._home_pan - cam_state.pan)
        if self._home_tilt is not None:
            pitch_err = self._home_tilt - cam_state.tilt

        distance = self._last_distance if self._last_distance is not None else self._cfg.laser.default_distance_m
        radial_velocity = self._last_radial_velocity

        yaw_rate, _, _ = self._axis_controllers["yaw"].solve(
            yaw_err,
            dt_actual=dt,
            distance=distance,
            lateral_velocity=0.0,
            radial_velocity=radial_velocity,
        )
        pitch_rate, _, _ = self._axis_controllers["pitch"].solve(
            pitch_err,
            dt_actual=dt,
            distance=distance,
            lateral_velocity=0.0,
            radial_velocity=radial_velocity,
        )

        return AxisPair(yaw_rate, pitch_rate), AxisPair(yaw_err, pitch_err)


# Backwards compatibility for existing imports relying on ControlLoop.
ControlLoop = PIDControlLoop

