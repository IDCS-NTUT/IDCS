"""Closed-loop controller that turns detections into pan/tilt commands."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import zmq

from common.control import (
    AxisPair,
    ControlConfig,
    LaserMountConfig,
    angular_error_from_pixel_delta,
    pixel_delta,
)
from common.geometry import (
    laser_ray_to_pixel,
    pixel_to_world_point,
    project_point_to_pixel,
    project_world_point_to_pixel,
    world_velocity_to_pixel_velocity,
)
from common.schemas import Box, CamState, ControlCmd, DetectionMsg
from common.tracker import PixelTracker, TrackingConfig, TrackerMeasurement, TrackerPrediction
from common.tracker_z import (
    TrackingZConfig,
    TrackingZMeasurement,
    TrackingZPrediction,
    ZTracker,
)
from common.tracker_world import (
    WorldTracker,
    WorldTrackerMeasurement,
    WorldTrackerPrediction,
)


_LOG = logging.getLogger("jetson.control")


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class _LatencyEstimator:
    def __init__(self, *, alpha: float = 0.2, initial_ms: Optional[float] = None) -> None:
        self._alpha = float(alpha)
        self._ema: Optional[float] = initial_ms

    def observe(self, sample_ms: float) -> None:
        if not math.isfinite(sample_ms):
            return
        sample_ms = max(0.0, float(sample_ms))
        if self._ema is None:
            self._ema = sample_ms
        else:
            self._ema = self._alpha * sample_ms + (1.0 - self._alpha) * self._ema

    def value(self) -> Optional[float]:
        return self._ema


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
    tracker_prediction: Optional[TrackerPrediction] = None
    range_prediction: Optional[TrackingZPrediction] = None
    world_prediction: Optional[WorldTrackerPrediction] = None


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
        tracking_cfg: Optional[TrackingConfig] = None,
        tracking_z_cfg: Optional[TrackingZConfig] = None,
        distance_alpha: Optional[float] = None,
        cli_json_logs: bool = False,
    ) -> None:
        self._cfg = config
        self._pub = pub
        self._laser_mount = laser_mount
        self._tracking_cfg = tracking_cfg
        self._tracking_z_cfg = tracking_z_cfg

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

        self._tracker: Optional[PixelTracker] = None
        self._tracker_time_s: Optional[float] = None
        self._tracker_prediction: Optional[TrackerPrediction] = None
        self._tracker_velocity_samples: list[Tuple[float, float]] = []
        self._tracker_warmup_required: int = 0
        self._tracker_warmup_std_thresh: float = 0.0
        self._tracker_warmup_ready: bool = True
        self._world_tracker: Optional[WorldTracker] = None
        self._world_tracker_time_s: Optional[float] = None
        self._world_prediction: Optional[WorldTrackerPrediction] = None
        self._latency_estimator: Optional[_LatencyEstimator] = None
        self._tracker_jump_limit_px = 0.25 * float(max(self._cfg.width, self._cfg.height))
        self._z_tracker: Optional[ZTracker] = None
        self._z_tracker_time_s: Optional[float] = None
        self._z_tracker_prediction: Optional[TrackingZPrediction] = None

        if tracking_cfg and tracking_cfg.enabled:
            if tracking_cfg.model == "cv":
                self._tracker = PixelTracker(
                    tracking_cfg,
                    fx_px=self._cfg.fx_px,
                    fy_px=self._cfg.fy_px,
                    cx_px=self._cfg.cx_px,
                    cy_px=self._cfg.cy_px,
                )
            elif tracking_cfg.model == "world_cv":
                self._world_tracker = WorldTracker(tracking_cfg)
            else:
                raise ValueError(f"unsupported tracking model: {tracking_cfg.model!r}")
            self._latency_estimator = _LatencyEstimator(initial_ms=tracking_cfg.predict_horizon_ms)
            self._tracker_warmup_required = max(1, int(tracking_cfg.warmup_measurements))
            self._tracker_warmup_std_thresh = max(0.0, float(tracking_cfg.warmup_velocity_std_px))
        else:
            self._tracker_warmup_required = 0
            self._tracker_warmup_std_thresh = 0.0
        self._reset_tracker_warmup()

        if tracking_z_cfg and tracking_z_cfg.enabled:
            self._z_tracker = ZTracker(tracking_z_cfg)

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
        prev_idx = self._latest_target_idx
        target_uv, target_box = self._select_target(msg)
        target_changed = prev_idx != self._latest_target_idx

        cam_rates = self._current_cam_rates()
        msg.cam_rates_radps = cam_rates.as_tuple()
        prediction_horizon_s = self._compute_prediction_horizon(now)

        msg.tracker_uv_pred = None
        msg.tracker_uv_vel = None
        msg.predict_horizon_ms = None
        if self._world_tracker is None:
            msg.tracker_world_pos_m = None
            msg.tracker_world_vel_mps = None
            msg.tracker_world_horizon_ms = None
        if self._z_tracker is None:
            msg.tracker_z_pred_m = None
            msg.tracker_z_vel_mps = None
            msg.tracker_z_source = None

        if self._latency_estimator is not None and msg.rx_ts_ms:
            sample_ms = max(0.0, now * 1000.0 - float(msg.rx_ts_ms))
            self._latency_estimator.observe(sample_ms)

        if target_changed:
            if self._tracking_cfg and self._tracking_cfg.reset_on_target_switch:
                self._reset_tracker_state()
            self._reset_z_tracker_state()

        if target_uv is None:
            self._distance_ema = None
            msg.target_distance_smoothed_m = None
            msg.laser_origin_px = None
            msg.laser_dot_px = None
            msg.laser_on_target = None
            msg.laser_range_m = None
            msg.laser_range_source = None
            msg.parallax_compensation_active = False
            msg.tracker_z_pred_m = None
            msg.tracker_z_vel_mps = None
            msg.tracker_z_source = None
            msg.tracker_world_pos_m = None
            msg.tracker_world_vel_mps = None
            msg.tracker_world_horizon_ms = None
            self._laser_overlay = None
            self._resolved_range = None
            self._z_tracker_prediction = None

        tracker_prediction: Optional[TrackerPrediction] = None
        world_prediction: Optional[WorldTrackerPrediction] = None
        z_prediction: Optional[TrackingZPrediction] = None

        if target_uv is not None:
            self._last_detection_ts = now
            if self._smoothed_uv is None or not self._tracking_active:
                self._smoothed_uv = target_uv
            else:
                self._smoothed_uv = self._smooth_uv(target_uv)

            range_measurements = (
                self._build_range_measurements(msg, target_box)
                if target_box is not None
                else {}
            )

            z_prediction = self._ingest_z_tracker_measurement(
                range_measurements,
                msg,
                now,
                prediction_horizon_s,
            )
            if z_prediction is not None:
                msg.tracker_z_pred_m = float(z_prediction.distance_m)
                msg.tracker_z_vel_mps = float(z_prediction.velocity_mps)
                msg.tracker_z_source = z_prediction.source
            else:
                msg.tracker_z_pred_m = None
                msg.tracker_z_vel_mps = None
                msg.tracker_z_source = None

            range_m, range_source, parallax_active = self._resolve_laser_range(
                msg.target_distance_smoothed_m,
                predicted=z_prediction,
                measurements=range_measurements,
            )
            msg.laser_range_m = range_m
            msg.laser_range_source = range_source

            if self._world_tracker is not None:
                world_prediction = self._ingest_world_tracker_measurement(
                    target_uv,
                    msg,
                    now,
                    range_m,
                    range_source,
                    prediction_horizon_s,
                )
                if world_prediction is not None:
                    msg.tracker_world_pos_m = tuple(float(v) for v in world_prediction.position_m)
                    msg.tracker_world_vel_mps = tuple(float(v) for v in world_prediction.velocity_mps)
                    msg.tracker_world_horizon_ms = world_prediction.horizon_s * 1000.0
                else:
                    msg.tracker_world_pos_m = None
                    msg.tracker_world_vel_mps = None
                    msg.tracker_world_horizon_ms = None

            if self._tracker is not None and target_box is not None:
                tracker_prediction = self._ingest_tracker_measurement(
                    target_uv,
                    target_box,
                    msg,
                    now,
                    cam_rates,
                    prediction_horizon_s,
                )
                if tracker_prediction is not None:
                    msg.tracker_uv_pred = (
                        float(tracker_prediction.uv[0]),
                        float(tracker_prediction.uv[1]),
                    )
                    msg.tracker_uv_vel = (
                        float(tracker_prediction.velocity[0]),
                        float(tracker_prediction.velocity[1]),
                    )
                    msg.predict_horizon_ms = tracker_prediction.horizon_s * 1000.0

            if self._world_tracker is not None and world_prediction is not None:
                world_px = self._project_world_prediction(world_prediction, target_uv)
                if world_px is not None:
                    if tracker_prediction is None:
                        tracker_prediction = world_px
                        msg.tracker_uv_pred = (
                            float(world_px.uv[0]),
                            float(world_px.uv[1]),
                        )
                        msg.tracker_uv_vel = (
                            float(world_px.velocity[0]),
                            float(world_px.velocity[1]),
                        )
                        msg.predict_horizon_ms = world_px.horizon_s * 1000.0

            self._resolved_range = range_m
            self._z_tracker_prediction = z_prediction
            self._world_prediction = world_prediction
            parallax_active = bool(parallax_active)
        else:
            msg.laser_range_m = None
            msg.laser_range_source = None
            self._resolved_range = None
            parallax_active = False
            self._world_prediction = None

        self._latest_detection = _DetectionState(
            frame_id=msg.frame_id,
            src_ts_ms=msg.src_ts_ms,
            timestamp=now,
            target_uv=target_uv,
            target_distance_m=msg.target_distance_smoothed_m,
            resolved_range_m=msg.laser_range_m,
            range_source=msg.laser_range_source,
            range_active=parallax_active,
            tracker_prediction=tracker_prediction,
            range_prediction=z_prediction,
            world_prediction=world_prediction,
        )
        self._last_frame_id = msg.frame_id
        self._last_src_ts_ms = msg.src_ts_ms

        range_m = self._latest_detection.resolved_range_m
        range_source = self._latest_detection.range_source

        if msg.predict_horizon_ms is None:
            msg.predict_horizon_ms = prediction_horizon_s * 1000.0

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
            self._refresh_tracker_prediction(detection, now)
            self._refresh_z_tracker_prediction(detection, now)
            tracking_ready = self._is_tracking_ready()
            cmd = self._build_tracking_cmd(detection, dt, now)
            self._tracking_active = tracking_ready
        else:
            if self._cfg.reinit_on_lost:
                self._prev_err = None
                self._integ = AxisPair(0.0, 0.0)
                self._smoothed_uv = None
                self._prev_rate = AxisPair(0.0, 0.0)
            cmd = self._build_hold_cmd(now)
            self._tracking_active = False

        self._last_cmd_time = now
        self._send_cmd(cmd)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _select_target(
        self, msg: DetectionMsg
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Box]]:
        boxes: Sequence[Box] = msg.boxes
        prev_idx = self._latest_target_idx
        self._latest_target_idx = None
        msg.target_idx = None
        msg.target_distance_smoothed_m = None

        if not boxes:
            self._distance_ema = None
            return None, None

        enumerated: Sequence[Tuple[int, Box]] = list(enumerate(boxes))
        if self._class_filter:
            enumerated = [pair for pair in enumerated if pair[1].cls == self._class_filter]
            if not enumerated:
                self._distance_ema = None
                return None, None

        if self._selector_strategy == "largest_area":
            best_idx, best = max(enumerated, key=lambda item: item[1].w * item[1].h)
        else:
            best_idx, best = max(enumerated, key=lambda item: item[1].conf)

        self._latest_target_idx = best_idx
        msg.target_idx = best_idx
        self._update_target_distance(msg, best, previous_idx=prev_idx)

        u = (best.x + (best.w / 2.0)) * msg.img_w
        v = (best.y + (best.h / 2.0)) * msg.img_h
        return (u, v), best

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
        self,
        smoothed_distance: Optional[float],
        *,
        predicted: Optional[TrackingZPrediction] = None,
        measurements: Optional[Mapping[str, TrackingZMeasurement]] = None,
    ) -> Tuple[Optional[float], Optional[str], bool]:
        if self._cfg.aim_mode != "laser_point" or self._laser_mount is None:
            self._resolved_range = None
            return None, None, False

        measurement = self._select_z_measurement(measurements or {})
        measurement_value: Optional[float] = None
        measurement_source: Optional[str] = None
        if measurement is not None:
            value = float(measurement.value_m)
            if math.isfinite(value) and value > 0.0:
                measurement_value = value
                measurement_source = measurement.source

        smoothed_value: Optional[float] = None
        if smoothed_distance is not None and math.isfinite(smoothed_distance):
            if smoothed_distance > 0.0:
                smoothed_value = float(smoothed_distance)

        predicted_value: Optional[float] = None
        predicted_source: Optional[str] = None
        if predicted is not None:
            value = float(predicted.distance_m)
            if math.isfinite(value) and value > 0.0:
                predicted_value = value
                predicted_source = predicted.source

        policy = self._cfg.laser.use_range
        default_distance = float(self._cfg.laser.default_distance_m)
        resolved: Optional[float]
        source: Optional[str]

        if policy in {"known_size", "auto"}:
            resolved = None
            source = None
            if predicted_value is not None:
                if policy == "auto" or (predicted_source and predicted_source == "known_size"):
                    resolved = predicted_value
                    source = f"tracker_z:{predicted_source or 'unknown'}"
            if resolved is None and measurement_value is not None:
                resolved = measurement_value
                source = measurement_source or "known_size"
            if resolved is None and smoothed_value is not None:
                resolved = smoothed_value
                source = measurement_source or "known_size"
            if resolved is None:
                resolved = self._blend_range_towards(default_distance)
                source = "default"
        elif policy == "ground_plane":
            if not self._warned_ground_plane:
                _LOG.warning(
                    "control.laser.use_range=ground_plane is not implemented; "
                    "falling back to known_size/default distances",
                )
                self._warned_ground_plane = True
            resolved = None
            source = None
            if predicted_value is not None and predicted_source == "ground_plane":
                resolved = predicted_value
                source = "tracker_z:ground_plane"
            if resolved is None and measurement_value is not None:
                resolved = measurement_value
                source = measurement_source or "known_size"
            if resolved is None and smoothed_value is not None:
                resolved = smoothed_value
                source = measurement_source or "known_size"
            if resolved is None:
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

    def _reset_tracker_state(self) -> None:
        if self._tracker is not None:
            self._tracker.reset()
        self._tracker_time_s = None
        self._tracker_prediction = None
        self._reset_tracker_warmup()
        if self._world_tracker is not None:
            self._world_tracker.reset()
        self._world_tracker_time_s = None
        self._world_prediction = None

    def _reset_z_tracker_state(self) -> None:
        if self._z_tracker is not None:
            self._z_tracker.reset()
        self._z_tracker_time_s = None
        self._z_tracker_prediction = None

    def _reset_tracker_warmup(self) -> None:
        self._tracker_velocity_samples = []
        if self._tracker is None or self._world_tracker is not None:
            self._tracker_warmup_ready = True
            return
        if self._tracker_warmup_required <= 1 or self._tracker_warmup_std_thresh <= 0.0:
            self._tracker_warmup_ready = True
        else:
            self._tracker_warmup_ready = False

    def _observe_tracker_velocity(self, velocity: Tuple[float, float]) -> None:
        if self._tracker is None or self._world_tracker is not None:
            return
        if self._tracker_warmup_ready:
            return
        required = max(1, self._tracker_warmup_required)
        self._tracker_velocity_samples.append((float(velocity[0]), float(velocity[1])))
        if len(self._tracker_velocity_samples) > required:
            self._tracker_velocity_samples.pop(0)
        if len(self._tracker_velocity_samples) < required:
            return
        mean_u = sum(sample[0] for sample in self._tracker_velocity_samples) / len(
            self._tracker_velocity_samples
        )
        mean_v = sum(sample[1] for sample in self._tracker_velocity_samples) / len(
            self._tracker_velocity_samples
        )
        variance = 0.0
        for sample_u, sample_v in self._tracker_velocity_samples:
            du = sample_u - mean_u
            dv = sample_v - mean_v
            variance += du * du + dv * dv
        variance /= len(self._tracker_velocity_samples)
        std_px = math.sqrt(max(0.0, variance))
        if std_px <= self._tracker_warmup_std_thresh:
            self._tracker_warmup_ready = True

    def _is_tracking_ready(self) -> bool:
        if self._world_tracker is not None:
            return True
        if self._tracker is None:
            return True
        return self._tracker_warmup_ready

    def _current_cam_rates(self) -> AxisPair:
        state = self._cam_state
        yaw = state.pan_rate if state and state.pan_rate is not None else None
        pitch = state.tilt_rate if state and state.tilt_rate is not None else None
        if yaw is None:
            yaw = self._prev_rate.yaw
        if pitch is None:
            pitch = self._prev_rate.pitch
        return AxisPair(float(yaw), float(pitch))

    def _compute_prediction_horizon(self, now: float) -> float:
        cfg = self._tracking_cfg
        if cfg is None:
            return 0.0
        max_ms = float(cfg.predict_horizon_ms)
        estimate = self._latency_estimator.value() if self._latency_estimator else None
        if estimate is None:
            horizon_ms = max_ms
        else:
            horizon_ms = estimate if max_ms <= 0.0 else min(estimate, max_ms)
        return max(0.0, horizon_ms) / 1000.0

    def _build_range_measurements(
        self,
        msg: DetectionMsg,
        box: Optional[Box],
    ) -> Mapping[str, TrackingZMeasurement]:
        measurements: dict[str, TrackingZMeasurement] = {}
        if box is None:
            return measurements
        if box.distance_m is not None and math.isfinite(box.distance_m) and box.distance_m > 0.0:
            measurements["known_size"] = TrackingZMeasurement(
                value_m=float(box.distance_m),
                source="known_size",
                box_size_px=(box.w * msg.img_w, box.h * msg.img_h),
                confidence=float(box.conf),
            )
        return measurements

    def _select_z_measurement(
        self, measurements: Mapping[str, TrackingZMeasurement]
    ) -> Optional[TrackingZMeasurement]:
        if not measurements:
            return None
        cfg = self._tracking_z_cfg
        if cfg is not None:
            for key in cfg.meas_src_priority:
                if key in measurements:
                    return measurements[key]
        for measurement in measurements.values():
            return measurement
        return None

    def _ingest_z_tracker_measurement(
        self,
        measurements: Mapping[str, TrackingZMeasurement],
        msg: DetectionMsg,
        now: float,
        prediction_horizon_s: float,
    ) -> Optional[TrackingZPrediction]:
        tracker = self._z_tracker
        if tracker is None:
            return None

        measurement = self._select_z_measurement(measurements)
        measurement_time_s = float(msg.rx_ts_ms) / 1000.0 if msg.rx_ts_ms else now
        if self._z_tracker_time_s is None:
            dt = 0.0
        else:
            dt = max(0.0, measurement_time_s - self._z_tracker_time_s)
        if dt > 0.0:
            tracker.predict(dt)
        self._z_tracker_time_s = measurement_time_s

        if measurement is not None:
            tracker.update(measurement)

        prediction = tracker.project(max(0.0, float(prediction_horizon_s)))
        if prediction is not None:
            prediction = self._clamp_z_prediction(measurement, prediction)
        self._z_tracker_prediction = prediction
        return prediction

    def _ingest_tracker_measurement(
        self,
        uv: Tuple[float, float],
        box: Box,
        msg: DetectionMsg,
        now: float,
        cam_rates: AxisPair,
        prediction_horizon_s: float,
    ) -> Optional[TrackerPrediction]:
        tracker = self._tracker
        if tracker is None:
            return None

        measurement_time_s = (
            float(msg.rx_ts_ms) / 1000.0 if msg.rx_ts_ms else now
        )
        if self._tracker_time_s is None:
            dt = 0.0
        else:
            dt = max(0.0, measurement_time_s - self._tracker_time_s)
        if dt > 0.0:
            tracker.predict(dt, cam_rates)
        self._tracker_time_s = measurement_time_s

        measurement = TrackerMeasurement(
            uv=(float(uv[0]), float(uv[1])),
            box_size_px=(box.w * msg.img_w, box.h * msg.img_h),
            confidence=float(box.conf),
        )
        accepted = tracker.update(measurement, cam_rates=cam_rates)

        horizon_s = max(0.0, float(prediction_horizon_s))
        prediction = tracker.project(horizon_s, cam_rates)
        if prediction is not None and uv is not None:
            prediction = self._clamp_tracker_prediction(uv, prediction)
        if accepted and prediction is not None:
            self._observe_tracker_velocity(prediction.velocity)
        self._tracker_prediction = prediction
        if not self._is_tracking_ready():
            return None
        return prediction

    def _ingest_world_tracker_measurement(
        self,
        uv: Tuple[float, float],
        msg: DetectionMsg,
        now: float,
        range_m: Optional[float],
        range_source: Optional[str],
        prediction_horizon_s: float,
    ) -> Optional[WorldTrackerPrediction]:
        tracker = self._world_tracker
        if tracker is None:
            return None

        measurement_time_s = float(msg.rx_ts_ms) / 1000.0 if msg.rx_ts_ms else now
        if self._world_tracker_time_s is None:
            dt = 0.0
        else:
            dt = max(0.0, measurement_time_s - self._world_tracker_time_s)
        if dt > 0.0:
            tracker.predict(dt)
        self._world_tracker_time_s = measurement_time_s

        if range_m is not None and range_m > 0.0:
            pan, tilt = self._current_camera_pose()
            try:
                position_world = pixel_to_world_point(
                    uv[0],
                    uv[1],
                    distance_m=float(range_m),
                    fx_px=self._cfg.fx_px,
                    fy_px=self._cfg.fy_px,
                    cx_px=self._cfg.cx_px,
                    cy_px=self._cfg.cy_px,
                    pan_rad=pan,
                    tilt_rad=tilt,
                )
            except ValueError:
                position_world = None
            if position_world is not None:
                meas_std = self._world_measurement_std(range_m, range_source)
                measurement = WorldTrackerMeasurement(
                    position_m=position_world,
                    position_std_m=meas_std,
                )
                tracker.update(measurement)

        prediction = tracker.project(max(0.0, float(prediction_horizon_s)))
        self._world_prediction = prediction
        return prediction

    def _refresh_tracker_prediction(
        self, detection: _DetectionState, now: float
    ) -> Optional[TrackerPrediction]:
        tracker = self._tracker
        if tracker is not None:
            cam_rates = self._current_cam_rates()
            if self._tracker_time_s is None:
                self._tracker_time_s = now
            else:
                dt = max(0.0, now - self._tracker_time_s)
                if dt > 0.0:
                    tracker.predict(dt, cam_rates)
                    self._tracker_time_s = now

            prediction = tracker.project(self._compute_prediction_horizon(now), cam_rates)
            if prediction is not None and detection.target_uv is not None:
                prediction = self._clamp_tracker_prediction(detection.target_uv, prediction)
            self._tracker_prediction = prediction
            if not self._is_tracking_ready():
                detection.tracker_prediction = None
                return None
            detection.tracker_prediction = prediction
            return prediction

        world_tracker = self._world_tracker
        if world_tracker is None:
            self._tracker_prediction = None
            detection.tracker_prediction = None
            detection.world_prediction = None
            return None

        if self._world_tracker_time_s is None:
            self._world_tracker_time_s = now
        else:
            dt = max(0.0, now - self._world_tracker_time_s)
            if dt > 0.0:
                world_tracker.predict(dt)
                self._world_tracker_time_s = now

        world_prediction = world_tracker.project(self._compute_prediction_horizon(now))
        detection.world_prediction = world_prediction
        self._world_prediction = world_prediction
        if world_prediction is None:
            self._tracker_prediction = None
            detection.tracker_prediction = None
            return None

        projected = self._project_world_prediction(world_prediction, detection.target_uv)
        self._tracker_prediction = projected
        detection.tracker_prediction = projected
        return projected

    def _current_camera_pose(self) -> Tuple[float, float]:
        state = self._cam_state
        if state is None:
            return (0.0, 0.0)
        return (float(state.pan), float(state.tilt))

    def _future_camera_pose(self, horizon_s: float) -> Tuple[float, float]:
        pan, tilt = self._current_camera_pose()
        if horizon_s <= 0.0:
            return (pan, tilt)
        rates = self._current_cam_rates()
        return (pan + rates.yaw * horizon_s, tilt + rates.pitch * horizon_s)

    def _world_measurement_std(
        self, range_m: float, range_source: Optional[str]
    ) -> Optional[float]:
        cfg = self._tracking_cfg
        if cfg is None or cfg.world is None:
            return None
        base = max(cfg.world.meas_noise_pos_m, abs(range_m) * 0.05)
        if range_source is None:
            scale = 3.0
        else:
            src = range_source.lower()
            if src.startswith("tracker_z"):
                scale = 2.0
            elif src == "default":
                scale = 4.0
            else:
                scale = 1.0
        return base * scale

    def _project_world_prediction(
        self,
        prediction: WorldTrackerPrediction,
        measurement_uv: Optional[Tuple[float, float]],
    ) -> Optional[TrackerPrediction]:
        horizon = max(0.0, prediction.horizon_s)
        pan, tilt = self._future_camera_pose(horizon)
        try:
            uv = project_world_point_to_pixel(
                prediction.position_m,
                fx_px=self._cfg.fx_px,
                fy_px=self._cfg.fy_px,
                cx_px=self._cfg.cx_px,
                cy_px=self._cfg.cy_px,
                pan_rad=pan,
                tilt_rad=tilt,
            )
        except ValueError:
            return None

        try:
            vel_px = world_velocity_to_pixel_velocity(
                prediction.position_m,
                prediction.velocity_mps,
                fx_px=self._cfg.fx_px,
                fy_px=self._cfg.fy_px,
                cx_px=self._cfg.cx_px,
                cy_px=self._cfg.cy_px,
                pan_rad=pan,
                tilt_rad=tilt,
            )
        except ValueError:
            vel_px = (0.0, 0.0)

        projected = TrackerPrediction(uv=uv, velocity=vel_px, horizon_s=horizon)
        if measurement_uv is not None:
            projected = self._clamp_tracker_prediction(measurement_uv, projected)
        return projected

    def _refresh_z_tracker_prediction(
        self, detection: _DetectionState, now: float
    ) -> Optional[TrackingZPrediction]:
        tracker = self._z_tracker
        if tracker is None:
            self._z_tracker_prediction = None
            detection.range_prediction = None
            return None

        if self._z_tracker_time_s is None:
            self._z_tracker_time_s = now
        else:
            dt = max(0.0, now - self._z_tracker_time_s)
            if dt > 0.0:
                tracker.predict(dt)
                self._z_tracker_time_s = now

        prediction = tracker.project(self._compute_prediction_horizon(now))
        if prediction is not None:
            prediction = self._clamp_z_prediction(None, prediction)
        self._z_tracker_prediction = prediction
        detection.range_prediction = prediction
        if (
            prediction is not None
            and detection.range_source is not None
            and detection.range_source.startswith("tracker_z")
        ):
            detection.resolved_range_m = prediction.distance_m
        return prediction

    def _clamp_tracker_prediction(
        self, measurement_uv: Tuple[float, float], prediction: TrackerPrediction
    ) -> TrackerPrediction:
        du = prediction.uv[0] - measurement_uv[0]
        dv = prediction.uv[1] - measurement_uv[1]
        dist = math.hypot(du, dv)
        if dist <= self._tracker_jump_limit_px or dist == 0.0:
            return prediction
        scale = self._tracker_jump_limit_px / dist
        clamped_uv = (
            measurement_uv[0] + du * scale,
            measurement_uv[1] + dv * scale,
        )
        return TrackerPrediction(uv=clamped_uv, velocity=prediction.velocity, horizon_s=prediction.horizon_s)

    def _clamp_z_prediction(
        self,
        _: Optional[TrackingZMeasurement],
        prediction: TrackingZPrediction,
    ) -> TrackingZPrediction:
        distance = prediction.distance_m
        if not math.isfinite(distance) or distance <= 0.0:
            distance = 0.1
        else:
            distance = max(0.1, distance)
        return TrackingZPrediction(
            distance_m=distance,
            velocity_mps=prediction.velocity_mps,
            source=prediction.source,
            horizon_s=prediction.horizon_s,
        )

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
        predicted = detection.tracker_prediction or self._tracker_prediction
        if not self._is_tracking_ready():
            predicted = None
        if predicted is not None:
            target_uv = predicted.uv
        else:
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

