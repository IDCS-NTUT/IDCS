"""Closed-loop controller that turns detections into pan/tilt commands."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import zmq

from common.control import (
    AxisPair,
    ControlConfig,
    LaserMountConfig,
    angular_error_from_pixel_delta,
    pixel_delta,
)
from common.geometry import (
    camera_cv_to_world_matrix,
    laser_ray_to_pixel,
    matrix_vector_mul,
    pixel_to_camera_ray,
    project_point_to_pixel,
)
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


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    if not math.isfinite(result):
        return None
    return result


def _safe_degrees(value: Optional[float]) -> Optional[float]:
    result = _safe_float(value)
    if result is None:
        return None
    return math.degrees(result)


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
    prediction: Optional[_MotionModelPrediction] = None
    predicted_uv: Optional[Tuple[float, float]] = None


@dataclass
class _LaserOverlay:
    origin_px: Optional[Tuple[float, float]]
    dot_px: Optional[Tuple[float, float]]
    on_target: Optional[bool]
    active: bool
    range_m: Optional[float]
    range_source: Optional[str]


@dataclass
class _TargetEstimate:
    """Internal representation of the current tracked target state."""

    uv: Tuple[float, float]
    velocity_px_s: Tuple[float, float]
    timestamp: float


@dataclass
class _MotionModelPrediction:
    """Predicted future target state in pixel space."""

    uv: Tuple[float, float]
    horizon_s: float
    state_timestamp: float
    age_s: float
    camera_shift_px: Tuple[float, float]
    velocity_px_s: Tuple[float, float]


@dataclass
class _WorldFrameTargetEstimate:
    """World-frame representation of the tracked target."""

    uv: Tuple[float, float]
    position_world_m: Optional[Tuple[float, float, float]]
    velocity_world_m_s: Optional[Tuple[float, float, float]]
    timestamp: float
    distance_m: Optional[float]


@dataclass
class _CameraPose:
    """Snapshot of the camera pose used for world-frame transforms."""

    yaw: float
    pitch: float
    rotation_cam_to_world: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]
    origin_world_m: Tuple[float, float, float]
    timestamp: float


class _NullMotionModel:
    """Fallback implementation when motion prediction is disabled."""

    def __init__(self) -> None:
        self._state: Optional[_TargetEstimate] = None

    def reset(self) -> None:
        self._state = None

    def update_cam_state(self, cam_state: CamState, timestamp: float) -> None:  # pragma: no cover - trivial
        return

    def update_target(
        self, uv: Tuple[float, float], timestamp: float, distance_m: Optional[float] = None
    ) -> _TargetEstimate:
        self._state = _TargetEstimate(uv=uv, velocity_px_s=(0.0, 0.0), timestamp=timestamp)
        return self._state

    def predict(self, now: float, horizon_s: float) -> Optional[_MotionModelPrediction]:
        state = self._state
        if state is None:
            return None
        horizon = max(horizon_s, 0.0)
        age_s = max(now - state.timestamp, 0.0)
        return _MotionModelPrediction(
            uv=state.uv,
            horizon_s=horizon,
            state_timestamp=state.timestamp,
            age_s=age_s,
            camera_shift_px=(0.0, 0.0),
            velocity_px_s=state.velocity_px_s,
        )

    def diagnostics(self, now: float, horizon_s: float) -> Dict[str, Any]:
        state = self._state
        if state is None:
            return {
                "has_state": False,
                "state_uv": None,
                "velocity_px_s": None,
                "state_timestamp": None,
                "state_age_s": None,
                "residual_px": None,
                "camera_shift_px": None,
                "last_update_ts": None,
            }

        age_s = max(now - state.timestamp, 0.0)
        return {
            "has_state": True,
            "state_uv": state.uv,
            "velocity_px_s": state.velocity_px_s,
            "state_timestamp": state.timestamp,
            "state_age_s": age_s,
            "residual_px": None,
            "camera_shift_px": None,
            "last_update_ts": state.timestamp,
        }


class _CameraFrameMotionModel:
    """Track target motion directly in pixel space with camera de-rotation."""

    _MIN_DT = 1e-3

    def __init__(self, config: ControlConfig) -> None:
        self._cfg = config
        self._state: Optional[_TargetEstimate] = None
        self._last_cam_state: Optional[CamState] = None
        self._last_residual: Optional[Tuple[float, float]] = None
        self._last_cam_shift: Optional[Tuple[float, float]] = None
        self._last_update_timestamp: Optional[float] = None

    def reset(self) -> None:
        self._state = None
        self._last_residual = None
        self._last_cam_shift = None
        self._last_update_timestamp = None

    def update_cam_state(self, cam_state: CamState, timestamp: float) -> None:
        self._last_cam_state = cam_state

    def update_target(
        self, uv: Tuple[float, float], timestamp: float, distance_m: Optional[float] = None
    ) -> _TargetEstimate:
        if self._state is None:
            self._state = _TargetEstimate(uv=uv, velocity_px_s=(0.0, 0.0), timestamp=timestamp)
            self._last_residual = (0.0, 0.0)
            self._last_cam_shift = (0.0, 0.0)
            self._last_update_timestamp = timestamp
            return self._state

        prev = self._state
        dt = max(timestamp - prev.timestamp, self._MIN_DT)
        cam_shift_u, cam_shift_v = self._camera_shift(dt)

        residual_u = uv[0] - prev.uv[0] - cam_shift_u
        residual_v = uv[1] - prev.uv[1] - cam_shift_v

        velocity = (
            residual_u / dt,
            residual_v / dt,
        )

        self._state = _TargetEstimate(uv=uv, velocity_px_s=velocity, timestamp=timestamp)
        self._last_residual = (float(residual_u), float(residual_v))
        self._last_cam_shift = (float(cam_shift_u), float(cam_shift_v))
        self._last_update_timestamp = timestamp
        return self._state

    def predict(self, now: float, horizon_s: float) -> Optional[_MotionModelPrediction]:
        state = self._state
        if state is None:
            return None

        if horizon_s < 0.0:
            horizon_s = 0.0

        target_dt = max((now + horizon_s) - state.timestamp, 0.0)
        cam_shift = self._camera_shift(target_dt)
        pred_u = state.uv[0] + state.velocity_px_s[0] * target_dt + cam_shift[0]
        pred_v = state.uv[1] + state.velocity_px_s[1] * target_dt + cam_shift[1]

        age_s = max(now - state.timestamp, 0.0)
        return _MotionModelPrediction(
            uv=(float(pred_u), float(pred_v)),
            horizon_s=horizon_s,
            state_timestamp=state.timestamp,
            age_s=age_s,
            camera_shift_px=(float(cam_shift[0]), float(cam_shift[1])),
            velocity_px_s=(float(state.velocity_px_s[0]), float(state.velocity_px_s[1])),
        )

    def _camera_shift(self, dt: float) -> Tuple[float, float]:
        cfg = self._cfg.motion_model.derotation
        if not cfg.enabled:
            return (0.0, 0.0)

        cam_state = self._last_cam_state
        if cam_state is None:
            return (0.0, 0.0)

        yaw_rate = cam_state.pan_rate
        pitch_rate = cam_state.tilt_rate
        if yaw_rate in (None, 0.0) and pitch_rate in (None, 0.0):
            return (0.0, 0.0)

        scale = cfg.rate_scale
        yaw_delta = 0.0 if yaw_rate is None else float(yaw_rate) * dt * scale
        pitch_delta = 0.0 if pitch_rate is None else float(pitch_rate) * dt * scale

        yaw_sign = self._cfg.yaw_sign if self._cfg.yaw_sign != 0 else 1.0
        pitch_sign = self._cfg.pitch_sign if self._cfg.pitch_sign != 0 else 1.0

        shift_u = -self._cfg.fx_px * yaw_delta / yaw_sign if yaw_delta != 0.0 else 0.0
        shift_v = -self._cfg.fy_px * pitch_delta / pitch_sign if pitch_delta != 0.0 else 0.0

        return (shift_u, shift_v)

    def diagnostics(self, now: float, horizon_s: float) -> Dict[str, Any]:
        state = self._state
        if state is None:
            return {
                "has_state": False,
                "state_uv": None,
                "velocity_px_s": None,
                "state_timestamp": None,
                "state_age_s": None,
                "residual_px": None,
                "camera_shift_px": self._last_cam_shift,
                "last_update_ts": self._last_update_timestamp,
            }

        age_s = max(now - state.timestamp, 0.0)
        return {
            "has_state": True,
            "state_uv": (float(state.uv[0]), float(state.uv[1])),
            "velocity_px_s": (
                float(state.velocity_px_s[0]),
                float(state.velocity_px_s[1]),
            ),
            "state_timestamp": state.timestamp,
            "state_age_s": age_s,
            "residual_px": self._last_residual,
            "camera_shift_px": self._last_cam_shift,
            "last_update_ts": self._last_update_timestamp,
        }


class _WorldFrameMotionModel:
    """Maintain target state in a world coordinate frame (scaffolding)."""

    _MIN_DT = 1e-3

    def __init__(self, config: ControlConfig) -> None:
        self._cfg = config
        self._state: Optional[_WorldFrameTargetEstimate] = None
        self._last_pose: Optional[_CameraPose] = None
        self._last_update_timestamp: Optional[float] = None

    def reset(self) -> None:
        self._state = None
        self._last_update_timestamp = None

    def update_cam_state(self, cam_state: CamState, timestamp: float) -> None:
        yaw = float(cam_state.pan)
        pitch = float(cam_state.tilt)
        rotation = camera_cv_to_world_matrix(yaw, pitch)
        origin = (
            0.0,
            float(self._cfg.motion_model.world_frame.camera_height_m),
            0.0,
        )
        self._last_pose = _CameraPose(
            yaw=yaw,
            pitch=pitch,
            rotation_cam_to_world=rotation,
            origin_world_m=origin,
            timestamp=timestamp,
        )

    def update_target(
        self,
        uv: Tuple[float, float],
        timestamp: float,
        distance_m: Optional[float] = None,
    ) -> _TargetEstimate:
        pose = self._last_pose
        world_position: Optional[Tuple[float, float, float]] = None
        ray_world: Optional[Tuple[float, float, float]] = None

        if pose is not None:
            ray_cam = pixel_to_camera_ray(
                uv[0],
                uv[1],
                fx_px=self._cfg.fx_px,
                fy_px=self._cfg.fy_px,
                cx_px=self._cfg.cx_px,
                cy_px=self._cfg.cy_px,
            )
            ray_world = matrix_vector_mul(pose.rotation_cam_to_world, ray_cam)
            if distance_m is not None and distance_m > 0.0:
                world_position = (
                    pose.origin_world_m[0] + ray_world[0] * distance_m,
                    pose.origin_world_m[1] + ray_world[1] * distance_m,
                    pose.origin_world_m[2] + ray_world[2] * distance_m,
                )
            else:
                default_height = self._cfg.motion_model.world_frame.default_target_height_m
                if default_height is not None and abs(ray_world[1]) > 1e-6:
                    t = (default_height - pose.origin_world_m[1]) / ray_world[1]
                    if t > 0.0:
                        distance_m = t
                        world_position = (
                            pose.origin_world_m[0] + ray_world[0] * t,
                            default_height,
                            pose.origin_world_m[2] + ray_world[2] * t,
                        )

        velocity_world: Optional[Tuple[float, float, float]] = None
        if (
            self._state is not None
            and world_position is not None
            and self._state.position_world_m is not None
        ):
            dt = max(timestamp - self._state.timestamp, self._MIN_DT)
            velocity_world = (
                (world_position[0] - self._state.position_world_m[0]) / dt,
                (world_position[1] - self._state.position_world_m[1]) / dt,
                (world_position[2] - self._state.position_world_m[2]) / dt,
            )

        self._state = _WorldFrameTargetEstimate(
            uv=uv,
            position_world_m=world_position,
            velocity_world_m_s=velocity_world,
            timestamp=timestamp,
            distance_m=distance_m,
        )
        self._last_update_timestamp = timestamp

        return _TargetEstimate(uv=uv, velocity_px_s=(0.0, 0.0), timestamp=timestamp)

    def predict(self, now: float, horizon_s: float) -> Optional[_MotionModelPrediction]:
        # Placeholder: world-frame prediction will be implemented in a later step.
        return None

    def diagnostics(self, now: float, horizon_s: float) -> Dict[str, Any]:
        state = self._state
        pose = self._last_pose
        if state is None:
            return {
                "has_state": False,
                "state_uv": None,
                "velocity_px_s": None,
                "state_timestamp": None,
                "state_age_s": None,
                "residual_px": None,
                "camera_shift_px": None,
                "last_update_ts": self._last_update_timestamp,
                "world_position_m": None,
                "world_velocity_m_s": None,
                "pose_timestamp": pose.timestamp if pose is not None else None,
            }

        age_s = max(now - state.timestamp, 0.0)
        return {
            "has_state": True,
            "state_uv": state.uv,
            "velocity_px_s": None,
            "state_timestamp": state.timestamp,
            "state_age_s": age_s,
            "residual_px": None,
            "camera_shift_px": None,
            "last_update_ts": self._last_update_timestamp,
            "world_position_m": state.position_world_m,
            "world_velocity_m_s": state.velocity_world_m_s,
            "pose_timestamp": pose.timestamp if pose is not None else None,
        }


class _MotionModelService:
    """Facade selecting the configured motion-model implementation."""

    def __init__(self, config: ControlConfig) -> None:
        self._cfg = config
        self._latency_ms: Optional[float] = None

        mode = config.motion_model.mode
        if mode == "camera_frame":
            self._impl: Any = _CameraFrameMotionModel(config)
            self._mode = "camera_frame"
        elif mode == "world_frame":
            self._impl = _WorldFrameMotionModel(config)
            self._mode = "world_frame"
        else:
            _LOG.warning(
                "motion_model.mode=%s is not implemented yet; falling back to raw pixels",
                mode,
            )
            self._impl = _NullMotionModel()
            self._mode = "raw_pixels"

    def reset(self) -> None:
        self._impl.reset()

    def update_cam_state(self, cam_state: CamState, timestamp: float) -> None:
        self._impl.update_cam_state(cam_state, timestamp)

    def update_target(
        self,
        uv: Tuple[float, float],
        timestamp: float,
        *,
        distance_m: Optional[float] = None,
    ) -> _TargetEstimate:
        return self._impl.update_target(uv, timestamp, distance_m)

    def update_latency(self, latency_ms: Optional[float]) -> None:
        self._latency_ms = latency_ms

    def predict(self, now: float) -> Optional[_MotionModelPrediction]:
        horizon_s = self._resolve_horizon_s()
        return self._impl.predict(now, horizon_s)

    @property
    def mode(self) -> str:
        return self._mode

    def diagnostics(self, now: float) -> Dict[str, Any]:
        horizon_s = self._resolve_horizon_s()
        impl_diag: Dict[str, Any]
        try:
            impl_diag = self._impl.diagnostics(now, horizon_s)
        except AttributeError:  # pragma: no cover - defensive
            impl_diag = {
                "has_state": False,
                "state_uv": None,
                "velocity_px_s": None,
                "state_timestamp": None,
                "state_age_s": None,
                "residual_px": None,
                "camera_shift_px": None,
                "last_update_ts": None,
            }

        diag: Dict[str, Any] = {
            "mode": self._mode,
            "horizon_s": horizon_s,
            "latency_ms": self._latency_ms,
        }
        diag.update(impl_diag)
        return diag

    def _resolve_horizon_s(self) -> float:
        motion_cfg = self._cfg.motion_model

        cfg_horizon_ms = motion_cfg.prediction_horizon_ms
        if cfg_horizon_ms is not None:
            horizon_ms = max(cfg_horizon_ms, 0.0)
        else:
            latency_ms = self._latency_ms
            if latency_ms is not None and latency_ms > 0.0:
                horizon_ms = latency_ms
            else:
                loop_dt = self._cfg.loop_dt
                if loop_dt is not None and loop_dt > 0.0:
                    horizon_ms = loop_dt * 1000.0
                else:
                    horizon_ms = 0.0

        horizon_ms = self._clamp_horizon_ms(horizon_ms)
        return horizon_ms / 1000.0

    def _clamp_horizon_ms(self, horizon_ms: float) -> float:
        if not math.isfinite(horizon_ms):
            horizon_ms = 0.0

        horizon_ms = max(horizon_ms, 0.0)

        motion_cfg = self._cfg.motion_model
        max_ms = motion_cfg.max_prediction_horizon_ms
        min_ms = motion_cfg.min_prediction_horizon_ms

        if max_ms is not None:
            horizon_ms = min(horizon_ms, max_ms)
        if min_ms is not None:
            horizon_ms = max(horizon_ms, min_ms)

        return horizon_ms


class ControlLoop:
    """Runs a rate-mode PID loop for yaw/pitch using the latest detection."""

    _MIN_DT = 1e-3
    _MAX_DT = 0.2
    _ARM_REQUIRED_MATCHES = 3
    _ARM_IOU_THRESHOLD = 0.5
    _GAIN_RAMP_DURATION_S = 0.3
    _DERIVATIVE_FREEZE_TICKS = 3
    _DERIVATIVE_MIN_DT = 5e-3

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
        self._armed = False
        self._arm_timestamp: Optional[float] = None
        self._arm_consecutive = 0
        self._arm_candidate_idx: Optional[int] = None
        self._arm_candidate_box: Optional[Tuple[float, float, float, float]] = None
        self._post_arm_ticks = 0

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

        self._latency_source: str = config.motion_model.latency_ms_source or "auto"
        self._latency_ms: Optional[float] = None
        self._latency_metrics: Dict[str, float] = {}

        self._motion_model = _MotionModelService(config)
        self._motion_model_target_idx: Optional[int] = None
        self._latest_prediction: Optional[_MotionModelPrediction] = None

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

    @property
    def latency_ms(self) -> Optional[float]:
        """Return the most recent latency estimate in milliseconds."""

        return self._latency_ms

    @property
    def latency_metrics(self) -> Mapping[str, float]:
        """Return the raw latency measurements captured on the Jetson."""

        return dict(self._latency_metrics)

    def update_latency_measurement(
        self,
        *,
        selected_ms: Optional[float],
        source: str,
        metrics: Mapping[str, float],
    ) -> None:
        """Record the latency figure that downstream prediction should use."""

        self._latency_source = source
        self._latency_ms = selected_ms if selected_ms is not None else None
        self._latency_metrics = dict(metrics)
        self._motion_model.update_latency(self._latency_ms)

    def update_detection(self, msg: DetectionMsg) -> None:
        """Consume the newest detection message."""

        now = time.monotonic()
        prev_motion_idx = self._motion_model_target_idx
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
            self._motion_model.reset()
            self._motion_model_target_idx = None
            self._latest_prediction = None
            self._disarm()

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

            if self._latest_target_idx != prev_motion_idx:
                self._motion_model.reset()
            self._motion_model_target_idx = self._latest_target_idx

            range_m, range_source, parallax_active = self._resolve_laser_range(
                msg.target_distance_smoothed_m
            )
            msg.laser_range_m = range_m
            msg.laser_range_source = range_source

            self._latest_detection.resolved_range_m = range_m
            self._latest_detection.range_source = range_source
            self._latest_detection.range_active = parallax_active

            self._motion_model.update_target(target_uv, now, distance_m=range_m)
        else:
            msg.laser_range_m = None
            msg.laser_range_source = None
            self._resolved_range = None
            parallax_active = False

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

        self._update_arming_state(msg, now)

        self._populate_detection_telemetry(msg, self._latest_detection, now)

    def update_cam_state(self, state: CamState) -> None:
        self._cam_state = state
        self._motion_model.update_cam_state(state, time.monotonic())
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

        if not target_recent:
            self._disarm()

        prediction: Optional[_MotionModelPrediction] = None
        if (
            detection
            and detection.target_uv is not None
            and target_recent
            and self._armed
        ):
            prediction = self._motion_model.predict(now)
            cmd = self._build_tracking_cmd(detection, dt, now, prediction)
            self._tracking_active = True
            self._latest_prediction = prediction
            self._post_arm_ticks += 1
        elif detection and detection.target_uv is not None and target_recent:
            self._prev_err = None
            self._integ = AxisPair(0.0, 0.0)
            cmd = self._build_idle_cmd(detection, now, reason="arming")
            self._tracking_active = False
            self._latest_prediction = None
        else:
            if self._cfg.reinit_on_lost:
                self._prev_err = None
                self._integ = AxisPair(0.0, 0.0)
                self._smoothed_uv = None
                self._prev_rate = AxisPair(0.0, 0.0)
            self._disarm()
            cmd = self._build_hold_cmd(now)
            self._tracking_active = False
            self._latest_prediction = None

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

    def _populate_detection_telemetry(
        self,
        msg: DetectionMsg,
        detection: Optional[_DetectionState],
        now: float,
    ) -> None:
        msg.track_mode = self._motion_model.mode
        msg.latency_ms_used_for_prediction = _safe_float(self._latency_ms)

        cam_state = self._cam_state
        if cam_state is not None:
            msg.cam_yaw_deg = _safe_degrees(cam_state.pan)
            msg.cam_pitch_deg = _safe_degrees(cam_state.tilt)
            msg.cam_yaw_rate_dps = _safe_degrees(cam_state.pan_rate)
            msg.cam_pitch_rate_dps = _safe_degrees(cam_state.tilt_rate)
        else:
            msg.cam_yaw_deg = None
            msg.cam_pitch_deg = None
            msg.cam_yaw_rate_dps = None
            msg.cam_pitch_rate_dps = None

        prediction: Optional[_MotionModelPrediction]
        if detection is not None and detection.target_uv is not None:
            prediction = self._motion_model.predict(now)
        else:
            prediction = None

        if prediction is not None:
            pred_u = _safe_float(prediction.uv[0])
            pred_v = _safe_float(prediction.uv[1])
            if pred_u is not None and pred_v is not None:
                msg.pred_px = (pred_u, pred_v)
            else:
                msg.pred_px = None
        else:
            msg.pred_px = None

        if detection is not None:
            pred_distance = detection.resolved_range_m
            if pred_distance is None:
                pred_distance = detection.target_distance_m
            if pred_distance is None:
                pred_distance = msg.target_distance_smoothed_m
            msg.pred_distance_m = _safe_float(pred_distance)
        else:
            msg.pred_distance_m = None

    def _is_target_recent(self, now: float) -> bool:
        if self._last_detection_ts is None:
            return False
        if self._lost_timeout_s <= 0.0:
            return True if self._latest_detection and self._latest_detection.target_uv else False
        return (now - self._last_detection_ts) <= self._lost_timeout_s

    def _disarm(self) -> None:
        self._armed = False
        self._arm_timestamp = None
        self._arm_consecutive = 0
        self._arm_candidate_idx = None
        self._arm_candidate_box = None
        self._post_arm_ticks = 0
        self._prev_err = None
        self._integ = AxisPair(0.0, 0.0)
        self._prev_rate = AxisPair(0.0, 0.0)

    def _update_arming_state(self, msg: DetectionMsg, now: float) -> None:
        detection = self._latest_detection
        if detection is None or detection.target_uv is None:
            self._disarm()
            return

        target_idx = self._latest_target_idx
        if target_idx is None or target_idx >= len(msg.boxes):
            self._disarm()
            return

        box = msg.boxes[target_idx]
        current_box = (float(box.x), float(box.y), float(box.w), float(box.h))

        if self._armed:
            self._arm_candidate_idx = target_idx
            self._arm_candidate_box = current_box
            return

        matched = False
        if self._arm_candidate_box is not None:
            if self._arm_candidate_idx == target_idx:
                matched = True
            else:
                iou = self._bbox_iou(self._arm_candidate_box, current_box)
                matched = iou >= self._ARM_IOU_THRESHOLD

        if matched:
            self._arm_consecutive += 1
        else:
            self._arm_consecutive = 1

        self._arm_candidate_idx = target_idx
        self._arm_candidate_box = current_box

        if self._arm_consecutive >= self._ARM_REQUIRED_MATCHES:
            self._armed = True
            self._arm_timestamp = now
            self._post_arm_ticks = 0
            self._prev_err = None
            self._integ = AxisPair(0.0, 0.0)
            self._prev_rate = AxisPair(0.0, 0.0)

    def _bbox_iou(
        self,
        prev_box: Tuple[float, float, float, float],
        cur_box: Tuple[float, float, float, float],
    ) -> float:
        prev_x, prev_y, prev_w, prev_h = prev_box
        cur_x, cur_y, cur_w, cur_h = cur_box

        prev_x2 = prev_x + prev_w
        prev_y2 = prev_y + prev_h
        cur_x2 = cur_x + cur_w
        cur_y2 = cur_y + cur_h

        inter_x1 = max(prev_x, cur_x)
        inter_y1 = max(prev_y, cur_y)
        inter_x2 = min(prev_x2, cur_x2)
        inter_y2 = min(prev_y2, cur_y2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        prev_area = max(prev_w, 0.0) * max(prev_h, 0.0)
        cur_area = max(cur_w, 0.0) * max(cur_h, 0.0)

        denom = prev_area + cur_area - inter_area
        if denom <= 0.0:
            return 0.0
        return inter_area / denom

    def _compute_dt(self, now: float) -> float:
        if self._last_cmd_time is None:
            dt = self._default_dt
        else:
            dt = now - self._last_cmd_time
        if not (dt > 0.0):
            dt = self._default_dt
        return _clamp(dt, self._MIN_DT, self._MAX_DT)

    def _gain_scale(self, now: float) -> float:
        if not self._armed or self._arm_timestamp is None:
            return 0.0
        if self._GAIN_RAMP_DURATION_S <= 0.0:
            return 1.0
        elapsed = max(now - self._arm_timestamp, 0.0)
        if elapsed >= self._GAIN_RAMP_DURATION_S:
            return 1.0
        return _clamp(elapsed / self._GAIN_RAMP_DURATION_S, 0.0, 1.0)

    def _should_use_derivative(self, dt: float) -> bool:
        if not self._armed:
            return False
        if dt <= self._DERIVATIVE_MIN_DT:
            return False
        return self._post_arm_ticks >= self._DERIVATIVE_FREEZE_TICKS

    def _build_tracking_cmd(
        self,
        detection: _DetectionState,
        dt: float,
        now: float,
        prediction: Optional[_MotionModelPrediction],
    ) -> ControlCmd:
        assert detection.target_uv is not None
        target_uv = self._resolve_control_target(detection, prediction)

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

        derivative_allowed = self._should_use_derivative(dt)

        if not self._tracking_active or self._prev_err is None or not derivative_allowed:
            derr = AxisPair(0.0, 0.0)
        else:
            derr = AxisPair(
                yaw=(err_rad.yaw - self._prev_err.yaw) / dt,
                pitch=(err_rad.pitch - self._prev_err.pitch) / dt,
            )

        gain_scale = self._gain_scale(now)
        kp_yaw = self._cfg.kp.yaw * gain_scale
        kp_pitch = self._cfg.kp.pitch * gain_scale
        ki_yaw = self._cfg.ki.yaw * gain_scale
        ki_pitch = self._cfg.ki.pitch * gain_scale
        kd_yaw = self._cfg.kd.yaw * gain_scale if derivative_allowed else 0.0
        kd_pitch = self._cfg.kd.pitch * gain_scale if derivative_allowed else 0.0

        integ_yaw = self._integrate(self._integ.yaw, err_rad.yaw, dt, ki_yaw, self._cfg.rate_limits.yaw)
        integ_pitch = self._integrate(
            self._integ.pitch, err_rad.pitch, dt, ki_pitch, self._cfg.rate_limits.pitch
        )
        self._integ = AxisPair(integ_yaw, integ_pitch)

        yaw_rate = (
            kp_yaw * err_rad.yaw
            + kd_yaw * derr.yaw
            + ki_yaw * integ_yaw
        )
        pitch_rate = (
            kp_pitch * err_rad.pitch
            + kd_pitch * derr.pitch
            + ki_pitch * integ_pitch
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

        motion_diag = self._motion_model.diagnostics(now)
        self._log_control_state(
            {
                "frame_id": detection.frame_id,
                "target_ok": True,
                "dt": round(dt, 6),
                "raw_uv": [
                    float(detection.target_uv[0]),
                    float(detection.target_uv[1]),
                ],
                "uv": [float(target_uv[0]), float(target_uv[1])],
                "aim_uv": [float(aim_uv[0]), float(aim_uv[1])],
                "err_px": [raw_px_err.yaw, raw_px_err.pitch],
                "err_rad": [err_rad.yaw, err_rad.pitch],
                "cmd_rate": [yaw_rate, pitch_rate],
                "range_m": detection.resolved_range_m,
                "range_src": detection.range_source,
                "parallax_active": detection.range_active,
                "pred_horizon_s": prediction.horizon_s if prediction else None,
                "pred_age_s": prediction.age_s if prediction else None,
                "pred_cam_shift_px": list(prediction.camera_shift_px)
                if prediction
                else None,
                "pred_velocity_px_s": list(prediction.velocity_px_s)
                if prediction
                else None,
                "motion": motion_diag,
                "armed": self._armed,
            },
            target_ok=True,
            now=now,
        )

        return cmd

    def _resolve_control_target(
        self,
        detection: _DetectionState,
        prediction: Optional[_MotionModelPrediction],
    ) -> Tuple[float, float]:
        if prediction is not None:
            detection.prediction = prediction
            detection.predicted_uv = (
                float(prediction.uv[0]),
                float(prediction.uv[1]),
            )
            if self._cfg.motion_model.apply_to_control:
                return detection.predicted_uv
        else:
            detection.prediction = None
            detection.predicted_uv = None

        if self._smoothed_uv is not None:
            return self._smoothed_uv

        return detection.target_uv

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

    def _build_idle_cmd(
        self, detection: _DetectionState, now: float, *, reason: str
    ) -> ControlCmd:
        uv_source: Optional[Tuple[float, float]]
        if detection.predicted_uv is not None:
            uv_source = detection.predicted_uv
        elif detection.target_uv is not None:
            uv_source = detection.target_uv
        else:
            uv_source = None

        if uv_source is None:
            uv = (self._cfg.cx_px, self._cfg.cy_px)
        else:
            uv = (float(uv_source[0]), float(uv_source[1]))

        cmd = ControlCmd(
            frame_id=detection.frame_id,
            src_ts_ms=detection.src_ts_ms,
            cmd_ts_ms=int(time.monotonic_ns() / 1e6),
            target_ok=False,
            target_uv=(float(uv[0]), float(uv[1])),
            err_uv=(0.0, 0.0),
            err_rad=(0.0, 0.0),
            pan_rate_cmd=0.0,
            tilt_rate_cmd=0.0,
            pan_abs_cmd=None,
            tilt_abs_cmd=None,
            laser_origin_px=self._laser_overlay.origin_px if self._laser_overlay else None,
            laser_dot_px=self._laser_overlay.dot_px if self._laser_overlay else None,
            laser_on_target=self._laser_overlay.on_target if self._laser_overlay else None,
            laser_range_m=self._laser_overlay.range_m if self._laser_overlay else None,
            laser_range_source=self._laser_overlay.range_source if self._laser_overlay else None,
            parallax_compensation_active=(
                self._laser_overlay.active if self._laser_overlay else False
            ),
        )

        self._prev_rate = AxisPair(0.0, 0.0)

        motion_diag = self._motion_model.diagnostics(now)
        self._log_control_state(
            {
                "frame_id": detection.frame_id,
                "target_ok": False,
                "dt": None,
                "uv": [float(uv[0]), float(uv[1])],
                "err_px": [0.0, 0.0],
                "err_rad": [0.0, 0.0],
                "cmd_rate": [0.0, 0.0],
                "pending_arm": reason == "arming",
                "armed": self._armed,
                "motion": motion_diag,
            },
            target_ok=False,
            now=now,
        )

        return cmd

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

        motion_diag = self._motion_model.diagnostics(now)
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
                "motion": motion_diag,
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

        motion = payload.get("motion")
        if isinstance(motion, dict):
            summary_parts = []
            mode = motion.get("mode")
            if mode is not None:
                summary_parts.append(str(mode))
            horizon = motion.get("horizon_s")
            if horizon is not None:
                summary_parts.append(f"h={self._format_float(horizon)}s")
            has_state = motion.get("has_state")
            if has_state is not None:
                summary_parts.append("state=track" if has_state else "state=idle")
            residual = motion.get("residual_px")
            if isinstance(residual, (list, tuple)) and len(residual) == 2:
                summary_parts.append(f"res={self._format_pair(residual)}")
            if summary_parts:
                parts.append("motion=" + ",".join(summary_parts))

        return " | ".join(parts)

    def _format_float(self, value: Any) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):.{self._log_float_precision}f}"
        return str(value)

    def _format_pair(self, values: Sequence[Any]) -> str:
        return f"({self._format_float(values[0])}, {self._format_float(values[1])})"

