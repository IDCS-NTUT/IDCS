"""Closed-loop controller that turns detections into pan/tilt commands."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import zmq

from common.control import (
    AxisPair,
    ControlConfig,
    angular_error_from_pixels,
    pixel_error,
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


@dataclass
class _DetectionState:
    frame_id: int
    src_ts_ms: int
    timestamp: float
    target_uv: Optional[Tuple[float, float]]


class ControlLoop:
    """Runs a rate-mode PID loop for yaw/pitch using the latest detection."""

    _MIN_DT = 1e-3
    _MAX_DT = 0.2

    def __init__(self, config: ControlConfig, pub: zmq.Socket) -> None:
        self._cfg = config
        self._pub = pub

        self._lost_timeout_s = config.lost_target_timeout_ms / 1000.0
        self._default_dt = (
            config.loop_dt if config.loop_dt is not None else 1.0 / max(1.0, config.loop_hz or 30.0)
        )

        self._latest_detection: Optional[_DetectionState] = None
        self._last_frame_id: int = 0
        self._last_src_ts_ms: int = 0
        self._last_detection_ts: Optional[float] = None

        self._smoothed_uv: Optional[Tuple[float, float]] = None
        self._prev_err: Optional[AxisPair] = None
        self._integ = AxisPair(0.0, 0.0)
        self._prev_rate = AxisPair(0.0, 0.0)
        self._last_cmd_time: Optional[float] = None
        self._tracking_active = False

        self._cam_state: Optional[CamState] = None
        self._home_pan: Optional[float] = None
        self._home_tilt: Optional[float] = None
        self._home_deadband = math.radians(0.5)

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
    def update_detection(self, msg: DetectionMsg) -> None:
        """Consume the newest detection message."""

        now = time.monotonic()
        target_uv = self._select_target(msg)

        self._latest_detection = _DetectionState(
            frame_id=msg.frame_id,
            src_ts_ms=msg.src_ts_ms,
            timestamp=now,
            target_uv=target_uv,
        )
        self._last_frame_id = msg.frame_id
        self._last_src_ts_ms = msg.src_ts_ms

        if target_uv is not None:
            self._last_detection_ts = now
            if self._smoothed_uv is None or not self._tracking_active:
                self._smoothed_uv = target_uv
            else:
                self._smoothed_uv = self._smooth_uv(target_uv)

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
            cmd = self._build_hold_cmd(now)
            self._tracking_active = False

        self._last_cmd_time = now
        self._send_cmd(cmd)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _select_target(self, msg: DetectionMsg) -> Optional[Tuple[float, float]]:
        boxes: Sequence[Box] = msg.boxes
        if not boxes:
            return None

        if self._class_filter:
            boxes = [b for b in boxes if b.cls == self._class_filter]
            if not boxes:
                return None

        if self._selector_strategy == "largest_area":
            best = max(boxes, key=lambda b: b.w * b.h)
        else:
            best = max(boxes, key=lambda b: b.conf)

        u = (best.x + (best.w / 2.0)) * msg.img_w
        v = (best.y + (best.h / 2.0)) * msg.img_h
        return (u, v)

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
        target_uv = self._smoothed_uv or detection.target_uv

        raw_px_err = pixel_error(target_uv[0], target_uv[1], self._cfg, apply_deadband=False)
        ctrl_px_err = pixel_error(target_uv[0], target_uv[1], self._cfg, apply_deadband=True)
        err_rad = angular_error_from_pixels(
            target_uv[0], target_uv[1], self._cfg, apply_deadband=True
        )

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
        )

        if _LOG.isEnabledFor(logging.INFO):
            _LOG.info(
                json.dumps(
                    {
                        "frame_id": detection.frame_id,
                        "target_ok": True,
                        "dt": round(dt, 6),
                        "uv": [float(target_uv[0]), float(target_uv[1])],
                        "err_px": [raw_px_err.yaw, raw_px_err.pitch],
                        "err_rad": [err_rad.yaw, err_rad.pitch],
                        "cmd_rate": [yaw_rate, pitch_rate],
                    }
                )
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
        )

        if _LOG.isEnabledFor(logging.INFO):
            _LOG.info(
                json.dumps(
                    {
                        "frame_id": self._last_frame_id,
                        "target_ok": False,
                        "dt": None,
                        "uv": [float(uv[0]), float(uv[1])],
                        "err_px": [0.0, 0.0],
                        "err_rad": [home_err.yaw, home_err.pitch],
                        "cmd_rate": [home_rates.yaw, home_rates.pitch],
                        "home": True,
                    }
                )
            )

        return cmd

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

