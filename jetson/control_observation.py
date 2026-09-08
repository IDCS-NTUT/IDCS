"""Shadow-only adapter from existing metadata to ``ControlObservation``.

The adapter intentionally refuses to infer missing input.  It keeps local
receipt timestamps for each source and marks each component invalid once it
ages out.  It neither publishes nor imports gimbal/serial drivers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from common.control import AxisPair, ControlConfig, angular_error_from_pixel_delta, pixel_delta
from common.schemas import (
    Box,
    CamState,
    ControlGimbalObservation,
    ControlObservation,
    ControlSafetyObservation,
    ControlTargetObservation,
    ControlTransportObservation,
    DetectionMsg,
    ManualControlState,
)


@dataclass(frozen=True)
class ObservationAgeLimits:
    target_s: float = 0.100
    gimbal_s: float = 0.100
    safety_s: float = 0.750


class ControlObservationAssembler:
    """Assemble latest-only legacy metadata into an atomic controller input."""

    def __init__(self, config: ControlConfig, *, age_limits: ObservationAgeLimits = ObservationAgeLimits()) -> None:
        self._config = config
        self._limits = age_limits
        self._detection: Optional[Tuple[DetectionMsg, float]] = None
        self._cam_state: Optional[Tuple[CamState, float]] = None
        self._manual: Optional[Tuple[ManualControlState, float]] = None
        self._sequence = 0

    def update_detection(self, message: DetectionMsg, *, received_at: float) -> None:
        self._detection = (message, float(received_at))

    def update_cam_state(self, state: CamState, *, received_at: float) -> None:
        self._cam_state = (state, float(received_at))

    def update_manual_state(self, state: ManualControlState, *, received_at: float) -> None:
        self._manual = (state, float(received_at))

    @staticmethod
    def _age_ms(now: float, received_at: float) -> float:
        return max(0.0, (now - received_at) * 1000.0)

    @staticmethod
    def _selected_box(message: DetectionMsg) -> Optional[Box]:
        if message.target_track_id is not None:
            return next(
                (box for box in message.boxes if box.track_id == message.target_track_id), None
            )
        if message.target_idx is not None and 0 <= message.target_idx < len(message.boxes):
            return message.boxes[message.target_idx]
        return None

    def _target(self, now: float) -> ControlTargetObservation:
        if self._detection is None:
            return ControlTargetObservation(valid=False)
        message, received_at = self._detection
        age_ms = self._age_ms(now, received_at)
        box = self._selected_box(message)
        if age_ms > self._limits.target_s * 1000.0 or box is None:
            return ControlTargetObservation(valid=False, source_age_ms=age_ms)
        target_u = (box.x + box.w / 2.0) * message.img_w
        target_v = (box.y + box.h / 2.0) * message.img_h
        err_px = pixel_delta(target_u, target_v, self._config.cx_px, self._config.cy_px,
                             self._config, apply_deadband=False)
        err_rad = angular_error_from_pixel_delta(err_px, self._config)
        rate = None
        if message.target_velocity_px_s is not None:
            # Velocity uses the same raw image axes as the target centre, so
            # apply the configured controller sign convention before turning
            # pixels/s into camera bearing rate.
            velocity = angular_error_from_pixel_delta(
                AxisPair(message.target_velocity_px_s[0] * self._config.yaw_sign,
                         message.target_velocity_px_s[1] * self._config.pitch_sign),
                self._config, linearize=True
            )
            rate = velocity.as_tuple()
        return ControlTargetObservation(
            valid=True, track_id=box.track_id, class_id=box.cls, confidence=box.conf,
            bearing_error_rad=err_rad.as_tuple(), bearing_rate_rad_s=rate, source_age_ms=age_ms,
        )

    def _gimbal(self, now: float) -> ControlGimbalObservation:
        if self._cam_state is None:
            return ControlGimbalObservation(valid=False)
        state, received_at = self._cam_state
        age_ms = self._age_ms(now, received_at)
        if age_ms > self._limits.gimbal_s * 1000.0:
            return ControlGimbalObservation(valid=False, sample_age_ms=age_ms)
        return ControlGimbalObservation(
            valid=True, yaw_rad=state.pan, pitch_rad=state.tilt,
            yaw_rate_rad_s=state.pan_rate, pitch_rate_rad_s=state.tilt_rate,
            sample_age_ms=age_ms,
        )

    def _safety(self, now: float) -> ControlSafetyObservation:
        if self._manual is None:
            return ControlSafetyObservation(valid=False, auto_allowed=False,
                                            manual_active=False, emergency_active=False)
        state, received_at = self._manual
        age_ms = self._age_ms(now, received_at)
        valid = age_ms <= self._limits.safety_s * 1000.0
        return ControlSafetyObservation(
            valid=valid,
            auto_allowed=bool(valid and not state.active and not state.emergency and state.control_cmd_enabled),
            manual_active=bool(state.active), emergency_active=bool(state.emergency), sample_age_ms=age_ms,
        )

    def build(self, *, now: float, serial_acceptance_ms: Optional[float] = None,
              last_command_age_ms: Optional[float] = None) -> ControlObservation:
        """Return a fully validated snapshot; no absent measurement is fabricated."""

        self._sequence += 1
        return ControlObservation(
            sequence=self._sequence, created_monotonic_ns=int(now * 1_000_000_000),
            target=self._target(now), gimbal=self._gimbal(now),
            transport=ControlTransportObservation(
                serial_acceptance_ms=serial_acceptance_ms, last_command_age_ms=last_command_age_ms,
            ), safety=self._safety(now),
        )
