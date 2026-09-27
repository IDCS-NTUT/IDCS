"""Causal video target-rate estimator with frame-time camera-pose alignment.

This module has no socket or motor authority. It does not invent PC/Jetson
clock bounds, exposure times, encoder samples, or missing target measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from common.schemas import CamState, ControlObservation
from jetson.control.feedforward import TargetRateKalman
from jetson.control.pose_history import CameraPoseHistory
from jetson.control.timing import ClockBounds


@dataclass(frozen=True)
class VideoFeedforwardEstimate:
    valid: bool
    reason: str
    yaw_rate_rad_s: float = 0.0
    pitch_rate_rad_s: float = 0.0
    capture_midpoint_ns: int | None = None
    clock_width_ns: int | None = None
    capture_camera_pose_rad: tuple[float, float] | None = None
    measured_target_world_rad: tuple[float, float] | None = None
    # Latency compensation: predicted target world angle and camera angle at
    # capture + predict * (decision - capture); their difference replaces the
    # raw frame bearing as the PID error when available.
    prediction_ns: int | None = None
    predicted_target_world_rad: tuple[float, float] | None = None
    predicted_bearing_error_rad: tuple[float, float] | None = None


class VideoTargetRateEstimator:
    def __init__(
        self, *, max_sample_age_s: float = 0.15,
        accel_sigma_rad_s2: float = 0.4,
    ) -> None:
        self.pose_history = CameraPoseHistory()
        # Match the video PID capture-age gate. Older observations cannot
        # contribute FF even if the Kalman state remains numerically stable.
        self._yaw = TargetRateKalman(max_sample_age_s=max_sample_age_s,
                                     acceleration_sigma_rad_s2=accel_sigma_rad_s2)
        self._pitch = TargetRateKalman(max_sample_age_s=max_sample_age_s,
                                       acceleration_sigma_rad_s2=accel_sigma_rad_s2)
        self._last_frame_id: int | None = None

    def reset(self) -> None:
        self.pose_history.reset()
        self._yaw.reset()
        self._pitch.reset()
        self._last_frame_id = None

    def observe_cam_state(self, state: CamState) -> bool:
        return self.pose_history.observe(state)

    def estimate(
        self, observation: ControlObservation, clock: ClockBounds | None,
        *, predict: float = 0.0,
    ) -> VideoFeedforwardEstimate:
        if not math.isfinite(predict) or not 0.0 <= predict <= 1.0:
            raise ValueError("predict must be in [0, 1]")
        if clock is None:
            return VideoFeedforwardEstimate(False, "clock_unavailable")
        if observation.source_identity_verified is not True:
            return VideoFeedforwardEstimate(False, "frame_identity_unverified")
        if observation.source_clock_domain != "pc_monotonic":
            return VideoFeedforwardEstimate(False, "source_clock_domain_invalid")
        if observation.source_frame_id is None or observation.source_time_ns is None:
            return VideoFeedforwardEstimate(False, "source_frame_time_missing")
        target = observation.target
        if not target.valid or target.track_id is None or target.bearing_error_rad is None:
            return VideoFeedforwardEstimate(False, "target_invalid")
        if self._last_frame_id is not None and observation.source_frame_id < self._last_frame_id:
            self._yaw.reset()
            self._pitch.reset()
            return VideoFeedforwardEstimate(False, "source_frame_regressed")
        try:
            capture = clock.map_pc_event(
                observation.source_time_ns,
                jetson_now_ns=observation.created_monotonic_ns,
            )
        except ValueError:
            return VideoFeedforwardEstimate(False, "capture_clock_mapping_invalid")
        pose, reason = self.pose_history.at(capture)
        if pose is None:
            return VideoFeedforwardEstimate(False, reason)
        camera_pose = (pose.yaw_rad, pose.pitch_rad)
        world_measurement = (
            pose.yaw_rad + target.bearing_error_rad[0],
            pose.pitch_rad + target.bearing_error_rad[1],
        )
        if observation.source_frame_id != self._last_frame_id:
            self._yaw.observe(
                track_id=target.track_id,
                angle_rad=world_measurement[0],
                sample_ns=pose.capture_midpoint_ns,
            )
            self._pitch.observe(
                track_id=target.track_id,
                angle_rad=world_measurement[1],
                sample_ns=pose.capture_midpoint_ns,
            )
            self._last_frame_id = observation.source_frame_id
        yaw = self._yaw.estimate(
            decision_ns=observation.created_monotonic_ns, track_id=target.track_id,
        )
        pitch = self._pitch.estimate(
            decision_ns=observation.created_monotonic_ns, track_id=target.track_id,
        )
        if not yaw.valid or not pitch.valid:
            return VideoFeedforwardEstimate(
                False, yaw.reason if not yaw.valid else pitch.reason,
                capture_midpoint_ns=pose.capture_midpoint_ns,
                clock_width_ns=pose.timing_width_ns,
                capture_camera_pose_rad=camera_pose,
                measured_target_world_rad=world_measurement,
            )
        prediction = self._predict(
            predict, pose.capture_midpoint_ns, observation.created_monotonic_ns,
            camera_pose, (yaw.position_rad, pitch.position_rad),
            (yaw.rate_rad_s, pitch.rate_rad_s),
        )
        return VideoFeedforwardEstimate(
            True, "ready", yaw.rate_rad_s, pitch.rate_rad_s,
            pose.capture_midpoint_ns, pose.timing_width_ns,
            camera_pose, world_measurement, *prediction,
        )

    def _predict(
        self, predict: float, capture_ns: int, decision_ns: int,
        capture_pose: tuple[float, float],
        target_at_decision: tuple[float, float],
        rate: tuple[float, float],
    ) -> tuple[int | None, tuple[float, float] | None, tuple[float, float] | None]:
        """Target and camera at capture + predict * age (constant velocity)."""
        if predict <= 0.0 or decision_ns < capture_ns:
            return None, None, None
        eval_ns = capture_ns + round(predict * (decision_ns - capture_ns))
        back_s = (decision_ns - eval_ns) / 1e9
        target = (target_at_decision[0] - rate[0] * back_s,
                  target_at_decision[1] - rate[1] * back_s)
        # Camera angle at eval time from measured samples; beyond the newest
        # sample the newest measured pose is held (no extrapolation).
        camera = self.pose_history.pose_at_or_latest(eval_ns) or capture_pose
        bearing = (target[0] - camera[0], target[1] - camera[1])
        return eval_ns, target, bearing

