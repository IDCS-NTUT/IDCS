"""Causal V3 video target-rate estimator with frame-time camera-pose alignment.

This module has no socket or motor authority. It does not invent PC/Jetson
clock bounds, exposure times, encoder samples, or missing target measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from common.schemas import CamState, ControlObservation
from jetson.control_v3.feedforward import TargetRateKalman
from jetson.control_v3.pose_history import AlignedPose, CameraPoseHistory
from jetson.control_v3.timing import ClockBounds, TimeInterval


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
        pose_source: Literal["encoder", "render", "frame"] = "encoder",
        accel_sigma_rad_s2: float = 0.4,
    ) -> None:
        self.pose_source = pose_source
        self.pose_history = CameraPoseHistory(
            pose_source="render" if pose_source == "frame" else pose_source,
        )
        # Match the video PID capture-age gate. Older observations cannot
        # contribute FF even if the Kalman state remains numerically stable.
        self._yaw = TargetRateKalman(max_sample_age_s=max_sample_age_s,
                                     acceleration_sigma_rad_s2=accel_sigma_rad_s2)
        self._pitch = TargetRateKalman(max_sample_age_s=max_sample_age_s,
                                       acceleration_sigma_rad_s2=accel_sigma_rad_s2)
        self._last_frame_id: int | None = None
        # Frame mode: the simulator renders relative_hil_pose(latest CamState)
        # (render pose if present, minus the bridge home). Keep the newest such
        # pose as the fresh camera angle for prediction, in the frame-pose frame.
        self._latest_relative: tuple[int, float, float] | None = None

    def reset(self) -> None:
        self._latest_relative = None
        self.pose_history.reset()
        self._yaw.reset()
        self._pitch.reset()
        self._last_frame_id = None

    def observe_cam_state(self, state: CamState) -> bool:
        accepted = self.pose_history.observe(state)
        if accepted and self.pose_source == "frame":
            relative = _relative_render_pose(state)
            if relative is not None and state.state_monotonic_ns is not None:
                self._latest_relative = (state.state_monotonic_ns, *relative)
        return accepted

    def estimate(
        self, observation: ControlObservation, clock: ClockBounds | None,
        *, frame_pose_rad: tuple[float, float] | None = None,
        predict: float = 0.0,
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
        if self.pose_source == "frame":
            width_ns = capture.latest_ns - capture.earliest_ns
            if width_ns > self.pose_history.max_interval_width_ns:
                return VideoFeedforwardEstimate(False, "capture_clock_interval_too_wide")
            if frame_pose_rad is None:
                return VideoFeedforwardEstimate(False, "sim_capture_pose_missing")
            if not all(math.isfinite(value) for value in frame_pose_rad):
                return VideoFeedforwardEstimate(False, "sim_capture_pose_invalid")
            pose = AlignedPose(
                float(frame_pose_rad[0]), float(frame_pose_rad[1]),
                (capture.earliest_ns + capture.latest_ns) // 2,
                width_ns, 0.0, 0.0,
            )
            reason = "exact_sim_frame_pose"
        else:
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
        if self.pose_source == "frame":
            latest = self._latest_relative
            if latest is None:
                return None, None, None
            camera = (latest[1], latest[2]) if latest[0] > capture_ns else capture_pose
            bearing = (target[0] - camera[0], target[1] - camera[1])
            return eval_ns, target, bearing
        latest = self.pose_history.latest()
        if latest is None or latest[0] <= capture_ns:
            camera = capture_pose
        elif eval_ns >= latest[0]:
            camera = (latest[1], latest[2])  # hold the newest measured pose
        else:
            aligned, _ = self.pose_history.at(TimeInterval(eval_ns, eval_ns))
            camera = capture_pose if aligned is None else (aligned.yaw_rad, aligned.pitch_rad)
        bearing = (target[0] - camera[0], target[1] - camera[1])
        return eval_ns, target, bearing


def _relative_render_pose(state: CamState) -> tuple[float, float] | None:
    """Same mapping as ``pc.streamer.relative_hil_pose`` (host renderer)."""
    if state.home_pan is None or state.home_tilt is None:
        return None
    use_render = state.render_pan is not None and state.render_tilt is not None
    pan = float(state.render_pan) if use_render else float(state.pan)
    tilt = float(state.render_tilt) if use_render else float(state.tilt)
    delta = pan - float(state.home_pan)
    return math.atan2(math.sin(delta), math.cos(delta)), tilt - float(state.home_tilt)
