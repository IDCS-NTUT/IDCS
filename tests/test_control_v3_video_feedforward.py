from __future__ import annotations

import pytest

from common.schemas import (
    CamState, ControlGimbalObservation, ControlObservation,
    ControlSafetyObservation, ControlTargetObservation,
    ControlTransportObservation,
)
from jetson.control_v3.timing import ClockBounds
from jetson.control_v3.video_feedforward import VideoTargetRateEstimator


def _observation(frame_id: int, source_ns: int, camera_yaw: float) -> ControlObservation:
    return ControlObservation(
        sequence=frame_id, created_monotonic_ns=source_ns + 60_000_000,
        source_frame_id=frame_id, source_time_ns=source_ns,
        source_clock_domain="pc_monotonic", source_identity_verified=True,
        target=ControlTargetObservation(
            valid=True, track_id=1, target_center_px=(500.0, 400.0),
            aim_reference_px=(500.0, 400.0), pixel_error=(0.0, 0.0),
            bearing_error_rad=(0.1 - camera_yaw, 0.0),
        ),
        gimbal=ControlGimbalObservation(valid=False),
        transport=ControlTransportObservation(),
        safety=ControlSafetyObservation(
            valid=False, auto_allowed=False, manual_active=False,
            emergency_active=False,
        ),
    )


def test_camera_motion_is_removed_before_estimating_target_world_rate() -> None:
    estimator = VideoTargetRateEstimator()
    result = None
    for index in range(12):
        capture_ns = 1_000_000_000 + index * 20_000_000
        yaw = index * 0.002
        assert estimator.observe_cam_state(CamState(
            frame_id=index * 2, src_ts_ms=0,
            state_monotonic_ns=capture_ns - 9_000_000,
            pan=yaw - 0.0009, tilt=0.0,
        ))
        assert estimator.observe_cam_state(CamState(
            frame_id=index * 2 + 1, src_ts_ms=0,
            state_monotonic_ns=capture_ns + 9_000_000,
            pan=yaw + 0.0009, tilt=0.0,
        ))
        obs = _observation(index + 1, capture_ns, yaw)
        clock = ClockBounds(0, 0, obs.created_monotonic_ns, 0.0)
        result = estimator.estimate(obs, clock)
    assert result is not None and result.valid
    assert result.yaw_rate_rad_s == pytest.approx(0.0, abs=0.002)
    assert result.pitch_rate_rad_s == pytest.approx(0.0, abs=0.002)
    assert result.capture_camera_pose_rad is not None
    assert result.measured_target_world_rad == pytest.approx((0.1, 0.0), abs=0.002)


def test_exact_sim_frame_pose_removes_camera_motion_without_interpolation() -> None:
    estimator = VideoTargetRateEstimator(pose_source="frame")
    result = None
    for index in range(12):
        capture_ns = 1_000_000_000 + index * 20_000_000
        yaw = index * 0.002
        obs = _observation(index + 1, capture_ns, yaw)
        clock = ClockBounds(0, 0, obs.created_monotonic_ns, 0.0)
        result = estimator.estimate(obs, clock, frame_pose_rad=(yaw, 0.0))
    assert result is not None and result.valid
    assert result.yaw_rate_rad_s == pytest.approx(0.0, abs=0.002)
    assert result.measured_target_world_rad == pytest.approx((0.1, 0.0), abs=0.002)
    assert estimator.estimate(obs, clock).reason == "sim_capture_pose_missing"


def test_video_feedforward_refuses_unbracketed_pose_and_missing_clock() -> None:
    estimator = VideoTargetRateEstimator()
    obs = _observation(1, 1_000_000_000, 0.0)
    assert estimator.estimate(obs, None).reason == "clock_unavailable"
    clock = ClockBounds(0, 0, obs.created_monotonic_ns, 0.0)
    assert estimator.estimate(obs, clock).reason == "pose_history_warmup"


def test_video_rate_age_matches_150ms_video_capture_gate() -> None:
    estimator = VideoTargetRateEstimator()
    latest = None
    for index in range(4):
        source_ns = 1_000_000_000 + index * 20_000_000
        for offset in (-9_000_000, 9_000_000):
            assert estimator.observe_cam_state(CamState(
                frame_id=index * 2 + int(offset > 0), src_ts_ms=0,
                state_monotonic_ns=source_ns + offset, pan=0.0, tilt=0.0,
            ))
        latest = _observation(index + 1, source_ns, 0.0)
        clock = ClockBounds(0, 0, latest.created_monotonic_ns, 0.0)
        estimator.estimate(latest, clock)
    assert latest is not None
    source_ns = latest.source_time_ns
    assert source_ns is not None
    near = latest.model_copy(update={"created_monotonic_ns": source_ns + 149_000_000})
    far = latest.model_copy(update={"created_monotonic_ns": source_ns + 151_000_000})
    assert estimator.estimate(near, ClockBounds(0, 0, near.created_monotonic_ns, 0.0)).valid
    assert estimator.estimate(far, ClockBounds(0, 0, far.created_monotonic_ns, 0.0)).reason == "sample_stale"
