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
            state_monotonic_ns=capture_ns - 9_000_000, pan_sample_monotonic_ns=capture_ns - 9_000_000, tilt_sample_monotonic_ns=capture_ns - 9_000_000,
            pan=yaw - 0.0009, tilt=0.0,
        ))
        assert estimator.observe_cam_state(CamState(
            frame_id=index * 2 + 1, src_ts_ms=0,
            state_monotonic_ns=capture_ns + 9_000_000, pan_sample_monotonic_ns=capture_ns + 9_000_000, tilt_sample_monotonic_ns=capture_ns + 9_000_000,
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
                state_monotonic_ns=source_ns + offset, pan_sample_monotonic_ns=source_ns + offset, tilt_sample_monotonic_ns=source_ns + offset, pan=0.0, tilt=0.0,
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


def _moving_target_run(estimator, *, predict, camera_yaw=0.0):
    """Target moves at 0.5 rad/s in the world; frames reach the estimator 60 ms late."""
    result = None
    for index in range(12):
        capture_ns = 1_000_000_000 + index * 20_000_000
        for offset in (-9_000_000, 9_000_000):
            estimator.observe_cam_state(CamState(
                frame_id=index * 2 + (offset > 0), src_ts_ms=0,
                state_monotonic_ns=capture_ns + offset, pan_sample_monotonic_ns=capture_ns + offset, tilt_sample_monotonic_ns=capture_ns + offset, pan=camera_yaw, tilt=0.0))
        target_world = 0.5 * (capture_ns - 1_000_000_000) / 1e9
        obs = _observation(index + 1, capture_ns, camera_yaw)
        obs = obs.model_copy(update={"target": obs.target.model_copy(
            update={"bearing_error_rad": (target_world - camera_yaw, 0.0)})})
        clock = ClockBounds(0, 0, obs.created_monotonic_ns, 0.0)
        result = estimator.estimate(obs, clock, predict=predict)
    return result


def test_full_prediction_moves_target_to_decision_time() -> None:
    result = _moving_target_run(VideoTargetRateEstimator(accel_sigma_rad_s2=2.0), predict=1.0)
    assert result.valid
    capture_s = (result.capture_midpoint_ns - 1_000_000_000) / 1e9
    # Decision is 60 ms after capture; camera is static at 0.
    assert result.predicted_target_world_rad[0] == pytest.approx(0.5 * (capture_s + 0.06), abs=0.003)
    assert result.predicted_bearing_error_rad[0] == pytest.approx(0.5 * (capture_s + 0.06), abs=0.003)
    assert result.prediction_ns == result.capture_midpoint_ns + 60_000_000


def test_half_prediction_evaluates_halfway_through_frame_age() -> None:
    result = _moving_target_run(VideoTargetRateEstimator(accel_sigma_rad_s2=2.0), predict=0.5)
    capture_s = (result.capture_midpoint_ns - 1_000_000_000) / 1e9
    assert result.predicted_bearing_error_rad[0] == pytest.approx(0.5 * (capture_s + 0.03), abs=0.003)


def test_no_prediction_by_default() -> None:
    assert _moving_target_run(VideoTargetRateEstimator(), predict=0.0).predicted_bearing_error_rad is None


def test_prediction_uses_newest_camera_pose_not_capture_pose() -> None:
    estimator = VideoTargetRateEstimator()
    result = _moving_target_run(estimator, predict=1.0)
    # The camera has since moved to 0.02 rad (newest sample after the capture).
    estimator.observe_cam_state(CamState(frame_id=999, src_ts_ms=0,
                                         state_monotonic_ns=result.prediction_ns - 1, pan_sample_monotonic_ns=result.prediction_ns - 1, tilt_sample_monotonic_ns=result.prediction_ns - 1, pan=0.02, tilt=0.0))
    obs = _observation(12, result.capture_midpoint_ns, 0.0)
    obs = obs.model_copy(update={"target": obs.target.model_copy(
        update={"bearing_error_rad": (result.measured_target_world_rad[0], 0.0)})})
    again = estimator.estimate(obs, ClockBounds(0, 0, obs.created_monotonic_ns, 0.0), predict=1.0)
    assert again.predicted_bearing_error_rad[0] == pytest.approx(again.predicted_target_world_rad[0] - 0.02, abs=1e-9)


def test_predict_outside_unit_interval_is_rejected() -> None:
    obs = _observation(1, 1_000_000_000, 0.0)
    with pytest.raises(ValueError):
        VideoTargetRateEstimator().estimate(obs, None, predict=1.5)
