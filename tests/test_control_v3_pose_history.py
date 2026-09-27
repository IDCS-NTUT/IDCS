from __future__ import annotations

import pytest

from common.schemas import CamState
from jetson.control_v3.pose_history import CameraPoseHistory
from jetson.control_v3.timing import TimeInterval


def _state(timestamp: int, pan: float, tilt: float) -> CamState:
    return CamState(frame_id=1, src_ts_ms=0, state_monotonic_ns=timestamp,
                    pan=pan, tilt=tilt)


def test_pose_history_interpolates_bracketed_capture_interval() -> None:
    history = CameraPoseHistory()
    assert history.observe(_state(1_000_000_000, 0.0, 0.0))
    assert history.observe(_state(1_100_000_000, 0.01, -0.02))
    pose, reason = history.at(TimeInterval(1_045_000_000, 1_055_000_000))
    assert reason == "aligned"
    assert pose is not None
    assert pose.yaw_rad == pytest.approx(0.005)
    assert pose.pitch_rad == pytest.approx(-0.01)
    assert pose.yaw_uncertainty_rad == pytest.approx(0.0005)


def test_pose_history_rejects_unbounded_or_unbracketed_capture() -> None:
    history = CameraPoseHistory()
    history.observe(_state(1_000_000_000, 0.0, 0.0))
    history.observe(_state(1_100_000_000, 0.01, 0.0))
    assert history.at(TimeInterval(1_020_000_000, 1_050_000_000))[1] == "capture_clock_interval_too_wide"
    assert history.at(TimeInterval(1_095_000_000, 1_105_000_000))[1] == "capture_not_bracketed_by_encoder"
    assert not history.observe(_state(1_050_000_000, 0.0, 0.0))


def test_pose_history_rejects_fast_motion_uncertainty() -> None:
    history = CameraPoseHistory(max_motion_uncertainty_rad=0.003)
    history.observe(_state(1_000_000_000, 0.0, 0.0))
    history.observe(_state(1_100_000_000, 0.2, 0.0))
    assert history.at(TimeInterval(1_045_000_000, 1_055_000_000))[1] == "pose_motion_uncertainty_exceeded"


def test_pose_history_rejects_sparse_encoder_bracket_even_if_stationary() -> None:
    history = CameraPoseHistory()
    history.observe(_state(1_000_000_000, 0.0, 0.0))
    history.observe(_state(1_200_000_000, 0.0, 0.0))
    assert history.at(TimeInterval(1_095_000_000, 1_105_000_000))[1] == "pose_bracket_too_sparse"


def test_render_pose_source_uses_rendered_not_encoder_angles_and_rejects_old_prediction() -> None:
    history = CameraPoseHistory(pose_source="render")
    for timestamp, render_pan in ((1_000_000_000, 0.02), (1_050_000_000, 0.04)):
        assert history.observe(CamState(
            frame_id=1, src_ts_ms=0, state_monotonic_ns=timestamp,
            pan=0.0, tilt=0.0,
            render_pan=render_pan, render_tilt=-render_pan,
            render_prediction_age_ms=10.0,
        ))
    pose, reason = history.at(TimeInterval(1_024_000_000, 1_026_000_000))
    assert reason == "aligned"
    assert pose is not None and pose.yaw_rad == pytest.approx(0.03)
    assert pose.pitch_rad == pytest.approx(-0.03)
    assert not history.observe(CamState(
        frame_id=2, src_ts_ms=0, state_monotonic_ns=1_100_000_000,
        pan=0.0, tilt=0.0,
        render_pan=0.05, render_tilt=-0.05,
        render_prediction_age_ms=101.0,
    ))
