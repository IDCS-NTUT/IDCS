from __future__ import annotations

import pytest

from common.schemas import CamState
from jetson.control.pose_history import CameraPoseHistory
from jetson.control.timing import TimeInterval


def _state(timestamp: int, pan: float, tilt: float, *, published: int | None = None,
           tilt_timestamp: int | None = None) -> CamState:
    return CamState(frame_id=1, src_ts_ms=0,
                    state_monotonic_ns=published if published is not None else timestamp,
                    pan_sample_monotonic_ns=timestamp,
                    tilt_sample_monotonic_ns=tilt_timestamp if tilt_timestamp is not None else timestamp,
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


def test_alignment_uses_measurement_time_not_publication_time() -> None:
    history = CameraPoseHistory()
    # Measured at 1.00 s and 1.04 s, but published 30 ms later each time.
    history.observe(_state(1_000_000_000, 0.0, 0.0, published=1_030_000_000))
    history.observe(_state(1_040_000_000, 0.04, 0.0, published=1_070_000_000))
    pose, reason = history.at(TimeInterval(1_020_000_000, 1_020_000_000))
    assert reason == "aligned" and pose.yaw_rad == pytest.approx(0.02)


def test_each_axis_uses_its_own_sample_times() -> None:
    history = CameraPoseHistory()
    history.observe(_state(1_000_000_000, 0.0, 0.0, tilt_timestamp=1_010_000_000))
    history.observe(_state(1_040_000_000, 0.04, 0.03, tilt_timestamp=1_070_000_000))
    pose, _ = history.at(TimeInterval(1_025_000_000, 1_025_000_000))
    assert pose.yaw_rad == pytest.approx(0.025)
    assert pose.pitch_rad == pytest.approx(0.0075)  # 15 ms of a 60 ms tilt segment


def test_republished_sample_is_not_a_new_measurement() -> None:
    history = CameraPoseHistory()
    assert history.observe(_state(1_000_000_000, 0.0, 0.0))
    assert not history.observe(_state(1_000_000_000, 0.0, 0.0, published=1_020_000_000))


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


def test_states_without_sample_times_are_not_positions() -> None:
    history = CameraPoseHistory()
    assert not history.observe(CamState(frame_id=1, src_ts_ms=0, state_monotonic_ns=1, pan=0.0, tilt=0.0))


def test_pose_at_or_latest_holds_newest_sample_beyond_the_data() -> None:
    history = CameraPoseHistory()
    history.observe(_state(1_000_000_000, 0.0, 0.0))
    history.observe(_state(1_040_000_000, 0.04, -0.04))
    assert history.pose_at_or_latest(1_020_000_000) == pytest.approx((0.02, -0.02))
    assert history.pose_at_or_latest(2_000_000_000) == pytest.approx((0.04, -0.04))
