from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from common.schemas import ControlObservation
from jetson.control_v3.pid import AxisPIDConfig, BasicPID
from jetson.control_v3.timing import ClockBounds
from jetson.control_v3.video_controller import VideoControllerCore, VideoControllerPolicy
from jetson.control_v3.video_feedforward import VideoFeedforwardEstimate
from tools.replay_control_v3_pid import _merge


FIXTURE = Path(__file__).parent / "fixtures" / "control_v3_pid_replay_v1.json"


def _controller(*, live: bool, scale: float) -> VideoControllerCore:
    axis = AxisPIDConfig(1.0, 0.0, 0.0, 0.0, 0.2, 3.5)
    controller = VideoControllerCore(
        BasicPID(axis, axis),
        VideoControllerPolicy(feedforward_scale=scale, live_authorized=live),
    )
    controller.feedforward.estimate = Mock(return_value=VideoFeedforwardEstimate(
        True, "ready", 0.05, -0.02,
    ))
    return controller


def _steps() -> tuple[list[ControlObservation], list[ClockBounds]]:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    exchange = ClockBounds.from_exchange(**fixture["clock_exchange"])
    observations = [ControlObservation.model_validate(_merge(
        fixture["base_observation"], step["observation"]
    )) for step in fixture["steps"][:3]]
    clocks = [ClockBounds(
        exchange.offset_min_ns, exchange.offset_max_ns,
        observation.created_monotonic_ns, 0.0,
    ) for observation in observations]
    return observations, clocks


def test_video_core_keeps_feedforward_separate_and_shadow_expired() -> None:
    observations, clocks = _steps()
    baseline = _controller(live=False, scale=0.0)
    combined = _controller(live=False, scale=0.5)
    for observation, clock in zip(observations[:2], clocks[:2]):
        off = baseline.decide(observation, clock)
        on = combined.decide(observation, clock)
    assert on.feedforward.valid
    assert on.applied_feedforward_rad_s == (0.025, -0.01)
    assert on.pid.pid.yaw.feedforward_rad_s == 0.025
    assert off.pid.pid.yaw.feedforward_rad_s == 0.0
    assert on.intent.yaw_rate_rad_s > off.intent.yaw_rate_rad_s
    assert on.intent.mode == "shadow"
    assert on.intent.valid_until_monotonic_ns == on.intent.issued_monotonic_ns


def test_explicit_per_tick_feedforward_scale_changes_only_separate_term() -> None:
    observations, clocks = _steps()
    controller = _controller(live=False, scale=0.0)
    controller.decide(observations[0], clocks[0], feedforward_scale=0.0)
    on = controller.decide(observations[1], clocks[1], feedforward_scale=0.5)
    assert on.applied_feedforward_rad_s == (0.025, -0.01)
    with pytest.raises(ValueError, match="feedforward scale"):
        controller.decide(observations[1], clocks[1], feedforward_scale=1.5)


def test_live_candidate_stops_on_unverified_frame_and_never_reuses_ff() -> None:
    observations, clocks = _steps()
    controller = _controller(live=True, scale=0.5)
    controller.decide(observations[0], clocks[0])
    valid = controller.decide(observations[1], clocks[1])
    assert valid.intent.mode == "live"
    assert valid.intent.reason == "tracking"
    assert valid.intent.yaw_rate_rad_s > 0
    assert valid.intent.valid_until_monotonic_ns - valid.intent.issued_monotonic_ns == 50_000_000
    invalid = controller.decide(observations[2], clocks[2])
    assert invalid.intent.mode == "live"
    assert invalid.intent.reason == "frame_identity_unverified"
    assert invalid.intent.yaw_rate_rad_s == invalid.intent.pitch_rate_rad_s == 0.0
    assert invalid.applied_feedforward_rad_s == (0.0, 0.0)


def test_video_core_without_clock_holds_even_if_estimator_claims_ready() -> None:
    observations, _ = _steps()
    controller = _controller(live=True, scale=0.5)
    result = controller.decide(observations[0], None)
    assert result.intent.reason == "clock_unavailable"
    assert result.intent.yaw_rate_rad_s == result.intent.pitch_rate_rad_s == 0.0


def test_video_capture_age_policy_is_explicitly_bounded() -> None:
    assert VideoControllerPolicy(max_capture_age_ns=250_000_000).max_capture_age_ns == 250_000_000
    with pytest.raises(ValueError, match="capture age"):
        VideoControllerPolicy(max_capture_age_ns=251_000_000)


def test_live_travel_envelope_stops_projected_outward_command() -> None:
    observations, clocks = _steps()
    controller = _controller(live=True, scale=0.5)
    controller.decide(observations[0], clocks[0])
    near_limit = observations[1].model_copy(update={
        "gimbal": observations[1].gimbal.model_copy(update={"yaw_rad": 0.149}),
    })
    decision = controller.decide(near_limit, clocks[1])
    assert decision.intent.reason == "travel_limit_hold"
    assert decision.intent.yaw_rate_rad_s == decision.intent.pitch_rate_rad_s == 0.0
    assert decision.applied_feedforward_rad_s == (0.0, 0.0)


def test_live_manual_safety_loss_stops_both_axes() -> None:
    observations, clocks = _steps()
    controller = _controller(live=True, scale=0.5)
    controller.decide(observations[0], clocks[0])
    denied = observations[1].model_copy(update={
        "safety": observations[1].safety.model_copy(update={"auto_allowed": False}),
    })
    decision = controller.decide(denied, clocks[1])
    assert decision.intent.reason == "safety_hold"
    assert decision.intent.yaw_rate_rad_s == decision.intent.pitch_rate_rad_s == 0.0


def test_live_sim_frame_pose_is_required_and_must_be_fresh() -> None:
    observations, clocks = _steps()
    axis = AxisPIDConfig(1.0, 0.0, 0.0, 0.0, 0.2, 3.5)
    controller = VideoControllerCore(
        BasicPID(axis, axis),
        VideoControllerPolicy(live_authorized=True, pose_source="frame"),
    )
    controller.feedforward.estimate = Mock(side_effect=lambda obs, _clock, **_kw:
        VideoFeedforwardEstimate(
            True, "ready", capture_midpoint_ns=obs.created_monotonic_ns - 60_000_000,
        ))
    controller.decide(
        observations[0], clocks[0], frame_pose_rad=(0.0, 0.0),
        frame_camstate_ns=observations[0].created_monotonic_ns - 80_000_000,
    )
    missing = controller.decide(observations[1], clocks[1])
    assert missing.intent.reason == "sim_capture_pose_hold"
    assert missing.intent.yaw_rate_rad_s == missing.intent.pitch_rate_rad_s == 0.0
    controller.reset()
    controller.decide(
        observations[0], clocks[0], frame_pose_rad=(0.0, 0.0),
        frame_camstate_ns=observations[0].created_monotonic_ns - 80_000_000,
    )
    stale = controller.decide(
        observations[1], clocks[1], frame_pose_rad=(0.0, 0.0),
        frame_camstate_ns=observations[1].created_monotonic_ns - 200_000_000,
    )
    assert stale.intent.reason == "sim_capture_pose_hold"
    controller.reset()
    controller.decide(
        observations[0], clocks[0], frame_pose_rad=(0.0, 0.0),
        frame_camstate_ns=observations[0].created_monotonic_ns - 80_000_000,
    )
    fresh = controller.decide(
        observations[1], clocks[1], frame_pose_rad=(0.0, 0.0),
        frame_camstate_ns=observations[1].created_monotonic_ns - 80_000_000,
    )
    assert fresh.intent.reason == "tracking"
