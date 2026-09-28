from __future__ import annotations

import pytest

from jetson.control.feedforward import TargetRateKalman
from jetson.control.pid import AxisPIDConfig, BasicPID, PIDInput
from jetson.control.timing import TimingVerdict


def test_kalman_recovers_constant_target_rate_with_timestamps() -> None:
    estimator = TargetRateKalman()
    start = 1_000_000_000
    for index in range(80):
        t = index * 0.02
        assert estimator.observe(track_id=7, angle_rad=0.1 * t,
                                 sample_ns=start + round(t * 1e9))
    result = estimator.estimate(decision_ns=start + 1_640_000_000, track_id=7)
    assert result.valid
    assert result.rate_rad_s == pytest.approx(0.1, abs=0.01)
    assert result.sample_age_s == pytest.approx(0.06)


def test_kalman_rejects_stale_future_step_and_track_change() -> None:
    estimator = TargetRateKalman()
    for index in range(5):
        estimator.observe(track_id=1, angle_rad=index * 0.001,
                          sample_ns=1_000_000_000 + index * 20_000_000)
    assert estimator.estimate(decision_ns=1_080_000_000, track_id=1).valid
    assert estimator.estimate(decision_ns=1_300_000_000, track_id=1).reason == "sample_stale"
    assert estimator.estimate(decision_ns=1_070_000_000, track_id=1).reason == "sample_in_future"
    assert not estimator.observe(track_id=1, angle_rad=1.0, sample_ns=1_100_000_000)
    assert estimator.estimate(decision_ns=1_100_000_000, track_id=1).reason == "estimator_warmup"
    assert estimator.estimate(decision_ns=1_100_000_000, track_id=2).reason == "target_uninitialized"


def test_feedforward_is_separate_and_subject_to_pid_limits() -> None:
    config = AxisPIDConfig(2.0, 0.0, 0.0, 0.0, 0.2, 10.0)
    pid = BasicPID(config, config)
    sample = PIDInput(
        decision_ns=1_000_000_000, track_id=1, error_rad=(0.01, 0.0),
        gimbal_rate_rad_s=(0.0, 0.0), timing=TimingVerdict(True, "test"),
        safety_allowed=True, gimbal_valid=True,
        feedforward_rad_s=(0.3, 0.0),
    )
    pid.decide(sample)
    decision = pid.decide(PIDInput(**{**sample.__dict__, "decision_ns": 1_020_000_000}))
    assert decision.yaw.proportional_rad_s == pytest.approx(0.02)
    assert decision.yaw.feedforward_rad_s == pytest.approx(0.3)
    assert decision.yaw.pre_limit_rad_s == pytest.approx(0.32)
    assert decision.yaw.final_rad_s == pytest.approx(0.2)
    assert decision.yaw.rate_limited


def test_kalman_predict_coasts_past_the_feedforward_staleness_limit() -> None:
    estimator = TargetRateKalman(max_sample_age_s=0.12)
    for index in range(20):
        estimator.observe(track_id=4, angle_rad=0.3 * index * 0.02,
                          sample_ns=1_000_000_000 + index * 20_000_000)
    last_ns = 1_000_000_000 + 19 * 20_000_000
    assert estimator.estimate(decision_ns=last_ns + 400_000_000, track_id=4).reason == "sample_stale"
    coast = estimator.predict(decision_ns=last_ns + 400_000_000, max_age_s=0.5)
    assert coast.valid and coast.position_rad == pytest.approx(0.3 * (19 * 0.02 + 0.4), abs=0.01)
    assert not estimator.predict(decision_ns=last_ns + 600_000_000, max_age_s=0.5).valid
    assert estimator.track_id == 4
