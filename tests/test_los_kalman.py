from __future__ import annotations

import pytest

from jetson.los_kalman import AxisLOSKalman, LOSKalmanConfig, TargetLOSKalman


def test_axis_estimator_recovers_constant_rate_and_predicts_to_query_time() -> None:
    estimator = AxisLOSKalman(LOSKalmanConfig(acceleration_spectral_density=0.01, measurement_variance_rad2=1e-6))
    for index in range(30):
        time_s = index * 0.04
        estimator.update(0.2 + 0.3 * time_s, sample_time_s=time_s)

    estimate = estimator.estimate(query_time_s=1.3)

    assert estimate is not None
    assert estimate.rate_rad_s == pytest.approx(0.3, abs=0.01)
    assert estimate.angle_rad == pytest.approx(0.2 + 0.3 * 1.3, abs=0.01)
    assert estimate.angle_variance_rad2 >= 0.0
    assert estimate.rate_variance_rad2_s2 >= 0.0
    assert estimate.last_update_accepted is True
    assert estimate.normalized_innovation_squared is not None


def test_target_switch_resets_rate_instead_of_cross_contaminating_tracks() -> None:
    estimator = TargetLOSKalman(LOSKalmanConfig())
    estimator.update(track_id=1, absolute_bearing_rad=(0.0, 0.0), sample_time_s=0.0)
    estimator.update(track_id=1, absolute_bearing_rad=(0.1, -0.1), sample_time_s=0.1)
    estimator.update(track_id=2, absolute_bearing_rad=(1.0, 2.0), sample_time_s=0.2)

    estimate = estimator.estimate(query_time_s=0.2)

    assert estimate is not None
    assert estimate[0].angle_rad == pytest.approx(1.0)
    assert estimate[0].rate_rad_s == 0.0
    assert estimate[1].angle_rad == pytest.approx(2.0)


def test_axis_estimator_rejects_out_of_order_measurement() -> None:
    estimator = AxisLOSKalman(LOSKalmanConfig())
    estimator.update(0.0, sample_time_s=1.0)

    with pytest.raises(ValueError, match="increase"):
        estimator.update(0.1, sample_time_s=1.0)


def test_large_innovation_is_gated() -> None:
    estimator = AxisLOSKalman(LOSKalmanConfig(innovation_gate_nis=9.0))
    estimator.update(0.0, sample_time_s=0.0)

    accepted = estimator.update(10.0, sample_time_s=0.02)
    estimate = estimator.estimate(query_time_s=0.02)

    assert accepted is False
    assert estimate is not None
    assert abs(estimate.angle_rad) < 0.1
    assert estimate.rejected_updates == 1
    assert estimate.last_update_accepted is False
    assert estimate.normalized_innovation_squared is not None
    assert estimate.normalized_innovation_squared > 9.0


def test_persistent_innovation_reinitializes_after_single_spike_is_rejected() -> None:
    estimator = AxisLOSKalman(LOSKalmanConfig(max_consecutive_rejections=2))
    estimator.update(0.0, sample_time_s=0.0)
    assert estimator.update(1.0, sample_time_s=0.02) is False

    assert estimator.update(1.01, sample_time_s=0.04) is True
    estimate = estimator.estimate(query_time_s=0.04)

    assert estimate is not None
    assert estimate.angle_rad == pytest.approx(1.01)
    assert estimate.rate_rad_s == 0.0
    assert estimate.reinitialized_updates == 1
    assert estimate.last_update_reinitialized is True
