import math

from common.tracker import (
    TrackingConfig,
    TrackingMeasurementNoise,
    TrackingProcessNoise,
    TrackingWorldParams,
)
from common.tracker_world import WorldTracker, WorldTrackerMeasurement


def _make_world_config() -> TrackingConfig:
    return TrackingConfig(
        enabled=True,
        model="world_cv",
        predict_horizon_ms=120.0,
        use_camera_derotation=False,
        meas_noise=TrackingMeasurementNoise(base_px=2.0, min_box_px=40.0),
        process_noise=TrackingProcessNoise(u=0.5, v=0.5),
        gate_chi2=11.34,
        reset_on_target_switch=True,
        warmup_measurements=1,
        warmup_velocity_std_px=0.0,
        world=TrackingWorldParams(process_noise_accel=0.75, meas_noise_pos_m=1.0),
    )


def test_world_tracker_initialises_from_measurement():
    tracker = WorldTracker(_make_world_config())
    measurement = WorldTrackerMeasurement(position_m=(1.0, 0.2, 8.0), position_std_m=0.5)
    accepted = tracker.update(measurement)
    assert accepted is True
    prediction = tracker.project(0.2)
    assert prediction is not None
    assert math.isclose(prediction.position_m[2], 8.0, rel_tol=1e-6)


def test_world_tracker_prediction_after_motion():
    tracker = WorldTracker(_make_world_config())
    measurement = WorldTrackerMeasurement(position_m=(0.0, 0.0, 10.0), position_std_m=0.3)
    tracker.update(measurement)
    tracker.predict(0.1)
    prediction = tracker.project(0.5)
    assert prediction is not None
    assert prediction.horizon_s == 0.5
    assert math.isclose(prediction.position_m[2], 10.0, rel_tol=1e-6)
