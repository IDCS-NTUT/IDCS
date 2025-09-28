import math

from common.tracker_z import (
    TrackingZConfig,
    TrackingZMeasurement,
    TrackingZMeasurementNoise,
    TrackingZProcessNoise,
    ZTracker,
)


def _make_config() -> TrackingZConfig:
    return TrackingZConfig(
        enabled=True,
        meas_src_priority=("known_size",),
        meas_noise=TrackingZMeasurementNoise(base_m=0.5, small_box_px=40.0),
        process_noise=TrackingZProcessNoise(z=0.3),
    )


def test_z_tracker_initialises_from_measurement() -> None:
    tracker = ZTracker(_make_config())
    measurement = TrackingZMeasurement(
        value_m=12.0,
        source="known_size",
        box_size_px=(80.0, 80.0),
        confidence=0.9,
    )
    updated = tracker.update(measurement)
    assert updated
    prediction = tracker.project(0.2)
    assert prediction is not None
    assert math.isclose(prediction.distance_m, 12.0, rel_tol=1e-3)
    assert prediction.velocity_mps == 0.0


def test_z_tracker_estimates_velocity() -> None:
    tracker = ZTracker(_make_config())
    tracker.update(
        TrackingZMeasurement(
            value_m=10.0,
            source="known_size",
            box_size_px=(100.0, 100.0),
            confidence=0.95,
        )
    )
    tracker.predict(0.1)
    tracker.update(
        TrackingZMeasurement(
            value_m=11.0,
            source="known_size",
            box_size_px=(100.0, 100.0),
            confidence=0.95,
        )
    )
    prediction = tracker.project(0.2)
    assert prediction is not None
    assert prediction.distance_m > 10.0
    assert prediction.distance_m < 12.0
    assert prediction.velocity_mps > 0.0


def test_z_config_from_raw_mapping() -> None:
    cfg = TrackingZConfig.from_raw_config(
        {
            "tracking_z": {
                "enabled": True,
                "meas_src_priority": ["ground_plane", "known_size"],
                "meas_noise_m": {"base": 0.4, "min_box_px": 30.0},
                "process_noise": {"z": 0.2},
            }
        }
    )
    assert cfg.enabled is True
    assert cfg.meas_src_priority == ("ground_plane", "known_size")
    assert math.isclose(cfg.meas_noise.base_m, 0.4)
    assert math.isclose(cfg.meas_noise.small_box_px, 30.0)
    assert math.isclose(cfg.process_noise.z, 0.2)
