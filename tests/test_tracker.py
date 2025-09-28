import math

from common.control import AxisPair
from common.tracker import (
    PixelTracker,
    TrackingConfig,
    TrackerMeasurement,
    TrackingMeasurementNoise,
    TrackingProcessNoise,
)


def _make_config() -> TrackingConfig:
    return TrackingConfig(
        enabled=True,
        model="cv",
        predict_horizon_ms=120.0,
        use_camera_derotation=True,
        meas_noise=TrackingMeasurementNoise(base_px=2.0, min_box_px=40.0),
        process_noise=TrackingProcessNoise(u=0.5, v=0.5),
        gate_chi2=9.21,
        reset_on_target_switch=True,
        warmup_measurements=1,
        warmup_velocity_std_px=0.0,
    )


def test_tracker_initialises_from_measurement():
    tracker = PixelTracker(_make_config(), fx_px=900.0, fy_px=900.0, cx_px=640.0, cy_px=360.0)
    meas = TrackerMeasurement(uv=(650.0, 360.0), box_size_px=(80.0, 80.0), confidence=0.9)
    updated = tracker.update(meas, cam_rates=AxisPair(0.0, 0.0))
    assert updated
    pred = tracker.project(0.1, AxisPair(0.0, 0.0))
    assert pred is not None
    assert math.isclose(pred.uv[0], 650.0, abs_tol=1e-3)
    assert math.isclose(pred.uv[1], 360.0, abs_tol=1e-3)


def test_tracker_rejects_large_outlier():
    tracker = PixelTracker(_make_config(), fx_px=900.0, fy_px=900.0, cx_px=640.0, cy_px=360.0)
    tracker.update(TrackerMeasurement(uv=(640.0, 360.0), box_size_px=(80.0, 80.0), confidence=0.9), cam_rates=AxisPair(0.0, 0.0))
    tracker.predict(0.05, AxisPair(0.0, 0.0))
    accepted = tracker.update(
        TrackerMeasurement(uv=(940.0, 700.0), box_size_px=(10.0, 10.0), confidence=0.2),
        cam_rates=AxisPair(0.0, 0.0),
    )
    assert not accepted
    pred = tracker.project(0.0, AxisPair(0.0, 0.0))
    assert pred is not None
    assert math.isclose(pred.uv[0], 640.0, abs_tol=1e-3)


def test_tracker_applies_camera_rotation():
    tracker = PixelTracker(_make_config(), fx_px=900.0, fy_px=900.0, cx_px=640.0, cy_px=360.0)
    tracker.update(TrackerMeasurement(uv=(640.0, 360.0), box_size_px=(80.0, 80.0), confidence=0.9), cam_rates=AxisPair(0.0, 0.0))
    tracker.predict(0.1, AxisPair(math.radians(30.0), 0.0))
    pred = tracker.project(0.0, AxisPair(0.0, 0.0))
    assert pred is not None
    assert pred.uv[0] < 640.0  # rotation to the right shifts prediction left
