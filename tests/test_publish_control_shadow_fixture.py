from __future__ import annotations

from common.control import AxisPair, ControlConfig, LaserAimingControlConfig
from jetson.control_observation import ControlObservationAssembler
from tools.publish_control_shadow_fixture import _manual, _snapshot


def _config() -> ControlConfig:
    return ControlConfig(
        mode="rate", loop_hz=50.0, fx_px=800.0, fy_px=820.0,
        cx_px=640.0, cy_px=360.0, aim_mode="camera_center",
        kp=AxisPair(0.0, 0.0), kd=AxisPair(0.0, 0.0), ki=AxisPair(0.0, 0.0),
        rate_limits=AxisPair(0.5, 0.5), accel_limits=AxisPair(3.5, 3.5),
        deadband_px=0.0, smooth_px_alpha=0.0, lost_target_timeout_ms=100,
        reinit_on_lost=True, target_selector="preselected", yaw_sign=1.0,
        pitch_sign=-1.0, frame_size=(1280, 720), fov_deg=None,
        laser=LaserAimingControlConfig(3.0, "infinite", 10.0),
    )


def test_fixture_produces_varied_native_v2_target_without_external_rate() -> None:
    first = _snapshot(1, 1_000_000_000, 0.0)
    second = _snapshot(2, 1_020_000_000, 0.4)
    assert first.selection is not None and first.selection.track_id == 7
    assert first.tracks[0].box != second.tracks[0].box

    assembler = ControlObservationAssembler(_config())
    assembler.update_perception_snapshot(first, received_at=1.0)
    assembler.update_manual_state(_manual(1000), received_at=1.0)
    observation = assembler.build(now=1.01)
    assert observation.target.valid
    assert observation.target.bearing_rate_rad_s is None
    assert observation.safety.auto_allowed
