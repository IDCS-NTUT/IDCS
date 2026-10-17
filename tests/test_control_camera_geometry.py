from __future__ import annotations

import pytest

from common.camera import CameraIntrinsics
from common.config import load_config_bundle, resolve_config_paths
from common.control import ControlConfig


def _base() -> dict:
    return load_config_bundle(resolve_config_paths('configs/base')).mutable_copy()


def test_aiming_uses_camera_fov_and_preserves_wide_angle_calibration() -> None:
    cfg = _base()
    aiming = ControlConfig.from_raw_config(cfg, (1280, 720))
    assert aiming.fov_deg == (135.0, 73.0)
    assert aiming.fx_px == pytest.approx(265.09667991878085)
    assert aiming.fy_px == pytest.approx(486.512077908)
    cfg['camera']['intrinsics']['fov_deg'] = {'h': 90.0, 'v': 90.0}
    aiming = ControlConfig.from_raw_config(cfg, (1280, 720))
    assert (aiming.fx_px, aiming.fy_px) == pytest.approx((640.0, 360.0))


def test_direct_intrinsics_follow_snapshot_resize() -> None:
    cfg = _base()
    cfg['camera']['intrinsics'] = {'source': 'direct', 'fx_px': 900.0, 'fy_px': 850.0,
                                   'cx_px': 620.0, 'cy_px': 350.0}
    aiming = ControlConfig.from_raw_config(cfg, (1280, 720)).for_frame_size((640, 360))
    assert (aiming.fx_px, aiming.fy_px, aiming.cx_px, aiming.cy_px) == (450, 425, 310, 175)
    assert aiming.laser.tolerance_px == 10.0


def test_simulated_camera_override_does_not_mutate_calibration() -> None:
    cfg = _base()
    intrinsics = CameraIntrinsics.from_vertical_fov((1920, 1080), 60.0)
    aiming = ControlConfig.from_raw_config(cfg, (1920, 1080), camera_intrinsics=intrinsics)
    assert (aiming.fx_px, aiming.fy_px) == pytest.approx((935.3074360871939, 935.3074360871939))
    assert aiming.fov_deg == pytest.approx((91.49284451967722, 60.0))
    assert cfg['camera']['intrinsics']['fov_deg'] == {'h': 135.0, 'v': 73.0}
