from dataclasses import FrozenInstanceError

import pytest

from common.camera import CameraIntrinsics
from common.ranging import KnownSizeRangingConfig, estimate_normalized_distance


def _config(*, aspect_limits=()):
    return KnownSizeRangingConfig(
        enabled=True,
        dimension="average",
        class_sizes_m={"drone": 0.5},
        min_pixels=1.0,
        ema_alpha=0.5,
        class_aspect_ratio_limits=dict(aspect_limits),
    )


def test_normalized_range_is_exact_and_immutable():
    intrinsics = CameraIntrinsics(
        fx_px=800.0,
        fy_px=820.0,
        cx_px=640.0,
        cy_px=360.0,
        fov_deg=None,
    )

    result = estimate_normalized_distance(
        class_id="0",
        width_norm=0.1,
        height_norm=0.2,
        frame_size=(1280, 720),
        label_map={"0": "drone"},
        intrinsics=intrinsics,
        config=_config(),
    )

    assert result is not None
    expected_height_m = 0.5 * 820.0 / 144.0
    expected_width_m = 0.5 * 800.0 / 128.0
    assert result.class_label == "drone"
    assert result.source == "average"
    assert result.distance_m == pytest.approx(
        (expected_height_m + expected_width_m) / 2.0
    )
    assert result.pixel_size_px == pytest.approx((144.0 + 128.0) / 2.0)
    with pytest.raises(FrozenInstanceError):
        result.distance_m = 1.0


def test_normalized_range_rejects_out_of_contract_aspect_ratio():
    result = estimate_normalized_distance(
        class_id="drone",
        width_norm=0.2,
        height_norm=0.05,
        frame_size=(1280, 720),
        label_map={},
        intrinsics=CameraIntrinsics(800.0, 820.0, 640.0, 360.0, None),
        config=_config(aspect_limits=(("drone", (0.5, 2.0)),)),
    )

    assert result is None
