import math

import pytest

from common.geometry import pixel_motion_from_camera_rotation


@pytest.mark.parametrize(
    "yaw_rate, pitch_rate, expected_sign",
    [
        (math.radians(30.0), 0.0, (-1, 0)),
        (0.0, math.radians(20.0), (0, 1)),
    ],
)
def test_pixel_motion_signs(yaw_rate, pitch_rate, expected_sign):
    du, dv = pixel_motion_from_camera_rotation(
        640.0,
        360.0,
        yaw_rate_rad_s=yaw_rate,
        pitch_rate_rad_s=pitch_rate,
        dt_s=0.05,
        fx_px=900.0,
        fy_px=900.0,
        cx_px=640.0,
        cy_px=360.0,
    )
    sign_u = 0 if abs(du) < 1e-6 else int(math.copysign(1, du))
    sign_v = 0 if abs(dv) < 1e-6 else int(math.copysign(1, dv))
    assert (sign_u, sign_v) == expected_sign


def test_pixel_motion_zero_dt():
    du, dv = pixel_motion_from_camera_rotation(
        100.0,
        200.0,
        yaw_rate_rad_s=0.5,
        pitch_rate_rad_s=-0.5,
        dt_s=0.0,
        fx_px=800.0,
        fy_px=800.0,
        cx_px=320.0,
        cy_px=240.0,
    )
    assert du == 0.0 and dv == 0.0
