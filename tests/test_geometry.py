import math

import pytest

from common.geometry import (
    camera_vector_to_world,
    pixel_motion_from_camera_rotation,
    pixel_to_world_point,
    project_world_point_to_pixel,
    world_vector_to_camera,
    world_velocity_to_pixel_velocity,
)


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


def test_world_camera_inverse():
    vec = (1.2, 0.4, 3.5)
    pan = math.radians(25.0)
    tilt = math.radians(-12.0)
    cam_vec = world_vector_to_camera(vec, pan_rad=pan, tilt_rad=tilt)
    world_vec = camera_vector_to_world(cam_vec, pan_rad=pan, tilt_rad=tilt)
    assert math.isclose(world_vec[0], vec[0], rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(world_vec[1], vec[1], rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(world_vec[2], vec[2], rel_tol=1e-6, abs_tol=1e-6)


def test_pixel_world_round_trip():
    fx = fy = 900.0
    cx = 640.0
    cy = 360.0
    pan = math.radians(15.0)
    tilt = math.radians(-8.0)
    u = 720.0
    v = 280.0
    distance = 12.0

    world_point = pixel_to_world_point(
        u,
        v,
        distance_m=distance,
        fx_px=fx,
        fy_px=fy,
        cx_px=cx,
        cy_px=cy,
        pan_rad=pan,
        tilt_rad=tilt,
    )
    u_back, v_back = project_world_point_to_pixel(
        world_point,
        fx_px=fx,
        fy_px=fy,
        cx_px=cx,
        cy_px=cy,
        pan_rad=pan,
        tilt_rad=tilt,
    )
    assert math.isclose(u_back, u, abs_tol=1e-6)
    assert math.isclose(v_back, v, abs_tol=1e-6)


def test_world_velocity_projection_matches_finite_difference():
    fx = fy = 900.0
    cx = 640.0
    cy = 360.0
    pan = math.radians(5.0)
    tilt = math.radians(3.0)
    position = (2.0, 1.0, 18.0)
    velocity = (0.5, -0.2, -1.0)

    du_dt, dv_dt = world_velocity_to_pixel_velocity(
        position,
        velocity,
        fx_px=fx,
        fy_px=fy,
        cx_px=cx,
        cy_px=cy,
        pan_rad=pan,
        tilt_rad=tilt,
    )

    dt = 1e-3
    next_position = (
        position[0] + velocity[0] * dt,
        position[1] + velocity[1] * dt,
        position[2] + velocity[2] * dt,
    )
    u0, v0 = project_world_point_to_pixel(
        position,
        fx_px=fx,
        fy_px=fy,
        cx_px=cx,
        cy_px=cy,
        pan_rad=pan,
        tilt_rad=tilt,
    )
    u1, v1 = project_world_point_to_pixel(
        next_position,
        fx_px=fx,
        fy_px=fy,
        cx_px=cx,
        cy_px=cy,
        pan_rad=pan,
        tilt_rad=tilt,
    )
    fd_du = (u1 - u0) / dt
    fd_dv = (v1 - v0) / dt

    assert math.isclose(du_dt, fd_du, rel_tol=1e-2, abs_tol=1e-2)
    assert math.isclose(dv_dt, fd_dv, rel_tol=1e-2, abs_tol=1e-2)
