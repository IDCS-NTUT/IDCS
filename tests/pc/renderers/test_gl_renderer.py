from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from pc.renderers.gl.renderer import build_camera_matrices


def test_build_camera_from_orientation() -> None:
    camera_state = {
        "position": (0.0, 0.0, 0.0),
        "orientation": {"yaw": 90.0, "pitch": 0.0, "roll": 0.0},
        "fov_y": 60.0,
    }

    matrices = build_camera_matrices(
        camera_state,
        aspect=16.0 / 9.0,
        near=0.1,
        far=100.0,
        world_up=(0.0, 1.0, 0.0),
    )

    assert matrices is not None
    np.testing.assert_allclose(matrices.forward, (-1.0, 0.0, 0.0), atol=1e-5)
    np.testing.assert_allclose(matrices.right, (0.0, 0.0, 1.0), atol=1e-5)
    np.testing.assert_allclose(matrices.up, (0.0, 1.0, 0.0), atol=1e-5)

    # Ensure the projection keeps the expected focal length relationship.
    f = 1.0 / math.tan(math.radians(60.0) / 2.0)
    assert math.isclose(matrices.projection[1, 1], f, rel_tol=1e-5)


def test_build_camera_from_target() -> None:
    camera_state = {
        "position": (0.0, 1.0, 0.0),
        "target": (0.0, 1.0, -1.0),
        "fov_y": 45.0,
    }

    matrices = build_camera_matrices(
        camera_state,
        aspect=1.0,
        near=0.1,
        far=50.0,
        world_up=(0.0, 1.0, 0.0),
    )

    assert matrices is not None
    np.testing.assert_allclose(matrices.forward, (0.0, 0.0, -1.0), atol=1e-5)
    np.testing.assert_allclose(matrices.up, (0.0, 1.0, 0.0), atol=1e-5)
    np.testing.assert_allclose(matrices.right, (1.0, 0.0, 0.0), atol=1e-5)

    # Points directly in front of the camera should have negative Z in camera space.
    view = matrices.view
    point = np.array((0.0, 1.0, -5.0, 1.0), dtype=np.float32)
    camera_space = view @ point
    assert camera_space[2] < 0.0


def test_build_camera_invalid_input() -> None:
    camera_state = {"position": (0.0, 0.0, 0.0)}
    matrices = build_camera_matrices(
        camera_state,
        aspect=1.0,
        near=0.1,
        far=100.0,
        world_up=(0.0, 1.0, 0.0),
    )
    assert matrices is None

