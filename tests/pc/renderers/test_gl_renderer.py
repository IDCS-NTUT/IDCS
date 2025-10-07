from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from pc.renderers.gl.assets import build_billboard_geometry, build_box_geometry
from pc.renderers.gl.renderer import build_camera_matrices, build_scene_primitives


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


def test_build_box_geometry_shapes() -> None:
    geometry = build_box_geometry((1.0, 2.0, 3.0))
    assert geometry.positions.shape == (24, 3)
    assert geometry.normals.shape == (24, 3)
    assert geometry.uvs.shape == (24, 2)
    assert geometry.indices.shape == (36,)
    assert np.isclose(geometry.positions[:, 0].max(), 1.0)
    assert np.isclose(geometry.positions[:, 0].min(), -1.0)
    assert np.isclose(geometry.positions[:, 1].max(), 2.0)
    assert np.isclose(geometry.positions[:, 1].min(), -2.0)
    assert np.isclose(geometry.positions[:, 2].max(), 3.0)
    assert np.isclose(geometry.positions[:, 2].min(), -3.0)
    lengths = np.linalg.norm(geometry.normals, axis=1)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-6)


def test_build_billboard_geometry_shapes() -> None:
    geometry = build_billboard_geometry(2.0, 4.0)
    assert geometry.positions.shape == (4, 3)
    np.testing.assert_allclose(geometry.positions[:, 2], 0.0, atol=1e-6)
    np.testing.assert_allclose(
        geometry.positions[:, 0],
        (-1.0, 1.0, 1.0, -1.0),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        geometry.positions[:, 1],
        (-2.0, -2.0, 2.0, 2.0),
        atol=1e-6,
    )
    assert geometry.indices.tolist() == [0, 1, 2, 0, 2, 3]


def test_build_scene_primitives_from_world() -> None:
    world = {
        "camera": {
            "position": (5.0, 5.0, 5.0),
            "target": (0.0, 0.0, 0.0),
            "fov_y": 60.0,
        },
        "objects": [
            {
                "type": "building",
                "base_centre": (0.0, 0.0),
                "footprint": (4.0, 6.0),
                "height": 10.0,
                "color": (128, 200, 220),
            },
            {
                "type": "billboard",
                "centre": (2.0, 1.0, -3.0),
                "size": (1.0, 2.0),
            },
        ],
    }

    camera = build_camera_matrices(
        world["camera"],
        aspect=16.0 / 9.0,
        near=0.1,
        far=100.0,
        world_up=(0.0, 1.0, 0.0),
    )

    assert camera is not None

    primitives = build_scene_primitives(world, camera, world_up=(0.0, 1.0, 0.0))
    assert len(primitives) == 2

    building = next(p for p in primitives if p.mesh == "box")
    billboard = next(p for p in primitives if p.mesh == "billboard")

    np.testing.assert_allclose(building.model_matrix[:3, 3], (0.0, 5.0, 0.0), atol=1e-6)
    assert building.dimensions == pytest.approx((2.0, 5.0, 3.0))
    expected_colour = tuple(c / 255.0 for c in (128.0, 200.0, 220.0))
    assert building.colour == pytest.approx(expected_colour)

    normal = billboard.model_matrix[:3, 2]
    direction = np.asarray(camera.position[:3], dtype=np.float32) - billboard.model_matrix[:3, 3]
    normal = normal / np.linalg.norm(normal)
    direction = direction / np.linalg.norm(direction)
    np.testing.assert_allclose(normal, direction, atol=1e-5)
    assert not billboard.lighting

