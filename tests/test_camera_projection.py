from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from pc.renderers._camera import build_camera_description, project_point
from pc.renderers.cpu import CPURenderer
from pc.sim_camera import SimCamera


def test_camera_projection_matches_cpu_renderer() -> None:
    camera = SimCamera(width=640, height=360)
    world = camera.describe_world(frame_id=12)
    camera_state = world["camera"]

    description = build_camera_description(
        camera_state,
        camera.width,
        camera.height,
        default_up=camera.world_up,
    )
    assert description is not None

    cpu_renderer = CPURenderer(context=camera)
    cpu_camera = cpu_renderer._build_camera(camera_state)
    assert cpu_camera is not None

    target_point = np.asarray(camera_state["target"], dtype=np.float32)

    cpu_projected = cpu_renderer._project_point(cpu_camera, target_point)
    gpu_projected = project_point(description, target_point)

    assert cpu_projected is not None
    assert gpu_projected is not None
    np.testing.assert_allclose(gpu_projected, cpu_projected, rtol=1e-5, atol=1e-4)

    view_projection = description.projection_matrix @ description.view_matrix
    np.testing.assert_allclose(description.view_projection_matrix, view_projection)

    point_h = np.concatenate([target_point.astype(np.float32), np.array([1.0], dtype=np.float32)])
    clip_space = description.gl_view_projection_matrix @ point_h
    w = float(clip_space[3])
    assert not np.isclose(w, 0.0)
    ndc = clip_space[:3] / w
    gl_x = (ndc[0] + 1.0) * 0.5 * (description.image_width - 1)
    gl_y = (1.0 - (ndc[1] + 1.0) * 0.5) * (description.image_height - 1)
    np.testing.assert_allclose((gl_x, gl_y), cpu_projected, rtol=1e-5, atol=1e-4)
