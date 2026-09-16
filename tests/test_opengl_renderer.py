from pathlib import Path

import numpy as np
import pytest

from pc.renderers.opengl import OpenGLRenderer, _build_unit_box
from pc.renderers.mesh import load_mesh


class _FakeContext:
    width = 160
    height = 120
    world_up = (0.0, 1.0, 0.0)
    renderer_opts = {}

    def describe_world(self, frame_id: int):
        return {
            "camera": {
                "position": (3.0, 3.0, 3.0),
                "target": (0.0, 0.0, 0.0),
                "fov_y": 60.0,
            },
            "objects": [],
        }


def test_opengl_renderer_smoke():
    ctx = _FakeContext()
    renderer = OpenGLRenderer(context=ctx)

    frame = np.zeros((ctx.height, ctx.width, 3), dtype=np.uint8)
    renderer.render(frame, frame_id=0)

    assert frame.shape == (ctx.height, ctx.width, 3)
    assert frame.dtype == np.uint8
    # Frame should not be all zeros even if OpenGL falls back to CPU/flat fill
    assert np.any(frame != 0)


def test_build_unit_box_has_uvs_for_each_face():
    vertices, indices = _build_unit_box()

    assert vertices.shape[1] == 11
    uv_columns = vertices[:, 6:8]
    assert np.unique(uv_columns, axis=0).shape[0] == 4
    assert np.all((uv_columns >= 0.0) & (uv_columns <= 1.0))
    assert indices.size == 36


def test_resolve_texture_path_uses_configured_search_paths(tmp_path):
    textures_dir = tmp_path / "assets" / "textures"
    textures_dir.mkdir(parents=True)
    texture_path = textures_dir / "ground_diffuse.png"
    texture_path.write_bytes(b"placeholder")

    renderer = OpenGLRenderer.__new__(OpenGLRenderer)
    renderer._repo_root = tmp_path
    renderer._context = type("Ctx", (), {"renderer_opts": {}})()
    renderer._renderer_cfg_path = tmp_path / "configs" / "renderer.yaml"
    renderer._renderer_cfg_mtime_ns = -1
    renderer._renderer_cfg = {"pbr": {"texture_search_paths": ["assets/textures"]}}

    resolved = renderer._resolve_texture_path("ground_diffuse.png")

    assert resolved == texture_path


def test_target_render_mode_can_preserve_detector_billboards():
    ctx = _FakeContext()
    ctx.renderer_opts = {"target_render_mode": "billboard"}
    renderer = OpenGLRenderer(context=ctx)

    assert renderer._uses_billboard_target({"type": "target", "sprite": "person"})
    assert not renderer._uses_billboard_target(
        {"type": "target", "sprite": "person", "render_mode": "mesh"}
    )


def test_person_mesh_bounds_are_centered_for_world_transform():
    asset = Path(__file__).resolve().parents[1] / "assets" / "meshes" / "person.obj"
    mesh = load_mesh(str(asset))
    midpoint = 0.5 * (mesh.vertices.min(axis=0) + mesh.vertices.max(axis=0))

    assert np.allclose(midpoint, np.zeros(3), atol=1e-5)


def test_person_target_scale_honors_requested_height_without_distortion():
    renderer = OpenGLRenderer.__new__(OpenGLRenderer)
    scale = renderer._target_mesh_scale(
        {"extents": (0.25, 0.8, 0.2)}, "person", width=0.7, height=1.6
    )

    assert scale == pytest.approx((2.0, 2.0, 2.0))
