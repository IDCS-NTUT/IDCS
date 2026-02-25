import numpy as np
import pytest

from pc.renderers.opengl import OpenGLRenderer


class _FakeContext:
    width = 160
    height = 120
    world_up = (0.0, 1.0, 0.0)

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
