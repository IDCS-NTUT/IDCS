"""Regression tests for the CPU billboard renderer safeguards."""

from __future__ import annotations

import types
import unittest
from typing import Any, Dict
from unittest import mock

import numpy as np

from pc.renderers.cpu import CPURenderer


def _build_renderer(width: int = 640, height: int = 480) -> CPURenderer:
    context = types.SimpleNamespace(width=width, height=height, world_up=(0.0, 1.0, 0.0))
    renderer = CPURenderer(context=context)

    sprite_bgr = np.full((8, 8, 3), 200, dtype=np.uint8)
    sprite_alpha = np.full((8, 8), 255, dtype=np.uint8)

    def _fake_sprite(_: Any) -> tuple[np.ndarray, np.ndarray]:
        return sprite_bgr, sprite_alpha

    renderer._get_sprite_image = types.MethodType(_fake_sprite, renderer)
    return renderer


def _build_camera(renderer: CPURenderer) -> Dict[str, Any]:
    camera_state = {
        "position": (0.0, 0.0, 0.0),
        "target": (0.0, 0.0, 1.0),
        "up": (0.0, 1.0, 0.0),
        "fov_y": 60.0,
    }
    camera = renderer._build_camera(camera_state)
    assert camera is not None
    return camera


class CPUBillboardSafetyTest(unittest.TestCase):
    def test_offscreen_billboard_skips_warp(self) -> None:
        renderer = _build_renderer()
        camera = _build_camera(renderer)
        frame = np.zeros((renderer.height, renderer.width, 3), dtype=np.uint8)

        billboard = {
            "centre": (5000.0, 0.0, 25.0),
            "size": (6.0, 6.0),
            "sprite": "dummy",
        }

        with mock.patch("cv2.warpPerspective") as warp_mock:
            renderer._draw_billboard(frame, camera, billboard)

        warp_mock.assert_not_called()

    def test_enormous_roi_skips_warp(self) -> None:
        renderer = _build_renderer()
        camera = _build_camera(renderer)
        frame = np.zeros((renderer.height, renderer.width, 3), dtype=np.uint8)

        billboard = {
            "centre": (0.0, 0.0, 15.0),
            "size": (20000.0, 20000.0),
            "sprite": "dummy",
        }

        with mock.patch("cv2.warpPerspective") as warp_mock:
            renderer._draw_billboard(frame, camera, billboard)

        warp_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()

