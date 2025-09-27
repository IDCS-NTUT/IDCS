"""Unit tests for geometric helper functions."""

from __future__ import annotations

import unittest

from common.geometry import (
    intersect_ray_with_depth,
    laser_ray_to_pixel,
    pixel_to_camera_ray,
    project_point_to_pixel,
)


class GeometryHelpersTest(unittest.TestCase):
    fx = 800.0
    fy = 820.0
    cx = 640.0
    cy = 360.0

    def test_pixel_to_camera_ray_center_aligned(self) -> None:
        ray = pixel_to_camera_ray(640.0, 360.0, fx_px=self.fx, fy_px=self.fy, cx_px=self.cx, cy_px=self.cy)
        self.assertAlmostEqual(ray[0], 0.0)
        self.assertAlmostEqual(ray[1], 0.0)
        self.assertAlmostEqual(ray[2], 1.0)

    def test_pixel_projection_round_trip(self) -> None:
        ray = pixel_to_camera_ray(712.0, 402.0, fx_px=self.fx, fy_px=self.fy, cx_px=self.cx, cy_px=self.cy)
        depth = 12.5
        scale = depth / ray[2]
        point = (ray[0] * scale, ray[1] * scale, depth)

        u, v = project_point_to_pixel(point, fx_px=self.fx, fy_px=self.fy, cx_px=self.cx, cy_px=self.cy)
        self.assertAlmostEqual(u, 712.0)
        self.assertAlmostEqual(v, 402.0)

    def test_intersect_ray_with_depth(self) -> None:
        origin = (0.1, -0.05, 0.0)
        direction = pixel_to_camera_ray(640.0, 360.0, fx_px=self.fx, fy_px=self.fy, cx_px=self.cx, cy_px=self.cy)
        hit = intersect_ray_with_depth(origin, direction, 10.0)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertAlmostEqual(hit[0], origin[0] + direction[0] * (10.0 - origin[2]) / direction[2])
        self.assertAlmostEqual(hit[2], 10.0)

    def test_intersect_ray_parallel_to_plane(self) -> None:
        origin = (0.0, 0.0, 0.0)
        direction = (1.0, 0.0, 0.0)
        self.assertIsNone(intersect_ray_with_depth(origin, direction, 5.0))

    def test_laser_ray_projection_with_offset(self) -> None:
        origin = (0.08, 0.0, 0.0)
        direction = pixel_to_camera_ray(640.0, 360.0, fx_px=self.fx, fy_px=self.fy, cx_px=self.cx, cy_px=self.cy)
        pixel = laser_ray_to_pixel(
            origin,
            direction,
            fx_px=self.fx,
            fy_px=self.fy,
            cx_px=self.cx,
            cy_px=self.cy,
            depth_m=15.0,
        )
        self.assertIsNotNone(pixel)
        assert pixel is not None

        expected_u = self.fx * ((origin[0]) / 15.0) + self.cx
        self.assertAlmostEqual(pixel[0], expected_u)
        self.assertAlmostEqual(pixel[1], self.cy)


if __name__ == "__main__":
    unittest.main()

