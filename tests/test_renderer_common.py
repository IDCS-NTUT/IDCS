"""Test shared camera logic consistency between renderers."""

from __future__ import annotations

import unittest
import numpy as np
from pc.renderers._common import (
    _build_camera,
    _projection_matrix,
    _view_matrix,
    _near_clip,
    _normalise,
    _vector_length,
)


class CommonCameraLogicTest(unittest.TestCase):
    """Test shared camera logic functions."""

    def test_near_clip_constant(self) -> None:
        """Test that near clip returns expected constant."""
        near = _near_clip()
        self.assertEqual(near, 0.05)

    def test_vector_length(self) -> None:
        """Test vector length calculation."""
        vec = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        length = _vector_length(vec)
        self.assertAlmostEqual(length, 5.0, places=5)

    def test_normalise(self) -> None:
        """Test vector normalization."""
        vec = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        normalized = _normalise(vec)
        self.assertAlmostEqual(_vector_length(normalized), 1.0, places=5)
        self.assertAlmostEqual(normalized[0], 0.6, places=5)
        self.assertAlmostEqual(normalized[1], 0.8, places=5)

    def test_normalise_zero_vector(self) -> None:
        """Test that normalizing zero vector returns zero vector."""
        vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        normalized = _normalise(vec)
        self.assertEqual(_vector_length(normalized), 0.0)

    def test_build_camera_target_based(self) -> None:
        """Test building camera from position and target."""
        camera_state = {
            'position': [0.0, 0.0, 0.0],
            'target': [0.0, 0.0, 1.0],
            'fov_y': 60.0,
        }
        camera = _build_camera(camera_state, 640, 480)
        
        self.assertIsNotNone(camera)
        self.assertEqual(camera['fov_y'], 60.0)
        self.assertAlmostEqual(camera['aspect'], 640.0 / 480.0, places=5)
        
        # Forward should point towards target
        np.testing.assert_array_almost_equal(
            camera['forward'],
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            decimal=5
        )

    def test_build_camera_orientation_based(self) -> None:
        """Test building camera from position and orientation."""
        camera_state = {
            'position': [0.0, 0.0, 0.0],
            'orientation': {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0},
            'fov_y': 45.0,
        }
        camera = _build_camera(camera_state, 800, 600)
        
        self.assertIsNotNone(camera)
        self.assertEqual(camera['fov_y'], 45.0)
        self.assertAlmostEqual(camera['aspect'], 800.0 / 600.0, places=5)
        
        # Default orientation should have forward pointing -Z
        np.testing.assert_array_almost_equal(
            camera['forward'],
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
            decimal=5
        )

    def test_build_camera_invalid_state(self) -> None:
        """Test that invalid camera state returns None."""
        # Missing position
        camera = _build_camera({}, 640, 480)
        self.assertIsNone(camera)
        
        # Missing both target and orientation
        camera = _build_camera({'position': [0, 0, 0]}, 640, 480)
        self.assertIsNone(camera)

    def test_projection_matrix_shape(self) -> None:
        """Test projection matrix has correct shape and type."""
        proj = _projection_matrix(60.0, 16.0/9.0, 0.1, 100.0)
        
        self.assertEqual(proj.shape, (4, 4))
        self.assertEqual(proj.dtype, np.float32)
        
        # Check key elements are non-zero
        self.assertNotEqual(proj[0, 0], 0.0)  # x scaling
        self.assertNotEqual(proj[1, 1], 0.0)  # y scaling
        self.assertNotEqual(proj[2, 2], 0.0)  # z mapping
        self.assertEqual(proj[3, 2], -1.0)     # perspective divide

    def test_view_matrix_shape(self) -> None:
        """Test view matrix has correct shape and type."""
        eye = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        view = _view_matrix(eye, forward, up)
        
        self.assertEqual(view.shape, (4, 4))
        self.assertEqual(view.dtype, np.float32)

    def test_camera_consistency_cpu_opengl(self) -> None:
        """Test that CPU and OpenGL renderers get same camera from same state."""
        camera_state = {
            'position': [1.0, 2.0, 3.0],
            'target': [0.0, 0.0, 0.0],
            'fov_y': 60.0,
        }
        
        # Build camera for both renderers
        camera_cpu = _build_camera(camera_state, 640, 480, None)
        camera_gl = _build_camera(camera_state, 640, 480, None)
        
        self.assertIsNotNone(camera_cpu)
        self.assertIsNotNone(camera_gl)
        
        # Compare all fields
        np.testing.assert_array_almost_equal(
            camera_cpu['position'], camera_gl['position'], decimal=5
        )
        np.testing.assert_array_almost_equal(
            camera_cpu['forward'], camera_gl['forward'], decimal=5
        )
        np.testing.assert_array_almost_equal(
            camera_cpu['right'], camera_gl['right'], decimal=5
        )
        np.testing.assert_array_almost_equal(
            camera_cpu['up'], camera_gl['up'], decimal=5
        )
        self.assertEqual(camera_cpu['fov_y'], camera_gl['fov_y'])
        self.assertEqual(camera_cpu['aspect'], camera_gl['aspect'])


if __name__ == "__main__":
    unittest.main()
