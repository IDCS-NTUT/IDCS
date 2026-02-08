"""Test mesh loading functionality."""

from __future__ import annotations

import unittest
import tempfile
import os
from pc.renderers.mesh import load_mesh, clear_cache, get_cache_size


class MeshLoaderTest(unittest.TestCase):
    """Test mesh loading and caching."""

    def setUp(self) -> None:
        """Clear cache before each test."""
        clear_cache()

    def test_load_obj_mesh(self) -> None:
        """Test loading an OBJ mesh file."""
        mesh_path = os.path.join(os.path.dirname(__file__), "..", "assets", "person.obj")
        if not os.path.exists(mesh_path):
            self.skipTest(f"Test mesh not found: {mesh_path}")
        
        vertices, normals, indices = load_mesh(mesh_path)
        
        # Check shapes
        self.assertEqual(vertices.shape[1], 3)
        self.assertEqual(normals.shape[1], 3)
        self.assertEqual(indices.shape[1], 3)
        
        # Check that we have data
        self.assertGreater(vertices.shape[0], 0)
        self.assertGreater(indices.shape[0], 0)
        
        # Check that normals match vertex count
        self.assertEqual(vertices.shape[0], normals.shape[0])

    def test_mesh_caching(self) -> None:
        """Test that loaded meshes are cached."""
        mesh_path = os.path.join(os.path.dirname(__file__), "..", "assets", "person.obj")
        if not os.path.exists(mesh_path):
            self.skipTest(f"Test mesh not found: {mesh_path}")
        
        # Load mesh first time
        self.assertEqual(get_cache_size(), 0)
        vertices1, normals1, indices1 = load_mesh(mesh_path)
        self.assertEqual(get_cache_size(), 1)
        
        # Load same mesh again
        vertices2, normals2, indices2 = load_mesh(mesh_path)
        self.assertEqual(get_cache_size(), 1)
        
        # Verify data is identical (same cached object)
        self.assertIs(vertices1, vertices2)
        self.assertIs(normals1, normals2)
        self.assertIs(indices1, indices2)

    def test_clear_cache(self) -> None:
        """Test clearing the mesh cache."""
        mesh_path = os.path.join(os.path.dirname(__file__), "..", "assets", "person.obj")
        if not os.path.exists(mesh_path):
            self.skipTest(f"Test mesh not found: {mesh_path}")
        
        # Load and cache
        load_mesh(mesh_path)
        self.assertEqual(get_cache_size(), 1)
        
        # Clear cache
        clear_cache()
        self.assertEqual(get_cache_size(), 0)

    def test_load_missing_file(self) -> None:
        """Test that loading a missing file raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            load_mesh("/nonexistent/path/to/mesh.obj")

    def test_mesh_normalization(self) -> None:
        """Test that mesh normalization works."""
        mesh_path = os.path.join(os.path.dirname(__file__), "..", "assets", "person.obj")
        if not os.path.exists(mesh_path):
            self.skipTest(f"Test mesh not found: {mesh_path}")
        
        # Load with normalization (default)
        vertices, _, _ = load_mesh(mesh_path, normalize=True)
        
        # Check that vertices are roughly in [-1, 1] range after normalization
        # (may be slightly outside due to aspect ratio)
        import numpy as np
        max_extent = np.abs(vertices).max()
        self.assertLess(max_extent, 3.0, "Normalized mesh should be reasonably scaled")

    def test_mesh_scale_parameter(self) -> None:
        """Test that mesh scale parameter works."""
        mesh_path = os.path.join(os.path.dirname(__file__), "..", "assets", "person.obj")
        if not os.path.exists(mesh_path):
            self.skipTest(f"Test mesh not found: {mesh_path}")
        
        # Clear cache to ensure fresh load
        clear_cache()
        
        # Load with scale factor
        vertices1, _, _ = load_mesh(mesh_path, normalize=True, scale=2.0)
        
        # Clear and load with different scale
        clear_cache()
        vertices2, _, _ = load_mesh(mesh_path, normalize=True, scale=1.0)
        
        # Scaled version should be larger
        import numpy as np
        extent1 = np.abs(vertices1).max()
        extent2 = np.abs(vertices2).max()
        self.assertGreater(extent1, extent2, "Scaled mesh should be larger")


if __name__ == "__main__":
    unittest.main()
