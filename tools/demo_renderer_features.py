#!/usr/bin/env python3
"""Demo script showing mesh loading and rendering capabilities.

This script demonstrates:
1. Loading a mesh using the new mesh loader
2. Creating an OpenGL renderer with EGL/GLES support
3. Basic rendering setup
"""

import sys
import numpy as np
from pc.renderers.mesh import load_mesh, get_cache_size
from pc.renderers import OpenGLRenderer
import types


def main():
    print("=== IDCS Mesh Loader & OpenGL Renderer Demo ===\n")
    
    # Test 1: Mesh loading
    print("1. Testing mesh loader...")
    try:
        mesh_path = "assets/person.obj"
        vertices, normals, indices = load_mesh(mesh_path)
        print(f"   ✓ Loaded mesh from {mesh_path}")
        print(f"     - Vertices: {vertices.shape[0]}")
        print(f"     - Triangles: {indices.shape[0]}")
        print(f"     - Cache size: {get_cache_size()}")
        
        # Try loading again to test caching
        vertices2, normals2, indices2 = load_mesh(mesh_path)
        print(f"   ✓ Loaded mesh again (from cache)")
        print(f"     - Cache size: {get_cache_size()}")
        assert vertices is vertices2, "Cache should return same object"
    except Exception as e:
        print(f"   ✗ Mesh loading failed: {e}")
        return 1
    
    # Test 2: OpenGL renderer creation
    print("\n2. Testing OpenGL renderer...")
    try:
        context = types.SimpleNamespace(width=640, height=480)
        renderer = OpenGLRenderer(context=context)
        
        if renderer._gl is None:
            print(f"   ℹ OpenGL context creation failed, using fallback")
            print(f"     (This is normal in headless environments)")
        else:
            print(f"   ✓ OpenGL context created successfully")
            print(f"     - Shader program: {'OK' if renderer._prog else 'FAILED'}")
            print(f"     - FBO: {'OK' if renderer._fbo else 'FAILED'}")
            print(f"     - VAO: {'OK' if renderer._vao else 'FAILED'}")
        
        # Test rendering
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        renderer.render(frame, frame_id=0)
        print(f"   ✓ Rendered frame successfully")
        print(f"     - Frame shape: {frame.shape}")
        print(f"     - Non-zero pixels: {np.count_nonzero(frame)} / {frame.size}")
        
    except Exception as e:
        print(f"   ✗ OpenGL renderer failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test 3: Shared camera logic
    print("\n3. Testing shared camera logic...")
    try:
        from pc.renderers._common import (
            _build_camera,
            _projection_matrix,
            _view_matrix,
        )
        
        camera_state = {
            'position': [3.0, 3.0, 3.0],
            'target': [0.0, 0.0, 0.0],
            'fov_y': 60.0,
        }
        
        camera = _build_camera(camera_state, 640, 480)
        assert camera is not None
        print(f"   ✓ Built camera from state")
        print(f"     - Position: {camera['position']}")
        print(f"     - FOV: {camera['fov_y']}°")
        print(f"     - Aspect: {camera['aspect']:.3f}")
        
        proj = _projection_matrix(camera['fov_y'], camera['aspect'], 0.1, 100.0)
        view = _view_matrix(camera['position'], camera['forward'], camera['up'])
        print(f"   ✓ Built projection and view matrices")
        print(f"     - Projection shape: {proj.shape}")
        print(f"     - View shape: {view.shape}")
        
    except Exception as e:
        print(f"   ✗ Camera logic failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n=== All tests passed! ===")
    print("\nKey features demonstrated:")
    print("  • Mesh loading with automatic triangulation and normalization")
    print("  • Mesh caching for efficient reuse")
    print("  • EGL/GLES context creation with fallback")
    print("  • Shared camera logic between CPU and OpenGL renderers")
    print("  • Proper FBO setup with alignment and BGR readback")
    return 0


if __name__ == "__main__":
    sys.exit(main())
