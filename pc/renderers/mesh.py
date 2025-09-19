# pc/renderers/mesh.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import numpy as np
import trimesh as tm

@dataclass
class LoadedMesh:
    vertices: np.ndarray  # (N,3) float32
    indices:  np.ndarray  # (M,3) uint32
    colors:   np.ndarray  # (N,3) float32 in [0,1]

def _collapse_scene(scene: tm.Scene) -> tm.Trimesh:
    # Concatenate all geometry in the scene into one mesh in world coordinates
    geoms = []
    for name, geom in scene.geometry.items():
        g = geom.copy()
        # Apply node transforms if any
        try:
            xf = scene.graph.get(world=name)[0]
            g.apply_transform(xf)
        except Exception:
            pass
        geoms.append(g)
    if not geoms:
        return tm.Trimesh()
    return tm.util.concatenate(geoms)

def load_obj(path: str, default_color: Tuple[float, float, float] = (0.9, 0.9, 0.9)) -> LoadedMesh:
    m = tm.load(path, force='mesh', process=True)

    if isinstance(m, tm.Scene):
        m = _collapse_scene(m)

    if not isinstance(m, tm.Trimesh):
        raise ValueError(f"Unsupported mesh type from '{path}': {type(m)}")

    # Ensure triangulated faces (Trimesh usually guarantees this, but be safe)
    if m.faces is None or (m.faces.ndim == 2 and m.faces.shape[1] != 3):
        m = m.triangulate()

    v = np.asarray(m.vertices, dtype=np.float32)
    i = np.asarray(m.faces,    dtype=np.uint32)

    # Try vertex colors; fall back to a solid color
    if hasattr(m, "visual") and m.visual is not None:
        if m.visual.vertex_colors is not None and len(m.visual.vertex_colors) == len(v):
            # vertex_colors may be RGBA uint8
            c = np.asarray(m.visual.vertex_colors, dtype=np.float32)
            c = c[:, :3] / 255.0
        elif hasattr(m.visual, "to_color"):
            c = np.asarray(m.visual.to_color().vertex_colors[:, :3], dtype=np.float32) / 255.0
        else:
            c = np.tile(np.array(default_color, dtype=np.float32), (v.shape[0], 1))
    else:
        c = np.tile(np.array(default_color, dtype=np.float32), (v.shape[0], 1))

    return LoadedMesh(vertices=v, indices=i, colors=c)
