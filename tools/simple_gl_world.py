"""Standalone OpenGL world preview (no SimCamera dependency)."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Tuple

import glfw
import moderngl
import numpy as np
import trimesh


_VERT_SHADER = """
#version 330
in vec3 in_position;
in vec3 in_normal;
uniform mat4 MVP;
out vec3 v_normal;
out vec3 v_pos;
void main() {
    gl_Position = MVP * vec4(in_position, 1.0);
    v_normal = in_normal;
    v_pos = in_position;
}
"""


_FRAG_SHADER = """
#version 330
in vec3 v_normal;
in vec3 v_pos;
out vec4 f_color;
uniform float u_grid;
uniform vec3 u_color;
void main() {
    vec3 n = normalize(v_normal);
    vec3 ldir = normalize(vec3(0.3, 1.0, 0.2));
    float l = max(dot(n, ldir), 0.0);
    float ambient = 0.35;
    float diffuse = 0.65;
    vec3 base = u_color;
    vec3 col = base * (ambient + diffuse * l);

    float gx = abs(fract(v_pos.x * 0.1) - 0.5);
    float gz = abs(fract(v_pos.z * 0.1) - 0.5);
    float grid = step(0.48, 0.5 - min(gx, gz)) * u_grid;
    col = mix(col, col * 0.5, grid);

    f_color = vec4(col, 1.0);
}
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone OpenGL world preview.")
    parser.add_argument("--width", type=int, default=1280, help="Frame width in pixels.")
    parser.add_argument("--height", type=int, default=720, help="Frame height in pixels.")
    parser.add_argument("--fps", type=float, default=30.0, help="Target display fps.")
    parser.add_argument("--frames", type=int, default=0, help="Number of frames to render (0 = run until quit).")
    parser.add_argument(
        "--window-title",
        default="Simple GL World",
        help="Window title for the preview.",
    )
    return parser.parse_args()


def _perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    nf = 1.0 / (near - far)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) * nf
    m[2, 3] = (2.0 * far * near) * nf
    m[3, 2] = -1.0
    return m


def _look_at(eye: Tuple[float, float, float], target: Tuple[float, float, float], up: Tuple[float, float, float]) -> np.ndarray:
    eye_v = np.array(eye, dtype=np.float32)
    target_v = np.array(target, dtype=np.float32)
    up_v = np.array(up, dtype=np.float32)

    f = target_v - eye_v
    f /= np.linalg.norm(f) + 1e-12
    u = up_v / (np.linalg.norm(up_v) + 1e-12)
    s = np.cross(f, u)
    s /= np.linalg.norm(s) + 1e-12
    u2 = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u2
    m[2, 0:3] = -f
    t = np.eye(4, dtype=np.float32)
    t[0:3, 3] = -eye_v
    return m @ t


def _build_ground_plane(size: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    half = size * 0.5
    positions = np.array(
        [
            (-half, 0.0, -half),
            (half, 0.0, -half),
            (half, 0.0, half),
            (-half, 0.0, half),
        ],
        dtype=np.float32,
    )
    normals = np.array([(0.0, 1.0, 0.0)] * 4, dtype=np.float32)
    vertices = np.hstack([positions, normals])
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return vertices, indices


def _build_unit_box() -> Tuple[np.ndarray, np.ndarray]:
    positions = [
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
        (0.5, -0.5, -0.5),
        (-0.5, -0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, -0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (-0.5, 0.5, 0.5),
        (-0.5, 0.5, -0.5),
        (0.5, -0.5, 0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, -0.5, 0.5),
        (-0.5, -0.5, 0.5),
    ]
    normals = [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, -1.0),
        (-1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, -1.0, 0.0),
    ]
    vertices = np.hstack(
        [np.array(positions, dtype=np.float32), np.array(normals, dtype=np.float32)]
    )
    indices = np.array(
        [
            0, 1, 2, 0, 2, 3,
            4, 5, 6, 4, 6, 7,
            8, 9, 10, 8, 10, 11,
            12, 13, 14, 12, 14, 15,
            16, 17, 18, 16, 18, 19,
            20, 21, 22, 20, 22, 23,
        ],
        dtype=np.uint32,
    )
    return vertices, indices


def _load_trimesh(path: Path) -> trimesh.Trimesh:
    if not path.exists():
        fallback = path.with_suffix(".stl")
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(f"Mesh asset not found: {path}")

    loaded = trimesh.load(str(path))
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded

    mesh.rezero()
    if mesh.scale > 1e-6:
        mesh.apply_scale(3.0 / mesh.scale)

    mesh.fix_normals()
    if mesh.faces.shape[1] != 3:
        mesh = mesh.triangulate()

    return mesh


def main() -> None:
    args = _parse_args()

    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(args.width, args.height, args.window_title, None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW window creation failed")
    glfw.make_context_current(window)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)

    ground_vertices, ground_indices = _build_ground_plane(1000.0)
    ground_vbo = ctx.buffer(ground_vertices.tobytes())
    ground_ibo = ctx.buffer(ground_indices.tobytes())

    mesh = _load_trimesh(Path("assets/meshes/drone.stl"))
    mesh_vertices = np.hstack([
        mesh.vertices.astype("f4"),
        mesh.vertex_normals.astype("f4"),
    ])
    mesh_vbo = ctx.buffer(mesh_vertices.tobytes())
    mesh_ibo = ctx.buffer(mesh.faces.astype("u4").tobytes())

    building_vertices, building_indices = _build_unit_box()
    building_vbo = ctx.buffer(building_vertices.tobytes())
    building_ibo = ctx.buffer(building_indices.tobytes())

    prog = ctx.program(vertex_shader=_VERT_SHADER, fragment_shader=_FRAG_SHADER)
    ground_vao = ctx.vertex_array(
        prog,
        [(ground_vbo, "3f 3f", "in_position", "in_normal")],
        index_buffer=ground_ibo,
    )
    if mesh is not None:
        mesh_vao = ctx.vertex_array(
            prog,
            [(mesh_vbo, "3f 3f", "in_position", "in_normal")],
            index_buffer=mesh_ibo,
        )
    building_vao = ctx.vertex_array(
        prog,
        [(building_vbo, "3f 3f", "in_position", "in_normal")],
        index_buffer=building_ibo,
    )

    proj = _perspective(60.0, args.width / args.height, 0.1, 100.0)
    model_ground = np.eye(4, dtype=np.float32)
    orbit_radius = 8.0
    orbit_height = 5.0
    orbit_speed = 0.35
    mesh_distance = 4.0
    building_specs = [
        {"base_centre": (0.0, -18.0), "footprint": (8.0, 6.0), "height": 12.0},
        {"base_centre": (-10.0, -26.0), "footprint": (10.0, 8.0), "height": 18.0},
        {"base_centre": (12.0, -28.0), "footprint": (12.0, 7.0), "height": 15.0},
    ]


    fps = max(1.0, float(args.fps))
    frame_delay = 1.0 / fps

    rendered = 0
    start_time = time.perf_counter()
    last_frame = start_time
    while not glfw.window_should_close(window):
        now = time.perf_counter()
        if now - last_frame < frame_delay:
            time.sleep(frame_delay * 0.5)
            continue
        last_frame = now

        elapsed = now - start_time
        angle = elapsed * orbit_speed
        eye = (
            math.cos(angle) * orbit_radius,
            orbit_height,
            math.sin(angle) * orbit_radius,
        )
        target = (0.0, 2.0, 0.0)
        view = _look_at(eye, target, (0.0, 1.0, 0.0))

        mvp_ground = proj @ view @ model_ground
        prog["MVP"].write(mvp_ground.T.astype("f4").tobytes())
        prog["u_grid"].value = 1.0
        prog["u_color"].value = (0.45, 0.7, 0.85)

        ctx.clear(1.0, 1.0, 1.0, depth=1.0)
        ground_vao.render()

        prog["u_grid"].value = 0.0
        prog["u_color"].value = (0.7, 0.7, 0.82)
        for spec in building_specs:
            base_x, base_z = spec["base_centre"]
            width, depth = spec["footprint"]
            height = spec["height"]
            model_building = np.eye(4, dtype=np.float32)
            model_building[0, 0] = float(width)
            model_building[1, 1] = float(height)
            model_building[2, 2] = float(depth)
            model_building[0:3, 3] = (
                float(base_x),
                float(height) * 0.5,
                float(base_z),
            )
            mvp_building = proj @ view @ model_building
            prog["MVP"].write(mvp_building.T.astype("f4").tobytes())
            building_vao.render()

        if mesh_vao is not None:
            forward = np.array(target, dtype=np.float32) - np.array(eye, dtype=np.float32)
            forward /= np.linalg.norm(forward) + 1e-12
            mesh_pos = np.array(eye, dtype=np.float32) + forward * mesh_distance
            model_mesh = np.eye(4, dtype=np.float32)
            model_mesh[0:3, 3] = mesh_pos
            mvp_mesh = proj @ view @ model_mesh
            prog["MVP"].write(mvp_mesh.T.astype("f4").tobytes())
            prog["u_grid"].value = 0.0
            prog["u_color"].value = (0.6, 0.7, 0.8)
            mesh_vao.render()

        glfw.swap_buffers(window)
        glfw.poll_events()

        rendered += 1
        if args.frames > 0 and rendered >= args.frames:
            break

    glfw.terminate()


if __name__ == "__main__":
    main()
