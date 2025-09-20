
"""
Diagnostic OpenGL Renderer (ModernGL + GLFW)

Purpose:
- Validate depth buffering, projection, and mesh health outside your main pipeline.
- Render either a procedural cube or a given OBJ to an offscreen FBO with a depth buffer.
- Optional depth visualization mode using gl_FragCoord.z (grayscale).
- Saves a PNG snapshot and prints depth stats.

Usage examples (run from your repo root):
    python -m pc.tools.diagnostic_renderer --mode cube --width 800 --height 600 --out cube.png
    python -m pc.tools.diagnostic_renderer --mode obj --obj assets/person.obj --out person_color.png
    python -m pc.tools.diagnostic_renderer --mode obj --obj assets/person.obj --depth-vis --out person_depth.png

Requires: glfw, moderngl, numpy, pillow
    pip install glfw moderngl numpy pillow
"""

import argparse
import math
import os
from dataclasses import dataclass

import glfw
import moderngl
import numpy as np
from PIL import Image


# ------------------------- Math Helpers -------------------------
def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    nf = 1.0 / (near - far)
    return np.array([
        [f / aspect, 0, 0,                       0],
        [0,          f, 0,                       0],
        [0,          0, (far + near) * nf, (2 * far * near) * nf],
        [0,          0, -1,                      0],
    ], dtype=np.float32)


def look_at(eye, target, up) -> np.ndarray:
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    f = target - eye
    f /= (np.linalg.norm(f) + 1e-8)
    s = np.cross(f, up); s /= (np.linalg.norm(s) + 1e-8)
    u = np.cross(s, f)

    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    return M @ T


def rotation_y(theta_rad: float) -> np.ndarray:
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ], dtype=np.float32)


# ------------------------- Geometry -------------------------
def make_cube():
    # Unit cube centered at origin, side length 2 (coords in [-1,1])
    V = np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],  # back
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],  # front
    ], dtype=np.float32)
    # 12 triangles
    F = np.array([
        [0,1,2], [0,2,3],  # back
        [4,6,5], [4,7,6],  # front
        [0,4,5], [0,5,1],  # bottom
        [3,2,6], [3,6,7],  # top
        [1,5,6], [1,6,2],  # right
        [0,3,7], [0,7,4],  # left
    ], dtype=np.uint32)
    return V, F


def load_obj_positions(path: str):
    vs, faces = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z, *rest = line.strip().split()
                vs.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                idxs = []
                for p in parts:
                    s = p.split('/')[0]
                    try:
                        idxs.append(int(s))
                    except:
                        pass
                if len(idxs) >= 3:
                    faces.append(idxs)
    V = np.array(vs, dtype=np.float32)
    F = []
    nV = len(V)
    for face in faces:
        # OBJ is 1-based; negatives are relative to end
        idxs = []
        for i in face:
            i = (nV + i) if i < 0 else (i - 1)
            idxs.append(i)
        # triangulate fan
        v0 = idxs[0]
        for a, b in zip(idxs[1:-1], idxs[2:]):
            F.append([v0, a, b])
    F = np.array(F, dtype=np.uint32) if F else np.zeros((0, 3), dtype=np.uint32)
    return V, F

# --- after V, F are loaded ---



# ------------------------- GL Shaders -------------------------
VS_DEFAULT = """
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

FS_COLOR = """
#version 330
uniform vec3 u_color;
out vec4 f_color;
void main() {
    f_color = vec4(u_color, 1.0);
}
"""

# Visualize depth using the actual depth buffer value
FS_DEPTH = """
#version 330
out vec4 f_color;
void main() {
    float d = gl_FragCoord.z; // [0,1], non-linear depth
    f_color = vec4(d, d, d, 1.0);
}
"""
VS_CLIP_Z = """
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
out float v_ndc_z;
void main() {
    vec4 clip = u_mvp * vec4(in_pos, 1.0);
    v_ndc_z = clip.z / clip.w;      // [-1, 1]
    gl_Position = clip;
}
"""

FS_NDC_Z = """
#version 330
in float v_ndc_z;
out vec4 f_color;
void main() {
    float g = clamp(v_ndc_z * 0.5 + 0.5, 0.0, 1.0);  // map [-1,1] -> [0,1]
    f_color = vec4(g, g, g, 1.0);
}
"""


# ------------------------- Settings -------------------------
@dataclass
class Settings:
    width: int = 800
    height: int = 600
    fov_y: float = 60.0
    near: float = 0.1
    far: float = 5000.0
    eye: tuple = (0.0, 0.0, 5.0)
    target: tuple = (0.0, 0.0, 0.0)
    up: tuple = (0.0, 1.0, 0.0)
    rotate_y_deg: float = 20.0
    color: tuple = (0.9, 0.9, 0.2)


# ------------------------- GL Setup -------------------------
def ensure_glfw():
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    # hidden window (offscreen rendering)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.DEPTH_BITS, 24)
    win = glfw.create_window(64, 64, "diag-offscreen", None, None)
    if not win:
        raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(win)
    return win


def make_fbo(ctx: moderngl.Context, w: int, h: int):
    color = ctx.texture((w, h), components=3)  # RGB8
    color.filter = (moderngl.NEAREST, moderngl.NEAREST)
    depth = ctx.depth_renderbuffer((w, h))
    fbo = ctx.framebuffer(color_attachments=[color], depth_attachment=depth)
    return fbo, color


# ------------------------- Rendering -------------------------
def render_snapshot(mode: str, obj_path: str, out_path: str, depth_vis: bool, s: Settings):
    win = ensure_glfw()
    try:
        ctx = moderngl.create_context()
        fbo, color_tex = make_fbo(ctx, s.width, s.height)

        prog_color = ctx.program(vertex_shader=VS_DEFAULT, fragment_shader=FS_COLOR)
        prog_depth = ctx.program(vertex_shader=VS_DEFAULT, fragment_shader=FS_DEPTH)

        if mode == "cube":
            V, F = make_cube()
            V = V * 1.5
        else:
            if not os.path.exists(obj_path):
                raise FileNotFoundError(f"OBJ not found: {obj_path}")
            V, F = load_obj_positions(obj_path)
        def center_and_scale(V, target_diag=2.0):
            # center vertices and scale so bbox diagonal ~ target_diag
            mn = V.min(axis=0); mx = V.max(axis=0)
            centroid = (mn + mx) * 0.5
            Vc = V - centroid
            diag = np.linalg.norm(mx - mn)
            scale = (target_diag / diag) if diag > 1e-8 else 1.0
            return Vc * scale, centroid, diag, scale

        # Center/scale for stability unless user wants raw coords
        V_fit, centroid, diag, scale = center_and_scale(V, target_diag=2.0)

        # Build buffers with V_fit instead of raw V
        vbo = ctx.buffer(V_fit.astype('f4').tobytes())
        ibo = ctx.buffer(F.astype('u4').tobytes()) if len(F) else None

        # Auto-fit camera distance from the (scaled) radius
        radius = np.linalg.norm(V_fit.max(axis=0) - V_fit.min(axis=0)) * 0.5
        radius = max(radius, 1.0)  # keep sane
        eye_dist = radius * 2.8  # was 2.2
        eye = np.array([0.0, 0.0, eye_dist], dtype=np.float32)

        # Tighter near/far for better depth precision
        near = max(eye_dist - radius * 1.2, 0.01)
        far  = eye_dist + radius * 1.2

        aspect = s.width / s.height
        P = perspective(s.fov_y, aspect, near, far)
        Vmat = look_at(eye, np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32))
        R = rotation_y(math.radians(s.rotate_y_deg))
        M = np.eye(4, dtype=np.float32) @ R
        MVP = P @ Vmat @ M

        ctx.enable(moderngl.DEPTH_TEST)
        fbo.use()
        ctx.viewport = (0, 0, s.width, s.height)
        fbo.clear(0.05, 0.07, 0.09, 1.0, 1.0)  # sky + depth=1

        if depth_vis:
            vao = ctx.vertex_array(prog_depth, [(vbo, '3f', 'in_pos')], index_buffer=ibo) if ibo else ctx.vertex_array(prog_depth, [(vbo, '3f', 'in_pos')])
            prog_depth['u_mvp'].write(MVP.T.tobytes())
            vao.render(moderngl.TRIANGLES)
        else:
            vao = ctx.vertex_array(prog_color, [(vbo, '3f', 'in_pos')], index_buffer=ibo) if ibo else ctx.vertex_array(prog_color, [(vbo, '3f', 'in_pos')])
            prog_color['u_mvp'].write(MVP.T.tobytes())
            prog_color['u_color'].value = s.color
            vao.render(moderngl.TRIANGLES)

        data = color_tex.read(alignment=1)
        img = np.frombuffer(data, np.uint8).reshape(s.height, s.width, 3)
        img = np.flipud(img)  # GL origin is bottom-left
        Image.fromarray(img, mode='RGB').save(out_path)

        # Tiny second pass to get a quick depth range estimate (using depth shader)
        fbo2, color2 = make_fbo(ctx, 64, 64)
        vao2 = ctx.vertex_array(prog_depth, [(vbo, '3f', 'in_pos')], index_buffer=ibo) if ibo else ctx.vertex_array(prog_depth, [(vbo, '3f', 'in_pos')])
        fbo2.use()
        ctx.viewport = (0, 0, 64, 64)
        fbo2.clear(0.0, 0.0, 0.0, 1.0, 1.0)
        prog_depth['u_mvp'].write(MVP.T.tobytes())
        vao2.render(moderngl.TRIANGLES)
        d_small = np.frombuffer(color2.read(alignment=1), np.uint8).reshape(64, 64, 3)
        r = d_small[..., 0].astype(np.float32) / 255.0
        print(f"[diag] depth(gl_FragCoord.z) ~ min={r.min():.4f} max={r.max():.4f} mean={r.mean():.4f}")
        print(f"[diag] wrote snapshot -> {out_path}")

        # create program
        prog_ndc = ctx.program(vertex_shader=VS_CLIP_Z, fragment_shader=FS_NDC_Z)

        # when rendering in NDC mode:
        vao = ctx.vertex_array(prog_ndc, [(vbo, '3f', 'in_pos')], index_buffer=ibo) if ibo else ctx.vertex_array(prog_ndc, [(vbo, '3f', 'in_pos')])
        prog_ndc['u_mvp'].write(MVP.T.tobytes())
        vao.render(moderngl.TRIANGLES)

        # after readback of the image, quickly estimate NDC z stats:
        # (re-render smaller for stats if you prefer, or reuse the big frame)
        img = np.frombuffer(color_tex.read(alignment=1), np.uint8).reshape(s.height, s.width, 3)
        r = img[...,0].astype(np.float32)/255.0
        print(f"[diag] NDC-z ~ min={r.min():.4f} max={r.max():.4f} mean={r.mean():.4f}")

    finally:
        glfw.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['cube', 'obj'], default='cube')
    ap.add_argument('--obj', type=str, default='')
    ap.add_argument('--width', type=int, default=800)
    ap.add_argument('--height', type=int, default=600)
    ap.add_argument('--fov', type=float, default=60.0)
    ap.add_argument('--near', type=float, default=0.1)
    ap.add_argument('--far', type=float, default=5000.0)
    ap.add_argument('--eye', type=float, nargs=3, default=[0.0, 0.0, 5.0])
    ap.add_argument('--target', type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument('--up', type=float, nargs=3, default=[0.0, 1.0, 0.0])
    ap.add_argument('--rotate-y', type=float, default=20.0)
    ap.add_argument('--color', type=float, nargs=3, default=[0.9, 0.9, 0.2])
    ap.add_argument('--depth-vis', action='store_true', help='Render depth grayscale instead of color')
    ap.add_argument('--out', type=str, default='diagnostic.png')
    args = ap.parse_args()

    settings = Settings(
        width=args.width, height=args.height,
        fov_y=args.fov, near=args.near, far=args.far,
        eye=tuple(args.eye), target=tuple(args.target), up=tuple(args.up),
        rotate_y_deg=args.rotate_y, color=tuple(args.color),
    )

    render_snapshot(args.mode, args.obj, args.out, args.depth_vis, settings)


if __name__ == "__main__":
    main()