# pc/renderers/gl.py
from __future__ import annotations
from typing import Any, List, Tuple
import time
import numpy as np
import cv2
import trimesh

from . import register_renderer

# ---------- math helpers ----------
def perspective(fov_y_rad: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(0.5 * fov_y_rad)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2.0 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M

def view_from_rt(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.astype(np.float32))
    t = tvec.reshape(3, 1).astype(np.float32)
    V = np.eye(4, dtype=np.float32)
    V[:3, :3] = R
    V[:3, 3:4] = t
    # flip Y and Z to match right-handed NDC (-Z forward)
    return np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32) @ V

def load_mesh_cpu(path: str, *, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    m = trimesh.load(path, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(m.dump())
    V = (m.vertices.astype(np.float32) * float(scale))
    F = m.faces.astype(np.int32)
    return V, F

# ---------- renderer ----------
class GLRenderer:
    def __init__(self, *, context: Any, finish_before_read: bool = True) -> None:
        import moderngl, glfw
        self._mgl, self._glfw = moderngl, glfw
        self._W = int(getattr(context, "width"))
        self._H = int(getattr(context, "height"))
        self._finish = bool(finish_before_read)

        # camera intrinsics
        self._fov = float(getattr(context, "fov"))
        self._aspect = float(getattr(context, "aspect"))

        # hidden window + GL context
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        if hasattr(glfw, "OPENGL_FORWARD_COMPAT"):
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        self._win = glfw.create_window(self._W, self._H, "IDCS-GL", None, None)
        if not self._win:
            glfw.terminate(); raise RuntimeError("Failed to create hidden GLFW window")
        glfw.make_context_current(self._win)
        glfw.swap_interval(0)

        self._ctx = moderngl.create_context(require=330)
        self._ctx.viewport = (0, 0, self._W, self._H)
        self._ctx.disable(self._mgl.CULL_FACE)
        self._ctx.disable(self._mgl.BLEND)
        self._ctx.color_mask = (True, True, True, True)

        self._sky = (180/255, 180/255, 210/255)
        self._grid_color = (150/255, 150/255, 150/255)
        self._boxes_color = (0.55, 0.55, 0.75)

        # programs
        self._grid_prog = self._ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;  // already NDC xy, z ignored
                void main(){ gl_Position = vec4(in_pos, 1.0); }
            """,
            fragment_shader="""
                #version 330
                uniform vec3 color;
                out vec4 fragColor;
                void main(){ fragColor = vec4(color, 1.0); }
            """,
        )
        self._tri_prog = self._ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;  // NDC xyz (z for depth)
                void main(){ gl_Position = vec4(in_pos, 1.0); }
            """,
            fragment_shader="""
                #version 330
                uniform vec3 color;
                out vec4 fragColor;
                void main(){ fragColor = vec4(color, 1.0); }
            """,
        )

        # grid (world → NDC on CPU)
        lines = []
        x_extent = 40.0
        z_near   = -2.0
        z_far    = -40.0
        step     = 2.0
        y = 0.0
        z = z_far
        while z <= z_near + 1e-6:
            lines += [-x_extent, y, z,   x_extent, y, z]
            z += step
        x = -x_extent
        while x <= x_extent + 1e-6:
            lines += [x, y, z_far,   x, y, z_near]
            x += step
        self._grid_world = np.array(lines, dtype=np.float32).reshape(-1, 3)
        self._grid_vbo = self._ctx.buffer(reserve=self._grid_world.size * 4)
        self._grid_vao = self._ctx.vertex_array(self._grid_prog, [(self._grid_vbo, "3f", "in_pos")])
        self._grid_count = self._grid_world.shape[0]
        self._ctx.line_width = 2.0

        # wire boxes (like CPU scene)
        self._boxes_world = self._build_boxes_from_context(context)
        self._boxes_vbo   = self._ctx.buffer(reserve=self._boxes_world.size * 4 if self._boxes_world.size else 4)
        self._boxes_vao   = self._ctx.vertex_array(self._grid_prog, [(self._boxes_vbo, "3f", "in_pos")])
        self._boxes_count = self._boxes_world.shape[0]

        # meshes (from context.actor_meshes), scaled via spec["scale"]
        self._mesh_items = []
        actor_meshes = getattr(context, "actor_meshes", []) or []
        for spec in actor_meshes:
            Vw, F = load_mesh_cpu(spec["path"], scale=float(spec.get("scale", 1.0)))
            color = tuple(map(float, spec.get("color", (0.8, 0.8, 0.8))))
            vbo = self._ctx.buffer(reserve=Vw.shape[0] * 3 * 4)  # NDC written every frame
            ibo = self._ctx.buffer(F.astype(np.int32).tobytes())
            vao = self._ctx.vertex_array(self._tri_prog, [(vbo, "3f", "in_pos")], index_buffer=ibo)
            self._mesh_items.append({"V_world": Vw, "F": F, "vbo": vbo, "ibo": ibo, "vao": vao, "color": color})
        if self._mesh_items:
            print(f"[gl] loaded {len(self._mesh_items)} mesh(es)")

        # optional per-frame model matrices
        self._get_actor_models = getattr(context, "get_actor_transforms", None)

        # projection
        self._proj = perspective(self._fov, self._aspect, near=0.05, far=2000.0).astype(np.float32)

        print(f"[gl] perspective grid ready {self._W}x{self._H}, fov={self._fov:.3f} rad, aspect={self._aspect:.3f}")

    # --- internals ---
    @staticmethod
    def _box_edges_world(cx, cy, cz, w, d, h) -> np.ndarray:
        hw, hd, hh = 0.5*w, 0.5*d, h
        p = np.array([
            [cx-hw, cy,    cz-hd],
            [cx+hw, cy,    cz-hd],
            [cx+hw, cy,    cz+hd],
            [cx-hw, cy,    cz+hd],
            [cx-hw, cy+hh, cz-hd],
            [cx+hw, cy+hh, cz-hd],
            [cx+hw, cy+hh, cz+hd],
            [cx-hw, cy+hh, cz+hd],
        ], dtype=np.float32)
        E = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
        out = []
        for a,b in E:
            out.append(p[a]); out.append(p[b])
        return np.vstack(out)

    def _build_boxes_from_context(self, context: Any) -> np.ndarray:
        boxes_world = []
        ctx_boxes = getattr(context, "boxes", None)
        if ctx_boxes:
            for (x, y, z, w, d, h), _ in ctx_boxes:
                boxes_world.append(self._box_edges_world(x, y, z, w, d, h))
        return np.vstack(boxes_world).astype("f4") if boxes_world else np.empty((0,3), dtype="f4")

    # --- main draw ---
    def render(self, frame: np.ndarray, /, *, rvec: np.ndarray, tvec: np.ndarray) -> None:
        self._glfw.make_context_current(self._win)
        self._ctx.viewport = (0, 0, self._W, self._H)
        self._ctx.clear(*self._sky, 1.0)

        # pose
        V   = view_from_rt(rvec, tvec)
        MVP = (self._proj @ V).astype(np.float32)

        # grid CPU → clip → NDC
        pts4 = np.concatenate([self._grid_world, np.ones((self._grid_world.shape[0],1), dtype="f4")], axis=1)
        clip = (MVP @ pts4.T).T
        ndc  = clip[:, :3] / clip[:, 3:4]
        ndc2 = np.stack([ndc[:, 0], ndc[:, 1], np.zeros_like(ndc[:, 0])], axis=1).astype("f4")
        self._grid_vbo.write(ndc2.tobytes())
        self._grid_prog["color"].value = self._grid_color
        self._grid_vao.render(self._mgl.LINES, vertices=self._grid_count)

        # boxes CPU → NDC
        if self._boxes_count:
            pts4 = np.concatenate([self._boxes_world, np.ones((self._boxes_world.shape[0],1), dtype="f4")], axis=1)
            clip = (MVP @ pts4.T).T
            ndc  = clip[:, :3] / clip[:, 3:4]
            ndc2 = np.stack([ndc[:, 0], ndc[:, 1], np.zeros_like(ndc[:, 0])], axis=1).astype("f4")
            self._boxes_vbo.write(ndc2.tobytes())
            self._grid_prog["color"].value = self._boxes_color
            self._boxes_vao.render(self._mgl.LINES, vertices=self._boxes_count)

        # meshes CPU → NDC (with depth)
        if self._mesh_items:
            self._ctx.enable(self._mgl.DEPTH_TEST)
            get_models = self._get_actor_models
            models = get_models() if get_models else [np.eye(4, dtype=np.float32)] * len(self._mesh_items)

            for item, M in zip(self._mesh_items, models):
                Vw  = item["V_world"]
                V4  = np.concatenate([Vw, np.ones((Vw.shape[0],1), dtype="f4")], axis=1)
                W4  = (M @ V4.T).T
                C4  = (MVP @ W4.T).T
                NDC = C4[:, :3] / C4[:, 3:4]
                item["vbo"].write(NDC.astype("f4").tobytes())
                self._tri_prog["color"].value = item["color"]
                item["vao"].render(self._mgl.TRIANGLES)
            self._ctx.disable(self._mgl.DEPTH_TEST)

        # readback
        if self._finish:
            self._ctx.finish()
        rgb = np.frombuffer(self._ctx.screen.read(components=3, alignment=1), np.uint8).reshape(self._H, self._W, 3)
        frame[:] = np.flip(rgb, 0)[:, :, ::-1]

    def __del__(self):
        try:
            if getattr(self, "_grid_vao", None): self._grid_vao.release()
            if getattr(self, "_grid_vbo", None): self._grid_vbo.release()
            if getattr(self, "_boxes_vao", None): self._boxes_vao.release()
            if getattr(self, "_boxes_vbo", None): self._boxes_vbo.release()
            for it in getattr(self, "_mesh_items", []):
                if "vao" in it: it["vao"].release()
                if "vbo" in it: it["vbo"].release()
                if "ibo" in it: it["ibo"].release()
            if getattr(self, "_ctx", None): self._ctx.release()
            if getattr(self, "_win", None): self._glfw.destroy_window(self._win)
            self._glfw.terminate()
        except Exception:
            pass

register_renderer("gl", GLRenderer)
__all__ = ["GLRenderer"]
