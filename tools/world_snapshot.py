
'''
USAGE (examples):
  # Minimal (GPU MVP for meshes)
  python tools/world_snapshot.py --width 1280 --height 720 --fov-deg 60 --rvec 0 0 0 --tvec 0 0 3 --mesh assets/person.obj --mesh-color 0.9 0.25 0.25 --out frame.png

  # Multiple boxes + grid config + CPU→NDC clipping for meshes (slower but exact)
  python tools/world_snapshot.py --width 1280 --height 720 --fov-deg 60 --near 0.05 --far 2000 --rvec 0 0 0 --tvec 0 0 6 --box 0 0 -8  4 4 2 --box 6 0 -14  5 6 3 --cpu-clip-mesh --mesh assets/person.obj --mesh-scale 1.0 --mesh-color 0.8 0.8 0.8 --out frame_cpuclip.png
  '''
import argparse, math, os, sys
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image

# Optional: only used if you want to build view from rvec/tvec Rodrigues
try:
    import cv2
except Exception:
    cv2 = None

# -------------- math (matches your renderer) --------------
def perspective(fov_y_rad: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(0.5 * fov_y_rad)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2.0 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M

def view_from_rt(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("cv2 is required for Rodrigues; install opencv-python")
    R, _ = cv2.Rodrigues(rvec.astype(np.float32))
    t = tvec.reshape(3, 1).astype(np.float32)
    V = np.eye(4, dtype=np.float32)
    V[:3, :3] = R
    V[:3, 3:4] = t
    # flip Y and Z to match right-handed NDC (−Z forward)
    return np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32) @ V

# -------------- scene builders (grid / boxes like your code) --------------
def build_grid_world(x_extent=40.0, z_near=-2.0, z_far=-40.0, step=2.0, y=0.0) -> np.ndarray:
    lines = []
    z = z_far
    while z <= z_near + 1e-6:
        lines += [-x_extent, y, z,  x_extent, y, z]
        z += step
    x = -x_extent
    while x <= x_extent + 1e-6:
        lines += [x, y, z_far,  x, y, z_near]
        x += step
    return np.array(lines, dtype=np.float32).reshape(-1, 3)

def box_edges_world(cx, cy, cz, w, d, h) -> np.ndarray:
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
    return np.vstack(out).astype(np.float32)

# -------------- quick OBJ loader (trimesh) --------------
def load_mesh(path: str, scale: float=1.0) -> Tuple[np.ndarray, np.ndarray]:
    import trimesh
    m = trimesh.load(path, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(m.dump())
    V = (m.vertices.astype(np.float32) * float(scale))
    F = m.faces.astype(np.int32)
    return V, F

# -------------- CPU→NDC helpers --------------
def project_ndc(MVP: np.ndarray, P3: np.ndarray, force_z=None) -> np.ndarray:
    """Project 3D pts -> NDC. If force_z is not None, replace NDC z with that constant."""
    P4 = np.c_[P3, np.ones((P3.shape[0], 1), dtype=np.float32)]
    C4 = (MVP @ P4.T).T
    ndc = C4[:, :3] / C4[:, 3:4]
    if force_z is not None:
        ndc[:, 2] = force_z
    return ndc.astype(np.float32)

# --- lightweight homogeneous clipping (triangle, OpenGL frustum) ---
def _clip_poly_plane(poly, pred, intersect):
    if not poly:
        return []
    out = []
    prev = poly[-1]; prev_in = pred(prev)
    for curr in poly:
        curr_in = pred(curr)
        if prev_in and curr_in:
            out.append(curr)
        elif prev_in and not curr_in:
            out.append(intersect(prev, curr))
        elif not prev_in and curr_in:
            out.append(intersect(prev, curr)); out.append(curr)
        prev, prev_in = curr, curr_in
    return out

def _clip_triangle_clipspace(v0, v1, v2, eps=1e-6):
    xle = lambda p: p[0] <=  p[3]         # x <= w
    xge = lambda p: p[0] >= -p[3]         # x >= -w
    yle = lambda p: p[1] <=  p[3]
    yge = lambda p: p[1] >= -p[3]
    zle = lambda p: p[2] <=  p[3]
    zge = lambda p: p[2] >= -p[3]
    wpos= lambda p: p[3] >   eps          # w > 0

    def _isect(a,b,sign,comp):  # solve sign*(a[comp]+t*(b-a)) = aw + t*(bw-aw)
        ax,ay,az,aw = a; bx,by,bz,bw = b
        dx,dy,dz,dw = bx-ax, by-ay, bz-az, bw-aw
        ca = (ax,ay,az)[comp]; da = (dx,dy,dz)[comp]
        num = aw - sign*ca
        den = sign*da - dw
        t = 0.0 if abs(den) < 1e-12 else float(num/den)
        t = min(max(t,0.0),1.0)
        return np.array([ax+dx*t, ay+dy*t, az+dz*t, aw+dw*t], dtype=np.float32)

    ixp = lambda a,b: _isect(a,b,+1,0); ixm = lambda a,b: _isect(a,b,-1,0)
    iyp = lambda a,b: _isect(a,b,+1,1); iym = lambda a,b: _isect(a,b,-1,1)
    izp = lambda a,b: _isect(a,b,+1,2); izm = lambda a,b: _isect(a,b,-1,2)

    poly = [v0,v1,v2]
    for pred, isect in [(xle,ixp),(xge,ixm),(yle,iyp),(yge,iym),(zle,izp),(zge,izm)]:
        poly = _clip_poly_plane(poly, pred, isect)
        if not poly:
            return []
    # guard w>0
    poly = _clip_poly_plane(poly, wpos, lambda a,b: np.array([a[0],a[1],a[2],max(eps, a[3])],dtype=np.float32))
    if not poly:
        return []
    out = []
    for i in range(1,len(poly)-1):
        out.append((poly[0], poly[i], poly[i+1]))
    return out

def clipmesh_cpu_to_ndc(MVP: np.ndarray, V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Return a packed triangle list (N,3) NDC by clipping in clip-space then dividing by w."""
    V4 = np.c_[V.astype(np.float32), np.ones((V.shape[0],1), np.float32)]
    C4 = (MVP @ V4.T).T
    tris_ndc = []
    for a,b,c in F.astype(np.int64):
        for t0,t1,t2 in _clip_triangle_clipspace(C4[a], C4[b], C4[c]):
            for t in (t0,t1,t2):
                x,y,z,w = t
                invw = 1.0 / max(w, 1e-12)
                tris_ndc.append((x*invw, y*invw, z*invw))
    return np.asarray(tris_ndc, dtype=np.float32)

# -------------- renderer (offscreen) --------------
def render_snapshot(width:int, height:int, fov_deg:float, near:float, far:float,
                    rvec:Tuple[float,float,float], tvec:Tuple[float,float,float],
                    grid_cfg, boxes:List[Tuple[float,float,float,float,float,float]],
                    meshes:List[Tuple[str,float,Tuple[float,float,float]]],
                    out_path:str, cpu_clip_mesh:bool=False):
    import moderngl, glfw
    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.DEPTH_BITS, 24)
    win = glfw.create_window(64, 64, "world-snapshot", None, None)
    if not win:
        glfw.terminate(); raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(win)

    ctx = moderngl.create_context(require=330)
    color = ctx.texture((width,height), components=3)
    color.filter = (moderngl.NEAREST, moderngl.NEAREST)
    depth = ctx.depth_renderbuffer((width,height))
    fbo = ctx.framebuffer(color_attachments=[color], depth_attachment=depth)

    # Shaders: lines (NDC) & tris (NDC) & tris with MVP (GPU)
    prog_lines = ctx.program(
        vertex_shader="""
            #version 330
            in vec3 in_pos;
            void main(){ gl_Position = vec4(in_pos, 1.0); }
        """,
        fragment_shader="""
            #version 330
            uniform vec3 u_color;
            out vec4 f_color;
            void main(){ f_color = vec4(u_color, 1.0); }
        """,
    )
    prog_ndc = ctx.program(
        vertex_shader="""
            #version 330
            in vec3 in_pos;
            void main(){ gl_Position = vec4(in_pos, 1.0); }
        """,
        fragment_shader="""
            #version 330
            uniform vec3 u_color;
            out vec4 f_color;
            void main(){ f_color = vec4(u_color, 1.0); }
        """,
    )
    prog_mvp = ctx.program(
        vertex_shader="""
            #version 330
            uniform mat4 u_mvp;
            in vec3 in_pos;
            void main(){ gl_Position = u_mvp * vec4(in_pos, 1.0); }
        """,
        fragment_shader="""
            #version 330
            uniform vec3 u_color;
            out vec4 f_color;
            void main(){ f_color = vec4(u_color, 1.0); }
        """,
    )

    aspect = width/height
    fov_rad = math.radians(fov_deg) if fov_deg > math.pi else fov_deg
    P = perspective(fov_rad, aspect, near, far)
    V = view_from_rt(np.array(rvec, np.float32), np.array(tvec, np.float32))
    PV = (P @ V).astype(np.float32)

    # Scene colors (match your defaults)
    sky = (180/255, 180/255, 210/255)
    grid_color = (150/255, 150/255, 150/255)
    boxes_color = (0.55, 0.55, 0.75)

    # Grid (CPU→NDC; z forced to 0 like your code)
    grid_world = build_grid_world(**grid_cfg)
    grid_ndc = project_ndc(PV, grid_world, force_z=0.0)
    vbo_grid = ctx.buffer(grid_ndc.astype("f4").tobytes())
    vao_grid = ctx.vertex_array(prog_lines, [(vbo_grid, "3f", "in_pos")])

    # Boxes (CPU→NDC; z forced to 0 like your code)
    box_lines = []
    for (x,y,z,w,d,h) in boxes:
        box_lines.append(box_edges_world(x,y,z,w,d,h))
    boxes_world = np.vstack(box_lines) if box_lines else np.empty((0,3), np.float32)
    boxes_ndc = project_ndc(PV, boxes_world, force_z=0.0) if len(boxes_world) else boxes_world
    vbo_boxes = ctx.buffer(boxes_ndc.astype("f4").tobytes() if len(boxes_ndc) else b"\x00"*4)
    vao_boxes = ctx.vertex_array(prog_lines, [(vbo_boxes,"3f","in_pos")])

    # Meshes
    mesh_vbos = []
    mesh_vaos = []
    for path,scale,col in meshes:
        V3,F = load_mesh(path, scale)
        if cpu_clip_mesh:
            tris_ndc = clipmesh_cpu_to_ndc(PV, V3, F)
            vbo = ctx.buffer(tris_ndc.astype("f4").tobytes() if len(tris_ndc) else b"\x00"*4)
            vao = ctx.vertex_array(prog_ndc, [(vbo,"3f","in_pos")])
            mesh_vbos.append((vbo, col, len(tris_ndc)))
            mesh_vaos.append(vao)
        else:
            vbo = ctx.buffer(V3.astype("f4").tobytes())
            ibo = ctx.buffer(F.astype(np.uint32).tobytes())
            vao = ctx.vertex_array(prog_mvp, [(vbo,"3f","in_pos")], index_buffer=ibo)
            mesh_vbos.append((vbo, col, ibo))
            mesh_vaos.append(vao)

    # ---- draw one frame ----
    fbo.use()
    ctx.viewport = (0,0,width,height)
    fbo.clear(*sky, 1.0)

    # Draw grid & boxes first (no depth write so meshes can appear on top)
    ctx.depth_mask = False
    prog_lines["u_color"].value = grid_color
    vao_grid.render(moderngl.LINES, vertices=grid_ndc.shape[0])
    if len(boxes_ndc):
        prog_lines["u_color"].value = boxes_color
        vao_boxes.render(moderngl.LINES, vertices=boxes_ndc.shape[0])
    ctx.depth_mask = True

    # Draw meshes with depth
    ctx.enable(moderngl.DEPTH_TEST)
    if cpu_clip_mesh:
        for (vbo,col,count), vao in zip(mesh_vbos, mesh_vaos):
            prog_ndc["u_color"].value = col
            vao.render(moderngl.TRIANGLES, vertices=count)
    else:
        for (vbo,col,ibo), vao in zip(mesh_vbos, mesh_vaos):
            prog_mvp["u_color"].value = col
            prog_mvp["u_mvp"].write(PV.T.tobytes())
            vao.render(moderngl.TRIANGLES)
    ctx.disable(moderngl.DEPTH_TEST)

    # Read back and save
    data = color.read(alignment=1)
    img = np.frombuffer(data, np.uint8).reshape(height, width, 3)
    img = np.flipud(img)
    Image.fromarray(img, "RGB").save(out_path)
    print(f"[snapshot] wrote {out_path}")

    # Cleanup
    try:
        for vao in mesh_vaos: vao.release()
        for v in mesh_vbos:
            if isinstance(v[2], moderngl.Buffer): v[2].release()
        vao_grid.release(); vbo_grid.release()
        vao_boxes.release(); vbo_boxes.release()
        fbo.release(); depth.release(); color.release(); ctx.release()
    finally:
        import glfw
        glfw.terminate()

# -------------- CLI --------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov-deg", type=float, default=60.0)
    ap.add_argument("--near", type=float, default=0.05)
    ap.add_argument("--far", type=float, default=2000.0)
    ap.add_argument("--rvec", type=float, nargs=3, default=[0,0,0], help="Rodrigues (radians)")
    ap.add_argument("--tvec", type=float, nargs=3, default=[0,0,6])
    ap.add_argument("--grid-xextent", type=float, default=40.0)
    ap.add_argument("--grid-znear", type=float, default=-2.0)
    ap.add_argument("--grid-zfar", type=float, default=-40.0)
    ap.add_argument("--grid-step", type=float, default=2.0)
    ap.add_argument("--box", type=float, nargs=6, action="append", default=[],
                    metavar=("x","y","z","w","d","h"))
    ap.add_argument("--mesh", type=str, action="append", default=[])
    ap.add_argument("--mesh-scale", type=float, default=1.0)
    ap.add_argument("--mesh-color", type=float, nargs=3, default=[0.8,0.8,0.8])
    ap.add_argument("--cpu-clip-mesh", action="store_true",
                    help="Use CPU→NDC with homogeneous clipping for meshes (slower).")
    ap.add_argument("--out", type=str, default="world.png")
    return ap.parse_args()

def main():
    args = parse_args()
    # Pack mesh specs (you can repeat --mesh to add more; share scale/color for simplicity)
    meshes = [(m, args.mesh_scale, tuple(map(float, args.mesh_color))) for m in args.mesh]
    grid_cfg = dict(x_extent=args.grid_xextent, z_near=args.grid_znear,
                    z_far=args.grid_zfar, step=args.grid_step, y=0.0)
    render_snapshot(
        width=args.width, height=args.height,
        fov_deg=args.fov_deg, near=args.near, far=args.far,
        rvec=tuple(args.rvec), tvec=tuple(args.tvec),
        grid_cfg=grid_cfg,
        boxes=[tuple(map(float,b)) for b in args.box],
        meshes=meshes,
        out_path=args.out,
        cpu_clip_mesh=args.cpu_clip_mesh
    )

if __name__ == "__main__":
    main()