import glfw
import moderngl
import numpy as np
import trimesh
from pyrr import Matrix44

VERT_SHADER = """
#version 330
in vec3 in_position;
in vec3 in_normal;
uniform mat4 MVP;
out vec3 v_normal;
void main() {
    gl_Position = MVP * vec4(in_position, 1.0);
    v_normal = in_normal;
}
"""

FRAG_SHADER = """
#version 330
in vec3 v_normal;
out vec4 f_color;
void main() {
    vec3 n = normalize(v_normal);
    vec3 ldir = normalize(vec3(1.0, 1.0, 1.0));
    float l = max(dot(n, ldir), 0.0);
    f_color = vec4(vec3(0.4, 0.7, 0.9) * l, 1.0);
}
"""

def load_trimesh(path):
    loaded = trimesh.load(path)

    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded

    mesh.rezero()
    mesh.apply_scale(3.0 / mesh.scale)

    mesh.fix_normals()

    if mesh.faces.shape[1] != 3:
        mesh = mesh.triangulate()

    return mesh

def main():
    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 600, "OBJ Renderer", None, None)
    glfw.make_context_current(window)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)

    # --- load mesh ---
    mesh = load_trimesh("assets/person.obj")

    # Vertex buffer (positions + normals)
    vertices = np.hstack([
        mesh.vertices.astype("f4"),
        mesh.vertex_normals.astype("f4"),
    ])

    vbo = ctx.buffer(vertices.tobytes())

    # Index buffer (faces)
    ibo = ctx.buffer(mesh.faces.astype("u4").tobytes())

    prog = ctx.program(
        vertex_shader=VERT_SHADER,
        fragment_shader=FRAG_SHADER,
    )

    vao = ctx.vertex_array(
        prog,
        [(vbo, "3f 3f", "in_position", "in_normal")],
        index_buffer=ibo,
    )

    proj = Matrix44.perspective_projection(60.0, 800 / 600, 0.1, 100.0)
    view = Matrix44.look_at(
        eye=(3, 3, 3),
        target=(0, 0, 0),
        up=(0, 1, 0),
    )

    while not glfw.window_should_close(window):
        ctx.clear(0.05, 0.05, 0.05)

        angle = glfw.get_time()
        model = Matrix44.from_y_rotation(angle)

        mvp = proj * view * model
        prog["MVP"].write(mvp.astype("f4"))

        vao.render()

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()

