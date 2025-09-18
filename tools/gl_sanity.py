# tools/gl_sanity.py
import sys, numpy as np, cv2, glfw, moderngl as mgl

W, H = 640, 360

def main():
    if not glfw.init():
        print("glfw.init() failed")
        sys.exit(1)

    # Hidden window solely to create a valid GL context on Windows
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    win = glfw.create_window(W, H, "hidden", None, None)
    if not win:
        print("glfw.create_window failed")
        glfw.terminate()
        sys.exit(1)

    glfw.make_context_current(win)
    ctx = mgl.create_context()
    print("GL version:", ctx.version_code, "vendor:", ctx.info["GL_VENDOR"], "renderer:", ctx.info["GL_RENDERER"])

    # Offscreen framebuffer with 3 components (RGB8)
    fbo = ctx.simple_framebuffer((W, H), components=3)
    fbo.use()
    ctx.viewport = (0, 0, W, H)

    # CLEAR to a BRIGHT color (not black)
    ctx.clear(0.2, 0.8, 0.1, 1.0)  # G-ish

    ctx.finish()
    raw = fbo.read(components=3, alignment=1)  # returns bytes, row 0 is bottom
    rgb = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3)
    bgr = rgb[:, :, ::-1]
    # Flip vertically so it looks natural
    bgr = np.flipud(bgr)

    cv2.imwrite("gl_sanity.png", bgr)
    print("Wrote gl_sanity.png")

    glfw.destroy_window(win)
    glfw.terminate()

if __name__ == "__main__":
    main()
