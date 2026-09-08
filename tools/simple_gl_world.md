# Simple GL World Preview

Standalone OpenGL preview for a minimal 3D world (ground plane only). This is intentionally decoupled from the SimCamera and streaming pipeline so it can serve as a reference for future Jetson-compatible rendering.

## Requirements

- Python with optional PC dependencies installed (moderngl, glfw, opencv-python is not required).

## Run

```bash
python tools/simple_gl_world.py --width 1280 --height 720 --fps 30
```

### Options

- `--frames N` to render N frames and exit (useful for smoke tests).
- `--window-title "Title"` to set the window name.

## Notes

- This script uses only glfw + moderngl + numpy. No SimCamera or renderer registry is used.
- Geometry is a single ground plane with a subtle grid-like shading pattern.
