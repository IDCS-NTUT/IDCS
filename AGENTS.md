# AGENTS.md — Guidance for Code Agents (Codex)

This document tells a coding agent how to work in this repo: how to run, test, and propose changes safely.

## Environments

- **PC (Windows, RTX 2070)**
  - Python 3.11 env `idcs` (Miniforge/Mamba)
  - GStreamer with NVENC (`nvh264enc`) and `avdec_h264`
  - OpenCV with GStreamer
- **Jetson (Xavier NX, JetPack 5.x)**
  - Python 3.8 `venv`
  - OpenCV 4.12 (CUDA+GStreamer), TensorRT, pycuda

## Canonical commands

### Run (three processes)
```bash
# Jetson
source ~/Desktop/project/venv/bin/activate
python -m jetson.server --config configs/dev.yaml

# PC sender
mamba activate idcs
python -m pc.streamer --config configs/dev.yaml

# PC UI
python -m pc.ui --config configs/dev.yaml
```

### Quick checks
```bash
# GStreamer elements
gst-inspect-1.0 nvh264enc avdec_h264 nvv4l2decoder nvv4l2h264enc

# OpenCV build info
python - << 'PY'
import cv2; print(cv2.getBuildInformation())
PY
```

## Coding conventions

- Python 3.8+ (Jetson) / 3.11 (PC)
- Type hints where practical
- Avoid blocking I/O in main loops; use non-blocking ZMQ with short polls
- GStreamer pipelines: prefer explicit caps and leaky queues before appsink/appsrc
- Keep **uplink** and **return** configs separate (width/height/fps/bitrate)

## Tests / validation

- **Smoke (PC only):**
  - `python -m pc.streamer --config configs/dev.yaml` with `source: sim` should run without error and print “Sent N frames…”.
- **Loopback (Jetson ingest):**
  - From PC, run sender; on Jetson, run server. Expect receiver to enter PLAYING and print FPS periodically; ZMQ PUB should start publishing.
- **UI:**
  - PC UI must open a window and display frames (return path). ESC exits cleanly.

(Planned) Unit tests:
- CUDA letterbox mapping: normalized coords map back to pixel coords correctly for arbitrary W×H.
- Message schema round-trip via `DetectionMsg.model_dump_json()`.

## Branching & PRs

- Create feature branches: `feat/<short>` or `fix/<short>`.
- Keep changes narrowly scoped and runnable.
- Include a brief description + how you tested (commands + expected outputs).
- Don’t commit large binaries; put models in `assets/` and keep ≤ ~10 MB per file.

## Task backlog (prioritized)

1. **Sim renderer backends**
   - Create `pc/renderers/` package.
   - Move current CPU drawing into `pc/renderers/cpu.py`.
   - Add config key `sim.renderer` and route from `pc/sim_camera.py`.

2. **Billboard targets (sprites)**
   - Add `assets/billboards/{person_*.png, drone_*.png}`.
   - Implement camera-facing quads; scale from FOV & distance; alpha-blend (CPU first).
   - Add YAML `sim.targets` with class, sprite, size_m, path params.

3. **OpenGL backend (ModernGL)**
   - `pc/renderers/gl.py`: offscreen context + FBO render of grid/boxes.
   - Read back to NumPy (RGB) → BGR → NVENC path unchanged.
   - Later: add billboards as textured quads.

4. **OBJ mesh targets (optional)**
   - Loader (trimesh or simple OBJ) → VAO/VBO
   - Draw low-poly `assets/models/{person.obj, drone.obj}` with flat shading.

5. **Stability polish**
   - Receiver reopen strategy (backoff) and better caps checks.
   - Return path: maintain FPS even under brief stalls (tune queues/vbv).

6. **Observability (optional)**
   - Structured JSON logs (src_ts, frame_id, stage, ms).
   - Periodic p50/p95 latency report every 60s.

## Ready-to-run prompts for a code agent

### A) Refactor: pluggable sim renderer
> Create `pc/renderers/cpu.py` and move drawing logic from `pc/sim_camera.py` into it. Keep the public API unchanged: `SimCamera.next_frame()` still returns a BGR frame. Add `sim.renderer` to `configs/dev.yaml` and let `pc/sim_camera.py` pick `cpu` by default. No functional changes expected; render output should be identical.

Acceptance:
- `python -m pc.streamer --config configs/dev.yaml` runs as before (`renderer: cpu`).
- Git diff shows drawing code moved out; `sim_camera.py` now delegates to the backend.

### B) Feature: billboard sprites
> Add billboard targets. Create `assets/billboards/` (placeholders OK). Extend `configs/dev.yaml` with a `sim.targets` list (cls, mode, sprite, size_m, path). Implement CPU billboard rendering (alpha compositing) and motion paths (circle/figure8/random). Draw billboards after background/grid.

Acceptance:
- With `source: sim` and configured targets, frames include sprites at expected sizes/positions.

### C) Feature: ModernGL backend (grid + boxes)
> Implement `pc/renderers/gl.py` using ModernGL. Render ground grid + boxes into an offscreen FBO at W×H. Return image as NumPy RGB. Wire selection via `sim.renderer: gl`.

Acceptance:
- `renderer: gl` runs; FPS higher or CPU lower vs `cpu`.

### D) Feature: OBJ mesh targets
> Load `assets/models/person.obj` and `drone.obj` (assume Y-up, in meters). Draw with a single shader (flat color). Per-frame model transform from target’s pose.

Acceptance:
- Targets render as 3D meshes in `renderer: gl` mode.

## Guardrails for agents

- Don’t change GStreamer pipelines unless necessary; when changed, explain exact caps/nodes added/removed.
- Keep uplink/return configs respected everywhere.
- Do not introduce blocking network calls in the main UI or server loops.
- Large refactors must preserve existing behavior (PC↔Jetson round trip still works).

## Appendix: Key files to read first

- `configs/dev.yaml`
- `pc/streamer.py` (NVENC pipeline, header PUSH)
- `jetson/server.py` (GRecv, TRT, ZMQ PUB, return encoder)
- `pc/ui.py` (return RX + HUD)
- `jetson/yolo_engine.py` (TRT bindings, preprocess)
