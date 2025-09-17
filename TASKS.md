# TASKS.md — Project Backlog (Actionable)

This backlog enumerates current and near-term work items for IDCS. Each task is sized for quick iteration and safe review.

## 0. Ground truth (docs & env)
- [X] Add `requirements_pc.txt` and `requirements_jetson.txt` with explicit versions.
- [V] Confirm `configs/dev.yaml` has separate `uplink` and `return` sections (width/height/fps/bitrate).
- [V] Ensure `README.md` and `AGENTS.md` are committed at repo root.

## 1. Pluggable sim renderer
- [ ] Create `pc/renderers/` package.
- [ ] Move current drawing logic from `pc/sim_camera.py` into `pc/renderers/cpu.py` (no behavior change).
- [ ] Add `sim.renderer` in YAML; update `pc/sim_camera.py` to select backend.
- [ ] Acceptance: `renderer=cpu` produces identical frames to before.

## 2. Billboard targets (sprites)
- [ ] Create `assets/billboards/` and add placeholder PNGs (alpha).
- [ ] Extend YAML with `sim.targets` list (cls, sprite path, size_m, motion path params).
- [ ] Implement CPU billboard compositing (scale from FOV & distance; alpha blend over background).
- [ ] Acceptance: with `source: sim` and configured targets, sprites appear stable at expected size.

## 3. ModernGL backend (grid + boxes)
- [ ] Add `pc/renderers/gl.py` using ModernGL (offscreen FBO at W×H).
- [ ] Render ground grid and boxes; `fbo.read()` → NumPy array (RGB) → BGR to NVENC.
- [ ] Config toggle: `sim.renderer: gl`.
- [ ] Acceptance: higher FPS or lower CPU than `cpu` path; image visually similar.

## 4. OBJ mesh targets (optional)
- [ ] Place cleaned models at `assets/models/person.obj` and `assets/models/drone.obj` (Y-up, meters, ~10–20k tris).
- [ ] In GL backend, load OBJ once into VBO/VAO; draw with a single shader (flat shading).
- [ ] Acceptance: meshes render at correct scale/position; pipeline FPS maintained.

## 5. Stability polish
- [ ] Receiver reopen: exponential backoff and more robust caps detection (appsink `NULL` handling).
- [ ] Return path: tune `vbv-size` and queue sizes to avoid periodic micro-stutter.
- [ ] Acceptance: continuous 30 FPS for >5 minutes under typical load.

## 6. Observability (optional)
- [ ] JSON logging (ts, frame_id, stage, ms) in streamer/server/ui.
- [ ] Periodic p50/p95 latency report every 60s.
- [ ] Acceptance: logs show stable latency; anomalies are visible.

## 7. Tests
- [ ] Unit: TRT postprocess box mapping back to W×H.
- [ ] Unit: schema JSON round-trip for `DetectionMsg`.
- [ ] Integration: sim source → Jetson → return loop smoke test script.
- [ ] Acceptance: tests pass locally; optional CI later.

---

### Prompts for code agents (copy/paste)

**A) Refactor (renderer plugin):** see AGENTS.md section “Ready-to-run prompts — A”.  
**B) Billboards:** see AGENTS.md section “Ready-to-run prompts — B”.  
**C) GL backend:** see AGENTS.md section “Ready-to-run prompts — C”.  
**D) OBJ meshes:** see AGENTS.md section “Ready-to-run prompts — D”.
