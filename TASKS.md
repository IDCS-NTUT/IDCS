# TASKS.md — Project Backlog (Actionable)

This backlog enumerates current and near-term work items for IDCS. Each task is sized for quick iteration and safe review.

## 0. Ground truth (docs & env)
- [V] Document core dependencies via `pyproject.toml` extras for PC and Jetson installs.
- [ ] Split `configs/dev.yaml` video settings into explicit `uplink` and `return` sections.
- [V] Keep `README.md` and `AGENTS.md` up to date at the repository root.

## 1. Pluggable sim renderer
- [V] Create `pc/renderers/` package.
- [V] Move current drawing logic from `pc/sim_camera.py` into `pc/renderers/cpu.py` (no behavior change).
- [V] Add `sim.renderer` in YAML; update `pc/sim_camera.py` to select backend.
- [V] Acceptance: `renderer=cpu` produces identical frames to before.

## 2. Billboard targets (sprites)
- [V] Add placeholder sprite PNGs (alpha) under `assets/` and expose them via `_SPRITE_ALIASES`.
- [V] Extend YAML with `sim.targets` list (cls, sprite path, size_m, motion path params).
- [V] Implement CPU billboard compositing (scale from FOV & distance; alpha blend over background).
- [V] Acceptance: with `source: sim` and configured targets, sprites appear stable at expected size.

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

## 8. Parallax-aware laser aim mode
- [ ] Extend `configs/*.yaml` with `control.aim_mode`, laser mount (offset + direction), range policy, tolerance, and render styling defaults.
- [ ] Update `common/control.ControlConfig` (and related dataclasses) to parse/validate the new laser configuration, enforce unit vectors, and surface defaults for legacy configs.
- [ ] Create shared geometry helpers to map pixels ↔ camera-frame rays, compute laser origin/direction with offsets, and intersect with range or ground plane fallbacks.
- [ ] Add laser-specific telemetry (`laser_dot_px`, `laser_on_target`, `parallax_compensation_active`, etc.) to `DetectionMsg` / `ControlCmd`, ensuring optional fields still serialize cleanly.
- [ ] Branch the Jetson controller so `aim_mode="laser_point"` uses laser-dot vs. target error, reuses PID gains, and flags on-target status using configured tolerance.
- [ ] Render laser beam/dot overlays on the return video using controller telemetry or shared helpers, with colours driven by config and clear status annotations.
- [ ] Handle range policies: prefer smoothed known-size distance, fall back to large-distance or ground-plane intersection (with camera height/pitch) when data is missing or jittery.
- [ ] Document calibration steps (laser offset/direction, range policy), guard rails (saturation, unreachable solutions), and multi-distance acceptance tests.

(V = Done, X = skipped)

---

### Prompts for code agents (copy/paste)

**A) Refactor (renderer plugin):** see AGENTS.md section “Ready-to-run prompts — A”.  
**B) Billboards:** see AGENTS.md section “Ready-to-run prompts — B”.  
**C) GL backend:** see AGENTS.md section “Ready-to-run prompts — C”.  
**D) OBJ meshes:** see AGENTS.md section “Ready-to-run prompts — D”.
