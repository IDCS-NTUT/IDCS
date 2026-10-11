# Configuration

Main video/controller/operator services load **`configs/base`** and their
**overlays** in order; recorder and utility entry points may use dedicated files.
For the layered stacks,
later files override earlier ones (mappings merge recursively, lists and
scalars replace). The files in `base/` hold disjoint top-level sections, so
their order does not matter (a test enforces this).

```
base/        always loaded
  network.yaml     source, video profiles, net endpoints
  camera.yaml      camera intrinsics + known-size ranging, laser mount, perception
  control.yaml     image-plane aiming (legacy control section)
  controller.yaml  live video controller: gains, FF, limits, coasting, idle return
  gimbal.yaml      gimbal axes and limits, serial I/O service
  deepstream.yaml  DeepStream runtime: model, tracker, coast gate, id stitching
  sim.yaml         simulator defaults (renderer, default scene, plant)
  swarm.yaml       threat evaluation and the swarm planner / learned policy
  panel.yaml       Raspberry Pi panel GPIO and return-video display
  operator.yaml    agent modes, source/session selection, PC channel and HUD relay
bench/       this hardware
  uncoupled.yaml   bench axis limits and mapping (uncoupled direct drive)
  tuned.yaml       output of the tuning procedure (do not edit by hand)
  mks_parameters.yaml  MKS SERVO42D parameter reference
controller/  controller deployment overlays
  hil.yaml         Jetson controller live on the HIL loop (truth-fed by default)
  detections.yaml  steer on DeepStream detections instead of simulator truth
  sim_mount.yaml   PC controller driving the simulated mount
  local_camera.yaml  the Jetson's own IMX219 (also switches DeepStream to Argus)
  pid_only.yaml    feedforward/prediction off (A/B baseline; inherited coasting remains)
  camera_open_loop.yaml  explicit live bench-camera trial, not the ordinary camera service
sim/         simulator scenes and modes
  v2_scene.yaml    the standard OpenGL drone scene
  target_fast.yaml 1.0 m/s path with vertical legs
  swarm.yaml       swarm engagement scene (planner_eval)
  mode_hil.yaml    render from the real mount's measured pose
  mode_simulated_mount.yaml  simulated mount, no hardware
  renderer.yaml    OpenGL renderer look (read by the renderer directly)
pc/          operator_local.yaml: PC camera viewer's local operator preferences
recorder/    flight recorder profiles (jetson, sim, hil_pc)
tuning/      tuning procedure plan
training/    swarm policy dataset and model
deepstream/  nvinfer, tracker and label files referenced by base/deepstream.yaml
```

Standard stacks (`--config configs/base --config-extra ...`):

| Service | Overlays |
|---|---|
| DeepStream RTP (Jetson) | none |
| DeepStream local camera (Jetson) | `controller/local_camera.yaml` |
| serial service (Jetson) | `bench/uncoupled.yaml` |
| gimbal bridge (Jetson) | `bench/uncoupled.yaml,bench/tuned.yaml` |
| controller, HIL (Jetson) | `bench/uncoupled.yaml,controller/hil.yaml,bench/tuned.yaml` |
| HIL streamer (PC) | `sim/v2_scene.yaml,sim/mode_hil.yaml` |
| sim streamer (PC) | `sim/v2_scene.yaml,sim/target_fast.yaml,sim/mode_simulated_mount.yaml` |
| sim controller (PC) | `sim/v2_scene.yaml,controller/sim_mount.yaml,bench/tuned.yaml` |
| swarm streamer (PC) | `sim/v2_scene.yaml,sim/target_fast.yaml,sim/mode_simulated_mount.yaml,sim/swarm.yaml` |
| swarm controller (PC) | `sim/v2_scene.yaml,controller/sim_mount.yaml,bench/tuned.yaml,controller/detections.yaml` |
| camera controller, shadow (Jetson) | `bench/uncoupled.yaml,bench/tuned.yaml,controller/local_camera.yaml` |
| SIM UI (PC) | `sim/v2_scene.yaml,sim/mode_simulated_mount.yaml` plus loopback CLI endpoints |
| camera UI (PC) | `pc/operator_local.yaml` |
| HIL UI (PC) | `sim/v2_scene.yaml` plus physical pose CLI endpoint |

Variants add overlays at the end: `controller/detections.yaml` (real
detections), `sim/swarm.yaml` (swarm scene), `controller/local_camera.yaml`
(local camera, on both DeepStream and the controller).

## Known configuration gaps (2026-10-11)

The SIM stacks above really do load hardware tuning; this is not the intended
independent stable-substitute design. Controller bind addresses now preserve
their configured host, including loopback. Bridge/serial startup rejects missing
overlays and duplicate keys through the strict bundle loader. Deployment drift
scanning now covers added config and `.yml` files; unit command order still
requires review in the plan.
See [configuration architecture](../docs/configuration_architecture.md) and
[review findings](../docs/codebase_review_20261011.md) before relying on these gates.

## Old names (journal before 2026-09-28)

The journal and older reports name config files from before the
reorganization (commit `c68c64d`). Where each one went:

| Old file | Now |
|---|---|
| `system.yaml` | split into `base/controller.yaml`, `base/gimbal.yaml`, `base/panel.yaml`, `base/sim.yaml`, `base/swarm.yaml` |
| `control.yaml` | `base/control.yaml` |
| `network.yaml` | `base/network.yaml` |
| `perception.yaml` | `base/camera.yaml` |
| `deepstream_runtime.yaml` | `base/deepstream.yaml` |
| `deepstream_pc_moving_tracking.yaml`, `control_sim.yaml`, `deepstream_pc_moving_tracking_opengl.yaml`, `deepstream_pc_moving_drone_opengl.yaml` | merged into `sim/v2_scene.yaml` |
| `sim_mode_hil.yaml`, `sim_mode_simulated_mount.yaml` | `sim/mode_hil.yaml`, `sim/mode_simulated_mount.yaml` |
| `sim_target_fast.yaml`, `sim_swarm.yaml`, `renderer.yaml` | `sim/target_fast.yaml`, `sim/swarm.yaml`, `sim/renderer.yaml` |
| `local_camera.yaml` | `controller/local_camera.yaml` |
| `controller_detections.yaml`, `controller_pid_only.yaml` | `controller/detections.yaml`, `controller/pid_only.yaml` |
| `controller_sim_hil.yaml`, `controller_sim_mount.yaml` | `controller/hil.yaml`, `controller/sim_mount.yaml` |
| `gimbal_bench_uncoupled.yaml`, `tuned_gimbal.yaml`, `mks_parameters.yaml` | `bench/uncoupled.yaml`, `bench/tuned.yaml`, `bench/mks_parameters.yaml` |
| `recorder_sim.yaml`, `recorder_jetson.yaml`, `recorder_hil_pc.yaml` | `recorder/sim.yaml`, `recorder/jetson.yaml`, `recorder/hil_pc.yaml` |
| `tuning_plan.yaml` | `tuning/plan.yaml` |
| `swarm_dataset.yaml`, `swarm_model.yaml` | `training/swarm_dataset.yaml`, `training/swarm_model.yaml` |
| `controller_sysid_safety.yaml`, `deepstream_argus_runtime.yaml`, `deepstream_detector_sweep.yaml`, `deepstream_drone_file_validation.yaml`, `deepstream_drone_sim_validation.yaml`, `deepstream_hil_feedforward_fixture.yaml`, `deepstream_pc_shadow.yaml`, `deepstream_person_sim_validation.yaml` | removed as unused (`c68c64d`); in git history |
| `sim_scene_drone_ellipse_opengl.yaml` | removed 2026-09-27 (`ffb5eea`) when the V2 scene was restored |
| `control_sim_estimator_{ideal,graybox}.yaml` | removed 2026-09-27 with the V2 controller (`724d987`) |
| `dev_validation_file.yaml`, `shadow_yolo26s_best_current_736.yaml` | removed 2026-09-21 (`079a38b`) |

Files named in the journal without a `configs/` path (for example
`hil_live_ff_20260927.yaml`, `hil_two_axis_20260926.yaml`,
`v3_live_ff_fast_fixture.yaml`) were trial overrides in the external
toolkit, not repository configs.
