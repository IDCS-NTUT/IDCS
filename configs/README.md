# Configuration

Main video/controller/operator services load **`configs/base`** and their
**overlays** in order; recorder and utility entry points may use dedicated files.
For the layered stacks,
later files override earlier ones (mappings merge recursively, lists and
scalars replace). The files in `base/` hold disjoint top-level sections, so
their order does not matter (a test enforces this).

Managed software services load `~/.config/idcs/live.yaml` last. Use
`python -m tools.idcs_config --host pc|jetson` on the intended host for validated
live/idle/restart-tier changes. Fixed algorithm/model-format values live in
`common/config_constants.py` and hash-bound model metadata; they are rejected
as YAML overrides. Bridge and serial services keep release-only configuration.
See [operator guide](../docs/live_configuration.md),
[all key tiers](../docs/live_configuration_keys.md) and
[implementation report](../docs/live_configuration_report_20261017.md).

```
base/        always loaded
  network.yaml     source, video profiles, net endpoints
  camera.yaml      camera intrinsics + known-size ranging, laser mount, perception
  control.yaml     aiming mode, laser aiming policy and image-axis signs
  controller.yaml  live video controller: gains, FF, required rate cap, coasting, idle return
  gimbal.yaml      gimbal axes and limits, serial I/O service
  deepstream.yaml  DeepStream runtime: model, tracker, coast gate, id stitching
  sim.yaml         simulator defaults; scene comes from sim/v2_scene.yaml
  swarm.yaml       threat evaluation and the swarm planner / learned policy
  panel.yaml       Raspberry Pi panel GPIO and return-video display
  operator.yaml    agent modes, source/session selection, PC channel, HUD relay, recorder and lock timeout
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
sim/         simulator scenes and modes
  v2_scene.yaml    the standard OpenGL drone scene
  target_fast.yaml 1.0 m/s path with vertical legs
  swarm.yaml       swarm engagement scene (planner_eval)
  mode_hil.yaml    render from the real mount's measured pose
  mode_simulated_mount.yaml  simulated mount, no hardware
  renderer.yaml    OpenGL renderer look (read by the renderer directly)
  visual_lock_trial.yaml  documented isolated manual ROI/VPI trial (docs/visual_lock.md)
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

## Configuration cleanup (2026-10-17)

Aiming focal lengths come from `camera.intrinsics`; the physical calibration
remains 135 by 73 degrees. `controller.camera_fov_y_deg`, when explicitly set,
provides an aspect-derived simulated camera model without rewriting config.
`control` retains `aim_mode`, `laser.*`, and the live `sign_convention` used by
aiming and planning. MPC/debug-overlay/negotiation and the old PID/filter/loop
settings have been removed. Offline trajectory benchmarks use local PID gains
and a 50 Hz default instead of the old control section.

All managed simulator streamers load `sim/v2_scene.yaml`; base no longer carries
a second scene. `sim/swarm.yaml` supplies the complete planner-evaluation policy.
`sim.freeze_frame` remains live in `pc.streamer`; `sim.render_fps` was unused.
The live controller clock-domain keys `local_clock` and `source_clock` remain,
alongside the separate `clock.*` policy and `operator.modes.*.source_clock`.

The tuning procedure's generated output is unchanged. Its `bench/tuned.yaml`
restates the bridge's real 0.8 rad/s yaw and pitch caps, which are also present
in base after the review follow-up in `245d103`, and supplies the controller's
3.5 rad/s^2 acceleration cap. The base controller rate cap remains 0.8 rad/s:
the strict controller parser requires it for stacks without the tuning overlay,
so that one base/tuned duplication remains pending a separate policy decision.

Serial device/baud defaults belong to `tools.serial_io_service --port/--baud`.
`rpi.manual_control` now contains shared GPIO/ADC helpers only; the panel runs
`rpi.runtime_control` and `rpi.operator_display`. The old `rpi.return_video`
module and settings are removed. `common.config_sync` retains local loading and
merge helpers for existing imports; it has no network synchronization path.

The five removed trial config files and every unit stack's before/after map
are listed in the [cleanup report](../docs/config_dead_removal_20261017.md).
Managed deployment fails closed on removed files: the next release needs an
explicit removal review. `controller/pid_only.yaml`, `sim/visual_lock_trial.yaml`,
`deepstream/nvinfer_yolo26n_960.txt`, and `deepstream/labels_coco.txt` remain for
manual/compatibility use.

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

## Managed visual-lock source modes (2026-10-20 candidate)

`operator.modes.camera_visual_lock` replaces the Argus detector owner with
`idcs-visual-lock-camera.service` plus the unchanged camera shadow controller.
`operator.modes.sim_visual_lock` replaces the RTP detector owner with
`idcs-visual-lock-sim.service` and starts the existing PC Swarm software session.
All source modes stop both possible visual-lock owners before starting a new
source. New per-mode `health_file` paths are separate from detector readiness;
V2 source clock and receiver session must match fresh health/return frames.
The detector target classes, planner selection, catalog lock expiry and score
reset do not select or release a visual box.

The new units use explicit receiver CLI arguments (camera sensor0/mode4,
flip2/WB3 matching `controller/local_camera.yaml`, return destinations .1/.3,
ports5555/5564/5595/5596). Review these arguments with local camera/network
configuration when staging; these CLI values do not hot-reload from live YAML.
They use release-local `build/visual-lock` native libraries. The PC SIM panel
unit reads Pi lease5582 and Jetson receiver status5595; only its existing
loopback software-mount output can be enabled by a managed visual-lock lease.
No Jetson physical unit gains SIM settings; a static test guards all four
hardware/shadow unit templates against `configs/sim/` overlays.

Box presets are display preferences, not shared config: Pi's
`~/.config/idcs/operator_display.json` key `roi_box_size`, values `Small`,
`Medium` (default), `Large`, `X-large`. Tracker gates, ABI4, Harris0.5, CUDA
input and overlay-queue defaults are unchanged. Plain `live_trial` stays on
isolated5597; only an explicit matching `--managed-mode` permits5564.
