# Configuration

Every service loads **`configs/base`** and then its **overlays** in order;
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
bench/       this hardware
  uncoupled.yaml   bench axis limits and mapping (uncoupled direct drive)
  tuned.yaml       output of the tuning procedure (do not edit by hand)
  mks_parameters.yaml  MKS SERVO42D parameter reference
controller/  controller deployment overlays
  hil.yaml         Jetson controller live on the HIL loop (truth-fed by default)
  detections.yaml  steer on DeepStream detections instead of simulator truth
  sim_mount.yaml   PC controller driving the simulated mount
  local_camera.yaml  the Jetson's own IMX219 (also switches DeepStream to Argus)
  pid_only.yaml    feedforward off (A/B baseline)
sim/         simulator scenes and modes
  v2_scene.yaml    the standard OpenGL drone scene
  target_fast.yaml 1.0 m/s path with vertical legs
  swarm.yaml       swarm engagement scene (planner_eval)
  mode_hil.yaml    render from the real mount's measured pose
  mode_simulated_mount.yaml  simulated mount, no hardware
  renderer.yaml    OpenGL renderer look (read by the renderer directly)
recorder/    flight recorder profiles (jetson, sim, hil_pc)
tuning/      tuning procedure plan
training/    swarm policy dataset and model
deepstream/  nvinfer, tracker and label files referenced by base/deepstream.yaml
```

Standard stacks (`--config configs/base --config-extra ...`):

| Service | Overlays |
|---|---|
| DeepStream (Jetson) | none |
| serial service (Jetson) | `bench/uncoupled.yaml` |
| gimbal bridge (Jetson) | `bench/uncoupled.yaml,bench/tuned.yaml` |
| controller, HIL (Jetson) | `bench/uncoupled.yaml,controller/hil.yaml,bench/tuned.yaml` |
| HIL streamer (PC) | `sim/v2_scene.yaml,sim/mode_hil.yaml` |
| sim streamer (PC) | `sim/v2_scene.yaml,sim/target_fast.yaml,sim/mode_simulated_mount.yaml` |
| sim controller (PC) | `sim/v2_scene.yaml,controller/sim_mount.yaml,bench/tuned.yaml` |
| operator UI (PC) | `sim/v2_scene.yaml` |

Variants add overlays at the end: `controller/detections.yaml` (real
detections), `sim/swarm.yaml` (swarm scene), `controller/local_camera.yaml`
(local camera, on both DeepStream and the controller).
