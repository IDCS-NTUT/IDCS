# IDCS

IDCS is a split PC/Jetson tracking system. DeepStream publishes immutable
perception snapshots and an annotated return stream; a fixed-rate video
controller turns observations into short-lease rate intents; the gimbal bridge
is the only process that turns intents into motor commands.

## Runtime topology

```text
PC camera/simulator (RTP H.264 + frame headers)  or  Jetson IMX219 (Argus, camera mode)
  -> Jetson DeepStream: YOLO26s (grey-letterbox TensorRT engine) + NvDCF + selector
       selector = swarm planner, overridden by the operator's lock and target classes
  -> PerceptionSnapshotV2 (:5564)
       -> PC UI, Pi panel screen, metadata monitor, operator agent
       -> jetson.control.video_runtime
  -> GPU OSD + RTP return video (:5002) to the PC UI and mirrored to the Pi screen

Pi panel (rpi.runtime_control): switches + joystick -> ManualControlState (:5559)
Pi screen (rpi.operator_display): menu -> OperatorCommand (:5590)
  -> jetson.operator_agent -> OperatorSelection (:5591, loopback) -> DeepStream selector
                           -> mode / recording unit start and stop

jetson.control.video_runtime
  <- CamState (:5558, step-count pose with per-axis sample times)
  <- ManualControlState (:5559, Pi safety panel)
  <- source-clock exchange (:5575)
  -> ControlIntent (:5557, live mode only)
  -> jetson.gimbal_bridge -> serial I/O service -> MKS SERVO42D motors
```

Panel controls: the safety switch is the master arm (no auto or manual motion
without it); the fire-control switch enables auto control; the control
switch selects manual, in which the joystick slews the gimbal through the
controller; the fire button records an engagement of the current target (no
effector); the E-stop blocks all motion. Details: `docs/operator_display.md`.

DeepStream is passive: it cannot publish a command or access serial hardware.
The controller cannot access serial hardware. The gimbal bridge is the only
consumer of `ControlIntent` messages, and the serial service is the only
process that owns the serial device.

## Main entry points

- `python -m pc.streamer`: source video, RTP, and correlated frame headers;
  in hardware-in-loop mode it renders from the measured gimbal pose.
- `python -m pc.ui`: operator display and return-video receiver.
- `python -m jetson.deepstream.runtime`: production detector/tracker/selector
  and GPU return-video pipeline.
- `python -m jetson.control.video_runtime`: fixed-rate video controller
  (PID + target-rate feedforward). All policy comes from the validated
  `controller` config section; `mode: shadow` never publishes.
- `python -m jetson.gimbal_bridge`: intent-to-F6 translation, axis enable with
  ACK check, limits, and step-count CamState.
- `python -m tools.serial_io_service`: sole owner of the RS485 bus.
- `python -m jetson.operator_agent`: the panel screen's commands (target lock,
  target classes, mode, recording).
- `python -m rpi.runtime_control` (Pi): panel switches and joystick.
- `python -m rpi.operator_display` (Pi): return video, status bar and menu on
  the panel screen.

## Deployment

Units live in `deploy/systemd/{jetson,rpi,pc}` and run from a clean
`IDCS-runtime` checkout at a tagged commit on each host:

```bash
# deploy a tag to the runtime checkouts (no restarts)
git tag deploy-N && scripts/deploy.sh deploy-N pc,jetson,pi

# Jetson (system units); idcs-operator-agent and idcs-deepstream-video start at boot
sudo systemctl start idcs-camera.target  # own IMX219: DeepStream, shadow controller, recorder
sudo systemctl start idcs-hil.target     # serial -> bridge -> controller (motors; bench only)
# Pi (user units, at boot): idcs-manual (panel), idcs-operator-display (screen)
# PC (user unit): idcs-hil-streamer.service, idcs-ui (return video)

# PC only, no hardware: the same controller drives a simulated mount
systemctl --user start idcs-sim.target  # sim streamer + sim panel + controller
```

Both loops render the V2 simulator scene (`deepstream_pc_moving_tracking.yaml`,
`control_sim.yaml`, `deepstream_pc_moving_tracking_opengl.yaml`,
`deepstream_pc_moving_drone_opengl.yaml`: OpenGL mesh drone, building, daylight
sky, 135x73 deg camera); `sim_mode_hil.yaml` or `sim_mode_simulated_mount.yaml`
selects the plant. The simulated mount runs intents through the measured F6
speed model, and `tools.sim_panel` supplies the armed state (loopback only).
The controller publishes read-only `ControlDiagnostics` for the HUD
(`idcs-ui` / `idcs-sim-ui`).
`idcs-sim.target` and `idcs-hil-streamer` conflict (shared ports).

On the Jetson, run `scripts/prepare_jetson_runtime.sh` once per new runtime
checkout: it builds the custom nvinfer parser, links the untracked models and
builds the TensorRT engines (swarm policy; detector with grey letterbox bars,
`tools/onnx_grey_letterbox.py`).

Every service runs `--check` as `ExecStartPre`. Missing or stale panel state,
master arm off, or emergency yields zero-rate intents (manual mode yields the
joystick's slew); stopping the controller publishes
explicit zero-rate intents and stopping the bridge de-energizes the axes.
Starting, stopping and deploying the services: `docs/launch_procedure.md`.

## Simulator contract

The simulator (`sim.use_jetson_cam_state`) has two modes:

- `stable_substitute` (false): a simulated mount for end-to-end video,
  detection, tracking, and UI evaluation. No in-tree controller drives it;
  controller gains come from `tools/latency_gain_sweep.py` and hardware sweeps.
- `hardware_in_loop` (true): the camera renders the world at `now - D` from
  the measured step-count pose, and `--sim-perception-pub` publishes exact
  ground-truth snapshots `--sim-total-latency-ms` after capture.

Detector and display verification should use synthetic or rendered targets
whose registration is controlled. Model acceptance, tracker behavior, UI,
control policy, and physical motion qualification remain separate evidence.

The simulator and hardware controller share `common.aiming`: selected target
pixels, known-size range provenance, configured mount offset, projected
parallax aim point, pixel error, and bearing error are computed once and
carried in `ControlObservation` without bearing-to-pixel reconstruction.

## Configuration

Every service loads `configs/base` (a directory of topic files with disjoint
sections) and then its overlays in order, e.g.
`--config configs/base --config-extra configs/bench/uncoupled.yaml,configs/bench/tuned.yaml`.
See `configs/README.md` for the layout and the stack each service uses.
How to start, stop, check and deploy the services: `docs/launch_procedure.md`.

Production perception uses `net.zmq_perception_v2`. There is no legacy result
socket or mutable detection-message transport. Return video has its own active
profile, currently allowing 60 FPS input with a 1280x720 30 FPS return stream.

## Verification

Prefer the lowest-risk layer that proves the behavior:

1. pure unit tests;
2. recorded V2 replay;
3. deterministic synthetic perception;
4. visual detector fixtures and rendered scenes;
5. control-free live video;
6. hardware-in-loop with actuation disabled;
7. bounded unloaded hardware only when explicitly authorized.

Useful checks:

```bash
python -m jetson.deepstream.runtime --check
python -m jetson.control.video_runtime --config configs/base --config-extra configs/controller/hil.yaml --check
python -m jetson.gimbal_bridge --config-extra ... --check
pytest -q
```

Gains, feedforward and actuator limits come from the standard procedure in
`docs/tuning_procedure.md` (`python -m tools.tuning`).

Current state, standing rules and open items for anyone picking up the work:
the top section of `AGENTS.md`.

See `docs/verification_strategy.md`, `docs/perception_architecture.md`,
`docs/controller_architecture.md`, and
`docs/deepstream_migration_journal.md` for contracts and evidence history.
