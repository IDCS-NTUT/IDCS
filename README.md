# IDCS

IDCS splits video, perception, operator input and motion authority across a PC,
Jetson and Raspberry Pi. This overview was cross-checked against the repository
on 2026-10-16; it is not a live host-status report.

## Runtime ownership

| Owner | Responsibility |
|---|---|
| PC `pc.streamer` | OpenGL scene/video source, correlated RTP identity and source timestamps; software mount or measured-pose HIL rendering |
| Jetson `jetson.deepstream.runtime` | Passive YOLO + NvDCF, V2 tracking/selection, GPU OSD and encoded return video |
| `jetson.control.video_runtime` | Fixed 50 Hz controller: proportional feedback through the basic PID core, separate Kalman target-rate feedforward, authority/timing/envelope gates |
| Jetson `jetson.gimbal_bridge` | Physical `ControlIntent` consumer, timed F6 commands and measured `CamState` |
| Jetson `tools.serial_io_service` | Managed RS485 bus owner; queueing, replies and execution events |
| Pi `rpi.runtime_control` | ADC/GPIO input, `ManualControlState`, indicator policy; no serial commands |
| Jetson `jetson.operator_agent` / PC `pc.mode_agent` | Source/session-scoped target locks and classes, coordinated source/UI modes and recording |
| PC/Pi viewers | Return-video presentation and display-only HUD; Pi `rpi.operator_display` sends menu commands through the agents |

DeepStream does not construct a controller or access serial hardware.
The software simulator is a separate `ControlIntent` consumer; it never sends
motor commands. `ControlCmd` remains compatibility/display code, not the active
physical control transport.

## Panel controls

GPIO12 fire-control enables automatic tracking. The latching GPIO5 control
switch independently selects manual joystick slew, requiring fresh panel
state, valid gimbal state and no E-stop. GPIO16 safety enables only the fire
button and its screen indication (`fire_arm`); it does not gate motion.
A tracked-target press while ARMED records `engage`; other presses record
`engage_refused`. No effector is connected. E-stop blocks all motion.

Only GPIO12/16 share power with their respective lamps. Lamp-off means the
contact is unreadable; auto/fire permission fails closed until a fresh powered
sample. Screen priority is **E-STOP > MANUAL > SAFE > ARMED** (or NO PANEL
when stale). See [panel and lamp semantics](docs/operator_display.md).

## Modes and video

- **Camera:** local Jetson Argus video and visual detections; the normal camera
  controller is shadow-only.
- **Simulator / Single:** software mount with controlled synthetic observations.
- **Simulator / Swarm:** software mount controlled from Jetson visual detections;
  simulator truth supplies scene scoring, not visual tracker identities.
- **HIL:** `sim.use_jetson_cam_state=true` renders from fresh measured physical
  pose, with no simulated fallback. This is not an ordinary console source mode.

The current source profiles target 60 fps; encoded return is 1280x720 at 30 fps.
The OpenGL scene uses 3D meshes. Learned detection accuracy, simulation camera
fidelity, controller performance and display latency are distinct qualifications.

**Standing bench restriction:** do not start motor/HIL, bridge or serial owners
until the user confirms repaired wiring and responses from every required RS485
motor. The pitch communication failure recorded on 2026-09-29 remains the
documented gate; this review did not re-test devices.

Software SIM is intended to be independent of hardware tuning. Its current units
still load `configs/bench/tuned.yaml` and the streamer quantizes motion through
the F6 model. This unresolved coupling, along with other confirmed issues, is
recorded in the [codebase review](docs/codebase_review_20261011.md).

## Configuration and deployment

Version-controlled `configs/base` is followed by ordered service overlays;
later values override earlier ones. See [config stacks](configs/README.md).
Strict immutable loading now covers the main video/controller/panel paths and
the bridge/serial startup paths, including required overlay checks:
[configuration architecture](docs/configuration_architecture.md).

Systemd templates under `deploy/systemd` use `@RELEASE@`. The managed
deployment workflow explicitly stages, validates, activates, verifies and can
restore units. Repository sync does not deploy services. Current-source evidence
is in host manifests, not dated trial paths copied from the journal:
[deployment](docs/deployment.md), [operations](docs/launch_procedure.md).

`scripts/prepare_jetson_runtime.sh` builds the custom nvinfer parser, links
the untracked models and builds the swarm-policy and grey-letterbox detector
TensorRT engines. See [engine build notes](docs/tensorrt_engine_build.md) for
the detector's frame-size constraints and rebuild requirements.

## Verification and documentation

Check running owners before any test; never stack video/GPU/serial processes.
Use synthetic contracts to isolate policy from learned detection. Passive local
configuration checks include:

```bash
python -m jetson.deepstream.runtime --check
python -m jetson.control.video_runtime --config configs/base --config-extra configs/controller/local_camera.yaml --check
```

A configuration check is not camera, motor or timing acceptance.
Native DeepStream/VPI/OpenGL integration requires the appropriate Linux/GPU
environment and separate bounded tests. See the [documentation index](docs/README.md),
[verification strategy](docs/verification_strategy.md),
[perception architecture](docs/perception_architecture.md), and
[2026-10-11 review results](docs/codebase_review_20261011.md) for coverage and gaps.
