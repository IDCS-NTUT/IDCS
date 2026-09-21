# IDCS

IDCS is a split PC/Jetson tracking system. The production video and metadata
pipeline is V2-only: DeepStream publishes immutable perception snapshots,
the host displays the returned H.264 stream and V2 telemetry, and a separate
fixed-rate controller may publish commands to the gimbal bridge.

## Runtime topology

```text
PC camera/simulator
  -> RTP H.264 + frame headers
  -> Jetson DeepStream detector + NvSORT + selector
  -> PerceptionSnapshotV2 (:5564)
       -> PC/RPi display and metadata monitor
       -> jetson.control_runtime
  -> GPU OSD + RTP return video (:5002)

jetson.control_runtime
  <- CamState (:5558)
  <- ManualControlState (:5559)
  -> ControlCmd (:5557)
  -> jetson.gimbal_bridge -> serial I/O service -> motor controller
```

DeepStream is passive: it cannot publish a command or access serial hardware.
The controller cannot access serial hardware. The gimbal bridge is the only
consumer of production `ControlCmd` messages, and the serial service is the
only process that owns the serial device.

## Main entry points

- `python -m pc.streamer`: source video, RTP, and correlated frame headers.
- `python -m pc.ui`: V2 operator display and return-video receiver.
- `python -m jetson.deepstream.runtime`: production detector/tracker/selector
  and GPU return-video pipeline.
- `python -m jetson.control_runtime`: production fixed-rate PID/MPC controller.
- `python -m jetson.sim_control_runtime`: loopback-only stable simulator
  controller; it refuses the production command endpoint.
- `python -m jetson.gimbal_bridge`: command-to-gimbal translation and CamState.

The convenience launchers are:

```bash
# Passive video only; no controller, bridge, or serial process.
scripts/run_jetson.sh --check
scripts/run_jetson.sh

# Explicit live hardware stack. This starts video, controller, bridge, and
# serial processes and stops all of them when any component exits.
scripts/run_jetson_with_gimbal.sh
```

`jetson.control_runtime` requires `--enable-control-publish` before it binds
the production command socket. Missing, stale, manual, or emergency authority
produces zero-rate commands; it never substitutes home motion for a disarmed
state.

## Simulator contract

Simulation has two intentionally separate control modes:

- The default baseline mode uses `sim.baseline_controller`, a bounded stable
  substitute for end-to-end video, detection, tracking, UI, and system-flow
  evaluation. It does not load or tune hardware controller artifacts.
- Hardware-controller observation mode may exercise the tuned controller
  interface against a simulated mount, but its results must not be used to
  tune physical hardware.

Detector and display verification should use synthetic or rendered targets
whose registration is controlled. Model acceptance, tracker behavior, UI,
control policy, and physical motion qualification remain separate evidence.

The simulator and hardware controller share `common.aiming`: selected target
pixels, known-size range provenance, configured mount offset, projected
parallax aim point, pixel error, and bearing error are computed once and
carried in `ControlObservation` without bearing-to-pixel reconstruction.

## Configuration

The usual merge order is:

```text
configs/network.yaml
configs/perception.yaml
configs/control.yaml
configs/system.yaml
runtime or simulator override
```

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
python -m jetson.control_runtime --check
python -m jetson.sim_control_runtime --check
pytest -q
```

See `docs/verification_strategy.md`, `docs/perception_architecture.md`, and
`docs/deepstream_migration_journal.md` for contracts and evidence history.
