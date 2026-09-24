# Estimator simulation environment

This environment runs the qualified LOS Kalman/feedforward policy through the
real V2 perception, observation, command-adaptation, simulated-camera, and UI
interfaces. It is an analysis environment, not a physical-controller tuning or
qualification source.

## Safety boundary

- Simulator commands bind only to `tcp://127.0.0.1:5571`.
- Simulated CamState binds only to `tcp://127.0.0.1:5572`.
- Read-only `ControlDiagnostics` binds only to `tcp://127.0.0.1:5573`.
- Deterministic simulator ground truth binds only to `tcp://127.0.0.1:5574`.
- `jetson.sim_control_runtime` rejects non-loopback endpoints and refuses
  `net.zmq_control`.
- The runtime imports no serial or gimbal driver. Both profiles set
  `sim.use_jetson_cam_state: false`.
- `study_only: true` is mandatory before a qualified controller artifact can
  be loaded. Simulation results cannot authorize changes to hardware tuning.

## Profiles

- `configs/control_sim_estimator_ideal.yaml` uses ideal rate integration. Use
  it to isolate estimator behavior from actuator-model error.
- `configs/control_sim_estimator_graybox.yaml` uses the independently validated
  asymmetric hardware-derived plant. Use it only as a robustness gate after
  the ideal profile.

Both profiles load the same immutable qualified estimator report as a
controller-under-test. The normal `configs/control_sim.yaml` baseline remains
estimator-free and continues to use ideal motion.

The service templates add
`configs/deepstream_estimator_drone_fixture.yaml` after the general moving
target profile. It supplies one close, slow CPU-rendered drone billboard whose
width matches the configured known-size range model. The drone class is
eligible under the production target-selection policy; `person` is deliberately
excluded there. OpenGL native-mesh overlays remain excluded because their
intermittent detector misses would confound estimator resets, feedforward, and
limiter analysis. They belong to a later end-to-end stress test after the
controlled profiles pass.

Controller/estimator acceptance does not consume the detector output. The
streamer projects the same rendered target into a guaranteed selected V2
snapshot on port 5574, while the rendered video still traverses DeepStream.
This separates motion-control correctness from model recall without bypassing
or disguising detector measurements; detector/tracker/selection coverage is
reported independently from passive captures of port 5564.

## Start and stop

The template instance is either `ideal` or `graybox`. Starting the UI starts
its required controller and streamer:

```text
systemctl --user start idcs-v2-estimator-sim-ui@ideal.service
```

Stop a capture explicitly in reverse order so reports and zero commands are
flushed:

```text
systemctl --user stop idcs-v2-estimator-sim-ui@ideal.service
systemctl --user stop idcs-v2-estimator-sim-controller@ideal.service
systemctl --user stop idcs-v2-estimator-sim-streamer@ideal.service
```

The service preflight guard refuses a second streamer, controller, or UI. Study
services do not automatically restart after a failure, preventing a rejected
second profile from retrying indefinitely.

## Evidence

Each controller instance writes:

- `logs/estimator-sim-<profile>-trace.jsonl`: observation, intent, command,
  CamState, and full `ControlDiagnostics` for every controller tick.
- `logs/estimator-sim-<profile>-report.json`: tracking/limiting metrics,
  estimator update counters, rejection/reinitialization maxima, and
  feedforward RMS/maximum contribution.
- `logs/estimator-sim-<profile>-ui-report.json`: HUD/video status on UI stop.

The trace can be replayed with `tools/analyze_estimator_feedforward_trace.py`
to compare raw PD, estimated error without feedforward, and the qualified
estimator/feedforward policy on identical observations. That replay is
non-causal; closed-loop comparisons require separate matched captures.

The HUD receives the simulation diagnostics directly and visualizes signed
yaw/pitch feedforward contribution without enabling the retired MPC overlay.

## 2026-09-24 validation

The first detector-coupled captures are retained as failed evidence. A passive
stationary-camera capture proved that those failures were not caused by the
controller: the original person track was ineligible under production policy,
and the replacement drone produced only 42% track coverage. Matching the
simulator to the real-camera 135 x 73 degree axis FOV corrected known-size
range, but model/tracker coverage remained scale-dependent (15% in the closer
fixture). Detector effectiveness therefore remains a separate open result.

With the guaranteed ground-truth observation channel, both controller studies
pass the unchanged integration gates:

- Ideal plant: 77.81 s, 3,889 commands, 99.974% tracking, 0.72 s acquisition,
  0.795 px steady RMS, 1.470 px steady p95, 0.849% rate-limited, zero command
  or diagnostics drops, and zero estimator rejection/reinitialization.
- Qualified gray-box plant: 101.19 s, 5,058 commands, 99.980% tracking,
  0.78 s acquisition, 0.693 px steady RMS, 1.406 px steady p95, 0.712%
  rate-limited, zero command or diagnostics drops, and zero estimator
  rejection/reinitialization.

These results qualify simulator integration only. They do not qualify the
detector, alter physical-mount tuning, or replace hardware verification.
