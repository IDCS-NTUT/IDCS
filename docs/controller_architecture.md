# Video controller architecture

Code/document cross-reference: 2026-10-17. This is the current implementation
description; the [previous document](archive/controller-architecture-before-20261011-review.md)
is retained as historical evidence.

## Ownership and signal path

```text
V2 perception + measured CamState + panel state + source-clock mapping
        -> 50 Hz video runtime / atomic observation
        -> basic PID feedback + separate Kalman target-rate feedforward
        -> authority, freshness, rate/slew and travel gates
        -> short-lease ControlIntent
             -> software SIM mount
             OR physical bridge -> serial service -> timed F6
```

The active entry point is `jetson.control.video_runtime`. DeepStream is passive,
the controller cannot open serial, and the physical bridge alone translates
intents into managed motor commands. `ControlCmd` remains a compatibility
payload; its unused MPC diagnostics and controller-mode fields were removed.
There is no MPC controller or MPC configuration path.

`ControlObservation` carries target geometry from `common.aiming`: selected
pixel centre, known-size range/provenance, projected parallax aim point,
pixel error and signed bearing error. Parallax adjusts the aim reference; it
does not imply a real laser/effector. No simulated truth is substituted into
the detection-fed Swarm controller.

## Feedback, estimator and loss behavior

The reusable `BasicPID` supports PID terms. The current runtime constructs it
with configured per-axis **Kp and zero Ki/Kd**. Thus current “PID” trials use
proportional feedback with rate/slew limits, not independently tuned I/D terms.

`VideoTargetRateEstimator` uses capture-time measurements and
`CameraPoseHistory` to separate target motion from camera motion. Pose history
retains per-axis sample times; repeating a CamState publication does not create
a new physical measurement. Kalman target angular rate supplies a separate
scaled feedforward contribution. Optional `predict` also advances the error
used by feedback; feedforward-only and prediction-only effects must therefore
be measured separately.

The checked-in bench tuning uses yaw Kp about 6.796, pitch Kp about 5.966,
feedforward scale 0.5, prediction 0.5 and acceleration-noise sigma 2.
These are bounded unloaded-trial results, not loaded/real-camera qualification.

Loss behavior is explicit:

- Configured coasting can continue the same target prediction through a short
  gap; base `coast_s` is 0.5.
- After the configured no-target delay, idle return seeks the initial accepted
  pose/envelope centre. This is **not hardware homing or encoder zeroing**.
- All such motion remains subject to panel, E-stop, feedback, timing and
  envelope gates. GPIO12 authority loss cannot be bypassed by lamp hysteresis.
- GPIO5 manual slew is independent of the auto-enable switch but uses the same
  safety/travel constraints.
- `configs/controller/pid_only.yaml` disables feedforward/prediction; it does
  not by itself disable inherited coasting or idle return.

## Timing and transport

The runtime samples latest inputs every 20 ms. Current policy uses a 50 ms
intent lease; the bridge also checks intent freshness and sends firmware-timed
F6 updates. Independent watchdogs address input age, panel/pose authority,
clock-map uncertainty and missing refresh. They are not one end-to-end
latency guarantee.

PC RTP source timestamps are joined to decoded frames using verified source
identity and mapped into the Jetson clock domain through timestamp exchanges.
Offset intervals, exchange freshness and allowed drift determine capture-age
bounds; arrival time is not guessed to be capture time.

The current base clock policy is `slew_limited_ntp`, 1500 ppm. The journal
records its basis as the configured PC chrony slew/frequency limits plus the
Jetson clock discipline limit. Runtime validation checks the declared policy
and numeric range, **not the hosts' current time-daemon configuration**.
Revalidate those assumptions after host/package/config changes. A small drift
observed in a survey is not a guarantee about future drift. The earlier
user-approved 1000 ppm test assumption is historical, not the current default.

Local Argus sensor frame number/time integration is implemented:
`jetson.deepstream.argus_sensor_meta` captures buffer metadata and joins it by
PTS, publishing verified `jetson_monotonic` V2 frames.
`configs/controller/local_camera.yaml` selects same-host timing, so no PC
clock conversion is required. The normal camera service remains shadow-only.
The sensor timestamp reference is not an independently established optical
exposure midpoint; do not relabel stage timings as photon-to-display latency.

PC/Pi HUDs use latest telemetry and lack an exact return-frame join.
Full camera optical-event-to-PC/Pi-display delay remains unmeasured here.

## Physical pose and command boundary

Default control pose comes from motor raw microstep count (0x33), with slower
encoder (0x31) cross-checks. The bridge anchors the step frame against encoder
pose at startup. Count agreement is not proof of mechanical coupling or loaded
tracking accuracy.

HIL uses the streamer's `MeasuredPoseTimeline`: render at a delayed time
supported by measured pose, never silently replace missing physical feedback
with synthetic motion. The old speculative render-pose/wire-prediction path
was removed. Timed F6 resolution and its empirically observed refresh behavior
remain actuator constraints; old pure one-RPM or F5/dither descriptions are not
the current implementation.

The bridge owns stop/release while serial IPC is alive. Serial-service shutdown
alone does not issue an automatic motor disable:
[serial lifecycle](serial_io_service_definition.md).

## Simulation and plant qualification

Software SIM is for system/video/perception/UI evaluation; its motion should
be a stable substitute independent of hardware tuning. **Implementation gap:**
both PC simulator controller units still load `configs/bench/tuned.yaml`,
and `pc.streamer` uses the F6 speed quantizer. Hardware gain changes therefore
change software-SIM behavior. This is R6 in the [review](codebase_review_20261011.md),
not a completed separation.

Offline plant fitting/gain-search tools and recorded unloaded HIL agreement
exist. Plant validity is specific to the identified command, delay, axis and
load envelope. It does not justify tuning a coupled production mount solely
from a visual simulator. The old low-range fit artifact is not present in this
checkout, so its regression test is not currently reproducible locally.

## Evidence: later results supersede the early handoff

These are **journal-recorded results**, not hardware tests rerun on 2026-10-11.

| Date / evidence | Result and scope |
|---|---|
| 2026-09-27 `v3_clean_*` ABBA | Capture-error RMS yaw/pitch PID 25.2/16.7 and 21.7/14.7 mrad; FF+prediction 11.3/9.6 and 11.5/11.2. About 51% yaw / 34% pitch improvement in bounded unloaded HIL |
| 2026-09-28 `tune-20260928-final` | Nine recorded tuning gates passed; live HIL ABBA PID 13.5/9.9 and 13.5/11.6 mrad, tuned 8.2/9.7 and 9.0/8.9 |
| 2026-09-28 clock survey | 600 s / 3000 exchanges; median interval width 3.9 ms; feasible constant drift -6.9 to +4.3 ppm over that window |
| 2026-09-28 local camera | 60 s / 3746 frames: sensor-reference to receipt p50/p95 6.8/8.4 ms; inference+tracker complete 21.4/22.3 ms; subscriber snapshot 22.9/23.5 ms. Not display latency or closed-loop physical tracking |
| 2026-09-29 onward | Wiring/RS485 pitch failure blocks further motor/HIL activation until repaired and confirmed |

The old handoff's “no consistent FF benefit” and “sensor timestamp integration
missing” conclusions were superseded by these later entries. Preserve both
chronologies without promoting unloaded evidence to production acceptance.

## Remaining gates

1. Repair/confirm wiring and responses from all required axes before any motor
   owner starts; no hardware was exercised in this review.
2. R1, R3, R4 and R5 in the [review](codebase_review_20261011.md) have local
   fixes and deterministic tests, but still need release validation; R2's
   operator-lock/score-job interleaving remains open.
3. Separate the stable software-SIM profile from bench gains/F6 dynamics.
4. Qualify loaded dynamics, coupled pitch behavior, gearing and actual camera
   tracking; validate clock assumptions and source/exposure semantics.
5. Measure real optical-to-return-display delay and representative detector
   accuracy separately.

See [tuning procedure](tuning_procedure.md),
[verification strategy](verification_strategy.md) and the
[journal](deepstream_migration_journal.md) for methods and retained artifacts.
