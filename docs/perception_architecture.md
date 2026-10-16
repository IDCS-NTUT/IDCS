# Perception and control architecture

Cross-checked against the repository on 2026-10-16.

## Contracts and ownership

`common.perception.PerceptionSnapshotV2` is the active detector/tracker/selector
wire contract. It separates raw detections, tracks, assessments and selection.
Frames carry source identity, clock domain and tracker session; selection retains
both its decision source frame and the frame where it was applied.

1. `jetson.deepstream.runtime` owns ingest, nvinfer YOLO, **NvDCF**, metadata,
   GPU OSD and return encoding. The configured default detector is YOLO26s
   with the grey-letterbox 736 engine (see [build notes](tensorrt_engine_build.md));
   `deepstream.tracker` defaults to NvDCF, with NvSORT still selectable.
2. `DeepStreamTargetSelector` computes known-size range and planner assessments.
   Its asynchronous wrapper uses a latest-only worker so policy work does not
   block the video probe. Planner selection is restricted to the operator's
   target classes. Operator selection through `jetson.operator_agent` is
   source/session scoped: a visible lock overrides selection (`operator_lock`).
   With Planner selection off, Operator mode picks the nearest visible eligible
   ranged target, holds its manual or automatic lock until lost, then reacquires
   after `operator.lock_lost_s`; it never falls back to Planner.
3. `SnapshotTransport` publishes V2 snapshots. Verified RTP joins SSRC/RTP
   timestamp to source headers and decoded PTS. For local Argus, sensor metadata
   is captured and joined by buffer PTS in the Jetson monotonic clock domain.
4. `jetson.control.video_runtime` samples perception, measured camera pose,
   panel authority and clock mappings at 50 Hz. `common.aiming` computes target
   centre, range provenance, offset aim point, pixel and signed bearing error.
5. The separate basic PID feedback and Kalman feedforward path produces bounded
   `ControlIntent`. The active runtime sets PID integral/derivative gains to
   zero; it does not run the old MPC controller.
6. Physical intents go to `gimbal_bridge` and the serial service. Software SIM
   intents go to the streamer's simulated mount. Passive perception has neither
   physical command authority nor a serial owner.

`ControlCmd` and some legacy/controller helpers remain for compatibility and
experimentation. Their presence does not make them the production wire path,
and the repository is not completely free of legacy code.

Missing or stale panel authority and E-stop produce zero-rate disarmed intents.
GPIO12 enables automatic tracking; GPIO5 independently selects slew-limited
manual rates with fresh panel and valid gimbal state. GPIO16 (`fire_arm`)
gates only the fire-button indication, not motion.

## Freshness and presentation boundaries

Asynchronous reuse of a recent planner result for matching current tracks is
intentional. The wrapper now caps both source-frame age and local reuse by
`swarm_eval.learned_model.max_result_age_ms` (default 150 ms), requires the same
tracker session and clock domains, and clears its cache on worker error. An
expired result no longer supplies selection or range assessment. See R3 in the
[review](codebase_review_20261011.md); this is a local test result, not a live
DeepStream qualification.

Jetson draws detector/tracker/target OSD. PC and Pi add their operational HUD
from available telemetry; Pi receives Jetson RTP directly. Simulator truth is
allowed in Single's synthetic-control path and score/HUD telemetry, not as
visual tracker evidence in Swarm. Internal target IDs are not operator labels;
multi-target priority rank is rendered in a solid bounding-box corner badge.

Latest HUD telemetry is not exactly joined to the returned video frame.
Sensor timestamp integration and frame identity likewise do not by themselves
measure optical exposure-to-PC/Pi-display delay.

## Verification

Use `common.synthetic_perception` for deterministic presence/loss, identity,
delay, duplication, reorder and policy tests. Use labeled representative real
imagery for detector-quality conclusions; rendered 3D scenes are supplementary.
Keep selection, tracker identity, transport, controller and physical acceptance
separate. See [verification strategy](verification_strategy.md) and
[controller architecture](controller_architecture.md).
