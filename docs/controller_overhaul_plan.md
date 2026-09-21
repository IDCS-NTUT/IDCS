# Controller Overhaul Plan

## Outcome

Replace the current frame-coupled PID/MPC invocation with a fixed-rate,
timestamp-aware controller service that consumes a single, versioned
observation snapshot; produces a bounded rate-command intent; is replayable
offline; and can be introduced in shadow mode before it is allowed to command
the physical gimbal.

This plan deliberately does **not** treat the existing `jetson.controller`
MPC as the redesigned controller.  It is a legacy implementation and a useful
baseline/reference only.

## Historical-data policy

All latency, tracking, tuning, and system-identification results produced by
the legacy frame-coupled implementation are **historical context only**.  They
may guide experiment design, but are not acceptance baselines and must not be
quoted as performance of the redesigned controller.  A result becomes current
only when it identifies the deployed code/configuration/model versions, clock
boundaries, hardware state, and trace/artifact, and is reproduced through the
new observation → policy → intent path.

## Status legend

- **Complete** — implemented and has recorded test evidence.
- **Partial** — useful code/evidence exists, but does not meet the overhaul
  interface or acceptance condition.
- **Not started** — no implementation or qualifying evidence.

## Work breakdown

| # | Work item and acceptance condition | Status | Existing evidence / gap |
| --- | --- | --- | --- |
| 1 | Freeze safety invariants: physical E-stop remains independent; manual/emergency always overrides automatic intent; every automatic command has expiry, rate, acceleration, and position bounds. | **Partial** | Emergency serial arbitration was measured on Jetson; RPi manual state and gimbal bridge gating exist. Automatic command expiry and one authoritative safety-state contract do not. |
| 2 | Define `ControlObservation v1`: target bearing/error and source age; selected target identity/confidence; encoder yaw/pitch/rates and sample age; command/serial timing; manual/emergency/safety state; validity flags and monotonically increasing sequence IDs. Validate it at ingress. | **Partial** | Immutable strict schema/assembler exist; a read-only three-address MKS encoder CamState publisher is live on Jetson. Manual state now has a trace-only mirror, but no capture yet contains selected target, encoder, and real manual state together. |
| 3 | Define `ControlIntent v1`: desired yaw/pitch rates, validity-until monotonic time, source observation sequence, saturation/reason codes, and shadow/live authority. Gimbal bridge must reject expired or out-of-order intents. | **Partial** | Strict intent schema and explicit shadow/live policy modes exist. The source bridge now rejects wrong-authority, expired, future, malformed, and out-of-order intents and has a local watchdog; deployment and wire validation remain. |
| 4 | Build one fixed-rate controller service with latest-only ingress and deterministic scheduling. On overload it must coalesce rather than replay stale ticks and export scheduler health. | **Partial** | The qualified V2 runtime now implements latest-only 50 Hz scheduling, health/report files, timestamp-derived restart-safe sequence epochs, and no serial imports. It passes check mode but is not deployed or qualified as a long-running Jetson service. |
| 5 | Establish time semantics: local monotonic timestamps at every Jetson boundary; preserve remote source timestamps only for diagnostics; measure PC↔Jetson clock offset/error if cross-host age is needed. | **Partial** | Controller input receipt time is explicit and trace recorder has receive monotonic time. No clock-offset measurement, unified timing schema, or full sensor→wire budget. |
| 6 | Build a lossless-enough recorder for observations, intents, encoder feedback, serial acceptance timing, manual/safety transitions, and scheduler misses; define trace format/version and retention. | **Partial** | Versioned passive recording exists and the production V2 runtime can now record its exact observation/intent pairs. Serial acceptance, real manual transitions, one qualified selected-target/encoder/manual capture, and retention policy remain. |
| 7 | Build deterministic replay: feed a trace through the fixed-rate service with recorded monotonic times; compare command sequence, stale-target decisions, saturations, and metrics against golden output. | **Partial** | Qualified-report native-V2 golden replay and an independent trace validator now reproduce every intent exactly. A bounded live synthetic three-input capture matched 170/170 intents at fixed rate; a qualified real selected-target/encoder/manual trace is still missing. |
| 8 | Finish plant identification: excitation that spans commanded operating range (PRBS/chirp/steps), measured command-to-wire delay, encoder sampling delay/jitter, repeatability, and uncertainty bounds for yaw and pitch. | **Partial** | A wire-timestamped unloaded bidirectional fit now independently qualifies both axes through 0.524 rad/s. Encoder reply latency and cadence are recorded. Sub-sample delay, uncertainty bounds, loaded response, and non-step excitation remain. |
| 9 | Create a versioned plant model package and simulation harness. It must replay recorded input delays, encoder cadence, limits, and disturbances; model parameters come only from a saved fit report. | **Partial** | The reusable harness is bound to the independently qualified asymmetric fit and replays 50 Hz dynamics plus rate/acceleration limits. Saved deterministic noise/latency/dropout scenarios exist; measured delay remains sub-sample and no loaded uncertainty model exists. |
| 10 | Select and specify the new controller architecture: estimator state/vector, target line-of-sight filter, reference generator, delay compensation, anti-windup, rate/acceleration limiting, fault/stale policy, and reset rules. | **Partial** | Raw PID, conditional anti-windup, rate/slew limits, and a timestamp-aware absolute-LOS constant-velocity Kalman/feedforward path are specified and offline-qualified. Live stale/loss integration and measured transport-delay compensation remain. |
| 11 | Implement the new estimator/controller behind a stable interface, initially with no publisher/serial dependency. Unit-test normal tracking, loss/reacquisition, encoder staleness, target identity switch, saturation, reset, and non-finite input rejection. | **Partial** | The rebuilt LOS estimator is transport-free and unit-tested. The V2 runtime now loads only the frozen qualified report and emits short-lived live intents; deployment, live-input qualification, and removal of the legacy policy implementation remain. |
| 12 | Run offline gain/model search against held-out traces and the simulator. Record objective metrics (RMS/p95 pointing error, overshoot, settle time, control effort, saturation, loss recovery) and preserve the selected parameters. | **Complete** | Versioned raw-PID and PID+LOS-Kalman searches preserve candidate tables, scenario traces, plots, holdout metrics, source hashes, and explicit pass/fail reports. Both axes pass; results remain unloaded/offline only. |
| 13 | Add shadow parity mode: run legacy and redesign from the same observation snapshots; publish neither to hardware, record command deltas and safety-decision mismatches, and define pass/fail thresholds. | **Partial** | The redesigned qualified policy now runs from atomic V2 observations in the passive fixed-rate recorder, and exact self-replay plus scheduler/source/rate gates exist. Same-snapshot legacy comparison, command-delta thresholds, and safety-decision mismatch thresholds remain. |
| 14 | Validate the transport/actuator chain: intent freshness rejection, serial command acceptance timing under scheduled encoder load, baud-rate soak decision, encoder health faults, and gimbal-limit behavior. | **Partial** | Source tests now cover intent freshness/order/watchdog rejection and manual-backed firmware-timed F6 payloads. Regular-command timing under encoder load, selected-baud soak, live watchdog stop timing, and limit-fault hardware tests remain. |
| 15 | Hardware acceptance, unloaded first then loaded: bounded trajectories, target-loss scenarios, emergency/manual takeover, and vision-in-the-loop tracking. Define new acceptance thresholds from redesign evidence before any cutover. | **Partial** | One unloaded bounded legacy-path 3D trajectory and connectivity test completed; it is historical context, not a redesign baseline. No redesigned-controller hardware run, loaded run, or vision-in-loop acceptance. |
| 16 | Controlled cutover/rollback: explicit feature flag, one command authority, startup state that never auto-zeros encoders, persisted configuration/version record, dashboard metrics, and tested rollback to a zero-command safe state. | **Partial** | Live publication and actuation require separate explicit flags. Serial startup is stop/query only; calibration and encoder zeroing require both tracked config and separate acknowledgements. Deployment, rollback drill, and dashboard qualification remain. |

## Completed foundations

1. All three MKS motors respond over Jetson RS485 at 38,400 baud.
2. The emergency request-to-wire path was measured under normal and adversarial
   queueing; it met the 25-ms same-host software acceptance target in that
   specific unloaded setup.
3. A low-speed, unloaded yaw/pitch step-data fit and a bounded 3D trajectory
   run have been recorded.
4. Controller inputs can now carry an explicit Jetson-local monotonic receipt
   time, and a shadow-only fixed-rate scheduler avoids catch-up bursts.
5. Existing metadata and manual authority transports are known and have shared
   schemas, but are not yet unified as controller input.

## Required implementation order

1. Items 1–3: safety and message contracts. No redesigned controller should
   command hardware before these are enforced at the bridge.
2. Items 4–7: service, timing, recording, replay. This creates the safe,
   repeatable environment for design work.
3. Items 8–9: excitation/fit/model and delay characterization.
4. Items 10–12: controller specification, implementation, and offline search.
5. Items 13–16: shadow parity, transport/hardware validation, and cutover.

## Immediate next milestone

Deploy the fixed-rate V2 runtime without the actuator bridge, using selected
detections, read-only encoder CamState, and real manual state. Record and
qualify one observation/intent/scheduler trace, then perform same-snapshot
legacy/redesign safety-decision parity. Only after that evidence may the new
bridge be deployed for a timed-command, zero-rate hardware canary; its live
actuation, calibration, and encoder-zero acknowledgements remain off by
default.
