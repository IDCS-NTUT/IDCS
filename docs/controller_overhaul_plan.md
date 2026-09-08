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
| 3 | Define `ControlIntent v1`: desired yaw/pitch rates, validity-until monotonic time, source observation sequence, saturation/reason codes, and shadow/live authority. Gimbal bridge must reject expired or out-of-order intents. | **Partial** | Immutable strict intent schema now exists and the shadow policy emits it. Gimbal bridge enforcement is intentionally not connected yet. |
| 4 | Build one fixed-rate controller service with latest-only ingress and deterministic scheduling. On overload it must coalesce rather than replay stale ticks and export scheduler health. | **Partial** | `FixedRateController` provides fixed monotonic cadence/coalescing. A passive redesigned observation → policy trace now exports scheduler health and recorded 963 ticks with zero missed periods; this is not yet the long-running controller service. |
| 5 | Establish time semantics: local monotonic timestamps at every Jetson boundary; preserve remote source timestamps only for diagnostics; measure PC↔Jetson clock offset/error if cross-host age is needed. | **Partial** | Controller input receipt time is explicit and trace recorder has receive monotonic time. No clock-offset measurement, unified timing schema, or full sensor→wire budget. |
| 6 | Build a lossless-enough recorder for observations, intents, encoder feedback, serial acceptance timing, manual/safety transitions, and scheduler misses; define trace format/version and retention. | **Partial** | Versioned recorder writes paired shadow intents and a final scheduler-health record. A live three-input capture reached 962 fresh encoder/safety samples with zero missed periods. Serial acceptance, recorded manual transitions, selected-target qualification, and retention policy remain. |
| 7 | Build deterministic replay: feed a trace through the fixed-rate service with recorded monotonic times; compare command sequence, stale-target decisions, saturations, and metrics against golden output. | **Partial** | Golden replay and a hold-vs-shadow parity comparator exist. A synthetic Jetson trace produced 239 tracking intents; no qualified real three-input trace or fixed-rate parity metric exists yet. |
| 8 | Finish plant identification: excitation that spans commanded operating range (PRBS/chirp/steps), measured command-to-wire delay, encoder sampling delay/jitter, repeatability, and uncertainty bounds for yaw and pitch. | **Partial** | New low-amplitude unloaded chirp fit exists for both axes with held-out metrics. Fit is still low-range/unloaded; 0-ms grid result does not identify true delay, and high-range, bidirectional, uncertainty, and loaded data remain. |
| 9 | Create a versioned plant model package and simulation harness. It must replay recorded input delays, encoder cadence, limits, and disturbances; model parameters come only from a saved fit report. | **Partial** | Offline 3D PID benchmark, new fit report, and frozen-fit validator exist. Yaw has one independent low-amplitude validation; pitch does not qualify due to limit-blocked validation. No reusable runtime plant-model API nor trace-driven closed-loop simulator. |
| 10 | Select and specify the new controller architecture: estimator state/vector, target line-of-sight filter, reference generator, delay compensation, anti-windup, rate/acceleration limiting, fault/stale policy, and reset rules. | **Partial** | `ShadowRatePolicy` specifies a bounded PD plus camera-relative bearing-rate feedforward law, latest-input fault holds, identity-switch reset, and hard travel-bound hold. Estimator, delay compensation, anti-windup, and live specification remain. |
| 11 | Implement the new estimator/controller behind a stable interface, initially with no publisher/serial dependency. Unit-test normal tracking, loss/reacquisition, encoder staleness, target identity switch, saturation, reset, and non-finite input rejection. | **Partial** | `ShadowRatePolicy` emits only shadow intents and has unit coverage for normal tracking, authority/loss hold, identity changes, rate/acceleration saturation, position-bound hold, and out-of-order snapshots. No publisher, serial dependency, estimator, or live authority exists. |
| 12 | Run offline gain/model search against held-out traces and the simulator. Record objective metrics (RMS/p95 pointing error, overshoot, settle time, control effort, saturation, loss recovery) and preserve the selected parameters. | **Not started** | Prior gain-search work is not a reproducible, accepted selection pipeline. |
| 13 | Add shadow parity mode: run legacy and redesign from the same observation snapshots; publish neither to hardware, record command deltas and safety-decision mismatches, and define pass/fail thresholds. | **Not started** | Current fixed-rate sidecar emits its own shadow commands but does not ingest atomic snapshots or compare controllers. |
| 14 | Validate the transport/actuator chain: intent freshness rejection, serial command acceptance timing under scheduled encoder load, 57,600-baud soak decision, encoder health faults, and gimbal-limit behavior. | **Partial** | 38,400 emergency p99/max evidence and a baud sweep exist. 57,600 soak, regular-command timing, stale-intent rejection, and limit-fault tests remain. |
| 15 | Hardware acceptance, unloaded first then loaded: bounded trajectories, target-loss scenarios, emergency/manual takeover, and vision-in-the-loop tracking. Define new acceptance thresholds from redesign evidence before any cutover. | **Partial** | One unloaded bounded legacy-path 3D trajectory and connectivity test completed; it is historical context, not a redesign baseline. No redesigned-controller hardware run, loaded run, or vision-in-loop acceptance. |
| 16 | Controlled cutover/rollback: explicit feature flag, one command authority, startup state that never auto-zeros encoders, persisted configuration/version record, dashboard metrics, and tested rollback to a zero-command safe state. | **Partial** | Shadow separation and serial emergency priority exist. Production integration, safe startup audit across launchers, feature flag, rollback drill, and dashboard are missing. |

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

Run one redesigned fixed-rate shadow service using selected detections,
read-only encoder CamState, and the mirrored real manual state; record
observation, intent, and scheduler health in one trace. Qualify it before
parity comparison. No change to `gimbal_bridge.py` command authority until
stale-intent rejection is enforced there and qualifying replay evidence exists.
