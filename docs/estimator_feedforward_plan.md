# Estimator-Driven Feedforward Plan

> Superseded (2026-09-27): the V2 LOS-Kalman controller this plan targeted was
> removed. Feedforward now lives in `jetson/control` (see
> `docs/controller_architecture.md`).

## Objective

Improve moving-target tracking with an estimator-derived target line-of-sight
(LOS) rate term while preserving the proven feedback controller, safety holds,
and rate/acceleration/position bounds. Estimator parameters may be selected
offline from the hardware-derived plant and controlled target trajectories, but
must not be tuned from simulated-camera motion. Simulation is an operations and
video-pipeline validation environment; bounded encoder-coupled HIL is a holdout
validation environment, not a controller-tuning source.

## Current baseline

The V2 controller already loads an estimator/feedforward profile from
`artifacts/controller_sim/los_kalman_feedforward_wire_20260914/`:

- independent constant-velocity absolute-LOS Kalman filters for yaw and pitch;
- state `[absolute_target_angle, absolute_target_rate]`;
- yaw process spectral density `0.001`, pitch `0.005`, measurement variance
  `9e-6 rad^2`, innovation gate `NIS <= 16`, and a `0.25 s` maximum gap;
- feedback gains yaw `(Kp=8, Kd=0.1)` and pitch `(Kp=6, Kd=0)`;
- target-rate feedforward gain `0.5` on each axis;
- 50 Hz controller and a qualification scenario that assumes 30 Hz vision,
  50 ms fixed latency, 3 mrad noise, and 10 percent dropout.

The active video path now supplies approximately 60 inference updates per
second while the deployable controller remains 50 Hz because the present
38,400-baud, three-axis acknowledged serial link cannot sustain 120 Hz. A
separate 120/60 offline profile exists, but is not deployable on this transport.

The 2026-09-24 encoder-coupled HIL run used the current estimator/feedforward
controller. It acquired the moving target and ended at 12.83 px error, but
post-warm-up error was 24.80 px RMS and 42.01 px p95. Acceleration limiting was
active on 70.9 percent of ticks and yaw rate limiting on 51.6 percent. Because
the controller requested up to 0.5 rad/s while the bounded HIL bridge clipped
motion to 0.2 rad/s, this run does not isolate estimator or feedforward value.

## Implementation status - 2026-09-24

The first non-authoritative slice and Gate-A tooling are implemented:

- `ControlObservation` now preserves optional Jetson-local frame receive and
  inference-observation timestamps with their clock domains. Existing payloads
  remain valid and the policy does not consume the new fields for commands.
- `LOSEstimate` now exposes predicted covariance, last innovation/variance/NIS,
  acceptance, rejection streak, and reinitialization state.
- `ControlDiagnostics v1` records local timing evidence, estimator state and
  uncertainty, update disposition, feedback/damping/feedforward contributions,
  pre/post-limit rates, and final commands. `jetson.control_runtime` writes it
  only when `--diagnostics-trace` is explicitly supplied.
- `tools/analyze_estimator_feedforward_trace.py` replays raw PD, estimator with
  feedforward disabled, and the qualified estimator/feedforward policy from
  identical observations. Its report explicitly forbids causal performance
  claims. The raw-PD gimbal-damping option is offline-only and defaults off.
- On the retained 2,177-tick HIL trace, the instrumented qualified policy
  reproduced all 2,177 recorded intents exactly. Of 2,017 estimator measurement
  ticks, yaw rejected 416 and reinitialized 325; pitch rejected 216 and
  reinitialized 156. Qualified feedforward reached `0.692 rad/s` yaw and
  `0.569 rad/s` pitch before output limits. Removing feedforward in replay
  reduced acceleration-limited ticks from 1,543 to 1,307, but replay cannot say
  whether that would improve closed-loop error.
- The complete host suite passes 350 tests and 12 subtests; focused native
  Jetson coverage passes 31 tests. The active qualified artifact, controller
  configuration, bridge caps, and motor processes were not changed.

## Correctness gaps to close first

### 1. Measurement time is not the frame measurement time

`ControlObservation.target.source_age_ms` currently measures time since the
latest perception snapshot reached the controller. `ShadowRatePolicy` subtracts
that value from the controller tick time and treats the result as the target
sample time. This omits the preceding stream/decode/inference/tracker latency.

`PerceptionFrameV2` already carries:

- the remote source timestamp and its clock domain;
- Jetson-local frame receive time;
- Jetson-local inference observation time; and
- both associated clock-domain labels.

The assembler must preserve a Jetson-local measurement-time estimate and its
uncertainty. Remote source time must never be subtracted from Jetson monotonic
time without a qualified clock mapping. When no mapping exists, use the local
frame-receive timestamp plus a separately measured/configured source-to-receive
delay, and retain the uncertainty explicitly.

### 2. Brief detector loss destroys useful estimator state

Every target-invalid hold currently resets the estimator immediately. The
2026-09-24 run had nine short loss events, at most 0.2 seconds. Control must
still go immediately to zero on target loss, but estimator state may coast
internally for a short bounded interval. It can be reused only if the same
track identity returns, covariance remains acceptable, and the configured gap
has not expired. Target identity changes, timestamp regressions, prolonged
loss, emergency/manual authority, and non-finite input must reset it.

### 3. Estimator confidence is unavailable to the policy

`LOSEstimate` exposes counters but not covariance, innovation, NIS, prediction
horizon, or reinitialization state. Feedforward should be enabled only after a
minimum accepted-update count and while rate variance, NIS, sample age, and
prediction horizon remain within explicit bounds. It must ramp in and out to
avoid a command step at acquisition or reacquisition.

### 4. Qualification cadence and actuator bounds differ from the live trial

The selected profile assumes 30 Hz measurements and 0.5 rad/s authority. The
current stream is approximately 60 Hz, and bounded HIL deliberately uses
0.2 rad/s. Qualification must produce separate, named profiles for:

- stable software simulation/operations evaluation;
- deployable real-hardware control at the actual serial command cadence and
  rate cap; and
- any higher-rate future transport.

No silent bridge-side clipping is allowed in performance qualification. The
controller profile and bridge cap must agree, even when the agreed value is a
temporary conservative safety value.

### 5. Secondary pitch telemetry has the wrong camera sign

Pitch-A and pitch-B encoder deltas agree in magnitude after calibration, but
the secondary B encoder angle changes opposite to A in camera coordinates.
Correct that sign and establish a common reference before using pitch-B for
estimator validation, divergence gating, or coupled acceptance. Pitch-A may
remain authoritative during isolated planning and offline work.

## Proposed estimator/control contract

For each axis, retain an inertial/base-frame target LOS state initially:

```text
x_target = [theta_target, omega_target]
z_k      = theta_gimbal(t_measurement) + bearing_error_k
```

The first candidate remains the constant-velocity model. A constant-
acceleration model is an experimental variant only and is adopted only if it
passes untouched holdouts without increasing command variation or saturation.

At controller time, predict both target and gimbal to the same estimated
command-effect time:

```text
t_effect  = t_tick + command_transport_delay + actuator_effect_delay
e_predict = theta_target_hat(t_effect) - theta_gimbal_hat(t_effect)
u_raw     = Kp * e_predict - Kd * omega_gimbal_hat + Kff * omega_target_hat
```

The result then passes through the existing rate, position, acceleration,
authority, expiry, and watchdog gates. `Kff` remains separately qualified from
the estimator noise parameters. It is not embedded into the estimator model.

The initial delay model should use measured components and uncertainty rather
than one fitted scalar:

- frame receive to inference/tracker output;
- snapshot delivery to the controller;
- controller scheduling age;
- intent publication to serial acceptance/wire execution; and
- command-to-observable encoder response.

If a timing component is unavailable or stale, prediction horizon is capped
and feedforward is reduced or disabled; feedback remains bounded and active.

## Required telemetry

Record per axis on every controller tick, without changing command authority:

- raw bearing error and reconstructed absolute LOS measurement;
- measurement time, clock source, age, and time uncertainty;
- estimated angle/rate and covariance diagonal;
- innovation, innovation variance, NIS, update accepted/rejected/reinitialized;
- target and gimbal prediction horizons;
- feedback, damping, and feedforward command contributions before limits;
- final rate/acceleration/position limit decisions;
- transport-delay inputs and whether each is measured, configured, or absent;
- loss-coast age, track identity, estimator readiness, and feedforward inhibit
  reason.

This should be a versioned controller-diagnostics record associated with the
observation and intent sequence IDs. Do not overload UI-only MPC term fields.

## Validation sequence

### Gate A - Identical-input, non-causal ablation

Replay the retained V2 observations through four policies:

1. raw bearing PD, no estimator and no feedforward;
2. estimator-predicted error, feedforward disabled;
3. estimator-predicted error plus current feedforward; and
4. proposed delay-aware estimator/feedforward.

Report command contribution, sign changes, limiting, safety-decision parity,
and estimator resets. This replay cannot claim tracking improvement because
the observations were generated under one active policy; it only isolates
decision behavior and catches unsafe or discontinuous changes.

### Gate B - Truth-based estimator tests

Use exact synthetic LOS truth, independent of detector effectiveness, with:

- constant rate, sine sweeps, ramps, reversals, and bounded acceleration;
- timestamp jitter, measured latency distributions, 30/60 Hz cadence, repeated
  frames, out-of-order frames, and 0-250 ms dropouts;
- measurement noise, isolated outliers, persistent maneuvers, track switches,
  and target reacquisition; and
- stationary targets while the camera moves, moving targets while the camera
  is stationary, and simultaneous target/camera motion.

Primary estimator metrics are rate RMSE/bias, effect-time angle prediction
RMSE/p95, NIS consistency, rejection/reinitialization rate, reacquisition time,
and computation p95. Camera-motion cancellation must be evaluated explicitly.

### Gate C - Hardware-derived gray-box closed loop

Use the frozen hardware plant package and measured delay/cadence distributions.
Select estimator parameters and `Kff` only on training trajectories, then lock
them before holdout evaluation. Compare all four policies above using identical
limits. Provisional acceptance:

- at least 10 percent aggregate RMS improvement over raw PD;
- no holdout scenario more than 10 percent worse in RMS or p95 error;
- command total variation no more than 1.25 times raw PD;
- no increase over five percentage points in rate/acceleration saturation;
- exact safety/hold parity; and
- estimator update and prediction p95 below 0.5 ms on Jetson.

Run separate searches for the deployable hardware profile and stable software
simulator profile. Do not transfer simulator-selected controller parameters to
hardware.

### Gate D - Controlled video-pipeline validation

Render deterministic moving targets with exact ground-truth trajectory and
variations in range, direction, speed, scale, background, and lighting. Use a
guaranteed detectable synthetic target for estimator/system tests, while also
recording separate 3D-drone detector results. Compare detector-derived LOS
estimates against renderer truth before involving motion control.

### Gate E - Shadow and bounded encoder-coupled HIL

First run the candidate estimator in shadow beside the current active policy,
with one command authority. Then run matched bounded A/B and B/A trials in
which real encoder movement drives the simulated camera. Use the same
deterministic target trajectories, initial pose, controller/bridge caps, and
duration. Parameters remain frozen; HIL is acceptance evidence, not tuning.

Only after estimator accuracy, safety parity, and bounded HIL pass should a
real-camera moving-target trial be considered.

## Implementation slices

1. **Timing/diagnostics only:** extend the observation/diagnostics contracts,
   preserve Jetson-local frame timing, expose estimator internals, and keep
   emitted intents byte-for-byte unchanged.
2. **Ablation tooling:** add deterministic policy variants and reports over
   existing traces; preserve a golden trace for timing/reset behavior.
3. **50/60 qualification:** regenerate a 50 Hz controller / 60 Hz vision
   profile using the hardware plant, measured delay distribution, and matching
   actuator caps.
4. **Bounded loss coasting:** separate immediate zero-command holds from
   estimator-memory reset, with identity, covariance, and timeout gates.
5. **Delay-aware prediction:** predict target/gimbal to a common measured
   effect horizon and add confidence-gated/ramped feedforward.
6. **Controlled video and shadow gates:** validate against exact renderer truth
   and then the live V2 stream without physical command authority.
7. **Encoder-coupled HIL:** correct pitch-B sign/reference, run matched bounded
   trials, and accept or reject the frozen candidate.

Each slice requires unit tests, deterministic replay, a versioned report with
source/config hashes, and a journal entry. No slice may silently change the
active qualified report or bridge rate cap.

## Immediate next work

Run a bounded diagnostics-enabled capture, without changing the active policy,
to quantify actual frame-receive-to-controller, inference, gimbal-sample,
intent-to-wire, and encoder-effect timing distributions. Use those distributions
to build the 50/60 truth-based validation scenario. Do not adjust process noise,
innovation gating, loss behavior, or feedforward gain from the non-causal HIL
replay; select candidates only against controlled truth and untouched holdouts.
