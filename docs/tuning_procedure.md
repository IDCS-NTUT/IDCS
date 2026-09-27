# Tuning procedure

The standard way to derive the controller gains, feedforward settings and
actuator limits, and to show they work. Run it after any mechanical change
(gear ratio, load, motor, mounting) and after the assembled system is built.

Everything lives in one **run directory**. `configs/tuning_plan.yaml` holds
every setting and gate threshold and is copied into the run, so a run is
reproducible from its directory alone. Each stage writes `<stage>/report.json`
with a pass/fail gate; a stage runs only after the stages it depends on
passed, and re-running a stage clears everything downstream of it.

```text
latency ─────────────────────────┐
sysid ── fit ────────────────────┼── sim ── hardware ── agreement ── emit ── live_ab
limits ──────────────────────────┘
```

```bash
python -m tools.tuning init runs/tune-YYYYMMDD [--plan configs/tuning_plan.yaml]
python -m tools.tuning status runs/tune-YYYYMMDD
```

Where stages run:
- **Jetson with the gimbal stack stopped** (they own the RS485 bus):
  `sysid`, `limits`, `hardware`. Stop with `sudo systemctl stop idcs-hil.target`.
- **Jetson with the stack running**: `live_ab` (it pauses only
  `idcs-controller` and restarts it afterwards).
- **Any host**: `latency`, `fit`, `sim`, `agreement`, `emit`.

## 1. latency — how old the frames are when the controller acts

Input: controller traces of the live hardware loop (the controller run with
`--trace`, HIL overlay, at least ~30 s of tracking). The stage takes the
capture age at each tracking decision (upper bound of the clock interval),
and reports p50/p95.

Gate: ≥ `latency.min_tracking_ticks` tracking ticks; clock-verified fraction
≥ `latency.min_clock_verified_fraction`.

```bash
python -m tools.tuning latency RUN --trace path/to/trace.jsonl
```

## 2. sysid — open-loop response data

Two `jetson.tools.gimbal_response_sweep` runs (training and an independent
validation set with different rates and step lengths), using the serial
service it starts itself.

Gate: both sweeps finish (`status: complete`).

## 3. fit — plant model, frozen and independently validated

`tools/fit_gimbal_response.py` fits the training data;
`tools/validate_gimbal_fit.py` scores the frozen fit on the validation data.

Gate: the validation report is `qualified` (omega/theta RMSE, direction bias
and sample-count limits; override thresholds under `fit:` in the plan).

The model's role is to screen gains in simulation and to check the simulator
against hardware (stage 7). The controller itself contains no plant model.

## 4. limits — fastest motion without losing steps

`jetson/tools/limit_probe.py` per axis: a grid of F6 speed levels x
acceleration bytes, forward and reverse. Each move compares the motor's step
count (0x33, the controller's position feedback) with the magnetic encoder
(0x31). A disagreement above 26 counts (~5 microsteps) is a lost step.

The rate limit is `limits.margin_levels` level(s) below the fastest level that
passed at `limits.accel_byte`, capped at `limits.max_rate_rad_s`, converted to
axis rate with the gear ratio. The controller uses the smaller of the axes.

Gate: a level passed at the chosen accel byte, with margin, and the resulting
rate ≥ `limits.min_rate_rad_s`.

**Must be re-run with the load applied** (camera, laser and mount on the
geared gimbal). The unloaded bench cannot lose steps, so its limits are only
the motor's. Under load, find the maximum acceleration *and deceleration* the
axes follow without step loss: the probe currently starts and stops each move
with the same accel byte, so a loaded run must also check stops from speed
(inertia over-running the motor is a deceleration failure), and the chosen
`accel_byte` and `loop.accel_limit_rad_s2` must stay below the measured limit
with margin.

## 5. sim — screen gains and feedforward

The real `BasicPID` and `TargetRateKalman` on the qualified plant, with the
measured F6 speed table, the measured latency (base p50, jitter up to p95),
the rate limit from stage 4, the gear ratio, and step-count angle feedback.
For every configuration in `feedforward_configs` (the first is PID only) a
P-gain sweep over `sim.kp_*` on the plan's `scenarios`. The configuration
with the lowest cost relative to PID only (summed over axes) is chosen.

Gate: neither the PID-only nor the chosen optimum sits at the edge of the gain
grid (widen the grid if it does).

## 6. hardware — the same sweep on the motors

`jetson/tools/step_count_gain_sweep.py` per axis, for PID only and the chosen
feedforward configuration: gains at `hardware_sweep.kp_factors` x the sim
optimum, the p50 latency injected, the same scenarios, rate limit, accel byte
and gear ratio.

Gate: every planned run completed without an abort (guard, bus error).

## 7. agreement — does the simulator predict the hardware?

Every hardware run is re-simulated with identical settings.

Gate, per axis and configuration:
- median |sim/hardware − 1| cost error ≤ `agreement.max_median_cost_error`;
- the gain the simulator would pick costs at most
  `agreement.max_sim_choice_penalty` more on hardware than the hardware best.

Selection (hardware decides): each configuration's gain is its hardware best;
feedforward is kept only if it beats PID only on hardware on every axis.

## 8. emit — the tuned configuration

Writes `emit/tuned_config.yaml`: `controller` gains, feedforward, rate and
acceleration limits; `gimbal` rate limits, accel bytes and gear ratios. Its
header records the plan and every stage report hash. Also writes
`emit/pid_only_overlay.yaml`, the live baseline.

Gate: the controller's config validation accepts the result.

Deploy: add `tuned_config.yaml` (copied into `configs/`) to the
`--config-extra` of both `idcs-bridge` and `idcs-controller`, then restart the
stack.

## 9. live_ab — does it work on the live loop?

With the stack running the tuned config and the PC on `idcs-hil-streamer`:

```bash
# on the PC: the running streamer's arguments with --check
python -m pc.streamer <idcs-hil-streamer arguments> --check > streamer-check.json
# copy it to the Jetson, then
python -m tools.tuning live-ab RUN --streamer-check streamer-check.json
```

The stage pauses `idcs-controller`, runs 30 s controller trials in ABBA order
(PID only, tuned, tuned, PID only) and scores each with
`tools/analyze_video_hil.py` against the simulator's exact target truth.

Gate: at least `live_ab.min_pairs` pairs; every tuned trial within
`live_ab.max_capture_rms_mrad`; with feedforward, the tuned trials beat PID
only on both axes. On success the run gets `qualified_config.yaml`.
