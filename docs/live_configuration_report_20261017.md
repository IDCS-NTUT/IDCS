# Live configuration implementation — 2026-10-17

## Follow-up 2026-10-18

Branch `codex/live-config-followup`, base `79c6b6c` (`lab/live-config-reviewed`),
including reviewer fixes `2a6b7e7` and `79c6b6c`. Tier/schema tables, deployment
gate, bridge/serial exclusion and wire schemas are unchanged. No lab host,
hardware interface, deployment, unit activation, push, tag or PR was used.

Reviewer Jetson evidence (2026-10-17): 125/126 target tests passed; the history
comparison failed only because the trial copy lacked `245d103`. File replay
applied a live edit within one 0.5 s health sample, journal correct, exit 0.
However, rebuild plus validation cost **64 ms median / 110 ms maximum** on
Orin's owning thread: **4–7 frames at 60 fps**, or **3–5 ticks at 50 Hz** per
change. The probe's 200 ms overlay poll cost approximately **1 ms**. Jetson has
`NOPASSWD: ALL`; restart safety must come from code, not sudoers.

### Changes

- `common/live_config.py`: one worker per service owns the update socket,
  200 ms file poll, stack reconstruction, validation, tier split, digesting,
  pure typed preparation, receipt heartbeat and application journal. The owner
  only checks a prepared slot without waiting, swaps objects and queues an ACK.
  No file I/O, hashing, YAML/schema parsing or policy construction on `tick()`.
  Worker publication is atomic; engagement/force epochs and base identity reject
  stale work, and the overlay is checked again after a slow build. File-only
  changes become known on the worker poll; there is no filesystem transaction
  spanning an independent writer and the owner tick. ACKs retain retired objects
  for worker reclamation. Only successful swaps are journaled, with their actual
  swap timestamp and digest. Transient journal/receipt failures retry; shutdown
  drains ACKs and joins the socket owner. Windows atomic writes retry reader
  sharing violations for at most 250 ms, entirely outside the hot path.
- `jetson/control/{live_settings,video_controller,video_runtime}.py`: worker
  prepares runtime/aim/laser settings, policy and PID coefficients; owner checks
  current-origin travel before assignment and swaps without resetting history.
- `jetson/deepstream/{live_settings,async_target_selection,pipeline}.py`: worker
  prepares shadow/stitch/selector policies; metadata probe only swaps references
  and adjusts the existing stitcher's gap, preserving track history. Selector
  process acknowledgement/revision semantics remain unchanged.
- `jetson/operator_agent.py`, `pc/{live_settings,streamer}.py`: agent and
  simulator pure parsing also uses preparation callbacks.
- Restart authorization is now the literal, host-scoped
  `RESTART_ALLOWED_UNITS` allow-list. Jetson permits only
  `idcs-deepstream-camera.service`, `idcs-deepstream-video.service`,
  `idcs-recorder.service`. PC permits `idcs-camera-ui.service`, `idcs-ui.service` (display only),
  `idcs-hud-relay.service`, `idcs-sim-streamer.service`,
  `idcs-sim-swarm-streamer.service`, `idcs-sim-ui.service`,
  `idcs-sim-controller.service`, `idcs-sim-swarm-controller.service`,
  `idcs-sim-panel.service`, `idcs-sim-recorder.service`. Every other/future unit
  is denied. SIM dependency grouping stays active-only and the expanded
  transaction is checked against the same literal allow-list. Before planning and
  executing a Jetson perception restart, refreshed `motors_live()` or active
  controller health reporting `tracking`/`coasting` vetoes it. Missing, stale or
  malformed active-controller health and failed unit refresh fail closed.
  Result: `cannot_restart`, reason `motors live`; overlay retained for manual
  restart, with no automatic retry when motion stops.
- `tests/test_live_config.py`: asynchronous handoff, stale builds/ready objects,
  engagement flips, no repeated engaged rebuild, typed rejection, socket thread
  ownership/teardown, nonblocking tick, ACK-only journaling and persistence
  retries, literal allow-list/unknown-unit denial and both restart vetoes.
  The history comparison clearly skips if `245d103` is absent; if present,
  every original stack comparison remains strict.

### Local verification

The 20-ready-change controller test guards file operations, stack loading,
schema validation, typed parsers and policy/coefficient constructors to raise
if invoked on the tick thread. It exercises the actual controller adapter,
preserves PID state, and requires every measured swap to stay below **0.5 ms**.
Measured on this Windows/Python 3.14.4 machine in the final full run:
**0.02750 ms median / 0.04130 ms
maximum**. This measures call duration; Jetson frame scheduling/GIL effects
still need the replay below.

The documented 22 s, 50 Hz software helper also completed locally in unchanged
and changed modes: **1,100 shadow decisions each, one gain swap in the changed
run**, no transport/device construction, helper exits 0. Changed-run `tick()`
call duration was **0.0155 ms median / 0.2881 ms maximum**, with preparation
**75.30 ms median / 94.29 ms maximum** on the worker (two builds). However,
owner scheduling around the edit worsened: maximum interval **34.08 ms changed
versus 21.26 ms baseline**, maximum deadline lateness **16.48 ms versus 1.39 ms**.
This does **not** establish the unchanged-jitter acceptance criterion. Background
Python work still shares interpreter/GIL and OS scheduling; the cheap method
duration alone cannot qualify timing. Jetson frame-gap and controller-jitter
acceptance remain open, with the unchanged criteria below. Raw local traces
remain in `.task-tmp/retime-controller-{baseline-local,local}/timing.json`.

Targeted command (all local/fake hardware and systemctl):

```sh
python -m pytest -q -s tests/test_live_config.py tests/test_control_video_controller.py \
  tests/test_control_video_runtime_check.py tests/test_deepstream_target_selection.py \
  tests/test_deepstream_pipeline_split.py tests/test_deepstream_snapshot_transport.py \
  tests/test_operator_agent.py tests/test_panel_indicators.py \
  --basetemp=.task-tmp/pytest-followup-target
```

Targeted result after the Windows sharing fix and passive UI inventory check:
**225 passed** (51.01 s). Two additional dependency-expansion authorization cases
were then added and passed in the final full suite. Full suite after the Windows
sharing fix, final allow-list enforcement and task-local dependency reuse:
**1,124 passed, 50 skipped, 1 failed, 4 collection errors, 20 subtests passed**
(74.00 s), using:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) '.task-tmp/deps')
python -m pytest -q -s --continue-on-collection-errors --basetemp=.task-tmp/pytest-followup-release-check
```

The remaining failure is the unchanged missing
`artifacts/gimbal_fit/controller_sysid_step_timed_unloaded_20260908/fit_report_external_selected.json`.
The four collection errors are `test_streamer_frame_ids`,
`test_streamer_gst_video_writer`, `test_streamer_hil_pose` and
`test_streamer_sim_mount`, requiring unavailable Windows `gi`/GStreamer.
The strict 21-stack history comparison passed here with `245d103` present;
its absent-history skip path is separately tested. Generated logs remain in
`.task-tmp/`; none are committed. The initial full run exposed the concurrent
Windows reader/replacement race (fixed above) and missing local Pillow/trimesh.
Preserved prior-task dependencies were copied to `.task-tmp/deps` without any
download or global environment change. Windows cannot qualify Linux IPC,
systemd ordering, DeepStream/GStreamer GPU execution or target scheduler jitter.

### Reviewer retiming commands and pass criteria (not run on Jetson here)

Use the same reviewed paced clip, engines, tracker and one-owner prerequisites
as the original replay below. Do not activate agents, controller units, bridge,
serial, HIL or motors. The helper below records samples in memory on the owner
and writes evidence only after cleanup. Save this Python block verbatim as
`.task-tmp/live-retime.py` in the reviewed release; it is measurement scaffolding,
not a deployed module. Run from the repository with `PYTHONPATH=.`.

```python
import json, os, statistics, sys, threading, time
from pathlib import Path
from types import SimpleNamespace
from common.live_config import LiveSettings, atomic_write, live_path, runtime_bundle
from common.config import resolve_config_paths

mode = sys.argv[1]
out = Path(os.environ["RETIME_DIR"])
out.mkdir(parents=True, exist_ok=True)
atomic_write(live_path(), "{}\n")
mutate = os.environ.get("RETIME_MUTATE", "1") == "1"
ticks, builds, frames, controller_times, change = [], [], [], [], []
timer = None
original_tick, original_prepare = LiveSettings.tick, LiveSettings._prepare
def timed_tick(self, **kw):
    began = time.monotonic()
    applied = original_tick(self, **kw)
    ticks.append((self.unit, began, (time.monotonic() - began) * 1000, applied))
    return applied
def timed_prepare(self, *a, **kw):
    began = time.monotonic()
    try:
        return original_prepare(self, *a, **kw)
    finally:
        builds.append((began, (time.monotonic() - began) * 1000))
LiveSettings.tick, LiveSettings._prepare = timed_tick, timed_prepare
def arm():
    global timer
    def update():
        change.append(time.monotonic())
        if mutate:
            value = ("deepstream:\n  shadow_tracks:\n    min_confidence: 0.6\n"
                     "  id_stitching:\n    max_gap_frames: 20\n" if mode == "video"
                     else "controller:\n  yaw_kp: 8.0\n")
            atomic_write(live_path(), value)
    timer = threading.Timer(12, update)
    timer.start()
try:
    if mode == "video":
        from jetson.deepstream import pipeline
        original_frame = pipeline.VerificationStats.record_frame
        def frame(self, meta, **kw):
            frames.append((time.monotonic(), int(meta.buf_pts)))
            if len(frames) == 1:
                arm()
            return original_frame(self, meta, **kw)
        pipeline.VerificationStats.record_frame = frame
        code = pipeline.run(sys.argv[2:])
    else:
        # Pure 50 Hz controller replay: no sockets, runtime service or device.
        from common.schemas import ControlObservation
        from jetson.control.pid import AxisPIDConfig, BasicPID
        from jetson.control.timing import ClockBounds
        from jetson.control.video_controller import VideoControllerCore, VideoControllerPolicy
        from jetson.control.live_settings import prepare_controller_settings, apply_controller_settings
        paths = resolve_config_paths(Path("configs/base"), [Path("configs/controller/sim_mount.yaml")])
        bundle = runtime_bundle(paths)
        core = VideoControllerCore(BasicPID(AxisPIDConfig(1, 0, 0, 0, .8, 3.5),
            AxisPIDConfig(1, 0, 0, 0, .8, 3.5)), VideoControllerPolicy(live_authorized=False))
        assembler, current = SimpleNamespace(), [None]
        def prepare(values):
            return prepare_controller_settings(values, (1280, 720), policy=core.policy)
        def apply(prepared):
            current[0] = apply_controller_settings(core, assembler, current[0], prepared)
        initial = prepare(bundle.mutable_copy())
        current[0] = initial[0]
        apply(initial)
        fixture = json.loads(Path("tests/fixtures/control_pid_replay_v1.json").read_text())
        clock = ClockBounds.from_exchange(**fixture["clock_exchange"])
        settings = LiveSettings(paths, unit="local/retime-controller", prepare=prepare,
            apply=apply, hot_prefixes=("controller", "laser"), initial_bundle=bundle)
        try:
            began = time.monotonic()
            arm()
            for i in range(1100):
                due = began + i * .02
                time.sleep(max(0, due - time.monotonic()))
                arrived = time.monotonic()
                controller_times.append((arrived, (arrived - due) * 1000))
                settings.tick(engaged=True)
                data = dict(fixture["base_observation"], sequence=i + 1, source_frame_id=100 + i)
                for key in ("created_monotonic_ns", "source_time_ns", "frame_received_time_ns", "frame_observed_time_ns"):
                    data[key] += i * 20_000_000
                observation = ControlObservation.model_validate(data)
                bounds = ClockBounds(clock.offset_min_ns, clock.offset_max_ns,
                                     observation.created_monotonic_ns, 0)
                assert core.decide(observation, bounds).intent.mode == "shadow"
            code = 0
        finally:
            settings.close()
finally:
    if timer is not None:
        timer.cancel()
        timer.join()
    def summary(values):
        return None if not values else dict(n=len(values), median=statistics.median(values), max=max(values))
    center = change[0] if change else float("inf")
    gaps = [(b[0] - a[0]) * 1000 for a, b in zip(frames, frames[1:]) if center - 1 <= b[0] <= center + 2]
    periods = [(b[1] - a[1]) / 1e6 for a, b in zip(frames, frames[1:]) if b[1] > a[1]]
    report = dict(ticks=ticks, preparations=builds, frames=frames,
        controller_times=controller_times, change=change,
        preparation_ms=summary([x[1] for x in builds]),
        tick_ms=summary([x[2] for x in ticks]), frame_gap_ms=summary(gaps),
        source_period_ms=statistics.median(periods) if periods else None,
        lateness_ms=summary([x[1] for x in controller_times if center - 1 <= x[0] <= center + 2]))
    (out / "timing.json").write_text(json.dumps(report, indent=2))
raise SystemExit(code)
```

On the Jetson, first rerun the contracts and guarded apply timing:

```sh
bin/python3 -m pytest -q -s tests/test_live_config.py tests/test_control_video_controller.py \
  tests/test_deepstream_runtime.py tests/test_deepstream_target_selection.py tests/test_operator_agent.py
```

Then run baseline and changed video passes, using a reviewed 20–25 s clip and
the same source period as the review (60 fps means 16.667 ms). Set `CLIP` first.
Use a new evidence directory for each repetition; retain previous results.

```sh
export PYTHONPATH=.
args=()
for f in configs/base/*.yaml; do args+=(--idcs-config "$f"); done
for kind in baseline changed; do
  export RETIME_DIR="$PWD/.task-tmp/retime-video-$kind"
  export IDCS_LIVE_CONFIG="$RETIME_DIR/live.yaml"
  export RETIME_MUTATE=0
  [ "$kind" = changed ] && export RETIME_MUTATE=1
  timeout --signal=INT 30s bin/python3 .task-tmp/live-retime.py video "$CLIP" \
    --paced --tracker nvdcf --tracker-config configs/deepstream/tracker_nvdcf.yml \
    --nvinfer-config configs/deepstream/nvinfer_yolo26s_736_drone_person_smoke.txt \
    --shadow-min-confidence 0.3 --shadow-max-age 30 --id-stitch-max-gap 45 \
    --target-selection --snapshot-result-bind "ipc://$RETIME_DIR/snapshots.sock" \
    --health-file "$RETIME_DIR/health.json" --report "$RETIME_DIR/report.json" "${args[@]}"
done
for kind in baseline changed; do
  export RETIME_DIR="$PWD/.task-tmp/retime-controller-$kind"
  export IDCS_LIVE_CONFIG="$RETIME_DIR/live.yaml"
  export RETIME_MUTATE=0
  [ "$kind" = changed ] && export RETIME_MUTATE=1
  timeout --signal=INT 30s bin/python3 .task-tmp/live-retime.py controller
done
```

Pass: normal replay exit **0**, worker cleanup completes, receipts/journal and
selector revision converge to the applied change, ready-change `tick()` stays
below **0.5 ms**, and **no metadata frame-arrival interval exceeds 2× the source
period in the interval from one second before to two seconds after the edit**
(33.333 ms at 60 fps). Inspect raw `frames`, `ticks`, `preparations` and
`controller_times` in each `timing.json`, not the coarse health sample. The
controller's 20 ms cadence/lateness distribution and missed-period count must
stay within repeated unchanged-run scheduler noise, with **no added missed
ticks around the change**. Rebuild duration itself may remain 64–110 ms; it is
now worker time, not owner time. The synthetic controller replay qualifies
software scheduling and state swaps only; it constructs no transport or motor
controller and cannot qualify physical control. If an existing authorized
software-runtime trace is also used, apply the same jitter criterion to it.

## Original implementation (2026-10-17)

Branch `codex/live-config`, task baseline `245d103` (`lab/config-cleanup-reviewed`).
No lab hosts, runtime network peers, serial/RS485, motor bridge, HIL or motors were
contacted. No deployment, activation, push, tag, PR or history rewrite was run.
Local tests use fakes and deterministic observations. Generated logs, merged
maps and fold examples are ignored under `.task-tmp/`, not committed.

## Design as built

- Explicit per-section field contract in `common/config_schema.py` and
  `common/config_fields.json`; existing typed validators are retained. Unknown
  paths, bad types, nonfinite values, invalid combinations and bridge-cap
  bypasses fail closed.
- `common/live_config.py` owns the final host overlay, atomic writes, lock/CAS,
  append-only journal, unit stacks parsed from systemd templates, per-unit
  receipts, host-local REQ/REP and update PUB/SUB, with durable-file recovery.
- `tools/idcs_config.py` implements get/set/unset/diff/revert/clear/status/fold,
  local full-stack preflight and agent revalidation. Folding prints a real
  applicable patch; a sample passed `git apply --check` without being applied.
- `jetson.operator_agent` accepts configuration on its existing REP channel and
  local IPC; `pc.mode_agent` adds local IPC while keeping its outbound mode poll.
  Restart planning considers effective changed values and conservative consumer
  sections. Only active software units are try-restarted, with one transaction
  for affected active SIM dependencies. Physical/HIL/current-agent units return
  `cannot_restart`. Startup/schema-only bridge/serial edits add no live loader,
  subscription, socket or device access.
- Controller tick swaps preserve core/PID/estimator state and sequence history;
  idle caps require fresh manual-panel proof and no preceding live motion.
  Safe-point changes happen before assembling the next observation. A startup
  snapshot is retained through initialization, preventing revision races.
- DeepStream swaps shadow/stitch policies in the metadata probe, retaining
  stitch history for parameter changes. Each queued selector request carries
  desired settings and revision so latest-only coalescing cannot drop an
  update. The worker reconfigures selection/timing without reloading the model.
  Old-revision results are discarded. Probe and worker digests are separate.
- Controller diagnostics/health/ready/recorder records, DeepStream health/ready/
  V2 recorder snapshots, agent state, and existing startup outputs expose
  provenance. Pi PanelState carries its startup digest. Constants restore the
  removed authored values for existing typed readers. The swarm model metadata
  pins model and external tensor bytes to normalization; no model was changed.
- `tools/deployment.py` remains a standalone installer. A nonempty host overlay
  blocks stage unless an explicit folded/discard choice is recorded, including
  exact file hash and contents. Verification retains the decision. Stage never
  clears runtime state or claims the patch has been folded.

## Full key-to-tier contract

See [the complete table](live_configuration_keys.md) for every field, type,
domain/default and tier. The table contains **834** entries including
structural maps/list members: **730 restart, 37 fixed, 60 live, 7 idle**. No removed cleanup key is restored.
Optional supported controller endpoint/manual/FOV settings and tracker-path
overrides are explicitly declared even where no committed overlay writes them.

## Keys moved to constants or model metadata

Values below are exactly those authored in baseline base files. Lists count as
one leaf in the merged-map audit. There are nine groups, 29 authored leaves.

| Former YAML path | Fixed value |
|---|---|
| `camera.known_size_ranging.class_aspect_ratio_limits` | `{"drone": {"max": 2.0, "min": 0.4}, "person": {"max": 4.5, "min": 1.2}}` |
| `camera.known_size_ranging.min_pixels` | `10.0` |
| `camera.known_size_ranging.ema_alpha` | `0.9` |
| `camera.known_size_ranging.quality` | `{"edge_margin_px": 2.0, "hold_s": 0.25, "jump_slack_fraction": 0.1, "max_fractional_change_per_s": 1.0}` |
| `laser.render` | `{"beam_length_m": 5.0, "color_bgr": [0, 0, 255], "hit_tolerance_px": 30.0, "thickness_px": 2}` |
| `gimbal.steps_per_rev` | `3200` |
| `gimbal.counts_per_rev` | `16384` |
| `rpi.runtime_control.indicators` | `{"control_stale_s": 0.5, "fire_control_off_delay_s": 1.0, "hardware_stale_s": 0.5, "perception_stale_s": 0.5, "safety_off_delay_s": 1.0}` |
| `swarm_eval.learned_model.normalization` | `{"max_angle_rad": 0.8, "max_breakthrough_time_s": 40.0, "max_closing_speed_m_s": 12.0, "max_damage_weight": 10.0, "max_distance_m": 80.0, "max_time_to_engage_s": 8.0, "max_track_age_s": 5.0, "max_track_observations": 10}` |

The checkpoint's own normalization matches these values (read safely with
`pickletools`, without executing/loading the checkpoint). Source metadata is
`assets/models/swarm/normalization.json`; engine sidecars are created only from
reviewed builds and checked on load. An existing unbound engine will use the
configured planner fallback until its provenance is supplied. This is a
required target preparation step, not evidence that the engine was qualified.

## Merged-config comparison

`python -m tools.audit_live_config` reconstructs the baseline with local
`git show`, validates all 21 systemd stacks, and compares complete resolved maps
with fixed policy materialized. **Zero effective value changes, additions or
removals in every row.** The raw authored maps only remove fixed leaves; bridge
and serial readers use the same named motor-format values. Recorder profiles
contain none of these leaves. Full maps and SHA-256 hashes are in
`.task-tmp/live-config-audit/`; the table below is the requested summary.

| Unit | Fixed authored leaves removed | Effective changes | Digest prefix |
|---|---:|---:|---|
| `jetson/idcs-bridge.service` | 29 | 0 | `058803f5f275` |
| `jetson/idcs-controller-camera.service` | 29 | 0 | `f0e7dad6bad2` |
| `jetson/idcs-controller.service` | 29 | 0 | `2b747c6de8c1` |
| `jetson/idcs-deepstream-camera.service` | 29 | 0 | `717a6ccfcf19` |
| `jetson/idcs-deepstream-video.service` | 29 | 0 | `1c207ad56e56` |
| `jetson/idcs-operator-agent.service` | 29 | 0 | `1c207ad56e56` |
| `jetson/idcs-recorder.service` | 0 | 0 | `0765c08771a0` |
| `jetson/idcs-serial.service` | 29 | 0 | `02b5e8c58262` |
| `pc/idcs-camera-ui.service` | 29 | 0 | `05b461a27a88` |
| `pc/idcs-hil-streamer.service` | 29 | 0 | `36548ae0f171` |
| `pc/idcs-hud-relay.service` | 29 | 0 | `1c207ad56e56` |
| `pc/idcs-pc-mode-agent.service` | 29 | 0 | `1c207ad56e56` |
| `pc/idcs-sim-controller.service` | 29 | 0 | `306b970df456` |
| `pc/idcs-sim-recorder.service` | 0 | 0 | `4ea5ca3f589d` |
| `pc/idcs-sim-streamer.service` | 29 | 0 | `e9d1b5dbdb51` |
| `pc/idcs-sim-swarm-controller.service` | 29 | 0 | `c2ba23c379ac` |
| `pc/idcs-sim-swarm-streamer.service` | 29 | 0 | `df549a52288b` |
| `pc/idcs-sim-ui.service` | 29 | 0 | `ff9b11acb599` |
| `pc/idcs-ui.service` | 29 | 0 | `4efcc9604b40` |
| `rpi/idcs-manual.service` | 29 | 0 | `1c207ad56e56` |
| `rpi/idcs-operator-display.service` | 29 | 0 | `1c207ad56e56` |

## Verification

Task-local dependencies: `python -m pip install --target .task-tmp/deps
Pillow==13.0.0 trimesh==5.1.1`; pip also installed numpy 2.5.4 in that isolated
directory. No global dependencies changed. For runs needing those packages:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) '.task-tmp/deps')
python -m pytest -q --continue-on-collection-errors --basetemp=.task-tmp/pytest-full-verified
```

Final full suite: **1,076 passed, 50 skipped, 1 failed, 4 collection errors,
20 subtests passed** (63.94 s). The failure is unchanged
`test_validate_gimbal_fit.py::test_existing_low_range_fit_is_not_offline_control_qualified`,
because the pre-existing external `artifacts/gimbal_fit/.../fit_report_external_selected.json`
is absent from this clone. Collection errors are the four streamer modules
(`test_streamer_frame_ids`, `test_streamer_gst_video_writer`,
`test_streamer_hil_pose`, `test_streamer_sim_mount`) requiring unavailable `gi`/
GStreamer on Windows. These were reported, not silently skipped or changed.

The final targeted run
`python -m pytest -q tests/test_live_config.py tests/test_control_video_runtime_check.py
 tests/test_deepstream_target_selection.py tests/test_panel_indicators.py
 --basetemp=.task-tmp/pytest-last-contract` passed **113 tests**. There are
**51 dedicated live-config contract cases**. A further
`python -m pytest -q tests/test_live_config.py tests/test_deepstream_pipeline_split.py
 tests/test_deepstream_snapshot_transport.py --basetemp=.task-tmp/pytest-probe-owner`
passed **59 tests**, including an actual callback test with detector/metadata
probes on different fake threads. Local merged-stack audit:
**21 stacks, zero effective changes**. Syntax compilation and `git diff --check`
pass. Full logs remain in `.task-tmp/`.

Coverage includes schema inventory/unknown paths, all stacks, gain application
without sequence reset, pending caps, fail-closed idle proof, invalid update and
callback rejection, affected active-only restarts, SIM dependency grouping,
protected units, HIL/remote-intent conversion refusal, durable restart overlay,
unset/revert/clear, CAS/lock, per-unit pending/apply receipts and journal,
recorded digest serialization, fake REQ/REP/PUB, pure DeepStream probe updates,
model-byte/normalization binding, startup races, fold output, and deployment
refusal/decision records. Existing GPIO/safety tests remain in the full suite.

## Needs Jetson (not run here)

Prerequisites: reviewer-approved code/environment, no competing GPU/video
owner, valid reviewed H.264 clip, model assets and engine metadata. These are
software/perception checks and do not authorize any controller, bridge, serial,
HIL or motor activation. Do not start a production agent merely to run the
isolated replay check.

1. Run the same fake contracts with target dependencies:

   ```sh
   bin/python3 -m pytest -q tests/test_live_config.py tests/test_deepstream_runtime.py \
     tests/test_deepstream_target_selection.py tests/test_operator_agent.py
   bin/python3 -m tools.swarm_model_metadata assets/models/swarm/swarm_policy.engine
   sudo -n -l
   ```

   Pass: tests pass, engine binding verifies, and the existing sudoers policy
   permits `systemctl try-restart` for the allowed passive units. Missing
   permission must produce `restart_failed`; there is no stop/start fallback or
   escalation to another unit. The sudoers policy was not edited here.

2. Passive file replay with isolated paths and no network command endpoint.
   Set `CLIP` to an existing reviewed clip lasting at least 20 s. Use the
   repository's built nvinfer profile/custom parser and approved model assets.

   ```sh
   export IDCS_LIVE_CONFIG=/tmp/idcs-live-canary/live.yaml
   mkdir -p /tmp/idcs-live-canary
   printf '{}\n' > "$IDCS_LIVE_CONFIG"
   args=()
   for f in configs/base/*.yaml; do args+=(--idcs-config "$f"); done
   timeout --signal=INT 25s bin/python3 -m jetson.deepstream.verify_pipeline "$CLIP" \
     --paced --tracker nvdcf --tracker-config configs/deepstream/tracker_nvdcf.yml \
     --nvinfer-config configs/deepstream/nvinfer_yolo26s_736_drone_person_smoke.txt \
     --shadow-min-confidence 0.3 --shadow-max-age 30 --id-stitch-max-gap 45 \
     --target-selection --snapshot-result-bind ipc:///tmp/idcs-live-canary/snapshots.sock \
     --health-file /tmp/idcs-live-canary/health.json \
     --report /tmp/idcs-live-canary/report.json "${args[@]}"
   ```

   In a second local terminal with the same environment, while frames run:

   ```sh
   bin/python3 -c 'from common.live_config import atomic_write,live_path; atomic_write(live_path(), "deepstream:\n  shadow_tracks:\n    min_confidence: 0.6\n  id_stitching:\n    max_gap_frames: 20\n")'
   cat /tmp/idcs-live-canary/health.json
   ```

   Pass: frames continue with one GPU/video owner; health/receipt digest changes
   to the validated desired revision; selector digest converges, pending worker
   acknowledgement clears, shadow/stitch values change in the probe without
   replacing its track history; exit/cleanup completes normally. No command
   transport, controller or motor service is involved. This checks hot delivery,
   not learned detector quality or hardware acceptance. Windows cannot run it.

3. After the reviewer has an authorized passive agent session, qualify real
   restart permissions and process ordering using the local config CLI:

   ```sh
   systemctl show idcs-deepstream-video.service idcs-deepstream-camera.service \
     idcs-controller.service idcs-controller-camera.service idcs-bridge.service \
     idcs-serial.service -p Id -p MainPID -p ActiveState
   bin/python3 -m tools.idcs_config --host jetson set deepstream.return_iframe_interval 15
   bin/python3 -m tools.idcs_config --host jetson status
   bin/python3 -m tools.idcs_config --host jetson unset deepstream.return_iframe_interval
   ```

   Pass: only the already-active affected passive DeepStream unit restarts per
   mutation. Inactive units remain inactive and all controller/bridge/serial
   PIDs/states stay unchanged. CLI reports denied permissions/unavailable units
   explicitly. Do not run this check until that passive software restart is
   separately authorized by the reviewer. This task ran its fake equivalent.

## Deviations and open decisions

- Bridge and serial are deliberately release-only, including startup. Initial
  automatic review rejected a proposed integration because it included them in
  overlay loading. The revised implementation gives them schema/constants only,
  with no overlay/subscription. Physical/HIL/current-agent restarts, including
  the camera controller even when shadow, require manual reviewed handling.
- PC had no inbound REQ/REP channel to extend. Its new endpoint is host-local
  IPC, preserving the existing no-inbound-PC-network architecture.
- Live/idle tiers describe capability by owning consumer. Legacy constructor
  consumers use a conservative affected-unit restart, explicitly reported.
  Learned-model lifecycle/scheduling, intrinsics, defended-asset position and
  enabling/changing operational modes remain restart tier.
- A host-wide overlay cannot change the software/HIL motion source or move a PC
  software controller's intent output off loopback. Those require a separately
  reviewed release, rather than an automatic restart of an active SIM session.
- Native nvinfer/label artifacts are restart-only opaque inputs; their internals
  were not converted into hot YAML keys or a new nvinfer parser (phase 2 remains
  deferred). The native tracker YAML fields are inventoried, but native-profile
  edits still follow release review.
- Worker application is asynchronous: the probe and worker expose separate
  applied digests while acknowledgement is pending. A worker fault drops old
  decisions and reports an error; there is no blocking GPU-probe round trip.
- No Pi config-command agent was requested/added. Pi software loads a manually
  provided local overlay on startup, with fixed safety policy; PC/Jetson agents
  cannot manage Pi units. A future Pi command path needs an explicit design.
- Stage records folded/discard intent; it does not clear the file or assert
  semantic folding. Operator review/clear/revalidation is required before
  activation. The standalone script has the explicit option; the multi-host
  shell wrapper does not silently choose it.
- Journal and overlay are separate durable files. Incomplete write-ahead
  transactions require digest/receipt review and explicit recovery; automatic
  revert includes completed transactions only. Target filesystem power-loss
  behavior and IPC permission/lifecycle handling need Linux qualification.
- Optional provenance fields extend strict V2 snapshots and diagnostics.
  Consumers must receive the reviewed code release together; older strict
  consumers may reject the new fields during a mixed-version rollout.

No physical bench test is needed to qualify the fake contracts. Wiring repair,
every RS485 motor responding, and all existing bench authorization gates remain
prerequisites for any later motor/HIL work.
