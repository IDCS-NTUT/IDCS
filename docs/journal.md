# Development Journal

Chronological evidence, decisions and test results for IDCS, newest entries at
the end. Each entry is dated and says what was run, on which host, with which
commit or release, and where its artifacts are kept. Old paths and results can
be superseded; do not execute historical commands as current procedures.

This journal continues the
[DeepStream and low-latency control migration journal](deepstream_migration_journal.md),
which covers 2026-08 to 2026-10-19 (DeepStream migration, controller V3,
deployments up to deploy-99, camera ROI visual lock, and the 2026-10-19
performance work). That file is closed; link to its entries by date.

State at the hand-over (2026-10-20):

- `deploy-99` (`11cf19e`) is running on PC, Jetson and Pi; source mode Camera;
  physical bridge/serial/controller/HIL units inactive (hardware hold).
- Visual lock: registered CUDA input is the default; live camera processing
  p50 12.9 ms (48 fps) with a static seed. It is not run by any deployed unit.
- Open performance work: the single GStreamer streaming thread runs tracker,
  overlay and conversions serially (~6.8 ms beyond the tracker per frame);
  VPI LK round trips cost ~7.9 ms per frame.

## 2026-10-20 - overlay queue A/B on SIM (CONTROL OFF, no motors)

- `live_trial --overlay-queue` (`b0b8084`, default off) puts a leaky
  two-buffer queue after the `visual_input` tracker probe, so overlay, NV12
  conversion and encode run on their own thread; the RGBA converter gets 8
  output buffers.
- SIM source per `docs/visual_lock.md`: PC `pc.streamer` with
  `v2_scene.yaml,visual_lock_trial.yaml`, Jetson receiver seed
  `547 255 74 46`, 60 s each, alternating, both hosts under the trial guard.
  Driver: `IDCS-build/sim-overlay-queue-20261020.sh`; Jetson outputs
  `deployments/idcs-sim-overlay-queue-20261020-*`.

  | run | processed / decoded | rate | consecutive | processing p50 / p95 |
  |---|---|---|---|---|
  | plain-1 | 3211 / 3221 | 53.5 fps | 91% | 10.78 / 13.04 ms |
  | queue-1 | 3449 / 3458 | 57.5 fps | 97% | 11.19 / 13.61 ms |
  | plain-2 | 3195 / 3204 | 53.3 fps | 90% | 10.90 / 12.97 ms |
  | queue-2 | 3468 / 3477 | 57.8 fps | 98% | 11.27 / 13.52 ms |

  Tracking held on every run (all but the 3 acquiring frames). Processing rises
  ~0.4 ms because overlay/encode now run concurrently with the tracker.
  Source-to-result lag is not comparable on SIM (source time is PC clock).
- GStreamer latency tracer (`GST_TRACERS=latency(flags=element)`) is unusable
  on this pipeline: with it loaded, nothing downstream of `nvv4l2decoder` got
  a buffer (elements up to `input_parse` logged 3126 frames), so no frame was
  verified and the run ended on "source startup timeout". Not pursued.
- RustDesk's hourly cycle restarted at 22:00 with the service restart; it now
  falls at ~:00 (00:00:23). Restarted processes stay IDL on core 5.

## 2026-10-20 - replenishment cadence evaluated and dropped

- Trial switch replenishing Harris features every N accepted frames (floor:
  replenish at once below 16 retained points), offline review set on the
  Jetson (`IDCS-trials/sim-overlay-queue-20261020/review-replenish-{1,2,3}`).
  N=1 reproduces the earlier review table exactly.
- Harris per frame on the review set: 0.40 ms (N=1), 0.28 (N=2), 0.24 (N=3).
- N=2: clutter-seg1159 coverage 0.075 -> 0.037, first loss frame 50 -> 27.
  N=3: clutter-seg1920 keeps "tracking" with error p50 92 px / p95 516 px
  (drift onto the wrong content). Saving 0.12-0.16 ms per frame does not justify
  either; the switch was removed, behaviour unchanged.
- Reference-check cadence was not implemented: fusion keeps its anchors only
  through the reference path, so skipping reference frames would make adjacent
  flow act as absolute geometry. It needs a fusion redesign, not a parameter.

## 2026-10-20 - RustDesk containment verified on SIM

- 120 s SIM run with `--overlay-queue` across the 01:00:27 RustDesk restart
  (`deployments/idcs-sim-overlay-queue-20261020-rustdesk`): 6969 processed,
  6966 tracking, 3 acquiring, no loss; 58 fps, 98% consecutive.
- Around the restart: one frame 35.6 ms and one 19.5 ms (otherwise 10-14 ms),
  largest frame gap 3 frames. Before the drop-in, the 18:42 restart gave ~20
  frames at 20-28 ms, a 150 ms source gap and a lost lock (camera run, no
  overlay queue, so not a like-for-like comparison).

## 2026-10-20 - overlay queue made the live-trial default

- `live_trial` now defaults to `--overlay-queue`; `--no-overlay-queue` restores
  the single streaming thread. Basis: the SIM A/B above. Pending: one camera
  A/B with the room lit (the camera input path differs from SIM). No deployed
  unit runs `live_trial`.

## 2026-10-20 - visual lock as a console feature: decisions, delegated to Codex

- Project-lead decisions: visual lock becomes source modes "Camera / Visual
  lock" and "Simulator / Visual lock" (SIM uses the Swarm software mount;
  primary demo path). In those modes the receiver publishes V2 on the normal
  perception port 5564, so the shadow controller / SIM mount follow it like a
  detection (aiming needs no class; no range follows camera centre). Release
  and loss return the box for the next selection; YOLO via the Source menu.
  Box size is a Pi menu setting: Small 64x48, Medium 96x72 (default), Large
  160x120, X-large 240x180 (small boxes rarely hold the 8 seed features; a
  larger box than the target adds background). SIM may keep using bench gains
  (R6 dropped); real hardware must never load `configs/sim/*`, which holds
  today and gets a guard test.
- Delegated to Codex: task `codex-bridge/tasks/visual-lock-console.md`, base
  `lab/visual-lock-console` (`3327594`). Review, builds and lab validation
  follow on return.

## 2026-10-20 - queued: visual-lock reference refresh (after the console milestone)

- Finding (code reading): the reference image and points are set once at
  seed (`vl_process` seed path) and never refreshed. When rotation, lighting or
  large scale change breaks the reference fit, frames are still accepted as
  "position agrees; scale held", but `update_template` requires confident
  geometry, so DCF learning and box size freeze while the reference disagrees.
  A lasting appearance change can therefore end in "weak DCF response" loss.
- Decision (project lead): try refreshing the reference (keyframing under
  strict agreement: DCF, adjacent flow with small FB error, confident fit)
  after the visual-lock console milestone is reviewed. Measure first on the
  review clips and a SIM run with a rotating/changing target: duration of
  untrusted reference, frozen learning before each loss.

## 2026-10-20 - managed visual-lock console candidate (offline Windows)

- Base `3327594`, branch `codex/visual-lock-console`; local Windows/Python3.14
  checkout only. No lab host, camera/GPU device, GPIO, serial, bridge, HIL or
  motor was contacted or activated. Nothing deployed, pushed, tagged or sent
  to another service. Existing recordings, model bytes and rollback sources
  are retained. This entry is candidate software evidence, not live acceptance.
- Added exclusive `camera_visual_lock` / `sim_visual_lock` source transactions,
  per-mode session/clock/video readiness, normal-source stop lists and exact
  previous-mode rollback. Camera retains the existing shadow controller; SIM
  uses the PC Swarm software session. Detector lock/expiry/catalog/class/score
  actions cannot alter visual selection, including a mixed/offline PC state.
- Only an explicit matching `--managed-mode` permits V2 port5564; all plain
  trials now default to5597 and refuse5564. Fixed seeds/NvDCF handoff remain
  isolated. Exact cache and return-frame identity, ABI4, Harris0.5, CUDA input,
  overlay queue and gates are unchanged. Loss/refusal releases request state,
  clears cached pixels and prepares resources without reseeding. Repeated
  request-scoped terminal status returns Pi to explicit box selection after a
  readable GPIO12 OFF; receiver shutdown publishes an empty selection.
- Normal Pi display/menu now supplies Select box / Release / Box size without
  a trial flag or replacement display. Pi-local JSON persists Small64x48,
  Medium96x72 (default), Large160x120 and X-large240x180; opening centres and
  clamps the preset, retaining optional fine resizing. Panel/source/video/
  session/E-stop fail-closed rules remain. Physical Pi output keeps its ROI
  auto block; only the PC loopback SIM panel bridge accepts a fresh matching
  managed lease plus receiver acceptance and physical-panel freshness. It
  never consumes a trial lease as managed authority. No controller code or
  physical-unit list changed. The aiming test uses real schema-valid V2
  `visual` tracks without a detector class/range and verifies camera-centre
  aiming with no parallax.
- New Jetson unit templates are not enabled at boot, conflict with both
  detector owners and use release-local native libraries. Runtime preparation
  builds the ABI4 tracker/live-surface adapter; deployment validation checks
  presence/digests without loading GPU libraries. Inventory accepts exactly
  the two named additions and rollback restores their prior absence even if
  installation failed early. Hardware/shadow templates have a SIM-overlay
  regression guard. Managed config fields are explicitly restart-tier.
- Managed evidence retains eight bounded session slots per source under
  `/var/lib/idcs/visual-lock/{camera,sim}`: frame/acquisition logs32/8MB,
  first64 seed diagnostics, input capture<=256MB/1805s, existing ~20MB return
  cap, bounded summaries/health/report. Full files stop collecting, not
  tracking; full session slots refuse startup until reviewed archival. No
  automatic pruning. Camera input capture stays unsupported/empty.
- Verification on this machine (logs under ignored
  `artifacts/visual-lock-console/`; pristine base retained under ignored
  `.task-tmp/visual-lock-console-baseline`, with its source tar also retained):
  - `python -m compileall -q common/return_frame_identity.py jetson/operator_agent.py jetson/visual_lock rpi/operator_display.py rpi/display tools/deployment.py tools/sim_panel.py`: passed.
  - `python -m pytest -q tests/test_managed_visual_lock.py tests/test_operator_agent.py tests/test_deployment.py tests/test_visual_lock_live_trial.py tests/test_roi_editor.py tests/test_roi_panel_workflow.py tests/test_camera_roi.py tests/test_sim_panel.py tests/test_operator_display.py tests/test_range_guard.py tests/test_return_frame_identity.py tests/test_visual_lock_acquisition.py tests/test_visual_roi_seed.py`: 219 passed,9 skipped.
  - `python -m pytest -q tests/test_live_config.py tests/test_deepstream_target_selection.py tests/test_panel_config_menu.py`:183 passed. Historical baseline comparison now permits exactly the two new source modes/stop-list additions and retains all other effective-value checks.
  - `python -m pytest -q tests/test_managed_visual_lock.py tests/test_deepstream_target_selection.py tests/test_shared_config.py tests/test_panel_config_menu.py`:188 passed at the config-integration checkpoint.
  - Full untouched-base suite with `python -m pytest -q --continue-on-collection-errors`:1322 passed,55 skipped,10 failed,8 collection errors,20 subtests passed. Eight collection errors are missing GI/Pillow. Failures include missing Pillow/trimesh, Windows temp-path casing, absent retained gimbal-fit fixtures, and one transient Windows git subprocess handle error in the old config comparison (that pristine-base test passed separately on retry).
  - Full candidate suite `python -m pytest -q tests --continue-on-collection-errors`: 1368 passed,55 skipped,9 failed,8 collection errors,20 subtests passed (162.44s). All nine failing test cases and eight collection-error modules also failed on the pristine base; no new failures/errors. Final log: `artifacts/visual-lock-console/final-reviewed-full.log`; comparison: `comparison.json` in that directory.
  - `git diff --check`: passed. New contract tests use deterministic backends,
    fake unit/PC runners, temporary files and headless menu imports; no actual
    systemd or rendering runs are represented by those passes.
- Not verified here: Jetson CMake/VPI/Argus build and native lifetime tests,
  real systemd plan/validate, Pi native rendering and persisted menu operation,
  PC OpenGL closed-loop SIM demo, camera quality/endurance and panel electrical
  operation. The exact Jetson/Pi/PC build, plan/validate, SIM demo, camera and
  recovery checklist is in `docs/visual_lock.md`; operator/launch/config/key
  inventory docs are updated. Review/bench qualification remains separate;
  no motor commands are in that checklist.

## 2026-10-20 - visual-lock console review fixes (reviewer)

- Review of Codex `9b15a38`: PC suite 1455 passed with the base failure set
  plus `test_runtime_check_reports_immutable_config_provenance`, which also
  fails alone on the base (order-dependent, pre-existing); Pi 34 menu/editor/HUD
  unittest cases pass; Jetson release-style native build and both managed
  `--check` runs pass; plain trials still refuse port 5564.
- Fixes on top (project-lead decisions):
  - Evidence: size-based rotation (4 GB per source, 512 MB reserve, refuse
    only below 5 GB free) instead of refusing after 8 sessions; rotation touches
    only managed `session-<hex>` directories.
  - Managed receivers read Argus sensor/mode/flip/white balance and return
    hosts from `--config` like the detector (camera unit adds
    `configs/controller/local_camera.yaml`); explicit `--argus-*` or
    `--return-host` are refused in managed mode. Camera unit gains PrivateTmp.
  - SIM visual lock uses one slow drone, not Swarm: new PC
    `idcs-sim-visual-lock.target` with a streamer on
    `v2_scene.yaml,visual_lock_trial.yaml,mode_simulated_mount.yaml` and a
    controller on `detections.yaml` (Jetson perception 5564). PC allowlist,
    operator modes and agent validation updated; all modes' `pc_stop` include
    the new target.
  - `test_all_21_systemd_stacks...` (skipped on Codex's machine for lack of
    the baseline commit) updated to 25 stacks.

## 2026-10-20 - deploy-100 not staged; deploy-101 carries the PC unit allowance

- Staging deploy-100 first stopped on a non-empty live overlay
  (`controller.coast_s: 2` on PC and Jetson, set via the menu at 12:26).
  Project lead: discard. Archived to `IDCS-build/deploy100-live-overlay-archive/`
  and cleared with `tools.idcs_config --host jetson unset controller.coast_s`
  (shared; both overlays now `{}`, back to the release 0.5 s).
- The planner then refused the three new PC units (`idcs-sim-visual-lock*`):
  only the Jetson visual-lock units were allowed as additions. Fixed with a
  per-role allowance and a test. Tag `deploy-100` (`64c88eb`) stays unused;
  nothing was installed from it.
- Side finding: Jetson `~/.config/idcs/live-journal.jsonl` is 112 MB.

## 2026-10-20 - deploy-101 rolled out; managed SIM visual lock checked unattended

- deploy-101 (`64c65e9`) staged with `IDCS-build/deploy-101-config-policy.json`
  (only `operator.yaml` mode lists changed); Jetson release assets linked as
  for deploy-99 plus the visual-lock native libraries built in the release.
  Validation: PC 369 passed/4 skipped, Jetson 392, Pi 82. Activation Jetson ->
  Pi -> PC. The first Pi activation failed its live gate on a cold-start
  overlay build of 16.4 ms (> 15 ms, single sample) and rolled back cleanly;
  the retry passed (video 30.2 fps, p95 12.0 ms). Audit: every unit on every
  host references deploy-101; bridge/serial/controller inactive.
- Pi overlay p95 was ~12 ms right after activation; settled to 3.3 ms in Camera mode (deploy-98: ~2.5 ms).
- Unattended check (`IDCS-build/sim_visual_lock_demo_check.py`, operator agent
  client as used by the Pi menu; truth used only to place the test seed and
  score error; reports `IDCS-build/sim-visual-lock-check-{1,2,3}.json`):
  Camera -> Simulator / Visual lock ready; seed accepted; release clears the
  request; loss returns the box (status terminal id + reason, no active
  request); reseed after loss tracked 872/873 frames for 15 s, error p50 1.4 px,
  p95 2.5 px; textureless seed refused; back to Camera ready in 10 s.
  Harness notes: the streamer delays truth ~100 ms; re-selecting the current
  mode restarts the receiver (run 2 seeded into a closing session).
- Tracking quality: 3 of 4 seeds on the single drone ended within ~1 s (or
  were refused) on "insufficient feature spread", despite 25-33 features and
  PSR ~60; the other tracked cleanly. Pose-dependent; to examine with the
  queued reference-refresh work.
- Not tested: SIM mount following (needs the Pi lease: Manual ON + GPIO12 ON
  at the panel) and the camera visual-lock mode.
