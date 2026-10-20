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

## 2026-10-20 - reference refresh measured and tried; not merged

- Measurement (baseline review replays and the recorded fusion calls from
  2026-10-19, replayed through an instrumented fusion copy): on real footage
  the reference fit is trusted only in the first 1-8 frames; afterwards every
  accepted frame is "position agrees; scale held" (41-254 frames), so DCF
  learning freezes almost at once and PSR declines before each loss
  (e.g. 25 -> 13). The failing condition is almost always an empty or
  near-empty reference set: 296 untrusted frames had 0 of 0 reference tracks
  (all failed LK status), others 1-3. Tracks decay over 10-35 frames
  (e.g. 25 -> 16 -> 11 -> 2 -> 0). When trusted, the fit is near perfect
  (residual and FB 0.00 px median).
- Trial (branch `reference-refresh`, `IDCS-trials/reference-refresh-20261020`):
  additive `vl_refresh_reference` (reference pyramid, points and box from the
  just-committed frame), fusion `rebase_reference`, session policy refreshing
  when valid reference tracks fall below 70% or every 30 accepted frames, on
  trusted frames ("trusted") or also strong adjacent flow + PSR >= 2x min
  ("confident"). Off reproduces the earlier review exactly.
- Result: "trusted" never fired on real clips (trust ends before the 70%
  point); "confident" changed little (seg1159 lost at frame 28 instead of 50,
  seg2520 at 146 instead of 141, others unchanged). Fixtures unaffected.
- Conclusion: every real-clip loss is adjacent-flow attrition ("insufficient
  forward/backward features" / "inconsistent feature motion") with few new
  corners on small targets; the empty reference is a symptom of the same
  fragile points. Not merged. Next candidate: LK initial-flow prediction from
  the DCF box motion (and/or more pyramid levels) for adjacent tracks.
- SIM "insufficient feature spread" losses were mostly the test seed: a padded
  truth box (~72x151) much taller than the drone's textured part (features
  spanned 33 px vertically against a 30 px threshold).

## 2026-10-20 - why real-clip locks end: point starvation; replenishment and min-points trials

- Per-point analysis of the recorded fusion calls (review clips): 98% of
  submitted adjacent points pass FB <= 1 px, LK status failures < 1%, and
  target motion before each loss is 1.5-6 px/frame (truth), well inside the
  3-level LK reach. Locks end because the point count drifts down (~25 -> 7-16):
  replenishment adds 0.5-1.5 new corners per frame on average and none in
  50-70% of frames, while 0.4-1 points per frame leave as fit outliers; most
  runs end with an abrupt collapse in the last 1-2 frames (e.g. 13 -> 5 -> 3).
  DCF PSR at loss is still 11-15 (minimum 6). LK initial-flow prediction was
  therefore not tried (motion is not the cause on these clips).
- Trials (branch `replenish-trial`, `IDCS-trials/replenish-trial-20261020`,
  full review set vs truth, off reproduces the earlier review):
  - `IDCS_VPI_REPLENISH` replenish-only Harris strength 0.2: identical to off
    (corner counts saturate below 0.5); NMS/spacing 2 px: small changes, same
    loss frames; 2x upsampled window: much worse (seg6565 lost at 15 vs 259,
    seg2520 at 54 vs 141; interpolated corners are unstable), +1.2 ms.
  - `min_points` 6 and 5 (review `--gates-json` pass-through): clip loss frames
    unchanged (abrupt collapse); open-tight hand now seeds and tracks; no false
    tracking anywhere; fixtures unchanged.
- Nothing merged. Candidate next step: a bounded DCF-only hold while features
  are too few (strict PSR floor, frame limit, size frozen), since DCF remains
  confident at every loss.

## 2026-10-20 - bounded DCF-only hold (trial, branch `dcf-hold`)

- `IDCS_VPI_DCF_HOLD="<psr floor>,<max frames>"` (default off, fusion only):
  when a tracking lock hits a feature-starvation reject (insufficient FB
  features, degenerate geometry, inconsistent motion, insufficient spread),
  keep it on the DCF position if PSR >= floor, the DCF jump is within the
  existing disagreement limit and the run is below max frames; size frozen,
  no template learning, surviving FB points retained so replenishment refills.
  Any normal accept resets the run; otherwise the normal loss path applies.
  8 unit tests.
- Review set vs truth (`IDCS-trials/dcf-hold-20261020`; off reproduces the
  earlier review): 12,30 and 12,60 identical: seg1159 coverage 0.08 -> 1.00
  (never lost, error p50 23 -> 8 px), seg1920 82 -> 85, seg2520 141 -> 143,
  seg5155 22 -> 25, seg6565 unchanged; 20 hold frames in total bridged gaps
  worth 560 more tracked frames. Fixtures/hands unchanged (occlusion and loss
  still lost at 57). 10,30 drifts: seg1920 coverage 0.69 with error p50 127 px,
  p95 310 px while the review's false-tracking count stays 0 (that count does
  not catch drift; use truth error). 14,30 loses the seg1159 gain (its PSR
  during starvation was ~13-14). Floor 12 is the only useful value on this
  set; margin is thin and the gain rests mostly on one clip.
- Timing: no cost (native p50 8.7 ms all variants; total p50 10.9 ms, p95
  13.3 -> 12.7 ms with the hold).
- Not merged; needs more footage (live SIM single drone, real camera) before a
  default.

## 2026-10-20 - DCF-only hold made default; deploy-102 rolled out

- Project lead: make the hold the default and try it live. `DEFAULT_DCF_HOLD =
  "12,30"`; `IDCS_VPI_DCF_HOLD=off` disables. PC suite 1467 passed, same
  environmental failure set.
- deploy-102 (`db21d48`): a first `deploy-102` tag was created on the wrong
  commit (before fast-forwarding `main`) and deleted within seconds, never
  staged or pushed, then recreated on `db21d48`. Config policy empty (no
  config change since deploy-101). Jetson assets linked as before plus the
  release-built visual-lock libraries. Validation PC 369/4 skipped, Jetson
  392, Pi 82; activation Jetson -> Pi -> PC verified first time; audit: all
  units reference deploy-102; bridge/serial/controller inactive.

## 2026-10-20 - Camera / Visual lock failed on deploy-102: PrivateTmp hid the Argus socket

- The project lead selected Camera / Visual lock and lost video: the receiver
  logged "Connecting to nvargus-daemon failed: No such file or directory",
  never produced a frame, and the operator agent rolled back to Standby
  ("incoming source/return video did not become ready").
- Cause (reviewer error): the review fix added `PrivateTmp=true` to
  `idcs-visual-lock-camera.service`; nvargus-daemon's socket is in /tmp, which
  the detector camera unit documents ("No PrivateTmp: ..."). The unattended
  checks only ran the SIM mode and `--check` for camera, so this was missed.
- Recovery: Camera mode restored through the operator agent (ready; Pi 29.9 fps).
  Fix: PrivateTmp removed with the same comment, plus a test that no Argus
  unit sets PrivateTmp. Shipping as deploy-103 with a real Camera / Visual
  lock mode switch as part of verification.

## 2026-10-20 - Pi return-video recording

- Finding: nothing recorded video; the menu's Recording runs the Jetson flight
  recorder (data only). The visual-lock session's `return.h264` evidence is
  capped at ~20 MB, i.e. its first ~8 s, so the 15:15 camera lock (open hand,
  X-large box, 11 s, 18 DCF-hold frames) has data and seed crops only
  (`IDCS-build/visual-lock-camera-20261020-1515/`).
- Project lead: record on the Pi (the PC is not always present).
  `rpi/display/recording.py`: tee after h264parse, one leaky-queue -> h264parse
  -> splitmuxsink(matroskamux) branch attached per recording, keyframe gate,
  EOS on stop, files `<start>-<segment>-return.mkv`, prune oldest beyond the
  budget. Config `rpi.operator_display.recording` {dir, segment_s 60,
  max_total_gb 15} (Pi 36 GB free). Display follows the agent's `recording`
  flag (last known value if the agent is briefly silent); report gains
  `recording`.
- Pi test with the real ReturnVideo on a spare port, fed RTP (pt 97) from the
  PC: on 5 s / off 3 s / on 14 s / abrupt stop. A static valve+split-now design
  named the second recording's file by the earlier stop or ~7 s late; the
  per-recording branch fixed it: files `154656-000` (150 frames, 5.00 s) and
  `154704-000` (234 frames, 7.80 s), both finalised and readable. An
  unfinalised Matroska file was also confirmed playable.

## 2026-10-20 - deploy-104 rolled out; Pi recording live

- deploy-104 (`70f2ddb`): Jetson/PC activated from a deploy-103 base; the Pi
  first refused staging ("unreviewed configuration drift") because the policy
  was keyed on deploy-103 while the Pi was still on deploy-102; Pi entry
  re-keyed, staged, second activation passed. Audit: all hosts deploy-104.
- Pi recording started immediately because the Jetson flight recorder was on:
  `~/idcs-recordings/video/20261020T160138-000-return.mkv` etc., 60 s
  segments of ~150 MB at ~19 Mbit/s (visual-lock receiver, all-IDR 20 Mbit/s;
  YOLO mode is 8 Mbit/s). 15 GB therefore holds ~1.8 h of visual-lock or ~4 h
  of YOLO footage before the oldest segments rotate out. Pi overlay p95 11.7 ms
  right after activation.

## 2026-10-20 - recordings did not play normally: per-segment zero start

- The project lead could not play the recordings normally (VLC). Cause:
  splitmuxsink keeps one timeline per recording, so segment -001 starts at
  1:00, -002 at 2:00, etc.; frame spacing also follows arrival (16-91 ms).
- Existing footage: joined per recording into MP4s starting at 0:00 on the PC
  (`~/idcs-recordings/export/`, GStreamer splitmuxsrc -> mp4mux, no re-encode).
  The hand test (16:07) is in `20261020T160545-return.mp4` at ~1:14-2:20; its
  Pi segments had already rotated out (continuous recording since 16:05 hit
  the 15 GB budget in ~2.6 h), so only the PC copies remain.
- Recorder rewritten: one queue -> h264parse -> matroskamux -> filesink branch
  per segment, pad offset so each file starts at 0:00, next segment attached
  before the old one's EOS. Pi test: 5 s file and 10.0/10.1/5.0 s segments, all
  start 0.000, 752 frames for 750 expected (no gap, ~2 frames overlap).
- Clock note: Jetson wall clock +2.2 s vs PC/Pi (not NTP-synced since 13 Oct).

## 2026-10-20 - resident Camera/SIM ROI, delegated review candidate

- Branch `codex/seamless-roi`, base `285f962` (deploy-105 source). Local Windows
  work only: no lab host, GPU/camera, serial, bridge, HIL or motor contact;
  no deployment/activation, push, tag or PR. Existing managed visual-lock modes
  and their defaults remain as fallback.
- DeepStream gains a mux routing barrier and one common RGBA frame owner.
  Pending/acquiring/tracking ROI buffers bypass both nvinfer and nvtracker;
  release uses YOLO on the next admitted frame, loss on the following one.
  Actual primary/tracker input counters and a bypass-violation guard distinguish
  real GPU bypass from merely hiding detector selection. Original detector
  configuration/input pixels remain unchanged while YOLO runs.
- Reuses exact 16-frame/400 ms GPU cache, bounded catch-up, VPI session/CUDA
  borrowing, unchanged gates/DCF hold12,30 and visual OSD. Exact return identity
  5595, seed/release5596 and normal V2 on5564 remain the existing ports. Optional
  strict `roi_state` carries capability/ownership; agent commands, lock expiry
  and score actions cannot override it. Acquisition status latches authority
  before a cold VPI output. Pi exposes the existing box workflow directly in
  normal Camera and Simulator. Single's existing loopback publisher relays ROI
  selection while owned; stale ROI withholds selection, never truth-falls-back.
- The learned selector also has a shared pause/work barrier: acquisition waits
  for its current job (500 ms timeout), then no queued policy work runs while
  locked. Epochs discard pending pre-ROI jobs/results on resume. Added threaded
  fake-worker tests prove pause ordering and stale-job rejection; process/service
  ownership does not change.
- Release/loss requires an explicit new box. Missing files/dependencies/ABI
  withdraw ROI capability visibly while YOLO/video continue. Evidence is per
  lock under separate `/var/lib/idcs/visual-lock/seamless/{camera,sim}` roots,
  using existing rotation/BoundedLog limits. Only the two passive software
  units gain StateDirectory; no physical/HIL fragment changed. A serialized
  frame-owner close replaces the ROI runtime's forced interpreter exit; native
  teardown faults remain visible and require Jetson lifetime qualification.
- Tests: pure seamless contracts exercise real session/fusion with fake native
  pixels, routing and socket dispatch, release/loss/arbitration, agent acquisition
  races, Camera/SIM menus, SIM lease/relay, evidence bounds and missing native
  files/dependencies/ABI. Both RTP and local-camera runtime `--check`, compileall
  and reviewer CLI `--help` run without opening video/control sockets.
- Baseline proof: an untouched `git archive HEAD` snapshot outside the worktree
  ran `python -m pytest -q --continue-on-collection-errors --tb=short`: **1382
  passed, 58 skipped, 9 failed, 8 collection errors**, 20 subtests. Failures:
  two Windows temporary-path casing assertions; five missing-Pillow tests; one
  missing trimesh test; one absent untracked gimbal fit report. Collection:
  four missing gi and four missing Pillow imports. The first candidate full
  run reproduced those exact failure names. Final results are recorded below.
- Cost is not hardware-qualified: no idle VPI calls/payloads in fake-native
  tests, but continuous exact-frame caching and serialization add overhead.
  Payload is 56.25 MiB (59.77 MiB transient clone-before-eviction), plus driver
  overhead. At60Hz, cache read+write traffic is ~442 MB/s; an assumed25 GB/s copy
  rate alone implies ~0.30 ms/frame. Existing live VPI was ~11-13 ms vs YOLO
  ~11 ms; locked frames bypass YOLO/NvDCF. Cold allocation (~160 ms in prior
  trial) plus bounded catch-up and destruction remain switch-frame risks.
  Idle-cost equality/unchanged cadence, 16.7 ms steady budget, cold freshness,
  GPU bypass, return continuity and native normal exit remain explicit reviewer
  gates. Do not relax source-gap/freshness guards to pass them.
- `docs/visual_lock.md` holds design, cost assumptions and the exact Jetson
  build, panel-less SIM programmatic5596 check (`tools.check_seamless_roi`), Pi,
  camera, missing-native and fallback/lifetime checklist. Hardware/panel
  acceptance and actual SIM/shadow follow were not exercised here.
- Final candidate verification:
  `python -m pytest -q --continue-on-collection-errors --tb=short` -> **1407
  passed, 57 skipped, the same 9 failed and 8 collection errors**, 20 subtests,
  175.96 s. No new failure names versus the untouched baseline. The additional
  baseline skip is the historical Git comparison, available in the real checkout
  and unavailable in the archive snapshot.
  `python -m pytest -q tests/test_seamless_roi.py tests/test_deepstream_target_selection.py tests/test_deepstream_runtime.py --tb=short`
  -> **45 passed**, including all **24 new seamless contracts**.
  `python -m pytest -q tests/test_seamless_roi.py tests/test_managed_visual_lock.py tests/test_operator_agent.py tests/test_deepstream_runtime.py tests/test_live_config.py::test_all_21_systemd_stacks_validate_and_keep_base_effective_values --tb=short`
  -> **121 passed** before the two final selector-barrier tests were added;
  the complete suite includes those additions. Both normal/local-camera
  `python -m jetson.deepstream.runtime --check` invocations, changed-module
  compileall, `python -m tools.check_seamless_roi --help` and `git diff --check`
  passed. Local baseline/candidate reports are preserved outside the repository
  in the Windows temporary `idcs-seamless-roi-base-9a72ef15f11345739f04f1bfc68b3ead`
  directory; no recordings or rollback sources were removed.

## 2026-10-20 - seamless ROI idle follow-up (offline review candidate)

- Branch `codex/seamless-roi-idle`, base `28bc4ef`. Local Windows work only;
  no lab hosts, hardware interfaces, deployment, activation, push, tag or PR.
  Preserved recordings, trial data, rollback sources and model files.
- Removed unconditional frame-owner waits and idle GPU cloning. Idle and
  selecting retain multiple outstanding YOLO/NvDCF frames. A pending seed
  drains every earlier metadata completion; active ROI remains serialized,
  bypassing both GPU elements. Release/loss drains the final ROI frame before
  reopening YOLO. Route decisions are retained by buffer PTS through metadata
  and the GPU-entry guards. Seeds arriving after the drain decision remain
  queued until the next drained boundary; they cannot enter ROI early.
- Pi publishes strict session-scoped `visual_roi_select_v1` cache intent on the
  existing seed5596 socket: 100 ms renewals, 500 ms heartbeat expiry and 15 s
  absolute unconfirmed deadline. Expired IDs do not restart from heartbeats.
  Idle has zero cache images/copies, VPI work or ROI timing samples. Cache is
  still bounded to 16 frames/400 ms while selecting; release, loss, refusal and
  cancellation clear it on the RGBA owner. A serialized expiry event also frees
  images without a following video frame, once that owner can proceed. Missing
  selecting cache and absent/stale exact images refuse explicitly, never seed
  another image. Managed fallback/trial workflows and ports are unchanged.
- Barrier delays now request cancellation and resume the waiting buffer on
  YOLO instead of dropping it or raising a fatal frame-owner timeout. Sideband
  immediately withdraws the accepted request and names the reason; native
  cancellation/close stays on the owner. Already executing native calls must
  finish before inference resumes, and their late selected output is suppressed.
  Python cannot safely interrupt a hung native driver. Shutdown wakes routing;
  EOS during a lock closes resources and acknowledges the owner. Native teardown
  faults still surface. Existing selection authority/selector epochs, quality
  gates, DCF hold12,30, evidence bounds and missing-library behavior remain.
- New measurement hooks: cache state/occupancy, cumulative cache copies, VPI
  seed steps/updates, waits/timeouts, selection timeouts and max in-flight count
  in health/5595/final reports. Health's `stage_timing_ms_latest` makes growing
  decode/infer delays observable at 1 Hz. Both on/off now stamp infer input at
  the primary sink; bypass inputs are stamped at mux admission. ROI timing
  arrays exclude idle callbacks and are explicitly scoped as workflow-only.
- Offline evidence: deterministic idle test admits 120 outstanding frames with
  its wait method configured to raise if called; zero waits, clones, borrows,
  native backends and ROI timing samples. Exact cached acquisition/release,
  expiry without another frame, absolute deadline, no-cache refusal, complete
  drain, late seed, slow metadata, executing-native cancellation and locked EOS
  all pass. Pi Camera/SIM lease renew/cancel/new-ID and fallback isolation pass.
  This proves the removed work, not Jetson cadence or timing equality.
- `deepstream.seamless_roi` is **false** in base YAML and runtime defaults.
  The enabled graph retains the ROI queue, downstream metadata placement,
  per-buffer identity bookkeeping and sideband traffic, so literal idle-path
  identity/timing equality is not guaranteed by construction. No offline GPU
  throughput claim is made. `docs/visual_lock.md` updates the current design,
  remaining limitations and exact reviewer **off/on/off/on,60 s each, camera
  YOLO with no box**, followed by the normal SIM selecting/seed/release5596
  check. `tools.check_seamless_roi` now opens/renews the cache and waits for
  cached images before seeding. Real Jetson timing/lifetime, return continuity,
  camera/Pi panel behavior and SIM applied-follow remain reviewer gates.
- Verification:
  `python -m pytest -q tests/test_seamless_roi.py tests/test_visual_roi_seed.py tests/test_managed_visual_lock.py tests/test_roi_panel_workflow.py tests/test_deepstream_runtime.py tests/test_deepstream_metadata_adapter.py tests/test_deepstream_target_selection.py tests/test_deepstream_pipeline_split.py tests/test_live_config.py --tb=short`
  -> **251 passed**. A preceding unredirected run encountered Windows invalid
  inherited stderr handle in the historical Git comparison; rerunning with real
  redirected stdout/stderr passed all251 without code changes for that issue.
  `python -m pytest -q --continue-on-collection-errors --tb=short` -> **1421
  passed,57 skipped,9 failed,8 collection errors**,20 subtests,178.04 s.
  An untouched `git archive HEAD` baseline -> **1406 passed,58 skipped, the
  same 9 failed and 8 collection errors**,20 subtests,169.68 s. Failure and
  collection name sets match exactly: two Windows temp-path casing assertions,
  five missing-Pillow tests, missing trimesh and absent gimbal fit report;
  collection has four missing gi and four missing Pillow imports. One baseline
  extra skip is the historical Git comparison unavailable in an archive.
  Final `python -m pytest -q tests/test_seamless_roi.py tests/test_deepstream_metadata_adapter.py tests/test_deepstream_runtime.py --tb=short`
  -> **74 passed** after the final status/health changes.
  Both default and explicit-on RTP/local-camera runtime `--check`, reviewer CLI `--help`,
  changed-module compileall and `git diff --check` pass without starting video
  or control. Baseline and candidate test logs remain outside the repository
  under Windows Temp: `idcs-seamless-roi-idle-baseline-i6kcty27/pytest.txt`,
  `idcs-seamless-roi-idle-final-tests.txt` and
  `idcs-seamless-roi-idle-focused-tests.txt`.

## 2026-10-20 Seamless ROI round 3: lease-scoped probes and owner queue

- Delegated Windows software work on `codex/seamless-roi-probes`, based on
  `8b4207f` with this entry's changes initially uncommitted. No lab host,
  serial/bridge, motor, HIL, service activation, deployment or remote Git
  operation was used. Local Python is 3.14.4. Inputs were deterministic
  repository unit fixtures / fake native images; no model or plant was run.
- `jetson/deepstream/pipeline.py` now constructs the identical camera/RTP
  element description for on/off, with ordinary tracker/primary-src metadata
  placement. `roi_pipeline.py` installs only that ordinary metadata callback
  at construction: no ROI per-frame/per-packet/event probe or owner queue.
  Idle `_metadata_probe` branches call no ROI handlers/wrapper. Cached
  capability is included in the existing snapshot copy; source freshness is
  stored by ordinary metadata for both flags and sampled by the control timer.
  The 10 ms command timer / 10 Hz status publisher remain, so this is a proof
  about added frame/packet callbacks and graph layout, not total CPU equality.
- A selecting lease temporarily blocks the metadata source, inserts the
  queue, moves metadata to its RGBA owner and attaches routing/GPU-entry probes.
  Encoder/return BUFFER_LIST probes exist only during the live selecting
  window, not while locked. Exact completed return identities are bounded to
  128 entries per window; unmapped/early seeds get an explicit warm-up refusal.
  Late callbacks keep the old mapper and cannot publish into a later window.
- At completed lease/lock/catch-up, mux admission drains, the ordinary source
  pad blocks and a serialized owner retirement releases native/cache resources.
  After queue-src IDLE acknowledgement, GLib stops/removes the queue, restores
  normal metadata and removes all ROI probes/blocks. Expiry without video,
  dropped-metadata bounded drain, loss without a following frame, transitions
  during shutdown and a new lease during removal have deterministic coverage.
  `roi_runtime.py` retirement rejects queued old-window seeds even when video
  stops with a heartbeat/seed still pending. Native teardown faults remain
  visible. GPU bypass, selector authority, cache and quality gates remain.
- `rpi/operator_display.py` and `tools/check_seamless_roi.py` bootstrap the first
  lease from fresh session-scoped status, avoiding dependence on an idle return
  identity. The editor may open during warm-up; a seed still requires exact
  displayed identity. Preview availability preserves lamp-powered GPIO12
  sampling during warm-up/after loss, so readable OFF can select again. Managed
  fallback/trial identity gates are preserved. Base/default seamless ROI stays
  **false**. `docs/visual_lock.md` contains the code-path argument, lifecycle
  fields and reviewer off/on/off/on 60 s camera A/B, plus returned-idle and
  repeated selection/expiry/release/loss checks in the same process.
- Verification on this machine:
  `python -m pytest -q tests/test_seamless_roi.py --tb=short` -> **63 passed,
  1 skipped**. The explicit skip is the new real-GStreamer synthetic queue
  insert/remove/reopen test because PyGObject is absent; it uses no GPU,
  hardware or real sockets and must pass on a suitable reviewer machine.
  `python -m pytest -q --continue-on-collection-errors --tb=short` -> **1447
  passed, 58 skipped, 9 failed, 8 collection errors**, 20 subtests, 177.09 s.
  An untouched `git archive HEAD` baseline -> **1420 passed, 58 skipped, the
  same 9 failed and 8 collection errors**, 20 subtests, 175.18 s. Failure/error
  names match exactly: two Windows temp-path casing assertions, five missing
  Pillow tests, missing trimesh and absent historical gimbal-fit report;
  collection has four missing gi and four missing Pillow imports. Baseline's
  historical-Git-comparison skip is replaced by the candidate's new Gst skip.
  After preserving preview lamp sampling and adding the loss/reselection
  assertion, final `python -m pytest -q tests/test_seamless_roi.py tests/test_roi_panel_workflow.py --tb=short`
  -> **68 passed, 1 skipped**.
  Default-off and explicit-on RTP/local-camera runtime `--check`, reviewer CLI
  `--help`, changed-module compileall and `git diff --check` pass without
  opening sockets, video or control. Logs remain under Windows Temp:
  `idcs-seamless-roi-probes-unit.txt`,
  `idcs-seamless-roi-probes-final-panel.txt`,
  `idcs-seamless-roi-probes-final-full.txt`,
  `idcs-seamless-roi-probes-baseline-msr9k3qj/pytest.txt` and
  `idcs-seamless-roi-probes-check-dhz3vsq6/`.
- Acceptance is limited to software contracts and idle construction. Real
  GStreamer/DeepStream dynamic relinking, NVMM/native lifetime, Pi behavior,
  return continuity and Jetson latency remain reviewer gates. No FPS, GPU
  timing or hardware acceptance is inferred from these results.

## 2026-10-20 Seamless ROI round 4: isolated faults and explicit native owner

- Delegated local review candidate on `codex/seamless-roi-lifecycle`, based on
  `0af9389e82d0c8fa7327e3777c25878ab409f0c6`; changes were uncommitted during
  verification. Host: this Windows workspace, Python 3.14.4. Inputs: deterministic
  fake NVMM images/backends, fake Gst graph/threads, fake helper sockets/clocks.
  No model/plant or GPU was run. No lab hosts, serial/RS485/bridge, motors, HIL,
  deployment, activation, remote Git operation or data upload was used.
- The lead supplied a passing Jetson round-3 idle A/B: off 60.20/60.21 fps,
  on 60.20/60.19 fps; infer+metadata p50/p95 11.58/13.30 vs 11.66/13.37 ms;
  decode-to-infer p50 2.65 vs 2.30 ms. The subsequent helper failed immediately
  on pre-handshake IMMEDIATE/NOBLOCK send; ROI teardown then failed its thread
  assertion and was promoted into a fatal pipeline outcome. Those bench
  observations were supplied with this task, not reproduced here.
- `jetson/deepstream/roi_runtime.py` gives each attached workflow one lazy
  single-thread native executor. Cloning/borrowing, VPI creation/update,
  cancellation, expiry, retirement and destruction share that context. RGBA
  callbacks synchronously retain their GstBuffer until native completion;
  the GLib timer submits at most one outstanding asynchronous service task.
  No Gst custom event or IDLE callback is assumed to execute on a frame thread.
  Retirement acknowledges and joins the worker before removing the temporary
  queue; no worker exists in initial or restored idle. Release/abort cleanup
  completes before reopening YOLO. Shutdown drops new routed frames while
  native work closes, preventing inference from overlapping teardown.
- `jetson/deepstream/roi_pipeline.py` contains callback/attach/cleanup faults,
  withdraws ROI, publishes terminal reasons and restores the ordinary path.
  Partial attach rollback preserves the existing metadata probe; failed queue
  stop falls back to pad deactivation before unlinking. Cleanup attempts every
  remaining pixel/session/log/report despite individual errors and does not
  retry uncertain native destructors. `pipeline.py` no longer quits the main
  loop or overwrites the pipeline outcome for ROI faults; ordinary metadata is
  installed before optional ROI construction. Genuine bus/source failures keep
  their normal semantics. Errors remain logged and visible through status.
- The matching session/selection lease now renews through pending/tracking;
  client abandonment expires a lock after 500 ms, including without a new
  video frame. The 15 s absolute bound still applies to unseeded selection.
  Cache copying/identity probes remain gated to selecting, never to locked
  heartbeat traffic. Unseeded terminal status includes `terminal_selection_id`.
  `rpi/operator_display.py` renews the lock lease and explicitly retires its ID
  on release/loss so immediate readable-OFF reselection creates a fresh ID.
  This client change must accompany the runtime during reviewer validation;
  an older client stops renewing at seed and its lock will now time out.
- `tools/check_seamless_roi.py` keeps IMMEDIATE but uses a 2 s bounded blocking
  send, renews the same lease through lock and attempts release then cancel on
  every error/interrupt. Cleanup intent is recorded before attempting a seed
  send. Failed cleanup delivery is explicit; PUSH queueing is not runtime
  acknowledgement. Bounded linger allows queued cleanup to drain.
- Final focused verification:
  `python -m pytest -q tests/test_seamless_roi.py tests/test_check_seamless_roi.py tests/test_managed_visual_lock.py tests/test_operator_agent.py tests/test_deepstream_runtime.py tests/test_return_frame_identity.py tests/test_roi_panel_workflow.py --tb=short`
  -> **209 passed, 1 skipped**, 6.75 s. This includes all final display changes,
  shutdown admission and unseeded-terminal request scoping. The skip is actual synthetic
  GStreamer queue integration: PyGObject is unavailable on this machine.
  Fake-thread tests distinguish GLib, fake Gst caller and native worker for:
  expiry before any frame, selecting-cache expiry, cancel during warm-up,
  client loss while locked, release during blocked catch-up, EOS and shutdown.
  Every fake native creation/use/close uses one worker; 60 normal metadata
  callbacks flow after restoration. Fault tests cover queue creation/link/
  start/stop, probe/sideband attach, cache copy, seed, lock, selector, metadata,
  pixel/session destruction and retirement. Helper tests cover delayed peer,
  send failures, errors before/after seed, interruption and cleanup failure.
- Full suite:
  `python -m pytest -q --capture=sys --continue-on-collection-errors --tb=short`
  -> **1478 passed, 58 skipped, 9 failed, 8 collection errors**, 20 subtests,
  177.92 s. An untouched `git archive` of the base with a local Git-directory
  reference, running the default-capture equivalent, produced **1447 passed,
  58 skipped, the same 9 failures and 8 collection errors**, 20 subtests,
  190.15 s. Failure/error names match: two Windows path-casing assertions,
  five missing Pillow failures, missing trimesh and absent historical gimbal-fit
  report; collection has four missing gi and four missing Pillow imports.
  The earlier default-capture candidate run also had a Windows Git subprocess
  stderr-handle failure (WinError 6/50); that test passes alone and with
  `--capture=sys`, which removes this harness issue without skipping the test.
  Full-suite collection preceded only the final unseeded-terminal scoping fix/
  regression; the final focused command above covers it. No tests requiring
  unavailable dependencies are represented as passes.
  Logs remain under Windows Temp: `idcs-seamless-roi-lifecycle-syscapture-full.txt`,
  `idcs-seamless-roi-lifecycle-final-full.txt`,
  `idcs-seamless-roi-lifecycle-baseline-full.txt` and
  `idcs-seamless-roi-lifecycle-baseline-failures.txt`. The preserved baseline
  snapshot is `idcs-roi-lifecycle-baseline-b33c090266be4663bbca905cfe65ad09/`.
- Default-off `python -m jetson.deepstream.runtime --config configs/base --check`
  passes (digest `c6d47d1b39a47c47fd4d5300adaf2a81c3da192d9ff23c841a903712760a9de2`).
  Explicit-on RTP and local-camera check-only invocations pass, digests
  `8cb45a380f4726359935423b7c3c4b00ea4cfec2e0f3e12e261af73eaaf9a0da`
  and `dc4ba0395b2e6fffb632ec0ff90ed6689f93435eb889fdcd30311f036d709c6a`.
  Resolved reports/overlay remain in Windows Temp under
  `idcs-roi-lifecycle-check-km8y8x34/`. Helper `--help`, changed-module
  compileall and `git diff --check` pass. Default stays **false**; the idle
  graph/probe set, both-GPU-element bypass and managed fallbacks remain covered.
- `docs/visual_lock.md` describes the thread design and reviewer sequence:
  repeat exclusive off/on/off/on idle A/B (60 s each), then the 10 s single-drone
  helper (`530 247 96 72`) requiring accepted lock, selected frames and detector
  return; then kill only the helper during warm-up and accepted lock. Require
  terminal status, detached graph, empty cache, resumed selector, advancing
  return video, unchanged PID/session/SSRC, another 60 s at baseline rate and
  successful fresh explicit selection. Also review cancel, catch-up release,
  EOS and both fallback modes. None of these host procedures was executed here.
- Actual Jetson NVMM/CUDA lifetime, worker-handoff cost, real GStreamer dynamic
  relinking, Pi behavior and throughput remain bench gates, with all motors/HIL
  prohibited. A native driver hang/process abort is not recoverable by Python:
  shutdown logs a delay after 2 s and waits for acknowledgement before releasing
  EGL. No off-thread destruction, forced exit or automatic reseed hides it.
