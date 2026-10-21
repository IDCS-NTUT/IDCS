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

## 2026-10-20 - seamless YOLO <-> ROI: decisions, delegated to Codex

- Project lead wants switching between YOLO and an operator box without a
  source-mode switch. Today that is impossible: the DeepStream runtime and
  the visual-lock receiver are separate processes that each need the camera.
- Decisions: one DeepStream pipeline with an idle visual-lock stage; YOLO
  (and detector selection) off while a box is acquiring/tracking, back on
  release or loss (YOLO ~11 ms + visual lock ~11-13 ms do not fit 16.7 ms);
  visual lock owns the V2 selection on 5564 while locked; Select box /
  Release / Box size directly in Camera and SIM; release/loss return the box;
  managed visual-lock modes kept as fallback.
- Delegated: `codex-bridge/tasks/seamless-roi.md`, base `lab/seamless-roi`
  (`285f962`).

## 2026-10-20 - seamless ROI v1 measured: not shippable; deploy-105 complete

- Review of Codex `28bc4ef` (branch `lab/seamless-roi-v1`): PC 1494 passed,
  same failure set as main; Jetson build and both runtime `--check` pass. YOLO
  is bypassed while locked (mux-src probe chains frames past nvinfer and
  nvtracker, with a contract check); but it is on by default and a one-frame
  "frame-owner barrier" runs on every frame.
- Live A/B, YOLO camera, no box, 60 s x 2 each
  (`IDCS-trials/seamless-roi-20261020/ab-*`): off 60.2 fps, infer+metadata
  11.6/13.3 ms, decode->infer 2.4 ms; on 50.4 fps, 14.4/16.4 ms,
  decode->infer p50 4.8 s / p95 8.7 s (growing backlog); one "on" run died on
  "ROI frame-owner barrier timed out". Not merged; follow-up delegated
  (`tasks/seamless-roi-idle-cost.md`): zero idle cost, cache only while the
  operator is selecting, barrier only while pending/locked, no fatal timeout.
- The first A/B attempt was refused by the trial guard because Camera / Visual
  lock owned the camera; the guard now also pauses/restores the managed
  visual-lock units. With the lead away and permission given, the system was
  switched to Camera mode; the Pi then activated deploy-105 on the first try
  (all hosts deploy-105) and new Pi segments start at 0.000.

## 2026-10-20 - seamless ROI v2 measured; third round delegated

- Codex `8b4207f` (branch `lab/seamless-roi-v2`): idle waits, cache copies and
  VPI work removed; Pi selecting lease over 5596 gates the frame cache;
  barrier delays cancel the ROI instead of failing the pipeline; default
  `seamless_roi: false`. PC suite 1508 passed, same failure set.
- Live A/B with an overlay enabling it (YOLO camera, no box, no lease, 60 s x 2;
  `IDCS-trials/seamless-roi-idle-20261020/ab-*`): fps 60.20 both, no backlog,
  no errors; but decode->infer 2.5 -> 3.5 ms p50 and infer+metadata
  11.6/13.3 -> 14.8/18.3 ms (p50/p95), i.e. ~4 ms p50 / ~6 ms p95 added to
  every YOLO frame. Likely the ROI Python probes attached for the pipeline's
  lifetime (per frame at several pads, per RTP packet on return_wire).
- Round 3 delegated (`tasks/seamless-roi-idle-probes.md`): attach the probes
  only while the Pi's selecting lease, a pending seed or a lock is live.
  Not merged; production stays on deploy-105.

## 2026-10-20 - seamless ROI v3: idle cost gone; lease teardown crashes; round 4

- Codex `0af9389` (`lab/seamless-roi-v3`): idle uses the off graph; probes and
  owner queue attach per lease. PC suite 1535 passed, same failure set.
- Idle A/B, YOLO camera, 60 s x 2 (`IDCS-trials/seamless-roi-probes-20261020/ab-*`):
  off 60.20/60.21 fps, infer+metadata 11.58/13.30 ms; on 60.20/60.19 fps,
  11.66/13.37 ms (difference < 0.1 ms). Idle cost resolved.
- SIM lifecycle (Simulator / Visual lock PC scene, Jetson runtime swapped in
  under the guard, seamless on): `tools.check_seamless_roi` failed after 0.2 s
  ("Resource temporarily unavailable", PUSH with IMMEDIATE), its lease
  expired, and the runtime died: "ROI teardown fault: ROI frame-owner teardown
  failed: ROI teardown moved off its frame owner". Not shippable; round 4
  delegated (`tasks/seamless-roi-lifecycle.md`): no ROI fault may stop YOLO,
  teardown on the frame owner in every exit path, robust helper.
- System returned to Camera mode (Jetson ~55-60 fps, Pi 29 fps). Pi display
  self-restarts on video loss: NRestarts 30 since deploy-105 activation.

## 2026-10-21 - seamless ROI v4: fault isolation works; reselect bug; round 5

- Codex `0bacf35` (`lab/seamless-roi-v4`): one workflow worker owns native
  work and teardown; ROI faults end only the ROI. PC suite 1567 passed, same
  failure set as main.
- Idle A/B (`IDCS-trials/seamless-roi-lifecycle-20261020/ab-*`): off 60.19 fps,
  infer+metadata 11.60/13.35 ms; on 60.19/60.21 fps, 11.65/13.38 ms.
- SIM lifecycle (two runtime runs, 6 + 15 min, both exit 0, no ROI error):
  isolated locks all tracked (~52 Hz selected); while locked pipeline 58.6
  fps, YOLO skipped, infer+metadata 5.7 ms, VPI 11.5-11.7 ms. Client killed
  mid-lock or mid-selection: ROI ended by itself within 3 s, YOLO unaffected,
  next lock worked. Textureless box refused.
- Bug: a lease arriving ~1.5 s after a release (previous teardown still
  "draining") is dropped and its renewals ignored, with no refusal; 2 of 5
  back-to-back locks never started. One unexplained early return (3 frames,
  no terminal reason). Round 5 delegated (`tasks/seamless-roi-reselect.md`).
- Test-harness note: an earlier stuck background loop resumed and ran its own
  cases concurrently with mine (lock3-5 of run 2); those refusals were correct
  single-operator behaviour, not bugs. `pgrep -f`/`pkill -f` patterns match
  their own shell; use exact process checks.
- System back in Camera mode (Jetson ~56-60 fps, Pi 30 fps).

## 2026-10-21 - seamless ROI v5: reselect still fails; root cause; round 6

- Codex `90df2dd` (`lab/seamless-roi-v5`): leases queued through teardown,
  FIFO timing fix. PC suite 1594 passed, same failure set as main.
- Idle A/B (`IDCS-trials/seamless-roi-reselect-20261021/ab-*`): off 60.18/60.20
  fps, infer+metadata 11.65/13.37 ms; on 60.20/60.19 fps, 11.63/13.32 ms.
  Locked: 59 fps, YOLO skipped, 5.5-5.9 ms, VPI ~12 ms. 30-37 ms spikes gone.
- Five locks ~1.5 s apart: 2 of 5 never started ("ROI lease retired").
  Minimal probe (`codex-bridge/runs/leaseprobe-20261021.py`, fresh lease id
  per trial, no seed) reproduces it: a lease admitted while the previous
  window tears down gets its cache for ~0.1 s, then is expired.
- Cause (reading `roi_runtime.py`): `_notify(..., False)`, `expire_cache()`
  and `retire()` act on the shared `selecting`, so the old window's teardown
  expires the new lease id. Round 6 (`tasks/seamless-roi-handover.md`):
  scope teardown to its window/generation.
- Runtime run exit 0, no faults. System back in Camera mode (Pi 30 fps).

## 2026-10-21 - seamless ROI v6: handover partly fixed; two new faults

- Codex `8e2953c` (`lab/seamless-roi-v6`): generation-scoped teardown.
  PC suite 1620 passed, same failure set as main.
- Idle A/B (`IDCS-trials/seamless-roi-handover-20261021/ab-*`): off 60.20/60.19
  fps, infer+metadata 11.62/13.37 ms; on 60.21/60.19 fps, 11.64/13.40 ms.
- Unseeded lease probe: 3/3 trials turned the cache on (fixed).
- Run 1, five seeded locks ~1.5 s apart: 1-2 tracked, then
  "ROI metadata fault: track_id values must be unique within a snapshot"
  disabled ROI for the rest of the run (YOLO kept running, ~57 fps).
  `metadata_adapter` only de-duplicates live objects against shadow tracks,
  not within either list; the duplicate appeared on a resume frame.
- Run 2, eight seeded locks: 5 tracked (~305 frames/6 s), 3 never started.
  Failed attempts: no status ever shows their selection id; status carries a
  retired lease from a later window (gen 3, after rep2's gen 1) and
  "ROI lease retired". Renewals dropped without refusal.
- Both runs exit 0. Camera mode restored (Pi 30 fps).
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

## 2026-10-21 - Seamless ROI round 5: release/reselection handover (local only)

- Base: `0bacf35` (`lab/seamless-roi-v4`); work stays on
  `codex/seamless-roi-reselect`. Verification used the dirty candidate on Windows,
  Python 3.14.4, injected monotonic time, schema-valid synthetic frames and fake
  Gst/native resources. No model/plant or physical camera was used. No lab host,
  serial/RS485/bridge/HIL/motor interface, deployment or activation was accessed.
- Reproduced the handover bug in all four removal phases. Router-side transition
  checks were insufficient: a timer/router call already submitted to the native
  executor could consume a new lease while the old graph was draining. Old
  metadata without source identity then cancelled that lease and tombstoned its
  ID, so every renewal was ignored. Abort cleanup also discarded queued control
  intent. The previous removal test did not execute late timer work.
- `jetson/deepstream/roi_runtime.py` and `roi_pipeline.py` now share a drain event
  set before mux blocking and cleared only after graph restoration. Worker-side
  service/routing/pixel processing honours it, preserving ordered control intent
  through draining/isolating/retiring/detaching. Retirement and abort discard
  old-window seeds with explicit refusal while retaining controls. Adjacent
  identical selection commands coalesce to their latest receipt, preserving
  release/cancel/seed ordering and bounding heartbeat traffic during teardown.
  Normal freshness checks resume after detachment. Queue-full refusal reports
  a selection-scoped reason and allows the same ID to retry. Explicit matching
  seed release allows same-ID reselection with a fresh seed; expiry/cancel/loss
  retain their tombstone policy. Nothing resumes the old tracker automatically.
- Owned cancellation stores request/selection/reason before clearing authority,
  emits the terminal result and queues a source-status snapshot. Queued status
  sequences are assigned in publication order with the periodic statuses.
  Pending -> acquiring no longer briefly clears both fields. An empty early
  return cannot leave ownership unexplained. Ordinary/orphan metadata while an
  ROI request owns selection ends it explicitly with
  `ROI released: frame routing ownership unavailable`, before any VPI update or
  detector-owned V2 publication. This guards the reported early-return symptom;
  its single lab occurrence was not reproduced here.
- Timing investigation found a distinct, reproducible measurement defect in
  `jetson/deepstream/pipeline.py`: one missing metadata callback leaves the old
  FIFO admission timestamp at the front, inflating every subsequent sample by
  one frame interval. The untouched base reports **28.2667 ms** twice for
  synthetic 60 Hz frames whose actual latency is **11.6 ms**. Candidate output
  is **11.6 ms** twice and one discarded input. Post-mux GstBuffer PTS now joins
  both ordinary and bypass admission to metadata; original frame-meta/source
  PTS is deliberately not used. Timing history is bounded to 128 entries;
  missing matches produce no sample, invalid/missing PTS retains FIFO fallback,
  and the report counts `discarded_infer_inputs`. Decode -> input remains FIFO.
  A real **37 ms** delay still reports 37 ms. Tests show no clones, VPI updates
  or ROI timing samples during removal despite late timer work and renewals.
  The supplied **30-37 ms** lab samples cannot be conclusively attributed from
  this machine: corrected PTS timing and actual GPU/queue profiling remain
  reviewer gates if they stay elevated. No claim of measured Jetson speedup.
- New fake-thread regressions in `tests/test_seamless_roi.py` cover all removal
  phases with and without abort, 25 deferred renewals, same/new ID release then
  immediate lease, blocked catch-up plus release/reselection, five consecutive
  locks both in one attached workflow and with full graph removal between locks
  (15 x 100 ms renewals during each drain), per-workflow native thread identity,
  all owned terminal paths, orphan/ordinary frame ownership, queued status wire
  order, and full-queue refusal/retry. Timing recovery, unknown PTS, bounded
  history and real queue delay are in `tests/test_deepstream_metadata_adapter.py`.
  `tests/test_live_config.py` now supplies a Gst buffer double with PTS.
- Final focused command:
  `python -m pytest -q --capture=sys tests/test_seamless_roi.py tests/test_check_seamless_roi.py tests/test_managed_visual_lock.py tests/test_operator_agent.py tests/test_deepstream_runtime.py tests/test_deepstream_metadata_adapter.py tests/test_return_frame_identity.py tests/test_roi_panel_workflow.py tests/test_live_config.py --tb=short`
  -> **363 passed, 1 skipped**, 37.25 s. The skip is the real synthetic
  GStreamer queue integration because PyGObject is absent. Default capture had
  the previously documented Windows Git subprocess stderr-handle failure;
  `--capture=sys` runs that test successfully. No unavailable test is a pass.
- Final full-suite command:
  `python -m pytest -q --capture=sys --continue-on-collection-errors --tb=short`
  -> **1506 passed, 58 skipped, 9 failed, 8 collection errors**, 20 subtests,
  182.72 s. The same nine failures and eight collection errors were reproduced
  in the corresponding untouched-base modules: **65 passed, 6 skipped, 9 failed,
  8 errors**, 3.72 s. Failures are two Windows path-casing assertions, five
  missing Pillow imports, missing trimesh and the absent historical gimbal-fit
  report. Collection errors are four missing gi and four missing Pillow imports.
  Baseline comparison command:
  `python -m pytest -q --capture=sys --continue-on-collection-errors --tb=short tests/test_hardware_config_preflight.py tests/test_hud_design.py tests/test_opengl_renderer.py tests/test_sim_score.py tests/test_ui_style.py tests/test_validate_gimbal_fit.py tests/test_streamer_frame_ids.py tests/test_streamer_gst_video_writer.py tests/test_streamer_hil_pose.py tests/test_streamer_sim_mount.py tests/test_tracking_reticle.py tests/test_ui_return_video.py tests/test_v2_hud_attitude.py tests/test_v2_ui_hud.py`.
  Base ZIP/directory are preserved under Windows Temp as
  `idcs-seamless-roi-reselect-base-0bacf35`; logs are
  `idcs-seamless-roi-reselect-final-full.txt` and
  `idcs-seamless-roi-reselect-base-failures.txt`. The earlier full log
  `idcs-seamless-roi-reselect-full.txt` also includes the subsequently fixed
  PTS-less settings test double; the final full run above includes its fix.
- `python -m jetson.deepstream.runtime --config configs/base --check` passes,
  default **false**, unchanged digest
  `c6d47d1b39a47c47fd4d5300adaf2a81c3da192d9ff23c841a903712760a9de2`.
  `python -m tools.check_seamless_roi --help`, changed-module compileall and
  `git diff --check` pass. Idle graph/probes, default-off configuration, both
  GPU-element bypass while locked and fault isolation remain covered.
- `docs/visual_lock.md` has the round-5 design and reviewer procedure: exclusive
  off/on/off/on idle A/B, 60 s each, then one-client **five** back-to-back locks
  roughly 1.5 s after each release (all must track), immediate Pi Select box
  after Release, then forced helper abandonment during warm-up and tracking.
  Require terminal reasons, detached graph/empty cache/resumed selector,
  unchanged PID/session/SSRC and progressing video, baseline timing after
  release and successful fresh selection. Actual NVMM/CUDA lifetime, native
  teardown cost, real GStreamer relinking, Pi behaviour and throughput remain
  lab review gates. No motors/HIL are needed or authorized for these gates.

## 2026-10-21 - Seamless ROI round 6: workflow generation isolation (local only)

- Base: `90df2dd` (`lab/seamless-roi-v5`), branch
  `codex/seamless-roi-handover`. Verification used the dirty candidate on Windows,
  Python 3.14.4, injected monotonic time, schema-valid deterministic frames and
  fake Gst/native resources. No model/plant, physical camera, lab host,
  serial/RS485/bridge/HIL/motor interface, deployment or activation was accessed.
- `jetson/deepstream/roi_runtime.py` closes admission at cancellation itself,
  before GLib enters draining. A successor lease stays in the ordered control
  queue, with adjacent renewals coalesced, until old native retirement, worker
  shutdown, queue removal and ordinary metadata restoration complete. Different
  unseeded selection IDs also require complete retirement. Explicit release
  keeps its same-ID reselect policy; cancel/expiry/loss retain tombstones. The
  window generation advances only at complete graph restoration. Native jobs,
  teardown methods, owner actions, attached probes and queued seeds capture/check
  their originating generation. Old events cannot cancel the successor, evict
  its cache, set its tombstone or acknowledge its retirement/close. A seed
  refusal sends its request result without retiring any selection. No automatic
  tracker restart/reseed or off-owner native destruction was added.
- `jetson/deepstream/roi_pipeline.py` exposes actual graph state to status reads,
  including while native work makes lifecycle return early. Queued terminal
  snapshots refresh live ownership/graph/counter fields at publication, keeping
  terminal request/selection/reason and `terminal_window_generation`. A stale
  graph snapshot, or inactive-probe snapshot with advanced routing/activation
  counters, increments `roi_counts.status_snapshot_mismatches` before correction.
  Current status names `roi_window_generation`, `roi_window_retiring` and
  `roi_selection_id`; obsolete events count as `stale_window_events`. Timer-only
  control processing no longer counts as routed `yolo_frames`/`yolo_skipped`.
- `tests/test_seamless_roi.py` reproduces admission before GLib draining for
  cancel, release, expiry and missing identity. Each regression fails on the
  archived base runtime at the premature-admission assertion (four failures),
  and passes on the candidate. Further fake-thread tests admit a successor,
  retain its cached pixels and seed, then deliver old retire, cache expiry,
  seed retirement/refusal/queued seed, owner retire/expire/service/close, Gst
  EOS/metadata and native job events. The lease/cache/tombstone survive and
  the successor actually reaches selected tracking. Existing removal-phase,
  abort, teardown-owner and fault tests remain. Five fresh leases each run for
  4 s with 100 ms renewals and 0.1, 0.5, 1.5 or 3 s gaps, both without seeds
  and with tracking. All five windows acquire cache/track in every case.
  Status regression publishes an idle snapshot after activation while its
  worker is blocked, and requires real active graph/ownership and mismatch
  evidence. A separate test distinguishes timer work from routed frame counts.
- Final focused command:
  `python -m pytest -q --capture=sys tests/test_seamless_roi.py tests/test_check_seamless_roi.py tests/test_managed_visual_lock.py tests/test_operator_agent.py tests/test_deepstream_runtime.py tests/test_deepstream_metadata_adapter.py tests/test_return_frame_identity.py tests/test_roi_panel_workflow.py tests/test_live_config.py --tb=short`
  -> **389 passed, 1 skipped**, 38.88 s. The skip is real synthetic GStreamer
  queue integration: PyGObject is absent. Tests retain identical idle graphs,
  no idle native/frame callbacks, both GPU elements bypassed while locked,
  terminal reasons, isolation of faults, and teardown on the native owner.
- Full-suite command:
  `python -m pytest -q --capture=sys --continue-on-collection-errors --tb=short`
  -> **1531 passed, 58 skipped, 9 failed, 8 collection errors**, 20 subtests,
  193.45 s. This run preceded only the final additive `roi_selection_id` status
  field and timer-count regression; the final focused command above covers
  both. The corresponding untouched-base modules reproduce **65 passed,
  6 skipped, 9 failed, 8 errors**, 4.11 s. Failure/error names match exactly:
  two Windows path-casing assertions, five missing Pillow failures, missing
  trimesh, the absent historical gimbal-fit report, and four gi/four Pillow
  collection errors. No unavailable dependency is represented as a pass.
  The baseline command uses the same 14 modules documented in the round-5
  entry. Base archive/tree and comparison logs are preserved in Windows Temp
  under `idcs-roi-handover-base-z8cstgpm/` (`admission-regression.txt`,
  `base-failures.txt`). Candidate logs are
  `idcs-seamless-roi-handover-full.txt` and
  `idcs-seamless-roi-handover-focused.txt` in Windows Temp.
- Check-only `python -m jetson.deepstream.runtime --config configs/base --check`
  passes with default **false** and unchanged digest
  `c6d47d1b39a47c47fd4d5300adaf2a81c3da192d9ff23c841a903712760a9de2`.
  Resolved settings are preserved as
  `idcs-seamless-roi-handover-config.json` in Windows Temp.
  Helper `--help`, changed-module compileall and `git diff --check` pass.
- `docs/visual_lock.md` describes the generation design, supplied round-5
  measurements (not new local measurements) and round-6 reviewer steps:
  exclusive off/on/off/on idle A/B, 60 s each; the exact no-seed probe with
  three fresh lease IDs, 100 ms renewals for 4 s, cancel and 2 s gap (all three
  must turn cache on); five back-to-back explicit locks at each requested gap
  (all five must track); and abandonment during warm-up and tracking, cancel,
  delayed catch-up release and EOS. Preserve statuses/generations/counters,
  terminal reasons, empty cache/restored graph/resumed selector, advancing
  video and unchanged PID/session/SSRC. Verify fresh selection after recovery.
  Actual Jetson timing, NVMM/CUDA lifetime, dynamic Gst recovery and Pi behavior
  remain reviewer gates; no hardware actuation is needed or authorized.

## 2026-10-21 - Seamless ROI round 7: Jetson lease and resume acceptance

- Base `8e2953cf21108b0bfdf1ed990cbd2b4eb58721f8` (`lab/seamless-roi-v6`),
  branch `codex/seamless-roi-lab`. Default `deepstream.seamless_roi` stays
  **false**. Check-only configuration digest stays
  `c6d47d1b39a47c47fd4d5300adaf2a81c3da192d9ff23c841a903712760a9de2`.
  This is a software/Jetson lifecycle qualification; no motor, serial, bridge,
  RS485, HIL, homing, step or sweep command, deployment, installed-unit/config
  edit, push, tag or PR. No Pi actuation or physical wiring qualification.
- `metadata_adapter.py` deduplicates native live objects and shadows by ID,
  preferring detector support then confidence, and protects the strict V2
  builder against duplicate stitcher output. The stitcher claims canonical
  identities before aliases, permits genuinely separated native hypotheses to
  split again, and never emits an already claimed fallback ID. The shadow
  metadata reader also deduplicates at its source. This fixes the resume-frame
  collision without suppressing two separated targets.
- `roi_runtime.py` skips unavailable source identities during **unseeded**
  selection, without copying pixels or resetting its original deadline.
  Pending/tracking requests still fail closed on unavailable identity. This
  removes the observed new-lease retirement on an ordinary frame during graph
  attachment. Admission/refusal/exit now logs the selection ID and generation.
  Status retains `lease_selection_id`, `lease_accepted`, `lease_reason` and
  `lease_window_generation` through publication-time graph refresh; unseeded
  exits immediately publish their own terminal status. Renewals stay coalesced
  and unlogged. `check_seamless_roi` waits for its own selection ID before
  seeding, records its lease, and surfaces a pre-seed termination directly.
- A metadata-probe exception ends the current workflow and withdraws capability
  until its native owner, queue and ordinary graph have completely retired.
  Two recoveries are allowed; the third metadata fault disables ROI with a
  reason. Destructor/native/graph/transport faults remain disabled when resource
  safety is uncertain. Tests exercise subsequent real fake-backend acquisition,
  full retirement, third-fault refusal and all existing generation/owner gates.
- Cold VPI acquisition initially took about 134-138 ms and the ordinary leaky
  two-frame decoded queue discarded enough following frames to trip the
  unchanged 150 ms source-gap gate. During a seed/lock barrier only,
  `roi_pipeline.py` retains up to **16 decoded frames / 400 ms**, non-leaky,
  through the existing source queue; its original settings return on release,
  loss, fault or retirement. Both off/on idle graph descriptions remain equal;
  no idle frame/packet callback or native owner was added. GPU input counters
  confirm neither primary nor tracker receives locked frames. No synthetic
  timestamp, automatic reseed, truth feedback or relaxed tracker gate.

### Hosts, preparation and preserved commands

- Git for Windows SSH (`C:/Program Files/Git/usr/bin/ssh.exe -n -o BatchMode=yes`)
  reached `idcs@140.124.39.68`; Jetson used nested
  `ssh -n -i ~/.ssh/id_ed25519_lan -o BatchMode=yes idcs@192.168.0.5`.
  Nontrivial operations were copied scripts, run by absolute path. Owners were
  checked before the build and every runtime with
  `systemctl list-units 'idcs-*' --state=active --no-pager`; initial Camera,
  shadow camera controller, operator and read-only recorder were active;
  physical/HIL owners were inactive. The managed manifest identified deploy-105.
- New folder on **both PC and Jetson**:
  `/home/idcs/Desktop/project/IDCS-trials/seamless-roi-lab-20261021/` (`T` below),
  with archive-copied branch source in `T/src`; modified files arrived as a
  separate tar overlay. No release/runtime checkout was edited. Linked engines:
  YOLO -> `/home/idcs/idcs-models/yolo/yolo26s_dataset2_e100_736_grey1280x720.engine`;
  swarm -> `/home/idcs/Desktop/project/IDCS-runtime/assets/models/swarm/swarm_policy.engine`;
  parser -> `deployments/native-assets/48eaa5e916ee4835d3998cc63dab8f3b7e3801891b3f80c56df7a20cb81f7ee3/libnvdsinfer_yolo26_parser.so`.
  These are resolved deploy-105 link targets, used read-only.
- Jetson build, from `T/src`:
  `cmake -S jetson/visual_lock/native -B build/visual-lock -DCMAKE_BUILD_TYPE=Release -DIDCS_BUILD_LIVE_SURFACE=ON -DIDCS_BUILD_TEST_SURFACE=OFF`
  then `cmake --build build/visual-lock -j4` -> **pass**, ABI4 main/live-surface
  libraries. `native-provenance.json` records all model/parser/library SHA256s.
- Source switches used the supplied kit's `set_mode.py` with the operator agent
  on 5590: original **Camera -> sim_visual_lock -> Camera**. No manual streamer
  or source-mode unit switching. `run_visual_lock_trial_guard.py` now also names
  the two managed visual-lock receiver units as software owners, so it can pause
  and restore the fallback selected by this authorized operator mode.
- Each video run used `T/sim.sh NAME DURATION` / `T/ab.sh`, a fresh
  `idcs-seamless-lab-NAME` transient unit, `RuntimeMaxSec=DURATION+120` (A/B 200),
  `TimeoutStopSec=40`, `KillMode=mixed`, UID/GID idcs, working directory `T/src`,
  `PYTHONPATH=T/src`, `PYTHONUNBUFFERED=1`, `XDG_RUNTIME_DIR=/run/user/1000`, and
  `bin/python3 -m tools.run_visual_lock_trial_guard --host jetson --output O/guard -- bin/python3 -m jetson.deepstream.runtime --config configs/base --config-extra T/seamless-on.yaml --duration-s DURATION --report O/report.json --health-file O/health.json`.
  The overlay only sets `deepstream.seamless_roi: true`. Camera A/B adds
  `configs/controller/local_camera.yaml`, omitting the ROI overlay for off.
  One receiver and one command helper at a time; no installed unit activation
  beyond the guard's scoped restoration and authorized operator mode switch.
- Exploratory transient runs `diagnostic-1/2/3` (240/240/120 s bounds),
  `acceptance-1/2` (350/140 s), then final `final-1/2` (**110 s each**) are all
  retained. Diagnostic-1/2 and acceptance-1 were ended early by SIGTERM to their
  own guard unit; their runtimes exited **0**. Every other runtime also exited
  **0**. Reports/logs document the early failures, not just successful cases:
  `diag1/diag4` cold source-gap losses, `diag2` success, `diag3` pre-seed identity
  loss, then `diag5/diag6` success after retention. `probe-matrix` ran after its
  short receiver had ended and timed out waiting for readiness; it seeded
  nothing. The first very-early `matrix-run2` lock lost to an actual source gap
  after 19 selected frames; its complete steady-source retry passed 15/15.
  Both later **final** runtimes passed their first and every subsequent lock.
  Startup/transient source gaps still terminate tracking; no gap gate was hidden.
- Guard restore attempts for managed `idcs-visual-lock-sim.service` returned a
  start-job error on some intermediate runs; the operator agent restored and
  the receiver was observed running between trials. These statuses and service
  journal are retained. Final Camera restoration was explicitly verified through
  the operator agent. The trial copy's linked swarm engine lacked its adjacent
  hash-bound normalization metadata, so runtime logged planner fallback. That
  does not feed the VPI tracker and is not learned-policy qualification.
- Session evidence generated by the kit's default ROI evidence root remains in
  `/var/lib/idcs/visual-lock/seamless/sim/` (about **14 MB** total, below its
  4 GB rotation budget). Own-trial sessions are additionally **copied**, never
  moved/deleted, to `T/native-session-evidence.tar.gz`. No recording, model,
  old trial, rollback source or deployment file was removed.

### Final Jetson acceptance numbers

The PC's task-local `acceptance.py` is one persistent test operator. It uses
truth only to choose each explicit seed box for that exact displayed source
frame/time; it supplies no ongoing tracker input. It records every status,
lease, seed and release, verifies unique published track IDs and stable GPU
input counters while locked, and waits the specified release-to-next-lease gap.
Commands used `PYTHONPATH=T/src venv/bin/python T/acceptance.py --output T/matrix-final1`
and `... --output T/matrix-final2`. Each seed runs 3 s before explicit release.

| Runtime | Gap | Tracked and detector returned | Selected frames per lock | Actual four gaps (s) |
| --- | --- | --- | --- | --- |
| final-1 | 0.5 s | **5/5** | 143, 152, 143, 146, 149 | .5073, .5050, .5016, .5065 |
| final-1 | 1.5 s | **5/5** | 155, 151, 154, 144, 148 | 1.5038, 1.5097, 1.5026, 1.5002 |
| final-1 | 3 s | **5/5** | 153, 148, 152, 145, 148 | 3.0097, 3.0095, 3.0055, 3.0038 |
| final-2 | 0.5 s | **5/5** | 139, 149, 148, 147, 147 | .5081, .5054, .5051, .5005 |
| final-2 | 1.5 s | **5/5** | 150, 145, 148, 155, 141 | 1.5050, 1.5007, 1.5065, 1.5061 |
| final-2 | 3 s | **5/5** | 140, 146, 150, 149, 146 | 3.0050, 3.0069, 3.0005, 3.0029 |

- Final runtimes: **6,271 / 6,251 frames**, **15 admissions, 15 activations and
  15 retirements each**, generations end at 15; **zero faults/roi_error**, empty
  cache, no probes or owner at exit; **return code 0 / 0**. VPI updates
  **2,231 / 2,200**; seed steps **45 / 46**. This does not claim sustained 60 Hz
  while locked: whole-run steady rates were **58.174 / 58.034 fps**.
- `PYTHONPATH=T/src bin/python3 T/probe.py` on Jetson, bounded with `timeout 30`:
  **3/3** fresh unseeded leases admitted and caches reached **16 frames each**.
  First cache-enabled status arrived after **4.249 / 1.600 / 5.906 ms**;
  100 ms renewal, 4 s selection, cancel, 2 s gap. Evidence `lease-probe.json`.
- PC `T/abandon.py` SIGKILLs its helper during selecting and during tracking:
  helper exits **-9 / -9**, each lease ends with **ROI lease expired**, full
  graph/cache cleanup after **0.807 / 1.508 s** from observed helper death.
  Subsequent fresh locks pass with **150 / 145 selected frames** and detector
  return. Evidence `abandon-*-result.json`, `after-abandon-*` and status JSONL.
- PC `acceptance.py --blank --count 1 --gaps .5 --output T/blank-run1`:
  **refused**, `seed: seed lacks texture`, **0 selected frames**, detector
  returned, next fresh locks above succeed. No runtime `roi_error`.
- Camera idle A/B: four exclusive **60 s** windows, off/on/off/on, with 15 s
  restoration gaps. Final table and comparison are recorded below.

| Window | Frames | Steady fps | Infer+metadata p50 / p95 (ms) | Decode-to-input p50 / p95 (ms) |
| --- | --- | --- | --- | --- |
| off-1 | 3,541 | 60.189 | 11.559 / 13.298 | 2.552 / 3.600 |
| on-1 | 3,542 | 60.195 | 11.646 / 13.356 | 2.226 / 3.649 |
| off-2 | 3,540 | 60.172 | 11.582 / 13.326 | 2.659 / 3.642 |
| on-2 | 3,535 | 60.200 | 11.645 / 13.333 | 2.276 / 3.643 |

On-minus-off p50: **+0.087 / +0.063 ms**, both below **0.2 ms**.
Both on windows have empty ROI counters, no cache/probes/owner, capability
true and `roi_error: null`; all four runtime exits are **0**.
The installed receiver preflight rejects a restore while the trial command
wrapper still lists its child module. Camera restore start-jobs returned **1**
while shadow-controller/operator restores returned **0**; the same symptom
appeared on the managed SIM fallback. After all wrappers exited, an explicit
operator-agent Camera switch succeeded, followed by stable status verification.
Final operator status is **camera**, not busy, motors not live; Camera video,
shadow controller and operator are active. Bridge, serial, motor controller and
HIL target are inactive. No trial unit remains active.


### Tests, provenance and limits

- Final Windows full suite:
  `python -m pytest -q --capture=sys --continue-on-collection-errors --tb=short tests`
  -> **1,537 passed, 58 skipped, 9 failed, 8 collection errors**, 20 subtests,
  **179.32 s**. Untouched-base comparison of the same 14 known failing modules
  (round-6 list above) -> **65 passed, 6 skipped, 9 failed, 8 errors**, **4.19 s**.
  Failure/error names match **exactly**: two Windows path-casing assertions,
  five Pillow failures, missing trimesh, missing historical gimbal-fit report,
  four gi and four Pillow collection errors. Unavailable dependencies are not
  counted as passes. An earlier full run exposed a separated-alias regression
  and an over-specific test call-count assertion; both were fixed before this
  final full run and final native matrices.
- Focused Windows command (round-6 nine modules plus
  `tests/test_visual_lock_live_trial.py tests/test_roi_seed_review.py`) ->
  **417 passed, 2 skipped**, **41.52 s** before the final canonical-alias fix;
  the final full suite covers that fix. Final smaller boundary run including
  `tests/test_deepstream_accepted_osd.py` -> **214 passed, 2 skipped**, **5.04 s**.
  Windows skips are missing PyGObject/gi for the real queue/videorate checks.
- Jetson task-local `jetson_tests.py`, bounded with `timeout 90`, runs
  `bin/python3 -m pytest -q tests/test_deepstream_metadata_adapter.py tests/test_seamless_roi.py tests/test_deepstream_accepted_osd.py --tb=short`
  equivalently through `pytest.main`: **184 passed**, **8.37 s**, including the
  real non-GPU GStreamer queue insert/remove/reopen test. The first plain
  absolute-path invocation hit the host's unrelated installed `tests` package
  during collection; the task-local runner scopes repository fixture imports
  without installing packages or changing system config.
- Check-only runtime/default-off config, helper `--help`, changed-module
  compileall and `git diff --check` pass. All lab commands and outputs are in
  the local task evidence's `actions.log`; PC/Jetson `*-evidence.tar.gz` preserve
  the scripts, health, reports, guard statuses, runtime logs and status traces.
  Windows evidence is preserved under `artifacts/seamless-roi-lab-20261021/`
  (including base archive/tree and suite logs). Remote task directories remain.
  `code-hashes.json` confirms all six runtime/helper modules exactly match the
  Jetson-tested bytes. `guard-restore-cause.txt` records the installed preflight's
  explicit false owners (sudo/systemd-run/guard wrappers listing the child).
- These checks qualify the scoped software ROI workflow and idle cost, not
  detector accuracy, controller tuning, cross-host transit latency, physical
  interlocks, Pi controls or hardware. Hardware hold remains in force.

## 2026-10-21 - seamless ROI v7 verified and merged (default off)

- Codex round 7 (`ea30bc1`, first round with lab access) reported 30/30
  back-to-back locks, lease probe 3/3, idle on-off +0.09/+0.06 ms.
- Reviewer re-check (`IDCS-trials/seamless-roi-verify-20261021`, own build,
  7 min run, exit 0): lease probe 3/3; 8/8 locks ~1.5 s apart (300-311
  selected frames per 6 s, detector returned each time); helper killed
  mid-lock, next lock tracked; no `roi_error`, no metadata fault. PC suite
  1625 passed, same failure set as main.
- Merged `lab/seamless-roi-v7` into integrate-99/main. `deepstream.seamless_roi`
  stays false; enabling it live (and a Pi panel Select box test in Camera
  mode) is a separate, bench-attended step.

## 2026-10-21 - deploy-107 (seamless ROI, off) on all hosts

- deploy-106 (`b18fc78`) staged only: Jetson validation failed
  `test_roi_lock_survives_late_image_metadata_but_releases_stale_receiver`
  (fake status lacked `agent_state`; PC skips the module). Fixed in `c87e023`,
  tagged deploy-107; 106 never activated, tag left in place.
- Jetson activation: stale `NRestarts=1` on idcs-visual-lock-sim from SIM trials
  refused verify (reset-failed), then plan+validate+activate passed.
- Side effect: the first attempt's DeepStream restart made PC `idcs-camera-ui`
  exit on return-stream loss with success (no restart) -> agent mode "mixed",
  Pi HUD UNAVAILABLE. Reselecting Camera fixed it. Same class as the Pi display
  exiting on video loss (proposed fix: keep UI processes alive / restart on
  video loss).
- Live: Jetson 59 fps, Pi 30 fps CAMERA, overlay p95 3.7 ms, recording on.
  `seamless_roi` false; enabling + Pi panel Select box test awaits the bench.

## 2026-10-21 - seamless ROI deployment mismatch corrected; menu hardening pending

- Root cause: PC/Jetson used deploy-109 (`5c18912`) but Pi still used deploy-107.
  Its Source & scene -> Camera / Visual lock sent `set_mode(camera_visual_lock)`,
  explicitly stopping DeepStream and starting the dedicated VPI camera owner.
  Operator logs show 12:37:36 -> 12:37:46 and 12:49:03 -> 12:49:13 transitions.
  The intended Targets -> Select box callback instead sends ROI lease/seed
  messages to the existing process. This is a deployment/UI routing issue,
  not a requirement to restart DeepStream for resident ROI.
- Activated the existing Pi deploy-109 through the managed installer. Initial
  activation refused a changed display owner (nine earlier automatic restarts);
  refreshed plan with the existing exact deploy109 config policy, revalidated
  83 tests in 4.709s, then activation/initial verify passed. No unit overrides,
  immutable release files or validation gates were bypassed.
- Returned once through the operator agent from dedicated Camera / Visual lock
  to normal Camera; readiness passed. That genuine source/SSRC change caused
  Pi `return_input` to fail `not-linked (-1)` and automatically restart once.
  Its subsequent strict managed verify correctly fails on NRestarts=1. Counter
  was retained. General source-change receiver recovery remains a separate
  known issue; do not claim an entirely clean deployment verification.
- Normal camera PID155062 and Pi display PID120252 stayed unchanged across
  five explicit three-second camera ROI acquire/track/release cycles. All five
  were accepted and returned detector ownership; selected observations:
  128,133,134,134,134. Tracker session `3e4c669567ba44d680cfd86ce77ebfb6` and
  return SSRC385895955 stayed identical. Camera restarts0 / Pi restarts1 before
  and after the repeated tests. No extra GPU/video owner or motor stack ran.
- Used an explicit textured poster region from the existing Pi compositor
  screenshot: source pixels x16,y210,w96,h150 at1280x720. No learned detection,
  SIM truth, GPIO injection, reseed or weakened quality/timing gate was used.
  Final cycle: command-to-first-selected observation129.43ms,
  release-to-detector28.69ms, maximum local perception receive gap91.47ms.
  These are short local protocol measurements, not photon/transit latency,
  moving-target accuracy, continuous60Hz or physical panel qualification.
- Evidence retained on PC outside source:
  `deployments/seamless-roi-fix-20261021/idcs-seamless-camera-20261021-01.json`
  through `-05.json`; local copies under
  `C:/Users/Lab412/idcs-dev/seamless-roi-evidence-20261021/`.
- Additional source hardening removes both managed visual-lock modes from the
  operator menu, including Simulator's nested fallback, even with stale/false
  capability or a currently active fallback. Maintenance API modes remain.
  Added regression coverage for both normal sources, no-source-command ROI
  callbacks, fallback states and capability loss. Extended the bounded check
  helper with explicit camera/source-clock, session/SSRC stability and timing
  reporting; it refuses an already active ROI workflow.
- Tests: focused Windows suite163 passed /22 skipped (missing real gi/cairo);
  all23 candidate real-menu tests pass on Pi with actual dependencies. Windows
  default pytest Temp directory was inaccessible; a fresh explicit task-only
  basetemp worked. The first isolated Pi runner needed the release cwd for
  relative config paths; corrected runner then passed. `git diff --check` passes.
- Source/doc changes are in local branch `fix/seamless-roi-menu-20261021`,
  worktree `C:/Users/Lab412/idcs-dev/seamless-roi-fix`, based on `5c18912`.
  No commit/push or new-release activation performed without explicit approval.
  Thus the existing deploy-109 camera fix is live, but additional unconditional
  menu hardening is not deployed. All physical/HIL units remain inactive.

## 2026-10-21 - Pi display receiver recovery implemented and verified, not deployed

- User requested display recovery. Existing uncommitted menu/ROI changes were
  preserved on `fix/seamless-roi-menu-20261021` based on `5c18912`; no commit,
  push, immutable-release edit or permanent service-path change was made.
- Split ReturnVideo into a persistent display/fallback/overlay/sink pipeline
  and a replaceable receiver bin containing RTP/RTCP, parse, decode and recording.
  Receiver ERROR/EOS no longer quits the application. EOS is intercepted at
  the receiver output before it can terminate the fallback. Screen/output
  errors and whole-output EOS remain explicit fatal failures.
- Fault handling clears the exact-frame join, local video freshness and ROI
  workflow, sends release/cancel and requires a fresh explicit operator lock.
  Menus stay open. Generation guards reject old receiver callbacks. Graph edits
  occur on the GLib/UI thread; replacement requires old decoder NULL ack.
  Initial receiver startup is isolated from the black/output startup too.
- Automatic rebuilds back off250ms,500ms,1s,2s,4s, then fail visibly after five
  retries. Ten healthy seconds reset the burst. Persistent failure leaves the
  UI alive, with Operations -> Reconnect video. Recovery state, attempts,
  rebuilds, generation and last error are reported. Retired RTP observers
  disconnect their bus handlers. Native recording finalization is polled,
  preserving the existing2s bound without a UI-thread sleep.
- Native trial development exposed/fixed actual Gst details: automatic bin
  ghosting captured RTP's not-yet-linked dynamic sink; explicit decoded-source
  ghosting fixes this. Input-selector pad names are allocated by the element,
  so use the actual fallback peer plus a requested live pad, not assumed indices.
  The old10fps fallback stalled on video-to-black transition with the hardware
  converter; matched30fps fallback passed. Initial failed trials01-04 and their
  source hashes/logs remain preserved, not counted as passes.
- Controlled native trials05 and07 passed with normal exit0 on the Pi,
  GStreamer1.26.2, real v4l2h264dec/v4l2convert/Wayland,640x360@30 CPU-generated
  H264/RTP. Three senders used SSRC70001/70002/70003. Two sender changes
  reproduced the real not-linked(-1) error; both recovered, as did explicit EOS
  and a posted receiver error. Final07 kept PID121641/display pipeline unchanged
  through four receiver rebuilds,150 decoded and536 output frames,19.21s total.
  Maximum output gap559.45ms includes the test's intentional500ms freshness wait.
  Every1.2s sender pause still produced at least21 fallback/output frames.
  Sender-change timings3.50/3.51s include recording finalization AND waiting for
  30 decoded frames; they are not first-frame or photon latency measurements.
- All five recorded native05 MKV segments start at0.000s, contain30/31 H264
  frames, and decode completely with the CPU h264 decoder without errors.
  Native05 recording-validation.json records each file. Final07 repeats the
  recording-enabled scenario with the final runtime bytes.
- Full operator-display/live-camera trials06 and08 passed with normal exit0.
  Trial08 PID121725 remained the same across receiver ERROR at4s and EOS at12s;
  its open menu survived both, local ROI/video gates were invalidated, and the
  20s report showed two rebuilds,30.12fps and overlay p95 9.34ms. The Jetson camera
  source was never switched or restarted. Active-lock invalidation is separately
  contract-tested; full UI trials did not inject a physical switch or live ROI.
- Each bounded trial paused only the previously active display unit and
  restored deploy-109 in finally; final07/08 also held an explicit exclusive
  Pi trial lock. GPIO process PID120170
  stayed unchanged. Final display restore PID121777. Jetson/PC source services,
  physical/HIL units and GPIO authority were not modified. Trial source is
  `IDCS-trials/display-recovery-20261021-01`; evidence is under
  `deployments/display-recovery-20261021-01` through `-08` on Pi, with native07
  and full-app08 recording the final runtime SHA256 hashes.
- Validation: Windows focused recovery/ROI/identity/RTP/recording set274 passed,
  32 skipped (missing real gi/cairo),10.48s. Real Pi managed-style unittest
  modules86 passed,5.225s. Final native/full-UI repeats qualify actual Gst usage.
  All generated files are outside source. No detector-quality, transit-time,
  physical-switch, motor or long-endurance acceptance is claimed.
- Recovery and earlier menu hardening remain uncommitted/undeployed, pending
  explicit commit approval and a clean managed release. Running services were
  restored to deploy-109, so current production still has the old recovery
  behavior until that rollout. No test owner was left running.

## 2026-10-21 - deploy-110 authorized and prepared

- User authorized deployment, testing and commit of receiver recovery and the
  preceding seamless-ROI menu hardening. Release110 is based on `5c18912`.
  No version-controlled configuration or native/model source changed.
- Added the receiver lifecycle, native RTP and Pi recording regression modules
  to managed PC/Jetson validation so subsequent deployments retain these gates.
- Final precommit Windows gate, including deployment safeguards:321 passed,
  32 optional-platform skips,10.95s. Native/full-display qualification from the
  previous entry uses the same runtime source bytes. Rollout receipt follows
  after stage, platform validation, ordered activation and live acceptance.

## 2026-10-21 - deploy-110 installed and live acceptance passed

- Runtime commit `a79a9b956bdb440d9b174bae7078b04ed527d362`, tag `deploy-110`,
  installed on PC/Jetson/Pi from their immutable IDCS-releases/deploy-110 trees.
  Local source branch is `fix/seamless-roi-menu-20261021`; the isolated PC Linux
  deployment checkout is `IDCS-wt-deploy110` (release/deploy-110). Existing dirty
  worktrees and the separate integration checkout were not reset or advanced.
- Archived the redundant Jetson live ROI override under
  `deployments/seamless-roi-fix-20261021/deploy110-jetson-overlay-before.yaml`,
  then removed it through the config agent. Base ROI remains enabled. Plans
  report zero version-controlled config differences on all roles. Parser/model
  links retain existing targets, and copied native visual-lock libraries have
  identical SHA256 hashes to deploy-109; no native/model source changed.
- Managed validation: PC402 passed/6 skipped (8 subtests,11.10s), Jetson431
  passed(16.57s), Pi86 unittest cases passed(5.223s). An early Pi activation
  invocation was safely refused before its validation receipt existed; no
  units changed. After all platform receipts passed, activation was ordered
  Pi -> PC -> Jetson, with exact unit backups and initial per-host verification.
  Repeat verification and reference audit after live tests passed all hosts.
- Updating Jetson first exercised a real sender change against the newly
  installed Pi app: not-linked(-1) triggered receiver rebuild1, not process
  exit. Deployed Camera -> Simulator/Single -> Camera then exercised rebuilds
  2 and3. Pi display PID122322 remained identical throughout, NRestarts0;
  recording stayed requested and return video recovered to28.52fps in SIM and
  30.12fps in Camera. The managed GPIO owner stayed PID122299 during acceptance.
- Three explicit three-second camera ROI locks/releases passed with128,133,133
  selected observations and detector return each time. Camera PID168302,
  tracker session02108d6cd8234cdc9460a44d5313194e and SSRC1107174390 stayed
  unchanged; camera NRestarts0. Pi PID/rebuild count stayed122322/3 during ROI.
  Command-to-first-selected273.37ms cold,127.49/133.18ms warm; release-to-detector
  30.49/43.92/2.05ms. These are local protocol timings, not display/transit bounds.
- Final authoritative manifests: all software source paths deploy-110, all
  automatic restart counters0, no source overrides. Jetson video57.96fps,
  return29.40fps, roi_error null; Pi30.29fps, overlay p95 3.07ms, receiver live.
  Current source is normal Camera. Dedicated visual-lock/SIM services and all
  physical/HIL services are inactive; the camera controller remains shadow-only.
- Evidence/coordinator/scripts are retained outside source under PC
  `deployments/seamless-roi-fix-20261021/`; live report and exact ROI reports are
  in its `deploy110-live-acceptance/`. Managed plans, tests, rollback, verification
  and audits are under each host's `deployments/managed/deploy-110/<role>/`.
  No previous releases, trials, recordings, model/native stores or useful scripts
  were deleted. No trial owner remains active.
- Code release tag is unchanged by this documentation-only receipt commit.
  No GitHub push was requested/performed. This is software rollout/recovery/ROI
  acceptance, not physical-panel, hardware, detector-quality or endurance proof.

## 2026-10-21 - real menu / simulated panel ROI acceptance exposes two blockers

- User requested the missing end-to-end menu test. Ran the actual deployed Pi
  OperatorDisplay/menu/editor against normal live Camera and resident Jetson ROI,
  with private IPC PanelState fixtures. Three separate joystick-right pushes
  opened Menu -> Targets -> Select box; simulated Select/joystick resized and
  positioned the box. Fixtures never reached operational GPIO, lamp or control
  endpoints. GPIO owner PID122299 stayed unchanged; all physical/HIL units
  remained inactive. No runtime/config source edits, commit or push were made.
- **Acceptance failed.** Deployed ReturnIdentityReceiver sets MAXMSGSIZE1024,
  but measured idle ROI status is already1139 bytes after a fresh camera start,
  growing to1649 bytes after prior trials. Side-by-side Pi subscribers received
  38/39 status messages in4s without that cap and zero with it. ZeroMQ disconnects
  the limited receiver; capability arrives through the separate operator agent,
  while ROI source freshness/exact displayed-frame identity remain unavailable.
  Prior direct Jetson protocol tests did not exercise this Pi receiver boundary.
- To isolate further faults, only the external harness substituted a bounded
  16384-byte receive limit in memory. Production release files stayed unchanged.
  One full camera cycle passed: real menu/editor, readable-OFF -> ON, accepted
  exact displayed source frame84982,3s hold with91 selected observations, then
  OFF release -> detector. Panel-ON-to-selected199.79ms; OFF-to-detector98.52ms
  in the Pi local observation clock, not qualified photon/transit latency.
  Initial ON and unreadable OFF followed by ON both correctly refused to lock.
- **Second blocker:** closing/reopening around lease retirement then yielded no
  exact frame. Subsequent trial also timed out before seeding. Jetson journal
  shows a queued successor lease admitted and cancelled in the same second;
  status remained roi_window_retiring=true with no probes/owner/transition and
  generation4. Inspection identifies an inactive-window cancellation path:
  _cancel sets retirement_requested, begin_frame refuses further work while it
  is set, but inactive lifecycle has no owner to deactivate/finish_window.
  This explains the persistent latch; targeted regression is still required.
- Proposed fixes, not implemented by this test-only request: share a bounded
  sideband message-size contract covering status/results/headers, test expanded
  statuses through the actual receiver socket, and explicitly finish cancellation
  of a never-activated window without bypassing active-owner drain/generation
  protection. Add queued select/cancel/reopen integration cases, then repeat all
  menu cycles. Do not treat the test-only cap or first successful cycle as rollout
  acceptance. Physical joystick/switch and moving-target acceptance remain open.
- Attempts01/02 preserved harness inproc-connect and then actual receive-cap
  failures;03 passed the first cycle and exposed retirement;04 confirmed the
  blocked successor. All Pi guards restored deploy-110 display. A single scoped
  camera-video restart cleared the test-induced retirement latch; camera PID
  changed168302 ->173414, shadow controller PID168317 stayed unchanged. Pi
  PID123277 recovered the resulting sender change in-process to29.98fps,
  receiver live/rebuild1, zero automatic service restarts. The deployed receive
  cap remains a known blocker; source is normal Camera, no test owner remains.
- Evidence and reusable harnesses: local
  `C:/Users/Lab412/idcs-dev/seamless-roi-evidence-20261021/`; PC
  `deployments/seamless-roi-fix-20261021/menu-e2e-results/`; original Pi artifacts
  `deployments/menu-roi-e2e-20261021-{01,02,03,04}/`. Preserved report/frame IDs,
  menu event traces, source/module/harness hashes, owner snapshots and journals.

## 2026-10-21 - resident YOLO/ROI mode rework and camera candidate checks

- In progress on fix/seamless-roi-menu-20261021 after cbfdb87; changes remain
  uncommitted and production remains deploy-110. User requested an explicit
  processing-mode switch, not another Camera/Visual-lock service transaction.
- Fixed both preceding blockers: shared 16384-byte sideband receive contract,
  tested through actual ZeroMQ; ownerless cancelled legacy windows now finish
  retirement before admitting a successor. Added revision/session-scoped mode
  commands and a Pi Targets -> Tracking mode: YOLO/ROI action. Release/loss does
  not silently switch back to YOLO. Source switching remains a separate operation.
- Candidate uses one fixed video queue/encoder path and a dedicated bounded ROI
  worker instead of dynamic graph relinking or waiting for VPI in video callbacks.
  Independent pre-OSD NVMM copies use a persistent CUDA-registered pool; no
  ordinary CPU readback. Exact source identities/times survive async completion;
  stale results cannot be relabelled as current observations. New explicit locks
  get distinct internal track IDs; arbitrary visual ROIs do not invent depth for
  parallax. V2 wire schema remains unchanged for existing consumers.
- Isolated camera snapshot03 and actual Pi menu/private panel fixtures passed
  three six-second locks/releases, then explicit return to YOLO. Tracker session
  b2f7da5024ad46b198c52d1120c159e2 stayed fixed. Pi return video measured YOLO
  30.025fps versus ROI 30.003fps; overall largest frame gap59.55ms, no receiver
  rebuild. ON-to-selected168.32/167.51/169.22ms; OFF-to-idle99.00/66.75/70.28ms,
  all Pi-local protocol timings, not photon or transit bounds. Initial ON and
  unreadable-OFF/ON refused to seed. GPIO PID122299 was untouched.
- Test-only verification of64 pooled CUDA frames against fresh EGL pixels had
  zero mismatches and zero maximum pixel difference. Native log recorded1089
  VPI updates across18s of lock holds. Worker processed1843/dropped19; test
  readbacks/warmup were enabled, so no production steady-state drop claim yet.
  Native main library SHA c0b7c0c8e5e68e13f79644c2bca77658f428ea7618518f62eff1ce3917c7e023;
  pool helper SHA8f59203925a21edcdf8fa96d2cc8b2cd0141c371077e14927c985a3857a84138.
- Snapshot03 full contracts initially failed one legacy dynamic-Gst reopen
  test; isolated rerun passed, then full rerun288 passed. This remains a recorded
  flaky legacy test, not a clean first-run result. Snapshot04 controller/panel/
  menu/agent contracts191 passed. Latest candidate retains explicit ROI locks
  when Manual is turned off, then uses the normal readable automatic gate;
  legacy isolated ROI auto suppression remains fail-closed.
- Added bounded moving-texture -> native VPI -> real observation/PID core ->
  stable software mount qualification tool. Initial trial failed before native
  tracking because standalone EGL was initialized on the producer rather than
  native owner; second exposed a test closure scoping error. Both harness faults
  are corrected, results retained, moving/control acceptance still pending.
- Trials stop only recorded software owners and restore them; physical/HIL/
  bridge/serial remain inactive. Snapshot03 guard restored deploy-110 sources,
  Pi display and unchanged GPIO owner. Source snapshots02/03/04 and evidence
  remain under IDCS-trials/resident-roi-20261021-* and deployments/resident-roi-
  runtime-20261021-03, menu-roi-e2e-20261021-07, resident-control-20261021-*.
  Coordinator/scripts/source archives also retained outside the local repo in
  idcs-dev/seamless-roi-evidence-20261021 and PC deployments/seamless-roi-fix-20261021.
- Not yet accepted: moving/controller loop, full software-SIM integration,
  no-readback cadence/endurance, physical panel/actuation. No commit, push or
  managed deployment performed in this rework; no historical assets removed.

### Resident follow-up: moving native loop and software-video integration

- Corrected standalone GPU verifier initializes/terminates EGL on the native
  owner, not the producer. resident-control-20261021-03 passed1080 frames at
  60.05Hz,1058 selected samples, zero queue drops and zero safety-zero failures.
  Decisions:851 tracking,120 safety_hold,60 manual,20 target_invalid and29
  capture_age_out_of_bounds during acquisition/startup. Localization p95
  0.610px; bearing-error p95 45.77mrad. Stable50ms software rate plant, PID2.5,
  no hardware gains, no LAN clock assumption. Previous harness failures retained.
- Snapshot04 camera / Pi menu attempt08 retained YOLO29.94/ROI29.99fps with
  no receiver rebuild, but first lock correctly failed seed texture. Saved crop
 96x151, extrema15..50, zero feature corners; earlier poster fixture is no longer
  usable in the dark. Camera acceptance of Manual-OFF retention remains pending.
  No quality gate relaxed; all guards restored deploy-110 and GPIO stayed122299.
- Further inspection corrected an initial diagnosis: existing SimRoiSelection
  already relays locked Jetson ROI on5574; Single SIM was not wholly disconnected
  from ROI. Candidate now feeds Single directly from Jetson V2, as Swarm already
  does, so normal YOLO and ROI use one visual input rather than the truth/ROI
  relay. Unit/config contracts cover all three software SIM controller templates;
  loopback intent endpoint and panel gates stay unchanged. No hardware tuning
  changes were made.
- Added deterministic textured software-video source (resident_sim_fixture)
  for transport/control checks without YOLO-recognition dependence. Snapshot05
  archive SHA4d71c529ec3fb4565bea62e2edb33e09791ea4b656b9ef390316e2bd1b6320ee.
  Pi menu tests passed including new explicit mode routing and Manual-OFF lock
  retention;159 local resident/managed/live-config tests passed;59 focused
  receiver/panel/protocol tests passed. Default Windows pytest temp directory
  was inaccessible; rerun used a new external evidence subdirectory, no deletion.
- Full PC fixture -> Jetson -> Pi menu / PC software-controller trial is in
  progress. Preserved harness faults: null optional controller values rejected
  by config schema before trial; stale fake operator receipt blocked UI readiness;
  seed fixture initially read5574, which is deliberately the ROI relay in ROI
  mode, not raw truth. Harness now uses source-frame catalog from5572 for seed
  geometry only, with matching frame ID and canonical source millisecond stamp;
  the actual seed still names the exact displayed nanosecond identity. No ongoing
  truth boxes enter the visual tracker or controller. All failed attempts restored
  prior software owners and kept physical/HIL services inactive.

### Resident acceptance receipt (software only, no managed rollout)

- Full software-video trial passed on snapshot05: PC60fps verified RTP fixture
  -> Jetson resident YOLO/ROI -> Pi actual menu/receiver -> PC real video
  controller -> stable50ms software mount. Pi private panel fixture alone
  supplied test authority, forwarded only to PC loopback; no GPIO publication,
  physical pose, bridge, serial or motor activation. One explicit seed per lock
  used source catalog geometry; ongoing VPI and control used only image-derived
  Jetson V2 observations. The PC truth/seed evidence relay was test-only and closed.
- Pi attempt12: three locks,252/180/180 selected observations across8.3/6/6s
  holds. First lock survived Manual OFF for2s, drove the software camera under
  ordinary automatic gate, then survived Manual ON. OFF released; explicit
  menu YOLO restored detection. Session986343ba1f354211a33da256a77eca05 stayed
  unchanged. ON-to-selected334.63/200.09/202.56ms (cold acquisition is finite,
  video does not wait); OFF-to-idle96.68/97.88/65.50ms, local Pi timings only.
- Return YOLO29.992fps versus ROI29.999fps,1440 decoded intervals, overall
  p95gap42.30ms/max60.96ms. No receiver rebuild, lost or late RTP packets.
  Native final report:3157 frames,2022 worker inputs processed, zero drops,
  zero pending, no worker/ROI error;1265 VPI updates. Tracking native p95
  17.45ms; worker completion lag p95 85.94ms, median17.55ms (bounded transient
  backlog, not a negligible-latency claim). Selected-frame exact truth scoring:
  1262 frames, localization p95 0.031px, VPI p95 17.50ms. Texture fixture is not
  real-drone detector-quality evidence.
- Real PC runtime:3104 ticks,101 automatic tracking ticks during the2s Auto
  interval, all on verified visual-class source observations; zero nonzero
  commands when test authority was disallowed. Median pointing-error magnitude
  first20 tracking ticks115.35mrad ->last20 8.61mrad; pose0/0 ->0.0904/0.0387rad.
  Tracking capture-age upper-end p95 78.96ms. This verifies actual networked
  perception/control plumbing, not physical gains or operational latency bounds.
- Final snapshot05 Jetson contracts346 passed(19.02s); Windows322 passed,
  24 optional platform skips(13.10s). Final fault-only teardown hardening ensures
  worker errors cannot skip pool/socket/probe cleanup; focused10 tests passed.
- Restoration audit caught a guard omission: SIM conflicts with camera.target,
  indirectly stopping its shadow controller; restoring source services alone
  left the target/shadow inactive. Preserved initial receipt, checked effective
  controller mode shadow/check-only/no motor authority, then restored ordinary
  camera.target. CameraPID214692, shadow216323, recorder167323 active on deploy-110;
  source/agent have no overrides, automatic restart counters0. GPIO122299 stayed
  unchanged; Pi deploy-110 display restoredPID126074. Physical/HIL units inactive.
  Guard inventory now includes camera.target, shadow controller and recorder;
  source overrides remain scoped to source/agent units only.
- Evidence consolidated on PC under deployments/seamless-roi-fix-20261021/
  resident-final-evidence-02 (summary, complete native shutdown journal/reports,
  three native acquisition sessions, Pi reports/events, controller trace and
  fixture truth, owner receipts and explicit restoration correction). Original
  artifacts remain in resident-roi-runtime-20261021-08, resident-sim-pc-20261021-05
  and menu-roi-e2e-20261021-12. Earlier harness failures and dark-camera refusal
  retained; no assets deleted. The first evidence copy hit socket-file warnings
  and a root-owned evidence-directory permission error; corrected collector
  copies regular Pi files and uses scoped sudo for retained native evidence only.
- Software candidate is ready for release review, not installed. Outstanding
  authority: explicit commit/deployment approval. Live camera repeat needs a
  visible textured target; physical panel/motor acceptance remains behind the
  wiring/bench hold. No restart-free physical motion or endurance claim is made.

### Final cleanup/qualification follow-up

- Snapshot06 first full Jetson run reproduced the previously recorded legacy
  dynamic-queue stall (346 passed/1 failed), despite resident live acceptance.
  Did not hide it by declaring a rerun clean. Added failure-only Gst pad/task/
  last-flow/bus diagnostics. Diagnostic-only fresh-process repetitions passed3x
  347, so the intermittent cause is not proven solely by those measurements.
- Hardened legacy removal ordering: after upstream idle blocks, unlink the queue
  inlet before setting NULL, leaving the outlet connected until the task stops.
  This isolates upstream from queue-deactivation flushing without allowing a
  live task to push into a disconnected return tail. A deterministic order
  regression asserts both conditions. Resident switching never calls this
  dynamic removal routine; its fixed queue remains in place.
- Windows legacy/resident contracts153 passed/1 optional Gst skip. Five fresh
  Jetson full-suite diagnostic runs with the exact candidate removal method
  applied in memory each passed347; logs and method source preserved under
  PC deployments/seamless-roi-fix-20261021/legacy-queue-diagnostics-02.
  This is repeated bounded evidence, not a proof that all scheduling races
  or every legacy teardown fault have been eliminated.
- An earlier ad-hoc repeated diagnostic runner omitted the repository cwd and
  produced unrelated config-path/fixture errors; its results are not source
  regressions or acceptance. Corrected to fresh processes with explicit cwd.
- Final status-only restoration check passed: ordinary Camera, busy null,
  motors_live false; Pi30.16fps/live/no receiver restart; one Jetson perception
  owner214692 and shadow controller216323, one Pi display126074, no PC trial
  source/controller. deploy-110 remains installed. Receipt is
  resident-final-evidence-02/post-restoration-check.json. Commit/deployment
  approval remains outstanding; no commit/push/managed activation performed.

- Final ordinary (no in-memory code substitution) snapshot07 Jetson validation:
  **348 passed in19.38s**; Pi menu unittest validation also exited0. Final source
  archive before this documentation-only receipt has SHA256
  4a0cfaa2cba8f3246179b89df80b49013670ca03f45c23871e70267a211fc7ef,
  staged as IDCS-trials/resident-roi-20261021-07 on PC/Jetson/Pi, not activated.
  It includes the cleanup/order regressions and reused verified native binaries
  on Jetson. Local evidence copies include the successful video/control summary,
  final restoration check and five diagnostic suite logs. Code remains dirty/
  uncommitted for review; the only difference from that archive is this receipt.

## 2026-10-21 - deploy-111 software rollout approved, preparing commit

- User replied proceed to the explicit commit/software-deploy question. Scope:
  resident YOLO/ROI mode rework, validated fixes/tests/tools/docs, PC/Jetson/Pi
  managed deployment. No GitHub push or physical motor/HIL activation authorized.
- Preflight restored Camera, busy null, motors_live false, Pi30.02fps; one
  Jetson perception owner214692 and shadow216323, one Pi display126074, no PC
  trial source/controller. Production remains deploy-110 at preparation time.
- No version-controlled YAML changes. Reviewed functional unit difference:
  Single SIM now directly subscribes Jetson visual perception via detections.yaml;
  loopback SIM intents and panel policy unchanged. Main VPI library unchanged;
  the validated surface helper has additive NVMM pool-copy/CUDA-view functions.
  Reuse only binaries whose native source matches this release and retain hashes.
- Expanded managed validation with resident-mode, receive-boundary, exact seed,
  legacy lifecycle, observation and PID-core tests, in addition to existing
  UI/operator/deployment checks. Validate all hosts before activating Pi -> PC
  -> Jetson; preserve rollback plans and never start inactive physical services.

## 2026-10-21 - deploy-111 committed, deployed and software acceptance passed

- Committed resident YOLO/ROI rework as d0e95029bc29ffbe239339a41ee9d7ae41a8af01,
  tag deploy-111, with Codex author/committer/trailer. Incremental Git bundle
  transferred to PC shared IDCS repository and clean detached IDCS-wt-deploy111;
  immutable IDCS-releases/deploy-111 staged on PC/Jetson/Pi. No user branch reset,
  useful-source deletion, GitHub push or physical-motion authorization inferred.
- Native asset safeguard initially refused tracker.cpp byte mismatch. Independent
  diff --strip-trailing-cr across the complete native directory found no source
  differences: Windows trial archive was CRLF, Git release LF. External rollout
  helper now compares normalized line endings, not arbitrary whitespace/code.
  Validated snapshot07 binaries copied unchanged: main VPI SHA256
  c0b7c0c8e5e68e13f79644c2bca77658f428ea7618518f62eff1ce3917c7e023;
  additive pool/CUDA helper SHA256
  8f59203925a21edcdf8fa96d2cc8b2cd0141c371077e14927c985a3857a84138.
  Model/parser links retain their existing bytes. Asset receipts preserved.
- Managed plans: zero YAML differences. Reviewed unit change is Single SIM's
  direct visual V2 input; no physical tuning changes. Managed validation passed
  PC592/6 optional skips/8 subtests, Jetson623, Pi88 unittest cases. Existing
  resident/native candidate evidence remains in resident-final-evidence-02.
  Activated Pi -> PC -> Jetson with rollback backups, then managed verification
  and audit passed all roles. No unmanaged overrides or automatic restarts.
- Deployed real Pi menu/input-path canary (menu-roi-e2e-20261021-15) passed
  three YOLO -> ROI -> YOLO cycles. Harness replaces only the scenario and
  private panel/control/lamp endpoints; no production module/receiver overrides,
  no physical GPIO publication, no motor APIs, no additional GPU/video owner.
  Sole display guard restores installed display in finally; GPIO PID126878
  unchanged during trial. Camera229302 and shadow229317 unchanged, same
  source session4a82d20b622f4313a250b420e436786d. Return YOLO29.976fps,
  ROI29.974fps, maximum decoded gap57.95ms, receiver rebuilds0; lost/late RTP0,
  one jitter-buffer latency drop. Idle ROI correctly has no invented selection;
  explicit YOLO restores detector ownership. This is switching/video acceptance,
  not a new camera texture lock, physical panel or photon-latency measurement.
  Raw harness acknowledgement wait times start after joystick pulse completion,
  so they must not be represented as full command-to-applied latency.
- Ordinary software source acceptance passed Camera -> Simulator/Single ->
  Camera via real operator/mode agents. Pi display127141/NRestarts0 unchanged
  across source changes; receiver-only rebuilds0 ->1 ->2 handled sender changes.
  SIM YOLO and idle ROI30.30fps; same perception231935/session
  c861a790899c4a648fe18d110438c9d9 throughout processing-mode changes, with
  no added receiver rebuild. Returned Camera29.68fps. Both source transactions
  succeeded; normal camera target/shadow restored, no trial units/overrides left.
- Current Camera/YOLO is software/shadow only; physical controller, bridge,
  serial and HIL target remain inactive. Bounded native moving-texture tracking
  and actual PC software-control evidence from snapshot05 still applies to the
  committed implementation: no claim of hardware tuning, real-drone quality,
  negligible worker delay or endurance acceptance. The actual camera's previous
  dark zero-corner seed refusal remains a scene prerequisite, not concealed.
- Receipts/scripts on PC: deployments/seamless-roi-fix-20261021/deploy111-*,
  including deploy111-menu-acceptance/report.json and
  deploy111-source-acceptance/report.json. Managed host receipts and authoritative
  current manifests under deployments/managed. Local copies retained outside
  repo in idcs-dev/seamless-roi-evidence-20261021. Documentation receipt updates
  current launch/menu/controller/config-stack guidance; legacy dynamic graph and
  check_seamless_roi's release-to-detector contract explicitly marked historical.
  Documentation-only follow-up does not move the deployed code tag.

## 2026-10-21 - comprehensive resident runtime migration plan

- User requested a plan covering all features/services after reviewing the26
  services and restart-based source groups. Added
  docs/resident_runtime_migration_plan.md and a short task-reference link.
  Planning only: no application code, unit/config, service state, deployment,
  hardware, commit or push changes in this task.
- Proposed final ownership: Jetson6, PC5, Pi2 (13 roles). Jetson controller and
  fake/real/disabled bridge share the operational control path; PC renders
  selected-backend pose without a second plant/controller. Stable fake motion
  has independent SIM settings, not bench gains/F6 quantization.
- Maps all26 current services and6 groups; inventories operator/menu/panel,
  YOLO/NvDCF/ROI, planner/ranging/parallax, PID/FF, timing, HUD, score/error,
  recording, live config, serial, recovery and useful maintenance tools.
  Explicit source/processing/backend/profile generations, real stop/queue
  fencing, alignment, panel authority and requested/applied receipts prevent
  process residency being mistaken for safe motion authority.
- Added ordered P0-P7 work packages, file/module boundaries, software acceptance
  matrix, warm-source resource/plugin-reset investigation gates, bounded and
  endurance tests, unit-inventory migration and rollback/removal policy. Real
  activation remains behind the existing wiring/RS485/bench gate. No automatic
  real/fake fallback, homing, synthetic physical availability or truth-driven
  operational control is permitted by the proposed design.
- Planning identifies two important migration traps: moving SIM control onto
  Jetson adds a pose-network timing leg; old serial queues also need epoch
  fencing, not only the bridge. No unmeasured zero-latency or universal seamless
  cold model/source startup claim. Existing deployed YOLO/ROI mode switch is
  preserved rather than rebuilt.
- Plan checks: PC14 and Jetson10 service mappings exactly match installed-unit
  templates; Pi2 are retained explicitly. Relative document links resolve and
  git diff --check passes. Software tests were not rerun for this documentation-
  only change; proposed acceptance thresholds and phases are not test results.

## 2026-10-21 - merge resident ROI branch into local main

- User requested current-branch merge first, then proceed. Committed the
  comprehensive migration plan/reference/journal changes as f6c727b with Codex
  author/committer/trailer. Local main was stale but had no independent commits;
  fast-forwarded it to fix/seamless-roi-menu-20261021 at f6c727b without conflicts.
  This includes the already validated deploy-110/deploy-111 work and receipts.
- origin/main was confirmed at5c18912; only five commits are new relative to
  that shared remote. Its push URL is the internal39.68 SSH repository, not
  GitHub. The root server checkout is clean on live-config; main is not checked
  out by any listed server worktree. Existing branches/worktrees were preserved.
- Remote sync command was rejected before execution by approval review as a
  potentially unauthorized default-branch/GitHub push. No push occurred and no
  workaround was used. Shared remote main remains5c18912 pending explicit
  approval; GitHub and deployed immutable deploy-111 sources are unchanged.
- Local merge has identical code to the validated branch; documentation links,
  complete service mappings and diff whitespace checks passed. No GPU/hardware
  tests or service restarts were needed/performed for the fast-forward and plan.
