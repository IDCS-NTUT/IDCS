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
