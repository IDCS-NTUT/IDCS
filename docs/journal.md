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
