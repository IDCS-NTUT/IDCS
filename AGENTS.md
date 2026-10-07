# 📑 AGENTS.md

## Current state and handover (updated 2026-10-07) — read first

This section is current. Everything under "Overview" below is older and is
kept for history; where they disagree, this section and the docs it names win.

### Standing rules

- **No motor or HIL commands** (bridge, serial service, `home_axes`,
  `motors_off`, step or sweep tools, `idcs-hil.target`) until the user says the
  panel and motor wiring is repaired and every motor answers on RS485. The pitch
  motor (address 2) stopped answering on 2026-09-29 after the wiring was
  disturbed. Camera, simulation and display work is fine.
- Hardware work stops at the bench: the user runs motor tests in person.
- Never send the user's email address to any service.
- Commits end with the co-author line the user's tooling asks for. Commit or
  push only when asked.

### Hosts and deploying

2026-10-07 **release76 deployed and verified**: code commit `22f9673`, tag
`deploy-76`, clean detached worktrees on PC/Jetson/Pi. PC viewer/HUD relay and
Pi display now use `IDCS-releases/deploy-76`; Pi has the mode-coloured reticle
and tracking-mode label. PC viewer restart recovered ONLY its software SIM
session (streamer/controller PIDs changed as expected, mode-agent unchanged).
Jetson video stays on nested-box05/PID1272298, with committed pipeline/adapter
text identical after ONLY CRLF normalization; byte hashes retained separately.
Release76 opt-in ROI OSD/tools are installed but no ROI workflow was activated.
PC194/Jetson166/Pi19 focused tests passed; Pi~30.23fps/build p95~8.72ms;
12.20s actual PC return survey706 single-track updates/zero duplicates.
Pi GPIO, Jetson video/operator/recorder owners unchanged; physical/HIL inactive.
Evidence/installer/backups are in all hosts' `deployments/release-76-20261007/`.
Rollback PC/Pi: restore ONLY backed-up viewer/relay units, daemon-reload/restart
previously active displays; recover `idcs-sim.target` on PC, never HIL/motors.
Keep release76 and nested-box05 while referenced. The pending/uncommitted
paragraphs below are historical and superseded by this completed receipt.

2026-10-07 **active nested-box fix (uncommitted source snapshot)**: ONLY
Jetson `idcs-deepstream-video.service` now uses
`IDCS-trials/nested-boxes-20261007-05`, PID1272298/zero automatic restarts,
via `/etc/systemd/system/idcs-deepstream-video.service.d/nested-boxes-20261007.conf`.
GPU OSD draws accepted V2 tracks once, not raw native/shadow duplicates.
Class-aware nested/IoU dedup keeps detector-backed/high-confidence geometry
and stable older IDs/aliases; detector-only verification remains available.
Final45s live2679 updates/59.53Hz/zero duplicate-track frames; subsequent PC
12s survey711 updates also zero, with one-box screenshots. Return~29.2fps.
PC source/controller/UI, Pi display/GPIO, Jetson operator/recorder owners were
unchanged; all physical/HIL units stayed inactive. No ROI console mode activated.
This supersedes ONLY the Jetson production-video source/PID claims below.
Stage/replay/native-metadata/tests/activation and rollback evidence is in
Jetson `deployments/nested-boxes-20261007-01/`; PC before/intermediate/final
captures in `deployments/pc-nested-boxes*-20261007/`. Read newest journal before
rollout or cleanup; keep the snapshot while the drop-in references it. Pending
reticle source below is still NOT deployed. No commit/tag/push was requested.

2026-10-07 **pending reticle update (not committed/deployed)**: shared
`common.hud_geometry.tracking_reticle_style` gives PC/Pi green YOLO, cyan ROI,
amber acquiring/uncertain, grey idle/lost/stale with textual mode/state.
SIM substitute truth is labelled SIM, not YOLO. ROI identity comes from
class `visual`, not its NvDCF backend; empty V2 cannot identify an ROI lifecycle.
Isolated ROI trial GPU OSD has explicit lifecycle information and the same
colours. PC display progress/250ms receive watchdog removes stale selection/aim
geometry; Pi preserves existing source/session freshness guards. No physical
authority or perception schema changed. ROI does not inherit detector depth
parallax. Pi isolated Cairo tests/benchmark/previews are in
`IDCS-trials/reticle-20261007-02/results`; live Pi display/GPIO PIDs unchanged.
The active release below remains authoritative until a requested deployment.

2026-10-07 **current four-group release**: `24c824f`, tag `deploy-75`, is
installed as clean detached worktrees at
`~/Desktop/project/IDCS-releases/deploy-75` on PC/Jetson/Pi. ROI reliability/
NvDCF handoff, capture/replay/scoring tools and scenarios are available opt-in;
Jetson ABI4 libraries are built at
`deployments/release-75-20261007/native-build/`. No console ROI integration or
new ROI video owner was activated. Shared PC HUD is active through SIM UI;
all PC viewer unit paths and HUD relay use the release. Pi display uses it;
GPIO/manual remains target01. Existing `IDCS-runtime` checkouts were NOT moved.
Pi holds ~30fps/build p95~8.4ms. Software source/controller retain target01
code/config but were restarted by the viewer's Requires dependency: PC SIM
session REQUIRES its UI, so UI restart propagates through session/target to
software workers. First strict-PID activation rolled back; final retry explicitly
recovered `idcs-sim.target`. Do not promise a viewer-only restart preserves
software worker PIDs. PC mode authority, Pi GPIO and Jetson video/operator/
recorder owners were unchanged; all physical/HIL units stayed inactive.
Final stage/preflight/activation/verification, unit backups and installer are
in each host's `deployments/release-75-20261007/` (installer also in parent
deployments). Read the latest journal receipt; do not bulk-install units over
tuned motor/video paths. The following deployment paragraphs are history.

| Host | Role | Units |
|---|---|---|
| PC 192.168.0.1 | simulator, streamers, PC UI | user units (`systemctl --user`) |
| Jetson 192.168.0.5 | DeepStream, controller, bridge, operator agent, recorder | system units (`sudo systemctl`; passwordless sudo) |
| Pi 192.168.0.3 | safety panel (`idcs-manual`), panel screen (`idcs-operator-display`) | user units; no sudo |

- SSH key: `~/.ssh/id_ed25519_lan`.
- Deploy a tagged commit: `git tag deploy-N && scripts/deploy.sh deploy-N pc,jetson,pi`
  (comma-separated hosts). It bundles the local repo, checks out
  `~/Desktop/project/IDCS-runtime` and installs units; nothing restarts.
- Tagged runtime checkouts remain Pi/Jetson `deploy-74`, PC `deploy-62`.
  **Latest software activation:** target-picker source is
  `~/Desktop/project/IDCS-trials/target-20261006-01` on all three hosts.
  Pi GPIO/display, Jetson operator/video/shadow-camera-controller and PC
  mode-agent/UI/normal+swarm simulator units now reference it. Camera mode and
  Planner selection were restored after the simulation tests. Actual HIL,
  bridge, serial and live physical controller remain inactive; the existing
  Jetson recorder was not restarted. Unit backups, final installed units,
  provenance and reports are in `.safety-deployment/`; installer rollback only
  restores units/reloads, not services. See latest journal entry before rollout.
  Snapshot exported from `57a54ef` + picker work before its authorized commit,
  with the retained startup patch. Snapshot paths are not tagged checkouts.
  The following safety/menu deployment paragraphs are historical, superseded
  by this entry. Keep their snapshots for rollback; never infer activation
  solely from runtime tags.
  New safety-activity policy was deployed/tested over SSH on 2026-10-06 as
  snapshot `~/Desktop/project/IDCS-trials/safety-20261006-01` (exported before commit):
  active Pi GPIO/display, Jetson operator agent and PC mode agent use it;
  normal/swarm simulator streamer units also point there (currently inactive).
  Existing video/UI/simulator-controller units retain menu trial 02. Camera
  mode was restored after live simulator and relay-loss tests. No motor unit
  started. Both snapshots must remain while units reference them.
  Scoped unit backups, installer and reports are in the new snapshot's
  `.safety-deployment/`; see journal for exact activation/provenance/rollback.
- Camera/simulator menu work is committed with the indicator release, but
  active Jetson video and PC video/simulator-controller services still use the tested
  snapshot `~/Desktop/project/IDCS-trials/menu-20261006-02`. PC runtime remains
  `deploy-62`; it was deliberately not migrated for this Pi indicator update.
  Tags alone therefore do not identify every active process. Camera → Simulator →
  Camera passed twice on 2026-10-06; camera mode was preserved during rollout.
  Trial backups/reports remain in `.deployment/`; Pi indicator unit backups,
  read-only verifier and live report are in
  `~/Desktop/project/deployments/indicators-20261006/`.
  Do not delete trial 02 while PC/Jetson services still use it.
  Check the runtime tag and running service versions before assuming activation.
- Operation, modes and checks: `docs/launch_procedure.md`. Configuration
  layout: `docs/configuration_architecture.md`, `configs/README.md`.
- History, results and the reasons behind every setting:
  `docs/deepstream_migration_journal.md` (newest entries at the end). Add an
  entry for each result.

### Architecture now

```
camera (IMX219 on Jetson) or PC streamer (sim / HIL video)
  -> jetson.deepstream.runtime: YOLO26s (grey-letterbox TensorRT engine, threshold 0.15)
     -> NvDCF -> target selection (swarm planner + operator lock / target classes)
     -> PerceptionSnapshotV2 on net.zmq_perception_v2 ; return video to PC UI and Pi screen
  -> jetson.control.video_runtime (shadow, or live in HIL): PID + feedforward, 50 Hz
     -> ControlIntent -> jetson.gimbal_bridge -> RS485 (MKS SERVO42, F6 speed mode)
Pi rpi.runtime_control: panel GPIO + joystick -> ManualControlState (PUSH to controller)
Pi rpi.operator_display: return video + status bar + menu on the panel HDMI screen
     -> OperatorCommand -> jetson.operator_agent -> OperatorSelection -> DeepStream selector
```

- Perception contract: `common/perception.py`. Control contract:
  `common/schemas.py`. Operator messages: `common/operator_commands.py`.
- Controller: `docs/controller_architecture.md`; tuning:
  `docs/tuning_procedure.md`.
- Panel screen, panel controls and the command path: `docs/operator_display.md`.
- Automatic console joystick: left opens a preview-only target picker,
  up/down browse, right confirms, left cancels. Fixed **Use planner** explicitly
  returns to Planner; closed-right still opens the ordinary menu. Operator
  selection is separate from GPIO5 manual movement. Locks are source/session
  scoped: camera NvDCF IDs and simulator ground-truth IDs are not interchangeable.
  Target loss clears the lock but retains Operator mode with no fallback;
  source/authority restarts cannot rebind old IDs. All physical GPIO roles and
  engagement permissions are unchanged. See latest SSH trial evidence below.
  Deployed 2026-10-07 menu-only polarity correction reverses vertical input in
  `PanelNavigator`: forward browses up, backward browses down. Manual pitch
  remains forward/down and backward/up; ADC, motion configuration, keyboard
  arrows and left/right are unchanged. Only Pi operator-display now uses
  `IDCS-trials/menu-pitch-20261007-02`; Pi GPIO/manual and PC/Jetson retain
  target snapshot 01. Checks and screenshots are in Pi
  `deployments/menu-pitch-20261007-02/`; no GPIO/motor service restarted.
  Pi operational HUD is now deployed locally from
  `IDCS-trials/panel-hud-20261007-01`; menu polarity is preserved. Only Pi
  display and the new PC user `idcs-hud-relay.service` use this snapshot.
  PC initiates read-only SIM telemetry PUSH to Pi5594 (pose5572,
  perception5574, diagnostics15565); Jetson RTP remains direct to Pi.
  Source/session/progress guards hide stale dynamic cues after250ms; no
  control/GPIO authority is added. Pi full-HUD RGB composition holds ~30fps,
  build p95 ~10ms; relay-loss/recovery passes with GPIO/source/controller PIDs
  unchanged. Camera full HUD is fixture-tested, not newly live-qualified.
  Exact decoded-frame join and network transit age remain unqualified and
  explicitly labelled on screen. Reports, backups, scoped installer and final
  source archive are in PC/Pi `deployments/panel-hud-20261007-01/`.
  Read `docs/operator_display.md` and latest journal before rollout; do not
  overwrite this snapshot with unrelated dirty VPI work or start motor units.
- Panel semantics:
  - Fire-control switch (GPIO12) = automatic tracking enable only; manual
    joystick authority comes from the latched control switch (GPIO5).
  - Safety switch (GPIO16) = fire-button enable only; it does not gate motion.
  - Control switch = manual: the joystick slews through the controller.
  - Fire button = screen indication and engagement record when enabled by
    GPIO16; no effector.
  - E-stop blocks everything.
  - ONLY the fire-control/safety contacts share lamp power (GPIO22 -> 12,
    GPIO25 -> 16). Fire-control lights for an available visual target.
    Lamp-off means unreadable/permission false, not physical switch-off; power
    recovery requires a fresh contact sample. GPIO5/manual, fire button and
    E-stop are independent. Yellow now means fresh physical motor feedback
    availability, not fire-button state. See docs/operator_display.md for
    watchdogs, scope and remaining bench qualification. No motor units started.
  - Active snapshot safety policy: safety means fresh applied live
    sim or gimbal target tracking, with manual mode as an exception. GPIO5
    manual selection works without any target/status; fresh applied live
    manual consumer status also qualifies. Visual selection/shadow-only does
    not. E-stop forces safety off; normal loss has 1s off delay and missing
    activity has a 0.5s watchdog. Fire-control/yellow and engagement rules
    are unchanged. Live simulator ON, relay-loss OFF/recovery and camera OFF
    passed over SSH. GPIO5 manual exception passed FakeGPIO tests, not physical
    switch operation. Physical bridge source is not activated/qualified;
    do not start it under the wiring prohibition. Read docs/operator_display.md
    before a future tagged rollout; the new relay endpoints need all consumers.

### Open items

1. Hardware verification, blocked by the wiring rule: manual slew and
   fire-button indication. The safety switch currently reads off; it only
   gates the fire button. GPIO12 auto authority and GPIO5 manual authority
   remain untested on the repaired bench.
2. Camera/simulator menu coordination is committed and live-tested via actual
   menu/client code and source/return frames. Physical joystick operation was
   not tested; PC/Jetson processes still use the tested trial snapshot. Their
   migration to tagged runtime is separate from this indicator release.
   Recording toggles from the screen menu have not been tried live.
   Hardware-camera <-> HIL handover is deferred while hardware is offline
   (user confirmed 2026-10-07). Current menu still refuses active motor/HIL
   switching; software SIM/camera readiness does not qualify this handover.
   Future work: explicit hold/stop acknowledgement, old command/target/PID/
   estimator invalidation, fresh calibrated hardware pose, HIL camera pose
   initialization, and timing/source readiness before restoring authority.
   Homing is separate and required only if the position reference is invalid.
   Deterministic fake-bridge tests may be done offline; actual stop/pose/
   reference continuity and both switch directions require repaired hardware
   and separately authorized bench tests. No guard removal or motor activation.
3. User decision pending: should an operator lock be limited to the selected
   target type? (Today it overrides it.)
4. Panel screen: the status bar overlaps the Jetson's "infer ms" OSD text, and
   the LOCK label overlaps the Jetson's TARGET label.
5. YOLO threshold 0.15: false-alarm rate on real clutter not yet measured.
6. Panel screen CPU on the Pi is about 33-42% (colour conversion and copy).
7. Detector-independent visual lock is an **experimental isolated engine**, not
   a deployed Pi menu/DeepStream mode. See `docs/visual_lock.md` and the newest
   journal entry: VPI CUDA DCF/Harris/LK pixel replay passes, including the actual
   OpenGL drone with a tight operator seed. Standalone CUDA/NVMM/EGL normal-exit
   interop now passes with early explicit EGL initialization; the no-bootstrap
   baseline still aborts. `StandaloneEGL` is NOT an extra display owner to insert
   into an existing DeepStream process. Keep live GPU input opt-in; never hide a
   finalizer failure with `_Exit`. An isolated control-free live NVMM/RTP/OSD
   trial now passes for the closer moving drone: 1,454/1,454 selected updates
   over 30 seconds, ~49.45Hz observations, ~29fps PC/Pi return, normal exit and
   scoped service restoration. The smaller ~38px drone loses reliable features;
   no automatic reseed or quality-gate relaxation. Exact displayed-frame ROI,
   console activation, sustained 60Hz and image-driven simulator control remain
   pending. See `jetson.visual_lock.live_trial`, `tools.run_visual_lock_trial_guard`
   and the newest journal entry. No production menu/motor rollout.
   **Extended live preview subsequently failed** after ~66s/3,248 selected
   observations: reliable features fell to seven and loss correctly latched.
   Return feed remains healthy, but long-run feature/scale retention must be
   fixed/qualified before operational acceptance. The two bounded short passes
   are not endurance evidence. Never hide this with automatic reseeding.
   Reliability rework is now ABI4: validated feature retention/refill plus
   immutable seed-image forward/backward reference LK. Frozen failure replay
   reproduces every prior box/state exactly; the new180s replay passes9,206/9,206
   with seed-relative position p95 0.573px (ABI3 failed7.27px/lost~150s). Native
   focused82 pass; Windows-compatible723 pass /19 skip /3 deselect /12 subtests.
   Final ABI4 live tests completed with normal exit and scoped restoration:
   15 minutes retained36,380/36,380 selected observations, seed-relative p95
   0.609px; a fresh five-minute twice-speed run retained12,084/12,084, p95
   2.966px (close to the predeclared3px gate). Neither rejected or lost a target.
   Evidence is in `IDCS-trials/visual-lock-reliability-v4-20261007/live-02`
   and `live-03`; the faster PC source is in
   `IDCS-trials/visual-lock-reliability-pc-02-20261007`.
   No VPI trial owner remains. Original camera/shadow/operator/mode-agent/UI
   services are restored; physical units remain inactive. Scoring joins opt-in
   source truth offline only, not tracker feedback. These measure drift of the
   initial ROI point, not target centring; raw centre error remains reported.
   Full live throughput is about40.4Hz, NOT sustained60Hz qualified. Console ROI/
   menu integration, real-camera qualification and image-driven simulator control
   remain pending. Read the latest journal before testing; no production rollout.
   Optional VPI-initialize -> native NvDCF handoff is now implemented in
   `jetson.visual_lock.nvdcf_handoff` and live/replay `--handoff-nvdcf`. One explicit
   external visual region, no YOLO or repeated injection; VPI closes on its owning
   thread, native output joins a bounded exact-frame queue, confirms and pins ID.
   Recorded120s/180s and live30s/300s cases pass; fast five-minute live has17,787
   selected, zero uncertain/lost,59.44Hz, seed-relative p95 1.419px. Six deterministic
   actual VPI/NvDCF GPU fixtures pass including immediate absence suppression and
   latched occlusion loss. This is a control-free isolated mode, NOT a Pi menu or
   resident-YOLO mode manager. Native scale accuracy and real-camera qualification
   remain pending. Final serialized-source fast30s run also passes,59.44Hz,
   seed-relative p95 1.169px. Final Jetson focused52/Windows756 tests pass; original
   software services restored and no trial/source remains. Read the latest journal
   for source digests, retained fixture/replay/live reports and restoration paths.

### Testing

- Visual-lock standalone acceptance (2026-10-07): Jetson 48 focused tests pass,
  including actual CUDA/NVMM/EGL wrappers; five unloaded + three service-loaded
  lifecycle repetitions exit normally with explicit early EGL initialization.
  Short ready-frame 60Hz replay passes unloaded, not alongside the camera GPU
  workload. Windows-compatible regression 690 pass / 18 skip / 3 deselect.
  Full live console integration remains pending; see `docs/visual_lock.md`.

- Live visual-lock trial (2026-10-07): 58 focused Jetson tests pass with actual
  CUDA/NVMM/EGL normal exit; Windows-compatible full regression 700 pass / 18 skip
  / 3 deselect / 12 subtests. Read the latest journal for transient viewing-unit
  names and expiry before starting another owner. `--duration-s` begins on the
  first verified input; startup has a separate default 60s fail-closed timeout.

- Windows-compatible suite: 647 pass on 2026-10-06 (documented environment
  exclusions in the journal). The PC has no cairo
  or Pi GStreamer elements, so a few display tests skip there; they run on the
  Pi.
- Safety-activity implementation: 116 Linux integration tests passed
  in the isolated PC snapshot `IDCS-trials/safety-activity-20261006`. It is not
  an active service source; no motor runtime or real GPIO was exercised.
- Active snapshot deployment: PC 58 / Jetson 103 focused tests and Pi 17
  FakeGPIO unittest cases passed. Live relay/lamp reports passed, Camera mode
  restored, no automatic restarts. Physical manual/E-stop switch exercise and
  real gimbal proof remain bench gates, not claims from these SSH tests.
- Target-picker trial: PC 70 initial Linux integration tests, final PC/Jetson
  57 focused tests each, and Pi 21 unittest cases passed. Real menu/client +
  synthetic joystick deflections confirmed target 2, no-command preview/cancel,
  no fallback after loss/session restart (including numeric-ID reuse), explicit
  Planner recovery and Camera return. Actual simulator controller target IDs and
  camera poses were sampled. Physical joystick/GPIO operation was not exercised.
- Follow the verification policy at the end of this file.

## Overview

### V2 video runtime boundary (authoritative)

- `pc.streamer` loads local configuration once through `common.config`, sends
  RTP96 video, and emits exactly one source-time/header record per encoded
  frame. SimCamera planner feedback consumes `PerceptionSnapshotV2` from
  `net.zmq_perception_v2`.
- `jetson.deepstream.runtime` is the passive production video runtime: YOLO,
  NvSORT, selection, V2 publication, GPU OSD, and RTP97 return video. It never
  constructs a controller. Legacy `net.zmq_results` output is opt-in rollback
  compatibility via `deepstream.legacy_display_output`.
- `pc.ui` and `pc.metadata_monitor` consume `PerceptionSnapshotV2` directly.
  They show or measure detection, track, and selection state without a legacy
  projection.
- Host streamer/UI processes never subscribe to production `net.zmq_control`
  implicitly. Optional simulation/debug command inputs require explicit CLI
  endpoints and reject that production endpoint.
- Use `--check` before live operation and `--duration-s` for bounded canaries.
  Start no serial, gimbal, laser, or controller process for video validation.

The older descriptions below document the rollback architecture and must not
be used as implementation guidance for the V2 runtime.
IDCS is a **distributed video AI system** with 3 main agents:

1. **PC Streamer**  
  Captures video (file or SimCamera in PC-originated modes) and streams
  compressed video → Jetson.  
   Publishes *frame headers* (`CamState`) via ZMQ.

2. **Jetson Server**  
   Receives video, runs YOLO inference, and publishes detection results.  
   Runs **Controller** to compute pan/tilt commands from detections.  
   Publishes *control commands* via ZMQ.

3. **PC UI**  
   Receives detection results and return video.  
   Displays video, overlay, and system status.  
   Integrates SimCamera physics (if used) by applying pan/tilt commands to update camera pose.

---

## Message Channels

### 🎥 Video (UDP / GStreamer RTP)
- **PC → Jetson**: Forward video stream (NVENC H.264 → RTP/UDP, payload 96).  
- **Jetson → PC**: Return video stream with drawn detections (NVENC H.264 → RTP/UDP, payload 97).

### 📨 Metadata (ZMQ JSON)
All metadata is exchanged via ZMQ sockets, “latest only” semantics.

#### 1. **PC → Jetson (headers & state)**  
**Socket**: PUSH (PC) → PULL (Jetson)  
**Content**:
```json
{
  "frame_id": 123,
  "src_ts_ms": 1727250000,
  "pan": 0.42,
  "tilt": -0.05
}
```

#### 2. **Jetson → PC (detections)**  
**Socket**: PUB (Jetson) → SUB (PC UI)  
**Content** (`DetectionMsg`, legacy display compatibility only):
```json
{
  "frame_id": 123,
  "src_ts_ms": 1727250000,
  "rx_ts_ms": 1727250010,
  "infer_ts_ms": 1727250035,
  "img_w": 1280,
  "img_h": 720,
  "boxes": [
    {
      "x": 0.25,
      "y": 0.32,
      "w": 0.15,
      "h": 0.23,
      "conf": 0.87,
      "cls": "0",
      "distance_m": 3.8,
      "distance_src": "height"
    }
  ],
  "target_idx": 0,
  "target_distance_smoothed_m": 3.7
}
```

The replacement DeepStream pipeline also publishes the authoritative internal
metadata stream on `net.zmq_perception_v2`. That endpoint carries strict
`PerceptionSnapshotV2` JSON and is the only perception input new controller,
trace, or simulation-sidecar code should consume. `net.zmq_results` remains a
separate compatibility projection for existing PC display consumers.

#### 3. **Jetson → PC (control commands)**  
**Socket**: PUB (Jetson) → SUB (PC UI / SimCamera)  
**Content** (`ControlCmd`):
```json
{
  "type": "ControlCmd",
  "frame_id": 123,
  "src_ts_ms": 1727250000,
  "cmd_ts_ms": 1727250038,
  "target_ok": true,
  "target_uv": [640, 360],
  "err_uv": [-12.4, 8.1],
  "err_rad": [-0.015, 0.010],
  "pan_rate_cmd": -0.35,
  "tilt_rate_cmd": 0.22,
  "controller_mode": "mpc",
  "mpc": {
    "yaw": {"status": "optimal", "u0": -0.35, "cost": 1.2},
    "pitch": {"status": "optimal", "u0": 0.22}
  }
}
```

---

## Agent Responsibilities

### PC Streamer
- Open PC-originated source (file/sim) for RTP uplink.
- Encode → RTP/UDP → Jetson.
- Send `frame_id` + `src_ts_ms` + `pan/tilt` state (if SimCamera).
- Gracefully stop on shutdown event.

Note: when `source` is Jetson-ingest mode (`webcam...` or `rpi...`), PC streamer
exits intentionally and ingest runs on Jetson/Pi-side.

### Jetson Server
- Receive video, decode on GPU.
- The replacement DeepStream runtime converts detector/tracker metadata
  directly to `PerceptionSnapshotV2`, performs selection in V2, and publishes
  V2 plus a separate legacy display projection.
- `jetson.server` is the legacy rollback runtime and must not run concurrently
  with the DeepStream service.
- Run **Controller**:
  - Select target from detections.
  - Compute pixel → angular error.
  - Run PID-like law → produce `pan_rate_cmd`, `tilt_rate_cmd`.
  - Publish `ControlCmd`.
- Encode annotated frame → return video.

### PC UI
- Subscribe to `DetectionMsg` (for overlays).
- Subscribe to `ControlCmd` (if SimCamera).
- Receive Jetson return video (RTP/UDP).
- Display video with overlays:  
  - e2e latency, FPS, error crosshairs.
- If in simulation: apply `ControlCmd` to update SimCamera pose, and publish updated `CamState`.

---

## Data Flow Summary
```
         (Video RTP 96)                  (Video RTP 97)
 PC Streamer  ───────────▶  Jetson Server  ───────────▶  PC UI
     │                             │                       │
     │ (CamState PUSH)             │ (DetectionMsg PUB)    │
     └────────────────────────────▶│                       │
                                   │ (ControlCmd PUB)      │
                                   └──────────────────────▶│
```

---

## Future Extensions
- Replace SimCamera with physical gimbal driver on PC or Jetson.  
- Add sensor fusion (IMU, encoder feedback) into `CamState`.  
- Security (ZMQ CURVE) for real deployments.  
- Multi-target policies (choose by class, priority).  

---

## Verification Policy

Follow `docs/verification_strategy.md` for all new work.

- Verify one responsibility at a time with deterministic inputs at its contract boundary.
- Do not make tracker, selector, controller, transport, or actuator tests depend on a
  learned detector recognizing a rendered target. Inject schema-valid synthetic
  detections or simulator ground truth for those tests.
- Core DeepStream modules must not import `common.schemas` perception types or
  `common.perception_compat`; those imports are restricted to named legacy
  compatibility modules and display sinks.
