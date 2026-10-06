# 📑 AGENTS.md

## Current state and handover (updated 2026-10-06) — read first

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

| Host | Role | Units |
|---|---|---|
| PC 192.168.0.1 | simulator, streamers, PC UI | user units (`systemctl --user`) |
| Jetson 192.168.0.5 | DeepStream, controller, bridge, operator agent, recorder | system units (`sudo systemctl`; passwordless sudo) |
| Pi 192.168.0.3 | safety panel (`idcs-manual`), panel screen (`idcs-operator-display`) | user units; no sudo |

- SSH key: `~/.ssh/id_ed25519_lan`.
- Deploy a tagged commit: `git tag deploy-N && scripts/deploy.sh deploy-N pc,jetson,pi`
  (comma-separated hosts). It bundles the local repo, checks out
  `~/Desktop/project/IDCS-runtime` and installs units; nothing restarts.
- Indicator release `deploy-74` is deployed to Pi/Jetson runtime checkouts
  (2026-10-06). Pi GPIO and display run from `IDCS-runtime`; only those two
  services were restarted. Jetson motor services remain inactive.
- Camera/simulator menu work is committed with the indicator release, but
  active Jetson video/operator and PC video/menu services still run the tested
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
- Panel semantics:
  - Fire-control switch (GPIO12) = automatic tracking enable only; manual
    joystick authority comes from the latched control switch (GPIO5).
  - Safety switch (GPIO16) = fire-button enable only; it does not gate motion.
  - Control switch = manual: the joystick slews through the controller.
  - Fire button = screen indication and engagement record when enabled by
    GPIO16; no effector.
  - E-stop blocks everything.
  - ONLY the fire-control/safety contacts share lamp power (GPIO22 -> 12,
    GPIO25 -> 16). Deployed deploy-74 lights fire-control for an available visual target and
    safety for an observed selected track, with independent 1s off delays.
    Lamp-off means unreadable/permission false, not physical switch-off; power
    recovery requires a fresh contact sample. GPIO5/manual, fire button and
    E-stop are independent. Yellow now means fresh physical motor feedback
    availability, not fire-button state. See docs/operator_display.md for
    watchdogs, scope and remaining bench qualification. No motor units started.

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
3. User decision pending: should an operator lock be limited to the selected
   target type? (Today it overrides it.)
4. Panel screen: the status bar overlaps the Jetson's "infer ms" OSD text, and
   the LOCK label overlaps the Jetson's TARGET label.
5. YOLO threshold 0.15: false-alarm rate on real clutter not yet measured.
6. Panel screen CPU on the Pi is about 33-42% (colour conversion and copy).

### Testing

- Windows-compatible suite: 608 pass on 2026-10-06 (documented environment
  exclusions in the journal). The PC has no cairo
  or Pi GStreamer elements, so a few display tests skip there; they run on the
  Pi.
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
