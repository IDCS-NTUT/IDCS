# IDCS Video AI Pipeline

## Overview
IDCS streams simulated or captured video from a development PC to a Jetson for
real-time object detection. Frames are encoded with GStreamer/NVENC on the PC,
decoded on the Jetson, processed by a TensorRT YOLO engine, then annotated
results and return video are sent back to the PC UI via ZeroMQ and RTP.

The repository currently targets a two-machine setup:

- **PC sender/UI (Linux/Windows with NVIDIA GPU, including WSL2 setups).** Generates frames via the
  simulation camera or file source (for PC-originated streaming modes), publishes
  frame headers over ZMQ, and encodes H.264 using NVENC for uplink RTP streaming.
- **Jetson Orin NX 8GB server.** Receives RTP video, runs YOLO inference via the
  custom TensorRT wrapper, republishes detections, and optionally streams an
  annotated return feed back to the PC.

## Repository layout
- `common/` – Shared utilities and Pydantic schemas for detection messages.
- `configs/` – Environment configuration split across `network.yaml`,
  `perception.yaml`, `control.yaml`, and `system.yaml`.
- `pc/` – PC-side tools: the simulation camera and renderers, uplink streamer,
  and monitoring UI.
- `jetson/` – Jetson-side receiver, YOLO engine loader, and inference server.
- `assets/` – Organized runtime assets:
  `models/yolo/` (YOLO engines/ONNX/PT), `models/swarm/` (swarm policy engine/ONNX),
  `meshes/` (OBJ/STL), `sprites/` (billboard PNGs), and `videos/` (sample inputs).
- `scripts/`, `tools/` – Miscellaneous utilities (not yet updated for public
  consumption).

## Installation
IDCS uses a PEP 621 project configuration. Install editable copies on each
machine so both share the same module layout.

```bash
# PC (Linux/Windows/WSL2, Python 3.11 via Miniforge/Mamba recommended)
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -e .[pc]

# Jetson (Orin NX 8GB, L4T 36.4.4 / JetPack 6.2.1, Python 3.10.12 venv)
python3 -m venv ~/Desktop/project/venv
source ~/Desktop/project/venv/bin/activate
pip install --upgrade pip
pip install -e .[jetson]
```

> **Note:** Jetson dependencies for CUDA 12.6.68, cuDNN 9.3.0.75, and TensorRT
> 10.3.0.30 are provided by JetPack 6.2.1 on L4T 36.4.4. Ensure GStreamer and
> OpenCV are installed with codec support on both machines (see `AGENTS.md` for
> inspection commands).

## Configuration
Runtime parameters are split across four files: `configs/network.yaml`
(shared video/network/source settings), `configs/perception.yaml`
(camera/YOLO/sensors/laser), `configs/control.yaml` (control/gimbal/serial IO),
and `configs/system.yaml` (RPi/runtime/logging/perf/sim). Duplicate the set per
environment. Key sections include:

- `video`: width, height, FPS, and NVENC bitrate for uplink/return streams.
- `net`: IP/port endpoints for RTP and ZeroMQ sockets between the PC and Jetson.
- `yolo`: TensorRT engine path, inference thresholds, and the optional
  `class_labels` mapping used to translate detector class IDs into human-readable
  labels for ranging and UI overlays. When `yolo.dual_tracker.enabled` is true,
  `yolo.dual_tracker.search` configures BoT-SORT multi-target identity tracking
  for `SEARCH/SLEW/RECOVER` (including `botsort` thresholds and CamState-first
  GMC), and `yolo.dual_tracker.track` configures single-target handoff/exit
  behavior for `TRACK` mode.
- `source` / `sim`: selects video ingest mode (for example `sim`, `file:<path>`,
  `webcam[:index]`, or `rpi`) and configures the simulation renderer (including
  debug orbit mode) when simulation is used. Set
  `sim.renderer` to `opengl` to enable the moderngl-backed renderer; it falls
  back to CPU automatically if GL init fails.
- `control`: PID gains, rate limits, and focal settings for the pan/tilt
  controller. The section is validated by `common.control.ControlConfig` so both
  Jetson and PC paths share the same interpretation of FOV and sign
  conventions. New fields such as `aim_mode` (now defaulting to
  `laser_point`, with `camera_center` retained for legacy behaviour) and the
  nested `laser` block (`tolerance_px`, `use_range`, and `default_distance_m`)
  are parsed and default to no-op values so existing setups continue to run.
- `laser`: physical mounting data for the emitter, including `offset_m` (meters
  from the camera centre, measured with +X to the right, +Y up, +Z forward),
  `dir_cam` (unit direction in the same frame), and optional render hints (beam
  length, colour, thickness, and hit tolerance).

### Network topology note (Jetson ↔ RPi)

The reference setup includes a **direct Ethernet link between the Jetson and
Raspberry Pi** for control/serial-adjacent services. Use these static IPs on
that point-to-point link:

- `Raspberry Pi`: `192.168.0.3`
- `Jetson`: `192.168.0.5`

Update other IP settings in the config files to match your full network layout
before running.

## Camera calibration and ranging setup

Distance estimates depend on accurate intrinsics and realistic real-world class
sizes. Follow the step-by-step calibration guide in
[`docs/camera_calibration.md`](docs/camera_calibration.md) to:

- Solve for `camera.intrinsics` via a chessboard calibration or by refining FOV
  measurements when a full solve is not available.
- Measure and record canonical object sizes for the `camera.known_size_ranging`
  lookup table.
- Optionally define `class_aspect_ratio_limits` to filter implausible
  detections whose `height / width` falls outside the expected range for the
  class.
- Validate the resulting distances in the field and adjust focal lengths or
  class sizes to remove residual bias.

Keep the guide handy when optics change or when you tune the ranging EMA and
pixel thresholds for new environments.

## Running the pipeline
Launch the RPi runtime, Jetson server, and (optionally) PC tools in separate
terminals. The commands below mirror the canonical setup described in `AGENTS.md`.

```bash
# Jetson server
source ~/Desktop/project/venv/bin/activate
python -m jetson.server --config configs/network.yaml --config-extra configs/perception.yaml,configs/control.yaml,configs/system.yaml

# Jetson server + gimbal bridge (CamState source selected by gimbal.camstate_source)
bash scripts/run_jetson_with_gimbal.sh configs/network.yaml configs/perception.yaml,configs/control.yaml,configs/system.yaml

# Jetson-only runtime source override (without editing YAML)
bash scripts/run_jetson.sh --source sim
bash scripts/run_jetson_with_gimbal.sh --source rpi

# RPi runtime (manual state uplink + return video display)
python -m rpi.runtime_control --config configs/network.yaml --config-extra configs/perception.yaml,configs/control.yaml,configs/system.yaml
python -m rpi.return_video --config configs/network.yaml --config-extra configs/perception.yaml,configs/control.yaml,configs/system.yaml

# PC sender (simulation source)
mamba activate idcs
python -m pc.streamer --config configs/network.yaml --config-extra configs/perception.yaml,configs/control.yaml,configs/system.yaml

# PC UI (optional return video)
python -m pc.ui --config configs/network.yaml --config-extra configs/perception.yaml,configs/control.yaml,configs/system.yaml
```

The streamer publishes frame headers via PUSH to the Jetson (`header_push`),
encodes frames with NVENC, and sends RTP video to `net.jetson_ip`. The Jetson
server drains the PUSH socket, runs YOLO inference, and publishes detection
messages with `common.schemas.detection_msg_to_json()` over a PUB socket. The
UI subscribes to results, decodes frames via
`common.schemas.detection_msg_from_json()`, overlays status text, and plays the
return video feed from the Jetson if enabled. The helpers omit unset optional
fields so downstream consumers that still expect the legacy schema do not see
unexpected `null` values.

To start the UI with config sync enabled (forcing a handshake with the Jetson),
run:

```bash
python -m pc.ui --config configs/network.yaml --config-extra configs/perception.yaml,configs/control.yaml,configs/system.yaml --config-sync-mode=force
```

Config sync policy is now source-dependent:

- Clients (`pc.streamer`, `pc.ui`, and `rpi.runtime_control`) wait indefinitely for Jetson sync by default (set `--config-sync-timeout=0` to skip locally).
- Jetson uses a short per-peer timeout by default (`--config-sync-timeout`, default `3.0s`) and either continues without missing peers or exits (`--config-sync-timeout-action=continue|exit`).
- `source: sim` → Jetson requires `pc`; other peers are optional.
- Non-`sim` sources (including `source: rpi`) → all peers are optional by default.
- Use `--required` on Jetson launch to require an explicit set, e.g. `--required pc,rpi`.

Manual/auto authority is controlled from `control.negotiation` (in
`configs/control.yaml`). Default runtime behavior is `rpi_priority`, where
Jetson suppresses auto control when Pi manual state indicates active or
emergency conditions, and emits zero-rate hold commands during manual authority.
Control-command output mode is also configured in `control.negotiation` via
`command_mode: always|toggle|off`. `off` forces zero-rate hold commands,
`always` keeps default behavior, and `toggle` follows the dedicated Pi GPIO
toggle published as `ManualControlState.control_cmd_enabled` (default pin
`rpi.runtime_control.control_toggle_pin`, 16).

### Streaming CLI usage and config keys
Use `pc.streamer` to send frames for PC-originated sources (`sim` and `file`).
When `source` is `webcam...` or `rpi...`, camera ingest is Jetson-side and
`pc.streamer` exits by design.

Example invocations:

```bash
# File playback (source: file:/path/to/video.mp4)
python -m pc.streamer --config configs/network.yaml --config-extra configs/perception.yaml,configs/control.yaml,configs/system.yaml

# Simulated camera with debug orbit enabled (source: sim)
python -m pc.streamer --config configs/network.yaml --config-extra configs/perception.yaml,configs/control.yaml,configs/system.yaml
```

```yaml
# configs/network.yaml
source: sim
```

```yaml
# configs/system.yaml
sim:
  renderer: cpu
  renderer_opts:
    theme: day
  debug: true
```

Expected configuration keys for streaming:

- `source`: `sim`, `file:<path>`, `webcam[:index]`, or `rpi`.
- `video`: `width`, `height`, `fps`, and `bitrate_kbps` (uplink stream settings).
- `net`: `jetson_ip`, `rtp_port`, `header_push`, and optional `zmq_control`.
  Set `pc_bind_ip` to the PC address on the Jetson link to source-bind PC RTP
  sockets, and set `pc_iface` on Linux to ask ZMQ sockets to bind to that
  interface when the installed pyzmq/libzmq exposes `BINDTODEVICE`.
- `sim` (when using `sim`): `renderer`, `renderer_opts`, and `debug`.

For Jetson camera ingest modes (`source: webcam...` or `source: rpi...`), run
Jetson server + UI (and Pi runtime when applicable) without relying on
`pc.streamer`.

When using Jetson-hosted IMU/magnetometer modules as the primary camera-state
source, set `gimbal.camstate_source: devices` in `configs/control.yaml`.
Configure IMU/magnetometer parameters in the top-level `camstate_devices`
section (for example in `configs/perception.yaml`):

- `mpu_bus`
- `mpu_addr`
- `mag_addr`
- `publish_hz` (used by `jetson.camstate_devices`; bridge publish cadence still follows `gimbal.feedback_hz`)

The IMU orientation model is fixed to the canonical `test.py` equations:
`pitch = atan2(ax, sqrt(ay^2 + az^2))`, `roll = atan2(-ay, az)`, magnetometer
alignment `(mx, my, mz) -> (my, mx, -mz)`, then tilt-compensated heading via
`atan2(my2, mx2)`.
To return to encoder-based CamState, set `gimbal.camstate_source: encoder`.
## Metadata schema summary (PC ↔ Jetson)
- **DetectionMsg (Jetson → PC)**: includes `frame_id` and timestamp fields
  (`*_ts_ms` in milliseconds), original image size (`img_w`/`img_h` in pixels),
  and a list of `Box` detections. Each `Box` uses normalized top-left `x`/`y`
  and normalized `w`/`h` (values in `[0, 1]`), with `conf` in `[0, 1]`. Optional
  fields carry target selection, ranging (`distance_m` in meters, `distance_src`
  method), laser overlay pixels, and predictive lead points. Optional keys are
  omitted during serialization for backward compatibility.
- **ControlCmd/MPC diagnostics (Jetson → PC)**: when `controller_mode="mpc"`,
  the `mpc` map includes per-axis `MpcAxisDiagnostic` entries. These provide
  solver `status`, optional `cost`, and `u0` (first MPC command, typically a
  rate in rad/s), with optional `slack`, `solver`, and `terms` maps for
  diagnostic breakdowns. Unset values are omitted to keep payloads compact.

## Testing and tuning the control loop
Follow the steps below to exercise the closed-loop controller with the simulator
and iterate on PID gains or filtering parameters:

1. **Start all three processes** using the commands in the section above. Keep
   the PC streamer on the `sim` source so the scene always produces a detectable
   target.
2. **Confirm command traffic** by watching the Jetson server logs. The control
   loop emits a compact status line per tick (tagged `jetson.control`) that
   includes the frame, target state, UV coordinates, pixel/angular errors, and
   commanded rates. When known-size ranging is active, the companion
   `jetson.ranging` logger summarizes each ranged detection with its class,
   distance, pixel measurement, confidence, and smoothed target distance. Set
   `logging.cli_json: true` in the config if you prefer the previous JSON dumps
   for downstream log ingestion.
3. **Monitor the return feed** in `pc.ui`. The crosshair should converge on the
   target centroid while the simulated camera pans/tilts in response to the
   Jetson’s `ControlCmd` messages.
4. **Adjust gains and limits** in `configs/control.yaml` under the `control`
   section. Useful knobs include:
   - `kp`, `kd`, `ki`: proportional/derivative/integral gains for yaw and
     pitch. Increase `kp` until you observe oscillation, then raise `kd` to
     damp. Leave `ki` at zero until steady-state error is unacceptable.
   - `rate_limits` and `accel_limits`: cap the commanded velocity and PID-level
     slew intent. Physical gimbal acceleration limits still act as the hardware
     ceiling.
   - `deadband_px` and `smooth_px_alpha`: suppress jitter from small centroid
     movements by widening the deadband or increasing the EMA smoothing factor.
   - `loop_hz`: raise to react faster when detections are frequent; lower if
     noise causes instability.
   - `sign_convention`: verify `pitch_positive` matches your rig. The simulator
     treats positive pitch commands as tilting the camera upward, so the dev
     config defaults to `up`; switch to `down` if your hardware expects the
     opposite sense.
5. **Apply changes by restarting** the Jetson server (and PC processes if they
   also consume control config). Configuration values are loaded on startup.
6. **Iterate and log** by capturing the Jetson server stdout to a file. Enable
   `logging.cli_json` when you need structured JSON for notebooks or automated
   analysis; otherwise, the default human-readable summaries keep the CLI easy
   to skim while you tune gains.

## RS485 gimbal bring-up (Jetson)
The Jetson side includes a minimal RS485 driver for MKS SERVO42D/57D_RS485
closed-loop stepper controllers plus a CLI exerciser for early hardware tests.
Serial signaling stays at 3.3 V TTL on the Jetson; an external transceiver
handles TTL↔RS485 conversion so the code uses a normal `pyserial.Serial`
instance without enabling `serial.rs485` mode.

- Configure the serial port, baud, and motor addresses in `configs/control.yaml`
  under the `gimbal` section. Defaults assume the Jetson GPIO UART
  (`/dev/ttyTHS0`) at `baudrate: 256000`, yaw address `1`, and a dual-pitch
  setup using two independent pitch motor addresses (2 and 3 by default).
  CamState publication source is selectable with `gimbal.camstate_source`:
  `encoder` keeps motor-encoder-based telemetry, while `devices` switches to
  Jetson-hosted IMU/magnetometer telemetry (`camstate_devices`). In `encoder`
  mode, the bridge also attempts to read IMU pitch (when available) and lock a
  one-time CamState tilt zero offset so `tilt=0` matches the gravity horizon.
  Yaw motor command direction can be adjusted independently using
  `yaw_motor_sign` (`+1` or `-1`) so hardware motor polarity changes do not
  require changing control-layer sign conventions.
  Pitch mirroring is defined in software via `pitch_motor_a_sign` and
  `pitch_motor_b_sign` so the two motors can run synchronized but opposite
  direction commands without relying on driver-menu `Dir` settings. Per-axis
  physical acceleration limits and rate clamps (`yaw_accel_limit_rad_s2`/
  `pitch_accel_limit_rad_s2` and `yaw_rate_limit_rad_s`/
  `pitch_rate_limit_rad_s`) are also configurable. The bridge converts physical
  acceleration intent into MKS acceleration bytes when translating ControlCmd
  rates into motor speed mode commands. Serial timeout/retry knobs (`timeout`,
  `retries`) and a
  `respond_on_writes` toggle exists for setups that re-enable motor
  acknowledgements. When both pitch encoders are wired,
  `pitch_divergence_thresh_rad` controls when the bridge
  logs warnings about disagreement between the authoritative and secondary
  pitch encoders (default ~5°). On startup the bridge issues the manual "Set
  current axis to zero" command (function `0x92`, manual page 26) so both axes
  treat their present position as zero before receiving control loop commands.
  With the motor "Respond" parameter set to `0`, write commands (F3/F6/F7/0x92)
  are sent without waiting for acknowledgements; only polling commands such as
  status/encoder reads return replies.
- Install Jetson extras with `pip install -e .[jetson]` to pull in the `pyserial`
  dependency (`>=3.5,<4.0`) for the USB-to-RS485 adapter.
- Run the CLI from the Jetson to validate link-layer communication before
  wiring it into the control loop. Examples:

```bash
# Command 0.5 rad/s with acceleration byte 10 for 2 seconds, then decelerate
python -m jetson.tools.test_mks_gimbal_serial speed --port /dev/ttyTHS0 --addr 1 --omega 0.5 --acc 10 --duration 2

# Read encoder counts and angle in radians from motor address 1
python -m jetson.tools.test_mks_gimbal_serial read-enc --port /dev/ttyTHS0 --addr 1

# Issue an emergency stop
python -m jetson.tools.test_mks_gimbal_serial estop --port /dev/ttyTHS0 --addr 1
```

The CLI reuses the shared `RS485Bus`/`MksServo42Axis` implementation so future
controller integrations can rely on the same protocol handling and safety
guards. Keep the serial session open while issuing stop commands so they reach
the motor before the port closes.

### Gimbal bridge runtime
Once the motors respond to serial commands, start the bridge that connects the
ControlCmd stream to the RS485 driver and republishes encoder-based telemetry.
You can run the bridge alone or launch it alongside the inference server:

```bash
python -m jetson.gimbal_bridge --config configs/network.yaml --config-extra configs/perception.yaml,configs/control.yaml,configs/system.yaml
# or to run bridge + inference together
./scripts/run_jetson_with_gimbal.sh configs/network.yaml configs/perception.yaml,configs/control.yaml,configs/system.yaml
```

The bridge subscribes to `net.zmq_control` for rate commands and publishes
`CamState` snapshots on `net.zmq_gimbal_state` at `gimbal.feedback_hz` (default
20 Hz). CamState `frame_id`/`src_ts_ms` come from the latest ControlCmd when
available; otherwise a local counter and monotonic timestamp are used. Keep the
pitch motor signs (`gimbal.pitch_motor_a_sign` / `gimbal.pitch_motor_b_sign`)
set for mirrored motion, and select the authoritative pitch encoder via
`gimbal.pitch_encoder_authority`. The bridge logs a heartbeat every few seconds
with the latest pan/tilt samples and ControlCmd frame IDs so you can monitor
connectivity headlessly.

### Operator checklist (Jetson RS485 gimbal)
1. **Motor menu setup**
   - Assign addresses: yaw Slave addr `1`, pitch A `2`, pitch B `3`; set both
     pitch motors to distinct Slave addresses.
   - Keep yaw at the default direction that matches the controller sign
     convention.
2. **Wiring**
   - Connect the Jetson 3.3 V UART (`/dev/ttyTHS0`) through an external RS485
     transceiver; no `serial.rs485` mode is required in software.
   - Keep A/B polarity consistent across all motors on the bus and ensure a
     shared ground between Jetson and the transceiver.
3. **Pre-flight checks**
   - With power applied, run the CLI to verify each motor individually before
     issuing closed-loop control writes:
     - `python -m jetson.tools.test_mks_gimbal_serial status --port /dev/ttyTHS0 --addr 1`
     - `python -m jetson.tools.test_mks_gimbal_serial read-enc --port /dev/ttyTHS0 --addr 2`
4. **Run**
   - Start the bridge alone (`python -m jetson.gimbal_bridge --config
    configs/network.yaml --config-extra configs/perception.yaml,configs/control.yaml,configs/system.yaml`) or with inference using
    `./scripts/run_jetson_with_gimbal.sh configs/network.yaml configs/perception.yaml,configs/control.yaml,configs/system.yaml`.
   - Watch startup logs for address/group configuration, divergence warnings,
     and heartbeat telemetry.
5. **Shutdown**
   - Stop with `Ctrl+C`; the bridge issues zero-speed and estop commands while
     the serial port remains open to prevent motors from coasting on exit.
   - If the process crashes, rerun the CLI `estop` command for each motor to
     guarantee a hard stop.

## Simulation camera
`pc.sim_camera.SimCamera` provides a minimal 3D scene with a configurable
renderer API. The default `cpu` renderer draws a ground grid, placeholder
buildings, orbiting billboards that use sprite assets, and an optional debug
mode with a spinning cube. Renderer selection is controlled through
`sim.renderer` and `sim.renderer_opts` in the config file, keeping the
`SimCamera.next_frame()` contract compatible with OpenCV sources.

Targets also support waypoint path movement in world coordinates:

```yaml
sim:
  scene:
    targets:
      - sprite: drone
        width: 0.4
        movement:
          type: path
          speed_m_s: 1.2
          points:
            - [0.5, 2.0, -2.5]
            - [2.0, 2.5, -5.0]
            - [-1.0, 1.8, -7.0]
```

For `movement.type: path`, each point is an absolute target centre
`[x, y, z]`. The simulator interpolates straight lines between consecutive
points and automatically closes the loop from the last point back to the first.
Path and circle targets can also opt into acceleration-limited dynamics so
they carry velocity instead of snapping exactly to the analytic movement curve:

```yaml
sim:
  scene:
    targets:
      - sprite: drone
        width: 0.4
        movement:
          type: path
          speed_m_s: 1.2
          points:
            - [0.5, 2.0, -2.5]
            - [2.0, 2.5, -5.0]
            - [-1.0, 1.8, -7.0]
          dynamics:
            enabled: true
            max_accel_m_s2: 2.0
            max_decel_m_s2: 3.0
            arrival_radius_m: 0.15
```

The same `dynamics` block works on `movement.type: circle`. Path movement uses
`speed_m_s` as the default dynamic speed cap; circle movement derives the
default cap from `radius * speed * fps_hz`. Either movement can override that
with `dynamics.max_speed_m_s`. When `dynamics.enabled` is omitted or false,
targets keep the exact legacy movement behaviour.

## Data products
Detections are serialized using `common.schemas.DetectionMsg`, which includes
per-frame timestamps, normalized bounding boxes, and optional ranging metadata.
Each `Box` exposes `distance_m` (meters) and `distance_src` (height/width/
average) when known-size ranging is active, while the message-level fields
`target_idx` and `target_distance_smoothed_m` surface the currently tracked
target’s smoothed distance. Control integration adds
`common.schemas.ControlCmd` (Jetson → PC rate commands) and
`common.schemas.CamState` (PC → Jetson pose feedback) so both sides share a
structured view of the gimbal state. Downstream consumers can subscribe to the
ZeroMQ endpoints configured in `configs/network.yaml` to monitor end-to-end latency
and control metadata. Laser-aware modes populate additional optional telemetry
including `laser_origin_px`, `laser_dot_px`, `laser_on_target`,
`laser_range_m`, `laser_range_source`, and `parallax_compensation_active` so
overlays and log pipelines can report the active parallax compensation policy
and assumed target distance. When MPC mode is active the controller also sets
`controller_mode: "mpc"` and emits a compact per-axis diagnostic map under the
`mpc` key (solver status, cost, first control input, slack activity) so UIs can
surface solver health without parsing Jetson logs. Lead estimation supplements
detection messages
with `target_velocity_px_s`, `target_lead_uv`, and `target_lead_time_s` so the
return video overlay can render a latency-compensated aim point alongside the
measured centroid.
