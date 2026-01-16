# IDCS Video AI Pipeline

## Overview
IDCS streams simulated or captured video from a development PC to a Jetson for
real-time object detection. Frames are encoded with GStreamer/NVENC on the PC,
decoded on the Jetson, processed by a TensorRT YOLO engine, then annotated
results and return video are sent back to the PC UI via ZeroMQ and RTP.

The repository currently targets a two-machine setup:

- **PC sender/UI (Linux/Windows with NVIDIA GPU).** Generates frames via the
  simulation camera or a webcam/file source, publishes frame headers over ZMQ,
  and encodes H.264 using NVENC for uplink RTP streaming.
- **Jetson Xavier NX server.** Receives RTP video, runs YOLO inference via the
  custom TensorRT wrapper, republishes detections, and optionally streams an
  annotated return feed back to the PC.

## Repository layout
- `common/` – Shared utilities and Pydantic schemas for detection messages.
- `configs/` – Environment configuration (`dev.yaml`) describing video, network,
  and YOLO settings.
- `pc/` – PC-side tools: the simulation camera and renderers, uplink streamer,
  and monitoring UI.
- `jetson/` – Jetson-side receiver, YOLO engine loader, and inference server.
- `assets/` – Sample YOLO models and billboard sprite images referenced by the
  simulation renderer.
- `scripts/`, `tools/` – Miscellaneous utilities (not yet updated for public
  consumption).

## Installation
IDCS uses a PEP 621 project configuration. Install editable copies on each
machine so both share the same module layout.

```bash
# PC (Linux/Windows, Python 3.11 via Miniforge/Mamba recommended)
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -e .[pc]

# Jetson (JetPack 5.x, Python 3.8 venv)
python3 -m venv ~/Desktop/project/venv
source ~/Desktop/project/venv/bin/activate
pip install --upgrade pip
pip install -e .[jetson]
```

> **Note:** Jetson dependencies for TensorRT and PyCUDA are provided by
> JetPack. Ensure GStreamer and OpenCV are installed with codec support on both
> machines (see `AGENTS.md` for inspection commands).

## Configuration
All runtime parameters live in `configs/dev.yaml` and should be duplicated per
environment. Key sections include:

- `video`: width, height, FPS, and NVENC bitrate for uplink/return streams.
- `net`: IP/port endpoints for RTP and ZeroMQ sockets between the PC and Jetson.
- `yolo`: TensorRT engine path, inference thresholds, and the optional
  `class_labels` mapping used to translate detector class IDs into human-readable
  labels for ranging and UI overlays.
- `source` / `sim`: selects `sim` (default), `webcam:<index>`, or `file:<path>`
  and configures the CPU simulation renderer (including debug orbit mode).
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

Update the IP addresses to match your network layout before running.

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
Launch the PC streamer, Jetson server, and PC UI in separate terminals. The
commands below mirror the canonical setup described in `AGENTS.md`.

```bash
# Jetson server
source ~/Desktop/project/venv/bin/activate
python -m jetson.server --config configs/dev.yaml

# PC sender (simulation source)
mamba activate idcs
python -m pc.streamer --config configs/dev.yaml

# PC UI (optional return video)
python -m pc.ui --config configs/dev.yaml
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
python -m pc.ui --config configs/dev.yaml --config-sync-mode=force
```

### Streaming CLI usage and config keys
Use `pc.streamer` to send frames from a webcam, file, or the simulator. Example
invocations:

```bash
# Webcam capture on device index 0 (source: webcam:0)
python -m pc.streamer --config configs/dev.yaml

# File playback (source: file:/path/to/video.mp4)
python -m pc.streamer --config configs/dev.yaml

# Simulated camera with debug orbit enabled (source: sim)
python -m pc.streamer --config configs/dev.yaml
```

```yaml
# configs/dev.yaml
source: sim
sim:
  renderer: cpu
  renderer_opts:
    theme: day
  debug: true
```

Expected configuration keys for streaming:

- `source`: `webcam:<index>`, `file:<path>`, or `sim` (defaults to `webcam:0`).
- `video`: `width`, `height`, `fps`, and `bitrate_kbps` (uplink stream settings).
- `net`: `jetson_ip`, `rtp_port`, `header_push`, and optional `zmq_control`.
- `sim` (when using `sim`): `renderer`, `renderer_opts`, and `debug`.
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
4. **Adjust gains and limits** in `configs/dev.yaml` under the `control`
   section. Useful knobs include:
   - `kp`, `kd`, `ki`: proportional/derivative/integral gains for yaw and
     pitch. Increase `kp` until you observe oscillation, then raise `kd` to
     damp. Leave `ki` at zero until steady-state error is unacceptable.
   - `rate_limits` and `accel_limits`: cap the commanded velocity and slew so
     the simulated mount remains smooth.
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

- Configure the serial port, baud, and motor addresses in `configs/dev.yaml`
  under the `gimbal` section. Defaults assume the Jetson GPIO UART
  (`/dev/ttyTHS0`) at `baudrate: 115200`, yaw address `1`, and a dual-pitch
  setup using a shared group address `0x50` (decimal `80`). Pitch motor A and B
  retain unique Slave addresses (2 and 3 by default) for encoder reads and
  diagnostics while sharing the group address for commands. Set motor A "Dir"
  to CCW and motor B "Dir" to CW in the driver menu so a single group F6
  command spins them in opposite mechanical directions. Per-axis acceleration
  bytes and rate clamps (`yaw_accel_byte`/`pitch_accel_byte` and
  `yaw_rate_limit_rad_s`/`pitch_rate_limit_rad_s`) are also configurable and are
  applied by the gimbal interface when translating ControlCmd rates into motor
  speed mode commands. Serial timeout/retry knobs (`timeout`, `retries`) and a
  `use_group_writes` toggle are available for bring-up to force individual
  writes if group addressing needs to be disabled temporarily, and a
  `respond_on_writes` toggle exists for setups that re-enable motor
  acknowledgements. When both pitch
  encoders are wired, `pitch_divergence_thresh_rad` controls when the bridge
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
python -m jetson.gimbal_bridge --config configs/dev.yaml
# or to run bridge + inference together
./scripts/run_jetson_with_gimbal.sh configs/dev.yaml
```

The bridge subscribes to `net.zmq_control` for rate commands and publishes
`CamState` snapshots on `net.zmq_gimbal_state` at `gimbal.feedback_hz` (default
20 Hz). CamState `frame_id`/`src_ts_ms` come from the latest ControlCmd when
available; otherwise a local counter and monotonic timestamp are used. Keep the
pitch group address (default `0x50`) consistent across both pitch motors so a
single F6 command drives the mirrored pair, and select the authoritative pitch
encoder via `gimbal.pitch_encoder_authority`. The bridge logs a heartbeat every
few seconds with the latest pan/tilt samples and ControlCmd frame IDs so you
can monitor connectivity headlessly.

### Operator checklist (Jetson RS485 gimbal)
1. **Motor menu setup**
   - Assign addresses: yaw Slave addr `1`, pitch A `2`, pitch B `3`; set both
     pitch motors to Group addr `0x50` (`80` decimal).
   - Set motor directions in the driver menu: Pitch A **CCW**, Pitch B **CW**;
     keep yaw at the default direction that matches the controller sign
     convention.
2. **Wiring**
   - Connect the Jetson 3.3 V UART (`/dev/ttyTHS0`) through an external RS485
     transceiver; no `serial.rs485` mode is required in software.
   - Keep A/B polarity consistent across all motors on the bus and ensure a
     shared ground between Jetson and the transceiver.
3. **Pre-flight checks**
   - With power applied, run the CLI to verify each motor individually before
     issuing group writes:
     - `python -m jetson.tools.test_mks_gimbal_serial status --port /dev/ttyTHS0 --addr 1`
     - `python -m jetson.tools.test_mks_gimbal_serial read-enc --port /dev/ttyTHS0 --addr 2`
   - If group writes misbehave during bring-up, launch the bridge with
     `gimbal.use_group_writes: false` to fall back to per-motor commands until
     wiring is validated.
4. **Run**
   - Start the bridge alone (`python -m jetson.gimbal_bridge --config
     configs/dev.yaml`) or with inference using
     `./scripts/run_jetson_with_gimbal.sh configs/dev.yaml`.
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
ZeroMQ endpoints configured in `configs/dev.yaml` to monitor end-to-end latency
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
