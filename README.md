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
  conventions. New fields such as `aim_mode` (choose between legacy
  `camera_center` or the upcoming `laser_point` behaviour) and the nested
  `laser` block (`tolerance_px`, `use_range`, and `default_distance_m`) are
  parsed and default to no-op values so existing setups continue to run.
- `laser`: physical mounting data for the emitter, including `offset_m` (meters
  from the camera centre), `dir_cam` (unit direction in camera coordinates), and
  optional render hints (beam length, colour, thickness, and hit tolerance).

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
and control metadata.

## Contributing
See `TASKS.md` for the current backlog. Focus work on renderer modularity, Jetson
stability, and observability improvements before expanding feature scope.
