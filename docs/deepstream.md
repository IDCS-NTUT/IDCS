# DeepStream 6.3 prerequisites

This document captures the baseline required to bring the Jetson server onto the
DeepStream SDK while keeping x86_64 development hosts in sync. It focuses on the
6.3.0 release so we can migrate the inference pipeline without surprises from
mixed driver or library versions.

## Release alignment

| Component                | Version / Notes                              |
|--------------------------|----------------------------------------------|
| DeepStream SDK           | 6.3.0 (`deepstream-app --version-all`)       |
| JetPack / L4T (Jetson)   | JetPack 5.1.2 / L4T 35.4.1 (ships CUDA 11.4) |
| CUDA runtime             | 11.4 (from JetPack 5.1.2)                    |
| TensorRT                 | 8.5 (bundled with DeepStream 6.3)            |
| cuDNN                    | 8.6 (bundled with JetPack 5.1.2)             |

DeepStream 6.3 is the last release targeting JetPack 5.x. All Jetson devices on
JetPack 4.x must be upgraded before adopting the DeepStream pipeline.

## Supported hardware

DeepStream 6.3 runs on the following hardware in our fleet:

- **Jetson Xavier NX** – current production target. Requires JetPack 5.1.2 (or
  newer in the 5.x family) with the NVIDIA-provided DeepStream 6.3 package.
- **Jetson AGX Xavier / Jetson Orin** – optional expansion targets. JetPack
  5.1.2 images for these platforms ship the same CUDA/TensorRT stack so the
  DeepStream configuration can be reused.
- **x86_64 development host** – Ubuntu 20.04 with an NVIDIA RTX/Tesla GPU
  (Turing or newer). Install the 525+ production driver along with CUDA 11.8 and
  the DeepStream 6.3 x86_64 tarball if you need to prototype pipelines away from
  the Jetson hardware.

> The repo is private; download DeepStream artifacts directly from NVIDIA
> Developer using your corporate credentials rather than redistributing them.

## Installation (Jetson)

1. **Update JetPack** to 5.1.2 (L4T 35.4.1). The NVIDIA SDK Manager handles the
   OS image and base CUDA/cuDNN/TensorRT stack for Xavier NX, AGX Xavier, and
   Orin devices.
2. **Install DeepStream 6.3** using the `deepstream-6.3_6.3.0-1_arm64.deb`
   package from NVIDIA. Transfer it to the Jetson and run:
   ```bash
   sudo apt install ./deepstream-6.3_6.3.0-1_arm64.deb
   sudo /opt/nvidia/deepstream/deepstream/user_additional_install.sh
   ```
3. **Verify the install** by activating the Jetson virtual environment and
   running `deepstream-app --version-all`. The expected output is:
   ```
   deepstream-app version 6.3.0
   DeepStreamSDK 6.3.0
   CUDA Driver Version: 11.4
   CUDA Runtime Version: 11.4
   TensorRT Version: 8.5
   cuDNN Version: 8.6
   libNVWarp360 Version: 2.0.1d3
   ```

## Installation (x86_64 optional)

1. Install Ubuntu 20.04 LTS and the NVIDIA 525+ production driver.
2. Install CUDA 11.8 (the minimum required for DeepStream 6.3 on discrete GPUs).
3. Download the `deepstream_sdk_v6.3.0_x86_64.tbz2` package and extract it to
   `/opt/nvidia/deepstream/deepstream-6.3/`.
4. Run the `sudo ./install.sh` script from the extracted directory, then add the
   DeepStream GStreamer plugins to `GST_PLUGIN_PATH` as shown below.

## Python bindings (`pyds`)

The DeepStream Python bindings are distributed as version-specific wheels:

- Jetson (Python 3.8):
  ```bash
  pip install /opt/nvidia/deepstream/deepstream/lib/python/3.8/dist-packages/pyds-1.1.6-py3-none-linux_aarch64.whl
  ```
- x86_64 (Python 3.8/3.9):
  ```bash
  pip install /opt/nvidia/deepstream/deepstream/lib/python/3.8/dist-packages/pyds-1.1.6-py3-none-linux_x86_64.whl
  ```

After installation, export the following environment variables before running
the DeepStream-enabled Jetson server:

```bash
export LD_LIBRARY_PATH="/opt/nvidia/deepstream/deepstream/lib:/opt/nvidia/deepstream/deepstream/lib/gst-plugins:$LD_LIBRARY_PATH"
export GST_PLUGIN_PATH="/opt/nvidia/deepstream/deepstream/lib/gst-plugins"
export PYTHONPATH="/opt/nvidia/deepstream/deepstream/lib/python/3.8/dist-packages:$PYTHONPATH"
```

Add the exports to the Jetson service unit (or shell profile) so they are set
for every launch.

## Runtime configuration

Enable the DeepStream backend by setting `deepstream.enabled: true` in
`configs/dev.yaml` or by launching the Jetson server with
`python -m jetson.server --pipeline deepstream`. The new configuration stanza
exposes the following knobs:

- `infer_config` / `engine_path` – point to the DeepStream `nvinfer`
  configuration and optional TensorRT engine file.
- `gpu_id` – select which GPU to run inference and encode on. Defaults to `0`
  (the integrated GPU on Jetson).
- `nvbuf_memory_type` – override the `nvstreammux`/`nvvideoconvert`
  `nvbuf-memory-type` property when you need CUDA device memory (e.g., `3`) or
  leave `null` to use the driver default.
- `return_stream` – fine-tune the outbound RTP encoder. Keys include
  `payload_type`, `bitrate_kbps`, optional GOP settings (`iframe_interval`,
  `idr_interval`), `insert_sps_pps`, manual `vbv_size`, `container` for
  recordings (`mp4` or `mkv`), and `record_path` when you want a persistent file
  sink instead of the auto-generated simulation capture.

Install optional Python dependencies with `pip install -e .[jetson,deepstream]`.
The wheel pulls in `PyGObject` for GStreamer bindings; NVIDIA's `pyds` package
still ships with the DeepStream SDK and must be installed manually from
`/opt/nvidia/deepstream/deepstream/lib/python/.../pyds-*.whl`.

## Smoke testing the pipeline

Use `tools/smoke_deepstream.py` to exercise the full PC streamer → Jetson
DeepStream → control loop path on a single host. The script launches the Jetson
server in DeepStream mode, starts the PC streamer against the configured source
(`sim` works best), and subscribes to the detection/control PUB sockets to
verify traffic. When the network section points at loopback (for example,
`net.pc_ip: 127.0.0.1`), the helper also binds a temporary UDP socket to the
return video port and asserts DeepStream is actively streaming the annotated
RTP feed back to the PC. Disable the return probe with
`--skip-return-feed-check` if you are targeting a remote PC IP or running a
file-based source that does not emit a return stream:

```bash
python tools/smoke_deepstream.py --config configs/dev.yaml --duration 45
```

The helper assumes loopback sockets (`tcp://127.0.0.1:<port>`) are reachable; if
your config still points at a dedicated Jetson IP, temporarily swap the
`net.*` addresses (including `net.pc_ip`) to `127.0.0.1` before running the
smoke test so the return-feed probe can bind locally. A successful run prints
the number of detection and control messages observed along with the most
recent payloads plus optional return-feed statistics so you can confirm
inference, controller updates, and the DeepStream return video are flowing
end-to-end.

## Legacy parity gaps

The DeepStream backend replaces the legacy OpenCV/TensorRT loop, but it does not
yet cover every feature the original pipeline exposed. Track these gaps while
qualifying DeepStream so operators understand which workflows still require the
legacy path:

| Feature | Legacy pipeline | DeepStream status | Impact |
|---------|-----------------|-------------------|--------|
| Offline / file-based sources | `jetson.server` can ingest `file:` sources via `FileVideoReader`, auto-deriving width/height/FPS and recording an annotated MP4 for offline review.【F:jetson/server.py†L1410-L1550】 | The DeepStream branch aborts when a file source is configured because only the live RTP ingest has been implemented so far.【F:jetson/server.py†L1063-L1075】 | Cannot exercise the DeepStream stack against canned clips or run the smoke test without a live PC streamer; legacy mode remains mandatory for dataset replays. |
| Return overlays (status + predictive aids) | The legacy overlay renders laser/parallax status text plus the controller's lead arrow and predictive box so operators can see ranging inputs, lead time, and the predictive tracker visually.【F:jetson/server.py†L204-L341】 | The DeepStream OSD currently draws detection boxes, labels, range text, and basic laser glyphs only, omitting the status ribbon, lead arrow, and predictive box entirely.【F:jetson/deepstream_server.py†L864-L966】 | Operators lose the at-a-glance cues that show whether the controller is leading targets, whether parallax compensation is active, or how predictive tracking is behaving. |
| Attitude overlay | Legacy returns a full azimuth/elevation ladder with degree ticks, numeric readouts, and FOV markers based on `CamState` and `CameraIntrinsics`.【F:jetson/server.py†L503-L724】 | DeepStream collapses attitude data into a single text string (`"Az xx | El yy"`) near the top of the frame; no ladder or tick marks are rendered.【F:jetson/deepstream_server.py†L954-L965】 | The simplified readout drops the visual horizon cues pilots relied on for situational awareness, making it harder to gauge pointing offsets at a glance. |

Reaching feature parity will require porting these UI elements (and file-source
ingest) into the DeepStream code path or keeping the legacy pipeline available
until replacements ship.
