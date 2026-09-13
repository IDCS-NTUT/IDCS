# DeepStream 60-FPS foundation

This directory is an isolated migration layer. It does **not** replace
`jetson.server`, publish `ControlCmd`, or actuate the gimbal.

The first proof of concept is deliberately narrow:

```text
H.264 file/RTP -> nvv4l2decoder -> NVMM -> nvstreammux -> nvinfer
               -> [optional NvSORT] -> NVMM OSD -> fakesink/encoder
```

It keeps decoded surfaces in NVMM and proves the model/parser/tracker path
before IDCS metadata publication and return-video migration are introduced.

## Build and validate on the Jetson

```bash
source /home/idcs/Desktop/project/bin/activate
cd /home/idcs/Desktop/project/IDCS/jetson/deepstream
make DEEPSTREAM_ROOT=/opt/nvidia/deepstream/deepstream-9.1
cd ../..
python -m jetson.deepstream.preflight
```

`preflight` performs no camera access and starts no service. It verifies the
engine, custom parser shared library, `nvinfer`, `nvtracker`, `nvstreammux`,
hardware decoder, OSD, and TensorRT plan compatibility.

## Replay smoke test

```bash
python -m jetson.deepstream.verify_pipeline /absolute/path/to/input.mp4 \
  --report artifacts/deepstream/detector-unpaced.json
python -m jetson.deepstream.verify_pipeline /absolute/path/to/input.mp4 --paced \
  --shadow-jsonl artifacts/deepstream/detector-shadow.jsonl \
  --report artifacts/deepstream/detector-paced.json
```

The first command measures capacity without source pacing. The second replays
according to source timestamps and reports both startup-inclusive and
steady-state FPS. Both commands read `NvDsFrameMeta` and `NvDsObjectMeta`
through PyDS, so a result proves metadata flow rather than only pipeline state.
The optional JSONL is an explicitly legacy, schema-valid `DetectionMsg`
compatibility artifact. It is replay-only and control-disabled: it opens no
ZMQ socket. The in-memory detector/tracker/selector path remains V2. In
that mode `src_ts_ms` is source-PTS-relative while `rx_ts_ms` and
`infer_ts_ms` are Jetson-monotonic; do not calculate cross-host latency from
those fields. The report's stage timing uses ordered single-source, batch-one
correlation because `nvstreammux` may rewrite GStreamer PTS.

After detector-only verification passes, add NvSORT as the first tracker
baseline:

```bash
python -m jetson.deepstream.verify_pipeline /absolute/path/to/input.mp4 \
  --paced --nvsort --shadow-jsonl artifacts/deepstream/nvsort-shadow.jsonl \
  --report artifacts/deepstream/nvsort-paced.json
```

`scripts/run_deepstream_file_smoke.sh` remains available for a native
GStreamer/OSD smoke test. Compare every tracker run against the detector-only
report before considering NvDCF or ReID.

To test a separate native engine without replacing the baseline profile, pass
its config explicitly. For example, the provisional two-class 960 model uses:

```bash
python -m jetson.deepstream.verify_pipeline /absolute/path/to/input.mp4 --paced \
  --nvinfer-config configs/deepstream/nvinfer_small_960_drone_person.txt \
  --shadow-jsonl artifacts/deepstream/small-960-shadow.jsonl
```

This only validates export/parser/schema compatibility. Do not treat a
partially trained checkpoint as a detector-quality or 60-FPS acceptance result.

## Bounded live CSI-camera smoke test

The verifier can also use the local Argus camera without starting IDCS services
or opening ZMQ/control sockets. This is a bounded 1280x720/60 IMX219 mode-4
check; set the sensor arguments explicitly if the connected hardware differs:

```bash
python -m jetson.deepstream.verify_pipeline --live-argus --duration-s 10 --nvsort \
  --nvinfer-config configs/deepstream/nvinfer_yolo26s_736_drone_person_smoke.txt \
  --shadow-jsonl artifacts/deepstream/live-shadow.jsonl \
  --report artifacts/deepstream/live-report.json
```

For live Argus mode, shadow timestamps are same-host camera/GStreamer values,
not PC-origin timestamps; cross-host end-to-end latency still requires live
`CamState` header matching.

## GPU OSD and hardware return-video verification

The verifier can attach neutral detector/tracker labels and a clear
control-disabled status line as `NvDsDisplayMeta`. `nvdsosd` renders those
metadata records on the GPU, after which the frame remains in NVMM and is
encoded by `nvv4l2h264enc`. This is the replacement direction for the current
CPU/OpenCV return-video overlay path; it does not yet replace `jetson.server`.

Use the local encoded fakesink first. Its report must show both nonzero
`frames` and nonzero `encoded_h264_buffers`:

```bash
python -m jetson.deepstream.verify_pipeline --live-argus --duration-s 12 --nvsort \
  --gpu-osd --return-h264 \
  --nvinfer-config configs/deepstream/nvinfer_yolo26s_736_drone_person_smoke.txt \
  --shadow-jsonl artifacts/deepstream/live-osd-shadow.jsonl \
  --report artifacts/deepstream/live-osd-report.json
```

For a controlled network handoff, add `--return-udp-host` and
`--return-udp-port`. The stream is RTP/H.264 payload type 97, matching the
IDCS return-video contract. Keep the first network test on Jetson loopback or
an explicitly approved receiver; it remains control-disabled and does not need
the PC simulator.

The OSD labels use only detector/tracker metadata currently available in this
verifier. Legacy controller authority, laser, lead, and predicted-aim graphics
must remain in the PC UI or wait for their equivalent Jetson metadata adapter;
they are intentionally not fabricated here.

## Control-free RTP shadow runtime

This is the current feature-gated Jetson runtime command. In its RTP profile it
receives the PC uplink contract (RTP/H.264 payload 96), publishes
header-correlated `PerceptionSnapshotV2` values and a separate legacy display
projection, returns GPU-annotated payload-97 H.264, and **never** opens a
control socket. Use distinct, non-production test
ports until the PC power supply is safe and end-to-end timing can be measured:

```bash
python -m jetson.deepstream.verify_pipeline --rtp-input-port 5000 --duration-s 30 \
  --nvsort --gpu-osd --return-h264 \
  --return-udp-host 127.0.0.1 --return-udp-port 5002 \
  --shadow-header-bind tcp://0.0.0.0:5555 \
  --shadow-result-bind tcp://0.0.0.0:5556 \
  --snapshot-result-bind tcp://0.0.0.0:5564 \
  --nvinfer-config configs/deepstream/nvinfer_yolo26s_736_drone_person_smoke.txt \
  --report artifacts/deepstream/rtp-shadow-runtime.json
```

Do not run `jetson.server` alongside this command: both bind the IDCS header
and result ports. The verifier's report records native V2 and legacy
publication separately, plus withheld, overflow, and non-monotonic-header
counts so correlation quality remains observable.

The `yolo26n` model configured here is generic COCO pretrained. It is only a
throughput/parser smoke-test model; its detections must not be used for gimbal
control. The trained two-class drone/person model will replace its engine and
labels without changing the DeepStream topology.

## Config-driven runtime and service

`jetson.deepstream.runtime` is the standalone replacement-launch boundary for
the passive video path. It derives the RTP input, header/V2/legacy metadata binds,
GPU OSD/return-video route, and selected nvinfer profile from
`configs/deepstream_runtime.yaml` plus the normal IDCS configuration files. It
does not import the legacy server or create a control/gimbal socket.

`jetson.deepstream.pipeline` owns the shared pipeline implementation. The
production launcher calls it directly; `jetson.deepstream.verify_pipeline` is
only a compatibility CLI for bounded verification commands.

`configs/deepstream_argus_runtime.yaml` is the equivalent complete profile for
the local IMX219/Argus source. It publishes the same V2 schema and passive result
stream, but deliberately has no PC header correlation: its identity and source
timestamps are Jetson-local.

Check the resolved deployment before opening any pipeline:

```bash
python -m jetson.deepstream.runtime --check
```

Use `--ready-file /run/idcs/deepstream-video.ready` when an external service
manager or health monitor needs a readiness signal. The file is written only
after the first DeepStream metadata frame (not merely process startup) and is
removed during shutdown or a failed no-frame run.

Use `--health-file /run/idcs/deepstream-video-health.json` for a rate-limited
live snapshot. It refreshes while frames arrive with frame count, last-frame
Jetson monotonic time, and pipeline FPS, then is removed on shutdown. A stale
or missing file is therefore a passive service-health failure.

For an explicitly simulation-only control canary, connect the sidecar to the
native V2 endpoint, never to the legacy display endpoint:

```bash
python -m jetson.deepstream.shadow_controller \
  --idcs-config configs/network.yaml \
  --idcs-config configs/perception.yaml \
  --idcs-config configs/control.yaml \
  --idcs-config configs/system.yaml \
  --snapshot-sub tcp://127.0.0.1:5564 \
  --sim-control-bind tcp://127.0.0.1:6551 --check
```

Remove `--check` only for a bounded simulator run with an opted-in SimCamera.
The sidecar cannot bind the production control port. It advances at
`control.loop_hz`; snapshot arrival does not set the command cadence.

Passive trace tools also consume the native V2 endpoint. Use
`--snapshot-sub` with `tools/record_control_protocol_trace.py`, or let
`tools/record_control_trace.py` resolve `net.zmq_perception_v2` from config
(override it with `--perception-endpoint`). The legacy `net.zmq_results`
display payload is deliberately unavailable to these controller/trace paths.

For systemd deployment, install
`deploy/systemd/idcs-deepstream-video.service` on the Jetson, then enable it.
The unit executes the same non-mutating `--check` as `ExecStartPre` and starts
only the control-free runtime. Keep `jetson.server` stopped: both paths bind
the RTP/header/result ports.

## Remaining replacement gates

The video runtime, GPU OSD, metadata publication, return encoder, service
health, bounded reconnect, and local-camera paths are complete and recorded.
Before default cutover, qualify the drone detector and scene-specific tracking,
run a longer PC hardware-encoder soak after PSU repair, and complete the
separate controller/physical-actuation redesign. ReID remains disabled until a
trained model establishes identity-quality evidence.
