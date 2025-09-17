# IDCS — PC↔Jetson Video + YOLOv12n(TensorRT) + Sim Camera

Real-time pipeline where the **PC** generates frames (webcam/file/sim), encodes (NVENC), and streams via **GStreamer RTP/UDP** to **NVIDIA Jetson (Xavier NX)**. Jetson decodes (NVDEC), runs **YOLOv12n TensorRT**, publishes detections over **ZeroMQ**, and returns an annotated video stream back to the PC. A simple UI on the PC displays results.

## Repo layout

```
IDCS/
├─ configs/
│  └─ dev.yaml                # central config (uplink/return/yolo/net/sim)
├─ common/
│  ├─ schemas.py              # dataclasses: Box, DetectionMsg
│  └─ shutdown.py             # graceful SIGINT handling (Ctrl+C)
├─ pc/
│  ├─ streamer.py             # source -> NVENC -> RTP/UDP to Jetson; header PUSH (ZMQ)
│  ├─ ui.py                   # receive Jetson return stream + SUB ZMQ to draw HUD
│  ├─ sim_camera.py           # CPU 3D-ish renderer (grid + boxes), FOV-aware
│  └─ renderers/              # (planned) pluggable backends: cpu/cuda/gl
├─ jetson/
│  ├─ receiver.py             # RTP/UDP ingest -> NVDEC -> RGBA appsink
│  ├─ yolo_engine.py          # TensorRT engine runner + CUDA letterbox preprocess
│  └─ server.py               # recv -> infer -> PUB detections + return video (NVENC)
└─ assets/
   ├─ models/                 # (planned) person.obj, drone.obj
   └─ billboards/             # (planned) PNG sprites with alpha
```

## Requirements

### PC (Windows, RTX 2070)
- NVIDIA driver + NVENC
- Python 3.10–3.11 (Miniforge/Mamba recommended)
- GStreamer (with `nvh264enc`, `avdec_h264`)
- OpenCV ≥ 4.6 (built with GStreamer)
- pyzmq, pyyaml, numpy

### Jetson (Xavier NX, JetPack 5.x)
- L4T with GStreamer (`nvv4l2decoder`, `nvv4l2h264enc`)
- Python 3.8 venv
- OpenCV 4.12 (built w/ CUDA & GStreamer)
- TensorRT (JetPack) + pycuda
- YOLO engine (`.engine`) with bindings: input `images [1,3,640,640]`, output `output0 [1,300,6]`

> Tip: keep `jetson_clocks` on while testing. Disable background `jtop` while benchmarking.

## Install

### PC
```bash
# create env (miniforge/mamba)
mamba create -n idcs python=3.11 -y
mamba activate idcs
pip install -r requirements_pc.txt
# ensure GStreamer is on PATH; verify:
gst-inspect-1.0 nvh264enc
python -c "import cv2; print(cv2.getBuildInformation())"
```

### Jetson
```bash
python3 -m venv ~/Desktop/project/venv
source ~/Desktop/project/venv/bin/activate
pip install -r requirements_jetson.txt
# verify critical elements:
gst-inspect-1.0 nvv4l2decoder nvv4l2h264enc
python - << 'PY'
import cv2, tensorrt, pycuda.driver as cuda
print(cv2.__version__, tensorrt.__version__)
PY
```

## Configuration (`configs/dev.yaml`)

```yaml
uplink:                 # PC -> Jetson
  width: 3200
  height: 1920
  fps: 30
  bitrate_kbps: 20000
  vbv_scale: 2

return:                 # Jetson -> PC
  width: 1280
  height: 720
  fps: 30
  bitrate_kbps: 6000
  vbv_scale: 2

net:
  jetson_ip: 192.168.0.10
  pc_ip:     192.168.0.2
  rtp_port:  5000
  rtp_return_port: 5001
  header_push:      tcp://192.168.0.10:5555  # PC -> Jetson
  zmq_results:      tcp://192.168.0.10:5556  # Jetson PUB -> PC SUB

yolo:
  engine_path: /home/idcs/Desktop/project/weights/y12n_640.engine
  input_size: 640
  conf_thres: 0.25
  iou_thres: 0.45
  preprocess_mode: bilinear

sim:
  renderer: cpu        # cpu | cuda | gl (future)
  fov_deg: 70
  # (future) targets: [...]
```

## Run

### 1) Start Jetson server
```bash
# on Jetson
source ~/Desktop/project/venv/bin/activate
python -m jetson.server --config configs/dev.yaml
```

### 2) Start PC sender
```bash
# on PC
mamba activate idcs
python -m pc.streamer --config configs/dev.yaml
```

### 3) Start PC UI
```bash
python -m pc.ui --config configs/dev.yaml
```

You should see detections and a return video window. Press **ESC** in the UI to exit cleanly.

## Notes

- **Graceful shutdown**: SIGINT handlers close sockets and pipelines; if a pipeline errors, the receiver will attempt reopen.
- **Latency**: We ship per-frame headers (`src_ts_ms`) via ZMQ to compute e2e latency on the PC UI (`Detections` overlay).
- **Large uplink res (3200×1920)**: adjust `uplink.bitrate_kbps`. If you see micro-stutter, tweak `vbv_scale` (1.5–3).

## Troubleshooting

- **PC NVENC error about properties**: verify `nvh264enc` element name/properties version; remove unsupported props (e.g., `tuning-info`).
- **Jetson decode “Unsupported Codec”**: ensure `h264parse` → `nvv4l2decoder` caps include `stream-format=byte-stream, alignment=au`.
- **appsink returns NULL caps**: wait for SPS/PPS (jitterbuffer), use a small reopen loop on read fail.
- **OpenCV Unicode errors (Windows cp950)**: always open files with `encoding="utf-8"`; set `PYTHONUTF8=1`.

## Roadmap (short)
- OpenGL renderer backend (ModernGL) for sim (grid/boxes/billboards/meshes)
- Billboard targets (person/drone) with alpha sprites
- Optional: OBJ mesh targets with simple shading
- (Later) gst-gl zero-copy GL→NVENC
