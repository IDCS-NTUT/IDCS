# 📑 AGENTS.md

## Overview
IDCS is a **distributed video AI system** with 3 main agents:

1. **PC Streamer**  
   Captures video (webcam, file, or SimCamera) and streams compressed video → Jetson.  
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
**Content** (`DetectionMsg`):
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
- Open video source (webcam/file/sim) for RTP uplink.
- Encode → RTP/UDP → Jetson.
- Send `frame_id` + `src_ts_ms` + `pan/tilt` state (if SimCamera).
- Gracefully stop on shutdown event.

### Jetson Server
- Receive video, decode on GPU.
- Run YOLO TensorRT → produce detections.
- Attach PC header to results and publish `DetectionMsg`.
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
