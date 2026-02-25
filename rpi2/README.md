RPi2 CSI streaming helpers
==========================

This directory contains simple scripts to stream a CSI camera from a Raspberry
Pi to the Jetson over RTP/UDP. The Jetson server already listens for H.264 RTP
on the configured `net.rtp_port` (default `5000`) so these scripts can be run
on the Pi to forward camera frames to the Jetson for inference.

Scripts
-------
- `stream_csi_gst.sh` — GStreamer-only pipeline using `v4l2src` and `x264enc`.
- `stream_csi_libcamera.sh` — uses `rpicam-vid` (Bookworm) or `libcamera-vid`
  (Bullseye) to capture H.264 and pipes into GStreamer for RTP.
  It also starts a lightweight side-channel publisher that sends
  `frame_id/src_ts_ms` to Jetson `net.header_push`.
  By default it now waits for Jetson config-sync readiness before starting
  transmission and adopts Jetson-configured stream settings.

Usage examples
--------------
From the Pi, run (replace IP and options as needed):

```bash
./stream_csi_gst.sh 192.168.55.1 5000 1280 720 30 4000
# or using libcamera:
./stream_csi_libcamera.sh 192.168.55.1 5000 1280 720 30 4000
# optional 7th arg: header_push port (default 5555)
./stream_csi_libcamera.sh 192.168.55.1 5000 1280 720 30 4000 5555

# force manual exposure/gain (env override)
CAM_SHUTTER_US=8000 CAM_GAIN=1.5 \\
  ./stream_csi_libcamera.sh 192.168.55.1 5000 1280 720 30 4000
```

Config-driven camera control (recommended):

```yaml
# configs/dev.yaml
camera:
  libcamera:
    tuning_file: "/usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json"
    shutter_us: null   # set e.g. 8000 to force manual exposure length
    gain: null         # set e.g. 1.5 to force manual analog gain
```

These are synced via `config_sync_gate.py` and applied by `stream_csi_libcamera.sh`.

Optional environment overrides (`stream_csi_libcamera.sh`):

- `CAM_TUNING_FILE` (default: `/usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json`)
- `CAM_SHUTTER_US` (shutter/exposure length in microseconds)
- `CAM_GAIN` (analog gain)

Leave `CAM_SHUTTER_US` and `CAM_GAIN` unset for auto exposure/gain.

Notes
-----
- Ensure `gst-launch-1.0` and camera capture tools (`v4l2` drivers,
  `rpicam-vid`, or `libcamera-vid`) are installed on the Pi.
- Ensure `python3` and `pyzmq` are installed on the Pi for the header
  side-channel helper (`rpi2/header_push.py`) and sync gate
  (`rpi2/config_sync_gate.py`).
- Tune encoder choices if your Pi supports a hardware H.264 encoder for
  lower CPU usage (e.g., `v4l2h264enc` or platform-specific encoders).
- The scripts stream to the Jetson IP/port. The Jetson should have its
  `net.rtp_port` set to match and run `jetson/server.py` to receive and
  process frames.

Systemd service (optional)
--------------------------
You can run the streamer at boot using the provided systemd unit templates.
Copy the appropriate unit file to `/etc/systemd/system/stream_csi_gst.service`
or `/etc/systemd/system/stream_csi_libcamera.service` on the Pi, edit the
`ExecStart` path to point to the installed script location (e.g., `/home/pi`),
then enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now stream_csi_gst.service
sudo systemctl status stream_csi_gst.service
```

Adjust the `ExecStart` arguments to match your Jetson IP, port, resolution,
and bitrate.

If `pyzmq` is installed only in a virtualenv, set `PYTHON_BIN` in the systemd
unit to that interpreter path, for example:

```ini
Environment=PYTHON_BIN=/home/idcs/Desktop/project/bin/python
```

Config-sync gate (first pass)
-----------------------------
`stream_csi_libcamera.sh` supports startup gating so the stream only starts
after Jetson responds on `net.config_sync` and config IDs are confirmed.

It also supports an always-on background watcher so the Pi remains ready to
sync again if Jetson restarts.

Service environment knobs:

```ini
Environment=ENABLE_CONFIG_SYNC_GATE=1
Environment=CONFIG_SYNC_ENDPOINT=tcp://192.168.55.1:5560
Environment=CONFIG_SYNC_TIMEOUT=60
Environment=CONFIG_SYNC_RETRY_INTERVAL=1
Environment=CONFIG_SYNC_CONFIG_IDS=dev.yaml dev_extra.yaml
Environment=CONFIG_SYNC_APPLY_JETSON_IP=0
Environment=CONFIG_SYNC_PEER_ID=rpi2
Environment=ENABLE_CONFIG_SYNC_WATCH=1
Environment=CONFIG_SYNC_WATCH_TIMEOUT=5
Environment=CONFIG_SYNC_WATCH_INTERVAL=2
# camera knobs are read from synced configs/dev.yaml -> camera.libcamera.*
# optional service-level overrides still work via CAM_* env vars if needed
```

Set `ENABLE_CONFIG_SYNC_GATE=0` to disable gating and always stream.
Set `ENABLE_CONFIG_SYNC_WATCH=0` to disable background re-sync attempts.
Set `CONFIG_SYNC_APPLY_JETSON_IP=1` only if you want `net.jetson_ip` from synced
config to override the script/service Jetson IP argument.

When Jetson `source: "rpi"`, the server can require specific sync peers before
startup proceeds (defaults to `pc` + `rpi2`). Optional config:

```yaml
net:
  config_sync_required_peers: ["pc", "rpi2"]
```

In strict `source: "rpi"` mode, Jetson now synchronizes one peer at a time
(all configs for `pc`, then all configs for `rpi2`) during startup.
