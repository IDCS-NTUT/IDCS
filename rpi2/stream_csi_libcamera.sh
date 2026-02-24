#!/usr/bin/env bash
# libcamera/rpicam-based streamer for Raspberry Pi OS (Bullseye/Bookworm)
# Uses rpicam-vid (or libcamera-vid) to capture and pipe into GStreamer for RTP streaming.

set -euo pipefail

JETSON_IP=${1:-192.168.0.5}
JETSON_PORT=${2:-5000}
WIDTH=${3:-1280}
HEIGHT=${4:-720}
FPS=${5:-30}
BITRATE_KBPS=${6:-4000}

echo "Streaming via libcamera to ${JETSON_IP}:${JETSON_PORT} at ${WIDTH}x${HEIGHT}@${FPS} (${BITRATE_KBPS} kbps)"

# rpicam-vid (Bookworm) or libcamera-vid (Bullseye) produces H.264 to stdout.
if command -v rpicam-vid >/dev/null 2>&1; then
  CAMERA_BIN="rpicam-vid"
elif command -v libcamera-vid >/dev/null 2>&1; then
  CAMERA_BIN="libcamera-vid"
else
  echo "Error: neither rpicam-vid nor libcamera-vid was found in PATH." >&2
  echo "Install rpicam-apps (Bookworm) or libcamera-apps (Bullseye)." >&2
  exit 127
fi

echo "Using camera binary: ${CAMERA_BIN}"

${CAMERA_BIN} -t 0 --inline --width ${WIDTH} --height ${HEIGHT} --framerate ${FPS} -o - \
  | gst-launch-1.0 -v fdsrc ! h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host=${JETSON_IP} port=${JETSON_PORT} sync=false async=false
