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
HEADER_PUSH_PORT=${7:-5555}

echo "Streaming via libcamera to ${JETSON_IP}:${JETSON_PORT} at ${WIDTH}x${HEIGHT}@${FPS} (${BITRATE_KBPS} kbps)"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HEADER_PUSH_SCRIPT="${SCRIPT_DIR}/header_push.py"
HEADER_PUSH_PID=""

cleanup() {
  if [[ -n "${HEADER_PUSH_PID}" ]]; then
    kill "${HEADER_PUSH_PID}" 2>/dev/null || true
    wait "${HEADER_PUSH_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if command -v python3 >/dev/null 2>&1 && [[ -f "${HEADER_PUSH_SCRIPT}" ]]; then
  HEADER_PUSH_ENDPOINT="tcp://${JETSON_IP}:${HEADER_PUSH_PORT}"
  echo "Starting header side-channel publisher to ${HEADER_PUSH_ENDPOINT}"
  python3 "${HEADER_PUSH_SCRIPT}" --endpoint "${HEADER_PUSH_ENDPOINT}" --fps "${FPS}" &
  HEADER_PUSH_PID=$!
else
  echo "Warning: header side-channel disabled (missing python3 or ${HEADER_PUSH_SCRIPT})" >&2
fi

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
