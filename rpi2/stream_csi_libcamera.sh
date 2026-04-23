#!/usr/bin/env bash
# libcamera/rpicam-based streamer for Raspberry Pi OS (Bullseye/Bookworm)
# Uses rpicam-vid (or libcamera-vid) to capture and pipe into GStreamer for RTP streaming.

set -euo pipefail

JETSON_IP=${1:-192.168.55.1}
JETSON_PORT=${2:-5000}
WIDTH=${3:-1280}
HEIGHT=${4:-720}
FPS=${5:-30}
BITRATE_KBPS=${6:-4000}
HEADER_PUSH_PORT=${7:-5555}
PYTHON_BIN=${PYTHON_BIN:-python3}
ENABLE_CONFIG_SYNC_GATE=${ENABLE_CONFIG_SYNC_GATE:-1}
CONFIG_SYNC_ENDPOINT=${CONFIG_SYNC_ENDPOINT:-tcp://${JETSON_IP}:5560}
CONFIG_SYNC_TIMEOUT=${CONFIG_SYNC_TIMEOUT:-60}
CONFIG_SYNC_RETRY_INTERVAL=${CONFIG_SYNC_RETRY_INTERVAL:-1}
CONFIG_SYNC_CONFIG_IDS=${CONFIG_SYNC_CONFIG_IDS:-"network.yaml perception.yaml control.yaml system.yaml"}
CONFIG_SYNC_APPLY_JETSON_IP=${CONFIG_SYNC_APPLY_JETSON_IP:-0}
CAM_TUNING_FILE=${CAM_TUNING_FILE:-/usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json}
CAM_SHUTTER_US=${CAM_SHUTTER_US:-}
CAM_GAIN=${CAM_GAIN:-}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HEADER_PUSH_SCRIPT="${SCRIPT_DIR}/header_push.py"
CONFIG_SYNC_GATE_SCRIPT="${SCRIPT_DIR}/config_sync_gate.py"
HEADER_PUSH_PID=""

cleanup() {
  if [[ -n "${HEADER_PUSH_PID}" ]]; then
    kill "${HEADER_PUSH_PID}" 2>/dev/null || true
    wait "${HEADER_PUSH_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "${ENABLE_CONFIG_SYNC_GATE}" == "1" ]]; then
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Error: ${PYTHON_BIN} not found; cannot run config sync gate." >&2
    exit 127
  fi
  if [[ ! -f "${CONFIG_SYNC_GATE_SCRIPT}" ]]; then
    echo "Error: config sync gate script missing: ${CONFIG_SYNC_GATE_SCRIPT}" >&2
    exit 127
  fi

  echo "Waiting for Jetson config sync readiness at ${CONFIG_SYNC_ENDPOINT}"
  SYNC_ARGS=()
  for cfg_id in ${CONFIG_SYNC_CONFIG_IDS}; do
    SYNC_ARGS+=(--config-id "${cfg_id}")
  done
  SYNC_EXPORTS=$("${PYTHON_BIN}" "${CONFIG_SYNC_GATE_SCRIPT}" \
    --endpoint "${CONFIG_SYNC_ENDPOINT}" \
    --timeout "${CONFIG_SYNC_TIMEOUT}" \
    --retry-interval "${CONFIG_SYNC_RETRY_INTERVAL}" \
    "${SYNC_ARGS[@]}" \
    --shell-output)
  eval "${SYNC_EXPORTS}"

  if [[ "${CONFIG_SYNC_APPLY_JETSON_IP}" == "1" ]]; then
    JETSON_IP="${STREAM_JETSON_IP:-${JETSON_IP}}"
  fi
  JETSON_PORT="${STREAM_JETSON_PORT:-${JETSON_PORT}}"
  WIDTH="${STREAM_WIDTH:-${WIDTH}}"
  HEIGHT="${STREAM_HEIGHT:-${HEIGHT}}"
  FPS="${STREAM_FPS:-${FPS}}"
  BITRATE_KBPS="${STREAM_BITRATE_KBPS:-${BITRATE_KBPS}}"
  HEADER_PUSH_PORT="${STREAM_HEADER_PUSH_PORT:-${HEADER_PUSH_PORT}}"
  CAM_TUNING_FILE="${STREAM_CAM_TUNING_FILE:-${CAM_TUNING_FILE}}"
  CAM_SHUTTER_US="${STREAM_CAM_SHUTTER_US:-${CAM_SHUTTER_US}}"
  CAM_GAIN="${STREAM_CAM_GAIN:-${CAM_GAIN}}"

  if [[ "${STREAM_SOURCE_ALLOWED:-1}" != "1" ]]; then
    echo "Config sync indicates Jetson source='${STREAM_EFFECTIVE_SOURCE:-unknown}'; skipping RPi stream startup."
    exit 0
  fi

  echo "Config sync confirmed. Using Jetson config: ${JETSON_IP}:${JETSON_PORT} ${WIDTH}x${HEIGHT}@${FPS} (${BITRATE_KBPS} kbps), header port ${HEADER_PUSH_PORT}"
fi

echo "Streaming via libcamera to ${JETSON_IP}:${JETSON_PORT} at ${WIDTH}x${HEIGHT}@${FPS} (${BITRATE_KBPS} kbps)"

if command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ -f "${HEADER_PUSH_SCRIPT}" ]]; then
  HEADER_PUSH_ENDPOINT="tcp://${JETSON_IP}:${HEADER_PUSH_PORT}"
  echo "Starting header side-channel publisher to ${HEADER_PUSH_ENDPOINT} via ${PYTHON_BIN}"
  "${PYTHON_BIN}" "${HEADER_PUSH_SCRIPT}" --endpoint "${HEADER_PUSH_ENDPOINT}" --fps "${FPS}" &
  HEADER_PUSH_PID=$!
else
  echo "Warning: header side-channel disabled (missing ${PYTHON_BIN} or ${HEADER_PUSH_SCRIPT})" >&2
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

CAMERA_ARGS=(-t 0 --inline --width "${WIDTH}" --height "${HEIGHT}" --framerate "${FPS}")

if [[ -n "${CAM_TUNING_FILE}" ]]; then
  CAMERA_ARGS+=(--tuning-file "${CAM_TUNING_FILE}")
fi
if [[ -n "${CAM_SHUTTER_US}" ]]; then
  CAMERA_ARGS+=(--shutter "${CAM_SHUTTER_US}")
fi
if [[ -n "${CAM_GAIN}" ]]; then
  CAMERA_ARGS+=(--gain "${CAM_GAIN}")
fi

CAMERA_ARGS+=(-o -)

echo "Camera controls: tuning='${CAM_TUNING_FILE}', shutter_us='${CAM_SHUTTER_US:-auto}', gain='${CAM_GAIN:-auto}'"

# Filter high-frequency camera status lines (e.g. "#123 (30.00 fps) exp ...")
# so systemd/journalctl logs remain readable while real warnings/errors pass through.
"${CAMERA_BIN}" "${CAMERA_ARGS[@]}" \
  2> >(grep -vE '^#[0-9]+ \([0-9]+\.[0-9]+ fps\) exp ' >&2) \
  | gst-launch-1.0 -v fdsrc ! h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host=${JETSON_IP} port=${JETSON_PORT} sync=false async=false
