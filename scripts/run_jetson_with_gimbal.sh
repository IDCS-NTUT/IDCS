#!/usr/bin/env bash
# Launch the Jetson inference server and the RS485 gimbal bridge together.

set -euo pipefail

CONFIG_PATH=${1:-configs/dev.yaml}

cleanup() {
  if [[ -n "${GIMBAL_PID:-}" ]]; then
    kill "$GIMBAL_PID" 2>/dev/null || true
    wait "$GIMBAL_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0:/usr/lib/aarch64-linux-gnu/libGL.so.1"
unset DISPLAY
export QT_QPA_PLATFORM=offscreen

echo "Starting gimbal bridge with config ${CONFIG_PATH}..." >&2
python -m jetson.gimbal_bridge --config "$CONFIG_PATH" &
GIMBAL_PID=$!

echo "Starting Jetson server with config ${CONFIG_PATH}..." >&2
python -m jetson.server --config "$CONFIG_PATH"
