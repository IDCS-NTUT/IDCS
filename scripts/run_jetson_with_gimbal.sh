#!/usr/bin/env bash
# Launch the Jetson inference server and the RS485 gimbal bridge together.

set -euo pipefail

CONFIG_PATH=${1:-configs/dev.yaml}

cleanup() {
  if [[ -n "${RS485_PID:-}" ]]; then
    kill "$RS485_PID" 2>/dev/null || true
    wait "$RS485_PID" 2>/dev/null || true
  fi
  if [[ -n "${GIMBAL_PID:-}" ]]; then
    kill "$GIMBAL_PID" 2>/dev/null || true
    wait "$GIMBAL_PID" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0:/usr/lib/aarch64-linux-gnu/libGL.so.1"
unset DISPLAY
export QT_QPA_PLATFORM=offscreen

echo "Starting RS485 service with config ${CONFIG_PATH}..." >&2
python -m jetson.rs485_service --config "$CONFIG_PATH" &
RS485_PID=$!

echo "Starting gimbal bridge with config ${CONFIG_PATH}..." >&2
python -m jetson.gimbal_bridge --config "$CONFIG_PATH" &
GIMBAL_PID=$!

echo "Starting Jetson server with config ${CONFIG_PATH}..." >&2
python -m jetson.server --config "$CONFIG_PATH" &
SERVER_PID=$!

set +e
wait -n "$RS485_PID" "$GIMBAL_PID" "$SERVER_PID"
FIRST_STATUS=$?
set -e

if ! kill -0 "$RS485_PID" 2>/dev/null; then
  echo "RS485 service exited early with status ${FIRST_STATUS}" >&2
  if kill -0 "$GIMBAL_PID" 2>/dev/null; then
    kill "$GIMBAL_PID" 2>/dev/null || true
    wait "$GIMBAL_PID" 2>/dev/null || true
  fi
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  exit "$FIRST_STATUS"
fi

if ! kill -0 "$GIMBAL_PID" 2>/dev/null; then
  echo "Gimbal bridge exited early with status ${FIRST_STATUS}" >&2
  # Ensure server stops as well
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  exit "$FIRST_STATUS"
fi

# Jetson server exited first; propagate its status after stopping the bridge
if kill -0 "$GIMBAL_PID" 2>/dev/null; then
  kill "$GIMBAL_PID" 2>/dev/null || true
  wait "$GIMBAL_PID" 2>/dev/null || true
fi
if kill -0 "$RS485_PID" 2>/dev/null; then
  kill "$RS485_PID" 2>/dev/null || true
  wait "$RS485_PID" 2>/dev/null || true
fi
exit "$FIRST_STATUS"
