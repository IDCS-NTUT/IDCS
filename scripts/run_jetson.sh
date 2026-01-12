#!/usr/bin/env bash
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0:/usr/lib/aarch64-linux-gnu/libGL.so.1"
unset DISPLAY
export QT_QPA_PLATFORM=offscreen
set -euo pipefail

CONFIG_PATH=${1:-configs/dev.yaml}

cleanup() {
  if [[ -n "${RS485_PID:-}" ]]; then
    kill "$RS485_PID" 2>/dev/null || true
    wait "$RS485_PID" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

echo "Starting RS485 service with config ${CONFIG_PATH}..." >&2
python -m jetson.rs485_service --config "$CONFIG_PATH" &
RS485_PID=$!

echo "Starting Jetson server with config ${CONFIG_PATH}..." >&2
python -m jetson.server --config "$CONFIG_PATH" &
SERVER_PID=$!

set +e
wait -n "$RS485_PID" "$SERVER_PID"
FIRST_STATUS=$?
set -e

if ! kill -0 "$RS485_PID" 2>/dev/null; then
  echo "RS485 service exited early with status ${FIRST_STATUS}" >&2
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  exit "$FIRST_STATUS"
fi

if kill -0 "$RS485_PID" 2>/dev/null; then
  kill "$RS485_PID" 2>/dev/null || true
  wait "$RS485_PID" 2>/dev/null || true
fi
exit "$FIRST_STATUS"
