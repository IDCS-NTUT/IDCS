#!/usr/bin/env bash
# Launch the Jetson inference server and the RS485 gimbal bridge together.

set -euo pipefail

CONFIG_PATH=${1:-configs/dev.yaml}

describe_status() {
  local status=$1
  if (( status == 0 )); then
    echo "exit 0"
  elif (( status > 128 )); then
    local sig=$(( status - 128 ))
    local name
    name=$(kill -l "$sig" 2>/dev/null || echo "SIG${sig}")
    echo "signal ${name}"
  else
    echo "exit ${status}"
  fi
}

cleanup() {
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

echo "Starting gimbal bridge with config ${CONFIG_PATH}..." >&2
python -m jetson.gimbal_bridge --config "$CONFIG_PATH" &
GIMBAL_PID=$!

echo "Starting Jetson server with config ${CONFIG_PATH}..." >&2
python -m jetson.server --config "$CONFIG_PATH" &
SERVER_PID=$!

set +e
wait -n "$GIMBAL_PID" "$SERVER_PID"
FIRST_STATUS=$?
FIRST_DESC=$(describe_status "$FIRST_STATUS")
set -e

if ! kill -0 "$GIMBAL_PID" 2>/dev/null; then
  echo "Gimbal bridge exited early (${FIRST_DESC}); shutting down Jetson server" >&2
  # Ensure server stops as well
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  exit "$FIRST_STATUS"
fi

# Jetson server exited first; propagate its status after stopping the bridge
if kill -0 "$GIMBAL_PID" 2>/dev/null; then
  echo "Jetson server exited first (${FIRST_DESC}); stopping gimbal bridge" >&2
  kill "$GIMBAL_PID" 2>/dev/null || true
  wait "$GIMBAL_PID" 2>/dev/null || true
fi
exit "$FIRST_STATUS"
