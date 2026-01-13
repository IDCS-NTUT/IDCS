#!/usr/bin/env bash
# Launch the Raspberry Pi RS485 service and joystick manual control loop.

set -euo pipefail

CONFIG_PATH=${1:-configs/dev.yaml}

cleanup() {
  if [[ -n "${RS485_PID:-}" ]]; then
    kill "$RS485_PID" 2>/dev/null || true
    wait "$RS485_PID" 2>/dev/null || true
  fi
  if [[ -n "${MANUAL_PID:-}" ]]; then
    kill "$MANUAL_PID" 2>/dev/null || true
    wait "$MANUAL_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

echo "Starting RPi RS485 service with config ${CONFIG_PATH}..." >&2
python -m rpi.rs485_service --config "$CONFIG_PATH" &
RS485_PID=$!

echo "Starting RPi manual control with config ${CONFIG_PATH}..." >&2
python -m rpi.manual_control --config "$CONFIG_PATH" "$@" &
MANUAL_PID=$!

set +e
wait -n "$RS485_PID" "$MANUAL_PID"
FIRST_STATUS=$?
set -e

if ! kill -0 "$RS485_PID" 2>/dev/null; then
  echo "RPi RS485 service exited early with status ${FIRST_STATUS}" >&2
  if kill -0 "$MANUAL_PID" 2>/dev/null; then
    kill "$MANUAL_PID" 2>/dev/null || true
    wait "$MANUAL_PID" 2>/dev/null || true
  fi
  exit "$FIRST_STATUS"
fi

if kill -0 "$RS485_PID" 2>/dev/null; then
  kill "$RS485_PID" 2>/dev/null || true
  wait "$RS485_PID" 2>/dev/null || true
fi
exit "$FIRST_STATUS"
