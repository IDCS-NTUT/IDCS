#!/usr/bin/env bash
# Launch the RPi serial I/O service and manual input client together.

set -euo pipefail

CONFIG_PATH=${1:-configs/dev.yaml}

cleanup() {
  if [[ -n "${INPUT_PID:-}" ]]; then
    kill "$INPUT_PID" 2>/dev/null || true
    wait "$INPUT_PID" 2>/dev/null || true
  fi
  if [[ -n "${SERIAL_PID:-}" ]]; then
    kill "$SERIAL_PID" 2>/dev/null || true
    wait "$SERIAL_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

SERIAL_PORT=$(python - "$CONFIG_PATH" <<'PY'
import sys
import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
gimbal_cfg = cfg.get("gimbal", {}) or {}
print(gimbal_cfg.get("serial_port_rpi") or gimbal_cfg.get("serial_port", "/dev/serial0"))
PY
)
SERIAL_BAUD=$(python - "$CONFIG_PATH" <<'PY'
import sys
import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get("gimbal", {}).get("baudrate", 115200))
PY
)
SERIAL_TIMEOUT=$(python - "$CONFIG_PATH" <<'PY'
import sys
import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get("gimbal", {}).get("timeout", 0.1))
PY
)
SERIAL_RETRIES=$(python - "$CONFIG_PATH" <<'PY'
import sys
import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get("gimbal", {}).get("retries", 1))
PY
)

echo "Starting RPi serial I/O service with config ${CONFIG_PATH}..." >&2
python -m tools.serial_io_service \
  --config "$CONFIG_PATH" \
  --port "$SERIAL_PORT" \
  --baud "$SERIAL_BAUD" \
  --timeout "$SERIAL_TIMEOUT" \
  --retries "$SERIAL_RETRIES" &
SERIAL_PID=$!

echo "Starting RPi manual input with config ${CONFIG_PATH}..." >&2
python -m rpi.manual_input --config "$CONFIG_PATH" &
INPUT_PID=$!

set +e
wait -n "$SERIAL_PID" "$INPUT_PID"
FIRST_STATUS=$?
set -e

if ! kill -0 "$SERIAL_PID" 2>/dev/null; then
  echo "Serial I/O service exited early with status ${FIRST_STATUS}" >&2
  if kill -0 "$INPUT_PID" 2>/dev/null; then
    kill "$INPUT_PID" 2>/dev/null || true
    wait "$INPUT_PID" 2>/dev/null || true
  fi
  exit "$FIRST_STATUS"
fi

if ! kill -0 "$INPUT_PID" 2>/dev/null; then
  echo "Manual input exited early with status ${FIRST_STATUS}" >&2
  if kill -0 "$SERIAL_PID" 2>/dev/null; then
    kill "$SERIAL_PID" 2>/dev/null || true
    wait "$SERIAL_PID" 2>/dev/null || true
  fi
  exit "$FIRST_STATUS"
fi

if kill -0 "$INPUT_PID" 2>/dev/null; then
  kill "$INPUT_PID" 2>/dev/null || true
  wait "$INPUT_PID" 2>/dev/null || true
fi
if kill -0 "$SERIAL_PID" 2>/dev/null; then
  kill "$SERIAL_PID" 2>/dev/null || true
  wait "$SERIAL_PID" 2>/dev/null || true
fi
exit "$FIRST_STATUS"
