#!/usr/bin/env bash
# Launch the Jetson inference server and the RS485 gimbal bridge together.

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: scripts/run_jetson_with_gimbal.sh [--config PATH] [--config-extra PATH] [--source VALUE]

Backward-compatible positional form is still supported:
  scripts/run_jetson_with_gimbal.sh [CONFIG_PATH] [EXTRA_PATH] [SOURCE]
EOF
}

CONFIG_PATH="configs/dev.yaml"
EXTRA_PATH="configs/dev_extra.yaml"
SOURCE_OVERRIDE="${JETSON_SOURCE:-}"

positionals=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -lt 2 ]] && { echo "--config requires a value" >&2; usage; exit 2; }
      CONFIG_PATH="$2"
      shift 2
      ;;
    --config-extra)
      [[ $# -lt 2 ]] && { echo "--config-extra requires a value" >&2; usage; exit 2; }
      EXTRA_PATH="$2"
      shift 2
      ;;
    --source)
      [[ $# -lt 2 ]] && { echo "--source requires a value" >&2; usage; exit 2; }
      SOURCE_OVERRIDE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      positionals+=("$1")
      shift
      ;;
  esac
done

if [[ ${#positionals[@]} -ge 1 ]]; then
  CONFIG_PATH="${positionals[0]}"
fi
if [[ ${#positionals[@]} -ge 2 ]]; then
  EXTRA_PATH="${positionals[1]}"
fi
if [[ ${#positionals[@]} -ge 3 ]]; then
  SOURCE_OVERRIDE="${positionals[2]}"
fi
if [[ ${#positionals[@]} -gt 3 ]]; then
  echo "Too many positional arguments" >&2
  usage
  exit 2
fi

cleanup() {
  if [[ -n "${SERIAL_PID:-}" ]]; then
    kill "$SERIAL_PID" 2>/dev/null || true
    wait "$SERIAL_PID" 2>/dev/null || true
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
export JETSON_WITH_GIMBAL=1

SERIAL_PORT=$(python - "$CONFIG_PATH" "$EXTRA_PATH" <<'PY'
import sys
import yaml

paths = sys.argv[1:]
cfg = {}
for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        cfg.update(yaml.safe_load(f) or {})
print(cfg.get("gimbal", {}).get("serial_port", "/dev/ttyTHS0"))
PY
)
SERIAL_BAUD=$(python - "$CONFIG_PATH" "$EXTRA_PATH" <<'PY'
import sys
import yaml

paths = sys.argv[1:]
cfg = {}
for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        cfg.update(yaml.safe_load(f) or {})
print(cfg.get("gimbal", {}).get("baudrate", 256000))
PY
)
SERIAL_TIMEOUT=$(python - "$CONFIG_PATH" "$EXTRA_PATH" <<'PY'
import sys
import yaml

paths = sys.argv[1:]
cfg = {}
for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        cfg.update(yaml.safe_load(f) or {})
print(cfg.get("gimbal", {}).get("timeout", 0.1))
PY
)
SERIAL_RETRIES=$(python - "$CONFIG_PATH" "$EXTRA_PATH" <<'PY'
import sys
import yaml

paths = sys.argv[1:]
cfg = {}
for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        cfg.update(yaml.safe_load(f) or {})
print(cfg.get("gimbal", {}).get("retries", 1))
PY
)

echo "Starting serial I/O service with config ${CONFIG_PATH} (+${EXTRA_PATH})..." >&2
python -m tools.serial_io_service \
  --config "$CONFIG_PATH" \
  --config-extra "$EXTRA_PATH" \
  --port "$SERIAL_PORT" \
  --baud "$SERIAL_BAUD" \
  --timeout "$SERIAL_TIMEOUT" \
  --retries "$SERIAL_RETRIES" &
SERIAL_PID=$!

echo "Starting gimbal bridge with config ${CONFIG_PATH} (+${EXTRA_PATH})..." >&2
python -m jetson.gimbal_bridge --config "$CONFIG_PATH" --config-extra "$EXTRA_PATH" &
GIMBAL_PID=$!

echo "Starting Jetson server with config ${CONFIG_PATH} (+${EXTRA_PATH})..." >&2
server_cmd=(python -m jetson.server --config "$CONFIG_PATH" --config-extra "$EXTRA_PATH")
if [[ -n "$SOURCE_OVERRIDE" ]]; then
  server_cmd+=(--source "$SOURCE_OVERRIDE")
fi
"${server_cmd[@]}" &
SERVER_PID=$!

set +e
wait -n "$SERIAL_PID" "$GIMBAL_PID" "$SERVER_PID"
FIRST_STATUS=$?
set -e

if ! kill -0 "$SERIAL_PID" 2>/dev/null; then
  echo "Serial I/O service exited early with status ${FIRST_STATUS}" >&2
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
if kill -0 "$SERIAL_PID" 2>/dev/null; then
  kill "$SERIAL_PID" 2>/dev/null || true
  wait "$SERIAL_PID" 2>/dev/null || true
fi
exit "$FIRST_STATUS"
