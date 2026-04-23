#!/usr/bin/env bash
# Launch the Jetson inference server and the RS485 gimbal bridge together.

set -euo pipefail

CONFIG_PATH="configs/network.yaml"
EXTRA_PATH="configs/perception.yaml,configs/control.yaml,configs/system.yaml"
SOURCE_OVERRIDE=""
REQUIRED_SYNC_PEERS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --config-extra)
      EXTRA_PATH="$2"
      shift 2
      ;;
    --source)
      SOURCE_OVERRIDE="$2"
      shift 2
      ;;
    --required)
      REQUIRED_SYNC_PEERS="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
Usage: scripts/run_jetson_with_gimbal.sh [--config PATH] [--config-extra LIST] [--source SOURCE] [--required LIST]

Options:
  --config PATH       Base config file (default: configs/network.yaml)
  --config-extra LIST Comma-separated extra config files
  --source SOURCE     Jetson-only runtime source override (sim/webcam/rpi/file:...)
  --required LIST  Comma-separated required peers (e.g. pc,rpi)

Legacy positional usage is still supported:
  scripts/run_jetson_with_gimbal.sh [CONFIG_PATH] [EXTRA_PATH]
EOF
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      if [[ "$CONFIG_PATH" == "configs/network.yaml" ]]; then
        CONFIG_PATH="$1"
      elif [[ "$EXTRA_PATH" == "configs/perception.yaml,configs/control.yaml,configs/system.yaml" ]]; then
        EXTRA_PATH="$1"
      else
        echo "Unexpected positional argument: $1" >&2
        exit 2
      fi
      shift
      ;;
  esac
done

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
SERVER_ARGS=(--config "$CONFIG_PATH" --config-extra "$EXTRA_PATH")
if [[ -n "$SOURCE_OVERRIDE" ]]; then
  SERVER_ARGS+=(--source-override "$SOURCE_OVERRIDE")
fi
if [[ -n "$REQUIRED_SYNC_PEERS" ]]; then
  SERVER_ARGS+=(--required "$REQUIRED_SYNC_PEERS")
fi
python -m jetson.server "${SERVER_ARGS[@]}" &
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
