#!/usr/bin/env bash
# Explicit live-hardware launcher: V2 video + V2 controller + actuator bridge.

set -euo pipefail

CONFIG_PATH="configs/network.yaml"
EXTRA_PATH="configs/perception.yaml,configs/control.yaml,configs/system.yaml,configs/deepstream_runtime.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --config-extra) EXTRA_PATH="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: scripts/run_jetson_with_gimbal.sh [--config PATH] [--config-extra LIST]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

cleanup() {
  for pid in "${VIDEO_PID:-}" "${CONTROL_PID:-}" "${GIMBAL_PID:-}" "${SERIAL_PID:-}"; do
    if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; fi
  done
  for pid in "${VIDEO_PID:-}" "${CONTROL_PID:-}" "${GIMBAL_PID:-}" "${SERIAL_PID:-}"; do
    if [[ -n "$pid" ]]; then wait "$pid" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT

IFS=',' read -r -a EXTRA_CONFIG_PATHS <<< "$EXTRA_PATH"
readarray -t SERIAL_SETTINGS < <(python - "$CONFIG_PATH" "${EXTRA_CONFIG_PATHS[@]}" <<'PY'
import sys
import yaml

cfg = {}
for path in sys.argv[1:]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg.update(yaml.safe_load(handle) or {})
gimbal = cfg.get("gimbal", {})
print(gimbal.get("serial_port", "/dev/ttyCH341USB0"))
print(gimbal.get("baudrate", 38400))
print(gimbal.get("timeout", 0.1))
print(gimbal.get("retries", 1))
PY
)

python -m tools.serial_io_service \
  --config "$CONFIG_PATH" \
  --port "${SERIAL_SETTINGS[0]}" --baud "${SERIAL_SETTINGS[1]}" \
  --timeout "${SERIAL_SETTINGS[2]}" --retries "${SERIAL_SETTINGS[3]}" &
SERIAL_PID=$!

python -m jetson.gimbal_bridge \
  --config "$CONFIG_PATH" --config-extra "$EXTRA_PATH" &
GIMBAL_PID=$!

python -m jetson.control_runtime \
  --config "$CONFIG_PATH" --config-extra "$EXTRA_PATH" \
  --enable-control-publish &
CONTROL_PID=$!

python -m jetson.deepstream.runtime \
  --config "$CONFIG_PATH" --config-extra "$EXTRA_PATH" &
VIDEO_PID=$!

set +e
wait -n "$SERIAL_PID" "$GIMBAL_PID" "$CONTROL_PID" "$VIDEO_PID"
STATUS=$?
set -e
echo "A V2 runtime component exited; stopping the remaining stack." >&2
exit "$STATUS"
