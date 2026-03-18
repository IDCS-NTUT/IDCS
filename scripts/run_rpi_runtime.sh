#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

CONFIG_PATH=${1:-configs/dev.yaml}
EXTRA_PATH=${2:-configs/dev_extra.yaml}

cleanup() {
  if [[ -n "${CONTROL_PID:-}" ]]; then
    kill "$CONTROL_PID" 2>/dev/null || true
    wait "$CONTROL_PID" 2>/dev/null || true
  fi
  if [[ -n "${VIDEO_PID:-}" ]]; then
    kill "$VIDEO_PID" 2>/dev/null || true
    wait "$VIDEO_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

cd "${REPO_ROOT}"

python -m rpi.runtime_control --config "$CONFIG_PATH" --config-extra "$EXTRA_PATH" &
CONTROL_PID=$!

python -m rpi.return_video --config "$CONFIG_PATH" --config-extra "$EXTRA_PATH" &
VIDEO_PID=$!

set +e
wait -n "$CONTROL_PID" "$VIDEO_PID"
FIRST_STATUS=$?
set -e

exit "$FIRST_STATUS"
