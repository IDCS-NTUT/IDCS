#!/usr/bin/env bash
# Launch the Raspberry Pi joystick manual control loop.

set -euo pipefail

CONFIG_PATH=""
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  CONFIG_PATH=$1
  shift
fi

ARGS=()
if [[ -n "$CONFIG_PATH" ]]; then
  ARGS+=(--config "$CONFIG_PATH")
fi

python -m rpi.manual_control "${ARGS[@]}" "$@"
