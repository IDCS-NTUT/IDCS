#!/usr/bin/env bash
# Start the passive V2 DeepStream video pipeline. This script never starts a
# controller, gimbal bridge, or serial service.

set -euo pipefail

CONFIG_PATH="configs/base"
EXTRA_PATH=""
RUNTIME_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --config-extra) EXTRA_PATH="$2"; shift 2 ;;
    --duration-s) RUNTIME_ARGS+=(--duration-s "$2"); shift 2 ;;
    --report) RUNTIME_ARGS+=(--report "$2"); shift 2 ;;
    --ready-file) RUNTIME_ARGS+=(--ready-file "$2"); shift 2 ;;
    --health-file) RUNTIME_ARGS+=(--health-file "$2"); shift 2 ;;
    --check) RUNTIME_ARGS+=(--check); shift ;;
    --help|-h)
      echo "Usage: scripts/run_jetson.sh [--config PATH] [--config-extra LIST] [DeepStream output options]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

exec python -m jetson.deepstream.runtime \
  --config "$CONFIG_PATH" \
  --config-extra "$EXTRA_PATH" \
  "${RUNTIME_ARGS[@]}"
