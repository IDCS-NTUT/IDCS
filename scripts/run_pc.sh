#!/usr/bin/env bash
set -euo pipefail
CONFIG_PATH=${1:-configs/dev.yaml}
EXTRA_PATH=${2:-configs/dev_extra.yaml}
python -m pc.streamer --config "$CONFIG_PATH" --config-extra "$EXTRA_PATH" &
python -m pc.ui --config "$CONFIG_PATH" --config-extra "$EXTRA_PATH"
