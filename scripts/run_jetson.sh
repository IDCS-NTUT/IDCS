#!/usr/bin/env bash
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0:/usr/lib/aarch64-linux-gnu/libGL.so.1"
unset DISPLAY
export QT_QPA_PLATFORM=offscreen
set -euo pipefail
CONFIG_PATH=${1:-configs/dev.yaml}
EXTRA_PATH=${2:-configs/dev_extra.yaml}
SOURCE_OVERRIDE=${3:-${JETSON_SOURCE:-}}

cmd=(python -m jetson.server --config "$CONFIG_PATH" --config-extra "$EXTRA_PATH")
if [[ -n "$SOURCE_OVERRIDE" ]]; then
	cmd+=(--source "$SOURCE_OVERRIDE")
fi

"${cmd[@]}"
