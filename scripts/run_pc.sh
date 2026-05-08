#!/usr/bin/env bash
set -euo pipefail
CONFIG_PATH=${1:-configs/network.yaml}
EXTRA_PATH=${2:-configs/perception.yaml,configs/control.yaml,configs/system.yaml}
PYTHON_BIN=${PYTHON:-python}

# Only one PC process should perform the config-sync handshake at startup.
# Streamer keeps sync enabled; UI skips by default to avoid lock contention and
# second-client timeouts after Jetson leaves sync mode.
UI_CONFIG_SYNC_MODE=${UI_CONFIG_SYNC_MODE:-skip}
UI_CONFIG_SYNC_TIMEOUT=${UI_CONFIG_SYNC_TIMEOUT:-0}

"$PYTHON_BIN" -m pc.streamer --config "$CONFIG_PATH" --config-extra "$EXTRA_PATH" &
STREAMER_PID=$!

cleanup() {
	if kill -0 "$STREAMER_PID" 2>/dev/null; then
		kill "$STREAMER_PID" 2>/dev/null || true
		wait "$STREAMER_PID" 2>/dev/null || true
	fi
}
trap cleanup EXIT INT TERM

"$PYTHON_BIN" -m pc.ui --config "$CONFIG_PATH" --config-extra "$EXTRA_PATH" \
	--config-sync-mode "$UI_CONFIG_SYNC_MODE" \
	--config-sync-timeout "$UI_CONFIG_SYNC_TIMEOUT"
