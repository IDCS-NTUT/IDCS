#!/usr/bin/env bash
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0:/usr/lib/aarch64-linux-gnu/libGL.so.1"
unset DISPLAY
export QT_QPA_PLATFORM=offscreen
set -euo pipefail

CONFIG_PATH="configs/network.yaml"
EXTRA_PATH="configs/perception.yaml,configs/control.yaml,configs/system.yaml"
SOURCE_OVERRIDE=""
FORCE_RPI_PEER=0

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
		--force-rpi-peer)
			FORCE_RPI_PEER=1
			shift
			;;
		--help|-h)
			cat <<'EOF'
Usage: scripts/run_jetson.sh [--config PATH] [--config-extra LIST] [--source SOURCE] [--force-rpi-peer]

Options:
	--config PATH       Base config file (default: configs/network.yaml)
	--config-extra LIST Comma-separated extra config files
	--source SOURCE     Jetson-only runtime source override (sim/webcam/rpi/file:...)
	--force-rpi-peer    Require rpi config-sync peer during startup

Legacy positional usage is still supported:
	scripts/run_jetson.sh [CONFIG_PATH] [EXTRA_PATH]
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

SERVER_ARGS=(--config "$CONFIG_PATH" --config-extra "$EXTRA_PATH")
if [[ -n "$SOURCE_OVERRIDE" ]]; then
	SERVER_ARGS+=(--source-override "$SOURCE_OVERRIDE")
fi
if [[ "$FORCE_RPI_PEER" == "1" ]]; then
	SERVER_ARGS+=(--force-rpi-peer)
fi

python -m jetson.server "${SERVER_ARGS[@]}"
