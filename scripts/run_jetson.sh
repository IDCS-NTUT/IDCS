#!/usr/bin/env bash
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0:/usr/lib/aarch64-linux-gnu/libGL.so.1"
unset DISPLAY
export QT_QPA_PLATFORM=offscreen
set -euo pipefail

usage() {
	cat >&2 <<'EOF'
Usage: scripts/run_jetson.sh [--config PATH] [--config-extra PATH] [--source VALUE]

Backward-compatible positional form is still supported:
	scripts/run_jetson.sh [CONFIG_PATH] [EXTRA_PATH] [SOURCE]
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

cmd=(python -m jetson.server --config "$CONFIG_PATH" --config-extra "$EXTRA_PATH")
if [[ -n "$SOURCE_OVERRIDE" ]]; then
	cmd+=(--source "$SOURCE_OVERRIDE")
fi

"${cmd[@]}"
