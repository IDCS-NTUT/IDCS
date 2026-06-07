#!/usr/bin/env bash
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0:/usr/lib/aarch64-linux-gnu/libGL.so.1"
unset DISPLAY
export QT_QPA_PLATFORM=offscreen
set -euo pipefail

CONFIG_PATH="configs/network.yaml"
EXTRA_PATH="configs/perception.yaml,configs/control.yaml,configs/system.yaml"
SOURCE_OVERRIDE=""
REQUIRED_SYNC_PEERS=""
RECORD_TRACE=0
RECORD_OUTPUT=""
RECORD_DURATION=""
RECORD_STATUS_INTERVAL=""

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
		--record|--record-control-trace)
			RECORD_TRACE=1
			shift
			;;
		--record-output)
			RECORD_OUTPUT="$2"
			shift 2
			;;
		--record-duration-s)
			RECORD_DURATION="$2"
			shift 2
			;;
		--record-status-interval-s)
			RECORD_STATUS_INTERVAL="$2"
			shift 2
			;;
		--help|-h)
			cat <<'EOF'
Usage: scripts/run_jetson.sh [--config PATH] [--config-extra LIST] [--source SOURCE] [--required LIST] [--record]

Options:
	--config PATH                     Base config file (default: configs/network.yaml)
	--config-extra LIST              Comma-separated extra config files
	--source SOURCE                   Jetson-only runtime source override (sim/webcam/rpi/file:...)
	--required LIST                   Comma-separated required peers (e.g. pc,rpi)
	--record, --record-control-trace  Run tools.record_control_trace with the Jetson session
	--record-output PATH             Trace JSONL output path
	--record-duration-s SECONDS      Stop recorder after N seconds
	--record-status-interval-s SEC   Recorder status print interval

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
if [[ -n "$REQUIRED_SYNC_PEERS" ]]; then
	SERVER_ARGS+=(--required "$REQUIRED_SYNC_PEERS")
fi

cleanup() {
	if [[ -n "${RECORD_PID:-}" ]] && kill -0 "$RECORD_PID" 2>/dev/null; then
		kill "$RECORD_PID" 2>/dev/null || true
		wait "$RECORD_PID" 2>/dev/null || true
	fi
}
trap cleanup EXIT INT TERM

if [[ "$RECORD_TRACE" -eq 1 ]]; then
	RECORD_ARGS=(--config "$CONFIG_PATH" --config-extra "$EXTRA_PATH")
	if [[ -n "$RECORD_OUTPUT" ]]; then
		RECORD_ARGS+=(--output "$RECORD_OUTPUT")
	fi
	if [[ -n "$RECORD_DURATION" ]]; then
		RECORD_ARGS+=(--duration-s "$RECORD_DURATION")
	fi
	if [[ -n "$RECORD_STATUS_INTERVAL" ]]; then
		RECORD_ARGS+=(--status-interval-s "$RECORD_STATUS_INTERVAL")
	fi
	echo "[run_jetson] starting control trace recorder..." >&2
	python -m tools.record_control_trace "${RECORD_ARGS[@]}" &
	RECORD_PID=$!
fi

python -m jetson.server "${SERVER_ARGS[@]}"
