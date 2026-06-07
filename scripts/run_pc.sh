#!/usr/bin/env bash
set -euo pipefail
CONFIG_PATH="configs/network.yaml"
EXTRA_PATH="configs/perception.yaml,configs/control.yaml,configs/system.yaml"
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
Usage: scripts/run_pc.sh [--config PATH] [--config-extra LIST] [--record] [record options]

Options:
	--config PATH                     Base config file (default: configs/network.yaml)
	--config-extra LIST              Comma-separated extra config files
	--record, --record-control-trace  Run tools.record_control_trace with the PC session
	--record-output PATH             Trace JSONL output path
	--record-duration-s SECONDS      Stop recorder after N seconds
	--record-status-interval-s SEC   Recorder status print interval

Legacy positional usage is still supported:
	scripts/run_pc.sh [CONFIG_PATH] [EXTRA_PATH]
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

# Only one PC process should perform the config-sync handshake at startup.
# Streamer keeps sync enabled; UI skips by default to avoid lock contention and
# second-client timeouts after Jetson leaves sync mode.
UI_CONFIG_SYNC_MODE=${UI_CONFIG_SYNC_MODE:-skip}
UI_CONFIG_SYNC_TIMEOUT=${UI_CONFIG_SYNC_TIMEOUT:-0}

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
	echo "[run_pc] starting control trace recorder..." >&2
	python -m tools.record_control_trace "${RECORD_ARGS[@]}" &
	RECORD_PID=$!
fi

python -m pc.streamer --config "$CONFIG_PATH" --config-extra "$EXTRA_PATH" &
STREAMER_PID=$!

cleanup() {
	if [[ -n "${RECORD_PID:-}" ]] && kill -0 "$RECORD_PID" 2>/dev/null; then
		kill "$RECORD_PID" 2>/dev/null || true
		wait "$RECORD_PID" 2>/dev/null || true
	fi
	if kill -0 "$STREAMER_PID" 2>/dev/null; then
		kill "$STREAMER_PID" 2>/dev/null || true
		wait "$STREAMER_PID" 2>/dev/null || true
	fi
}
trap cleanup EXIT INT TERM

python -m pc.ui --config "$CONFIG_PATH" --config-extra "$EXTRA_PATH" \
	--config-sync-mode "$UI_CONFIG_SYNC_MODE" \
	--config-sync-timeout "$UI_CONFIG_SYNC_TIMEOUT"
