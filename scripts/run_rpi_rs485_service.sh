#!/usr/bin/env bash
# Launch the Raspberry Pi RS485 service.

set -euo pipefail

CONFIG_PATH=${1:-configs/dev.yaml}

python -m rpi.rs485_service --config "$CONFIG_PATH"
