#!/usr/bin/env bash
set -euo pipefail
python -m pc.streamer --config configs/dev.yaml &
python -m pc.ui --config configs/dev.yaml
