#!/usr/bin/env bash
unset DISPLAY
export QT_QPA_PLATFORM=offscreen
set -euo pipefail
sudo nvpmodel -m 2 || true
sudo jetson_clocks || true
python -m jetson.server --config configs/dev.yaml
