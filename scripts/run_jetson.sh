#!/usr/bin/env bash
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0:/usr/lib/aarch64-linux-gnu/libGL.so.1"
unset DISPLAY
export QT_QPA_PLATFORM=offscreen
set -euo pipefail
sudo nvpmodel -m 2 || true
sudo jetson_clocks || true
python -m jetson.server --config configs/dev.yaml
