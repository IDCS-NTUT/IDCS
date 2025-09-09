#!/usr/bin/env bash
set -euo pipefail
sudo nvpmodel -m 2 || true
sudo jetson_clocks || true
python -m jetson.server --config configs/dev.yaml
