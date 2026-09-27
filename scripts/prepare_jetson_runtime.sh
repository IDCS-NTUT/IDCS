#!/usr/bin/env bash
# Prepare a fresh Jetson runtime checkout for DeepStream: build the custom
# YOLO26 nvinfer parser from source and link the (untracked) model directory.
# Without both, idcs-deepstream-video fails with NVDSINFER_CUSTOM_LIB_FAILED.
#
# Usage: scripts/prepare_jetson_runtime.sh [RUNTIME_CHECKOUT] [MODELS_DIR]
set -euo pipefail

runtime=${1:-/home/idcs/Desktop/project/IDCS-runtime}
models=${2:-/home/idcs/Desktop/project/IDCS/assets/models/yolo}

[[ -f $runtime/jetson/deepstream/nvdsinfer_yolo26_parser.cpp ]] || { echo "not a runtime checkout: $runtime" >&2; exit 2; }
[[ -d $models ]] || { echo "model directory missing: $models" >&2; exit 2; }

make -C "$runtime/jetson/deepstream"
link="$runtime/assets/models/yolo"
if [[ -L $link ]]; then
    [[ $(readlink -f "$link") == $(readlink -f "$models") ]] || { echo "$link points elsewhere" >&2; exit 3; }
elif [[ -e $link ]]; then
    echo "$link exists and is not a link; leaving it" >&2
else
    ln -s "$models" "$link"
fi
echo "ready: $runtime (parser built, models -> $(readlink -f "$link"))"
