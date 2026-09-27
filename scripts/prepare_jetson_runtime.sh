#!/usr/bin/env bash
# Prepare a fresh Jetson runtime checkout: build the custom YOLO26 nvinfer
# parser from source, link the (untracked) YOLO model directory, and build the
# target selector's TensorRT policy engine from its committed ONNX. Without the
# first two, idcs-deepstream-video fails with NVDSINFER_CUSTOM_LIB_FAILED;
# without the engine, target selection runs without its learned policy.
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
# TensorRT plans are device- and version-specific, so they are not committed.
engine="$runtime/assets/models/swarm/swarm_policy.engine"
onnx="$runtime/assets/models/swarm/swarm_policy.onnx"
if [[ ! -f $engine || $onnx -nt $engine ]]; then
    /usr/src/tensorrt/bin/trtexec --onnx="$onnx" --saveEngine="$engine" \
        --memPoolSize=workspace:1024M > "$engine.build.log" 2>&1 \
        || { echo "swarm policy engine build failed; see $engine.build.log" >&2; exit 4; }
fi
echo "ready: $runtime (parser built, models -> $(readlink -f "$link"), swarm engine built)"
