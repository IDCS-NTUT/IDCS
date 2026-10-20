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
models=${2:-/home/idcs/idcs-models/yolo}

[[ -f $runtime/jetson/deepstream/nvdsinfer_yolo26_parser.cpp ]] || { echo "not a runtime checkout: $runtime" >&2; exit 2; }
[[ -d $models ]] || { echo "model directory missing: $models" >&2; exit 2; }

make -C "$runtime/jetson/deepstream"
# Release-local ABI4 VPI tracker and the production borrowed-NVMM adapter.
# No test uploader, camera, GPU pipeline or service is started by this build.
cmake -S "$runtime/jetson/visual_lock/native" -B "$runtime/build/visual-lock" \
    -DCMAKE_BUILD_TYPE=Release -DIDCS_BUILD_LIVE_SURFACE=ON -DIDCS_BUILD_TEST_SURFACE=OFF
cmake --build "$runtime/build/visual-lock" -j2
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
# Rebuild when the hash-bound metadata sidecar is missing too: the binding is only
# written by a build that just ran from the exact ONNX.
if [[ ! -f $engine || $onnx -nt $engine || ! -f $engine.metadata.json ]]; then
    /usr/src/tensorrt/bin/trtexec --onnx="$onnx" --saveEngine="$engine" \
        --memPoolSize=workspace:1024M > "$engine.build.log" 2>&1 \
        || { echo "swarm policy engine build failed; see $engine.build.log" >&2; exit 4; }
    (cd "$runtime" && /home/idcs/Desktop/project/bin/python3 -m tools.swarm_model_metadata \
        "$engine" --built-from "$onnx")
fi
(cd "$runtime" && /home/idcs/Desktop/project/bin/python3 -m tools.swarm_model_metadata "$engine")
# Detector engine with grey letterbox bars: nvinfer pads with black, the model
# was trained with grey; tools/onnx_grey_letterbox.py fixes the bar rows for
# a 1280x720 stream (2026-10-04, migration journal).
src_onnx="$models/yolo26s_dataset2_e100_736.onnx"
grey_onnx="$models/yolo26s_dataset2_e100_736_grey1280x720.onnx"
grey_engine="$models/yolo26s_dataset2_e100_736_grey1280x720.engine"
if [[ -f $src_onnx ]] && [[ ! -f $grey_engine || $src_onnx -nt $grey_engine ]]; then
    PYTHONPATH="$runtime" /home/idcs/Desktop/project/bin/python "$runtime/tools/onnx_grey_letterbox.py" \
        "$src_onnx" "$grey_onnx" --frame 1280x720
    /usr/src/tensorrt/bin/trtexec --onnx="$grey_onnx" --fp16 --saveEngine="$grey_engine" \
        --memPoolSize=workspace:1024M > "$grey_engine.build.log" 2>&1 \
        || { echo "detector engine build failed; see $grey_engine.build.log" >&2; exit 5; }
fi
echo "ready: $runtime (parser built, models -> $(readlink -f "$link"), swarm and detector engines built)"
