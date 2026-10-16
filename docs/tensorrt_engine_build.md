# TensorRT engine build quick reference

Build TensorRT engines on the target device: serialized plans are specific to
the target TensorRT version and GPU architecture.

## Production (Jetson)

`scripts/prepare_jetson_runtime.sh` builds both production engines:

- the swarm policy (`assets/models/swarm/swarm_policy.engine`) from its
  committed ONNX;
- the detector, `yolo26s_dataset2_e100_736_grey1280x720.engine`
  (FP16):
  - `tools/onnx_grey_letterbox.py` first prepends a fixed mask to the ONNX, so
    the engine sees grey (114) letterbox bars where nvinfer pads with black.
  - The model was trained with grey bars; black bars cost about 7 percentage
    points of drone detection on the recorded fast-target clip
    ([journal](deepstream_migration_journal.md), 2026-10-04).
  - The engine is valid only for 1280x720 input with
    `maintain-aspect-ratio=1` and `symmetric-padding=1`.
  - It is used by
    `configs/deepstream/nvinfer_yolo26s_736_drone_person_smoke.txt` and
    `jetson/deepstream/preflight.py`.

The script builds missing engines and rebuilds when the source ONNX is newer
than the engine. A different frame size requires a matching mask/engine and
configuration; the script currently hard-codes `--frame 1280x720`.

## Generic commands

```bash
trtexec \
  --onnx=best.onnx \
  --saveEngine=best.engine \
  --explicitBatch \
  --fp16 \
  --workspace=4096
```

For Ultralytics export with embedded NMS:

```bash
yolo export model=best.pt format=engine half=true workspace=4096 nms=true \
  conf=0.25 iou=0.45
```

Use the current DeepStream migration journal for the verified Jetson-specific
commands and benchmark evidence; this page is a compact command reminder.
