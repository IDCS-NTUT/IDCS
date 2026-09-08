# TensorRT engine build quick reference

Build TensorRT engines on the target device: serialized plans are specific to
the target TensorRT version and GPU architecture.

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
