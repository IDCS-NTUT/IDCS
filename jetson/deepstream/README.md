# DeepStream V2 runtime

`python -m jetson.deepstream.runtime` is the production passive video process:
YOLO nvinfer, NvDCF, asynchronous target selection, GPU OSD, H.264 return,
source-identity correlation and immutable `PerceptionSnapshotV2` publication.
It cannot construct a motor controller or access serial hardware.

RTP sources use verified identity/header joins; local Argus captures sensor
frame/time metadata and joins it by PTS. Sensor-reference timing is not a
complete optical-to-display latency result.

Checked-in nvinfer profiles use checkout-relative asset paths. Runtime generates
a temporary absolute-path profile for that config tree; deploy external
models/parser/native libraries before validation.

Configuration-only check:

```bash
python -m jetson.deepstream.runtime --check
```

For a bounded native canary, first inspect and pause conflicting owners, then
provide `--duration-s 30 --report /absolute/external/artifact/canary.json`.
Do not write reports into an immutable release or stack another pipeline.
File replay via `jetson.deepstream.verify_pipeline` uses the same implementation.

Reports cover snapshot transport, target selection, GPU/encoder and stage/return
counters; `jetson.deepstream.acceptance` evaluates retained runtime/receiver
reports. It is not a detector-quality or hardware acceptance suite.

Known async-selector result-expiry defect and current contracts:
[perception architecture](../../docs/perception_architecture.md).
Operational phases and rollback: [deployment](../../docs/deployment.md).
