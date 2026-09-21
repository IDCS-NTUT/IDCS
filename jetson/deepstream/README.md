# DeepStream V2 runtime

`python -m jetson.deepstream.runtime` is the production passive video process.
It runs detector, NvSORT, optional V2 target selection, GPU OSD, H.264 return
video, source-header correlation, and `PerceptionSnapshotV2` publication.

It does not import a controller, create a command socket, open a gimbal, or
access serial hardware.

Validate configuration without opening video or sockets:

```bash
python -m jetson.deepstream.runtime --check
```

Run a bounded canary:

```bash
python -m jetson.deepstream.runtime \
  --duration-s 30 \
  --report artifacts/deepstream/canary.json
```

The config-driven wrapper supplies `--snapshot-result-bind`, `--header-bind`
for RTP sources, return-video settings, and `--target-selection` when enabled.
File replay may also call `jetson.deepstream.verify_pipeline`; it delegates to
the same pipeline implementation.

Runtime reports expose `snapshot_transport`, target-selection, GPU/encoder,
frame-stage, and return-output counters. Evaluate a report with
`python -m jetson.deepstream.acceptance`.
