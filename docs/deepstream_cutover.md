# V2 DeepStream validation and deployment

The managed passive runtime owns ingest, source identity, YOLO/NvDCF,
`PerceptionSnapshotV2`, GPU OSD and encoded return. It does not create a
physical control or serial owner.

## Controlled validation

1. Inspect current owners; do not start a second GPU/video pipeline on shared
   source/return/metadata ports.
2. Run `python -m jetson.deepstream.runtime --check` on the intended platform.
3. With the prior software owner deliberately paused, run a bounded canary
   (`--duration-s`) with an absolute `--report` path outside the release tree.
4. Retain a `pc.metadata_monitor` receiver report and evaluate:
   `python -m jetson.deepstream.acceptance REPORT.json --receiver-report PC_REPORT.json`.
5. Check frame/source identity, V2 publications, GPU OSD, encoded buffers and
   expected input/return cadence. Restore and verify the previous software mode.

Verified RTP uses source headers correlated with decoded frames. Local Argus
sensor frame/time integration is implemented and uses same-host timing; it is
not PC clock proof or an optical-to-display latency measurement.
A successful canary does not establish detector precision/recall, long-term
identity retention or physical controller readiness.

## Deployment and rollback

Use [managed deployment](deployment.md), including native/model preparation,
configuration review, platform validation and manifests. A runtime checkout,
native build and active service are different states.

Rollback restores the saved unit/override bytes and previous software owners.
Do not reset a dirty development tree or run old/new pipelines concurrently.
V2 is the perception wire contract; retained compatibility helpers are not a
second deployed detector pipeline.

The camera controller is normally shadow-only. Installing controller/bridge
units cannot grant motor authority; the standing repaired-wiring/RS485 gate
still applies. See [controller architecture](controller_architecture.md).
