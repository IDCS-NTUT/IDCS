# Perception and tracking architecture

The replacement pipeline separates four responsibilities that the legacy
`DetectionMsg` currently combines:

1. A detector registers raw boxes and class confidence for one source frame.
2. One tracker assigns identity and lifecycle state to those registrations.
3. A selector assesses tracks and chooses at most one according to explicit policy.
4. A fixed-rate controller consumes the latest valid selection and gimbal
   observation independently of video frame cadence.

`common.perception.PerceptionSnapshotV2` is the strict boundary for steps
1–3. Raw detections cannot carry tracker IDs, tracks cannot contain selector
state, and risk/range assessments are keyed separately by track identity. A
selection records the frame evaluated and the frame where an asynchronous
decision was applied. Every timestamp names its clock domain.

`common.perception` is V2-only. JSON serialization is validated at the V2
transport boundary, and legacy conversion lives separately in
`common.perception_compat`. The DeepStream metadata adapter, selector,
asynchronous worker, GPU OSD, simulation-sidecar ingress, and controller
observation assembler all exchange immutable V2 records. Both control trace
recorders consume the same V2 endpoint; neither reads the display projection.

The runtime publishes native `PerceptionSnapshotV2` on
`net.zmq_perception_v2`. It separately adapts the same snapshot to
`DetectionMsg` on `net.zmq_results` for the existing PC display and simulator
feedback consumers. No core pipeline module reads that legacy message back.
The legacy monolithic `jetson.server` remains a rollback implementation and is
not part of the replacement DeepStream pipeline.

The fixed-rate scheduling boundary accepts `ControlObservation`, not
`DetectionMsg`. The simulation-only sidecar assembles every valid V2 snapshot,
including snapshots with no selection so target loss is observable immediately,
then advances the compatibility controller at the configured control cadence.

## Deterministic verification source

`common.synthetic_perception` produces schema-valid registrations without
running a learned model or renderer. A versioned JSON scenario controls:

- linear normalized target motion and exact class/confidence;
- tracker identity, including deliberate identity changes;
- whole-frame or per-target absence/occlusion;
- dropped, duplicated, delayed, stale, and out-of-order delivery;
- source cadence, observation latency, clock domain, dimensions, and seed.

Canonical scenario content has its own SHA-256 digest for result provenance.

The source emits both raw detections and known-good tracks. Tracker tests use
the raw detections; selector and controller tests use known-good tracks. The
explicit legacy display adapter can feed old downstream code while guaranteeing
that model effectiveness is not part of the result.

The baseline fixture is `tests/fixtures/synthetic_tracking_v1.json`. Any test
that changes its expected sequence must state the fault being exercised and
assert the exact arrival order. Learned detector acceptance remains a separate
test against frozen labeled visual fixtures.
