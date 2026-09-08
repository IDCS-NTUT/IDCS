# Perception and tracking architecture

The replacement pipeline separates four responsibilities that the legacy
`DetectionMsg` currently combines:

1. A detector registers raw boxes and class confidence for one source frame.
2. One tracker assigns identity and lifecycle state to those registrations.
3. A selector chooses at most one current track according to explicit policy.
4. A fixed-rate controller consumes the latest valid selection and gimbal
   observation independently of video frame cadence.

`common.perception.PerceptionSnapshotV2` is the strict boundary for steps
1–3. Raw detections cannot carry tracker IDs, tracks cannot contain selector
state, and a selection must identify a track present in the same source frame.
Every timestamp names its clock domain.

The legacy `DetectionMsg` remains a compatibility transport during migration.
New logic should use the V2 types internally and adapt only at an old endpoint.

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
legacy adapter can feed current downstream code while guaranteeing that model
effectiveness is not part of the result.

The baseline fixture is `tests/fixtures/synthetic_tracking_v1.json`. Any test
that changes its expected sequence must state the fault being exercised and
assert the exact arrival order. Learned detector acceptance remains a separate
test against frozen labeled visual fixtures.
