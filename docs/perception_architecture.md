# Perception and control architecture

## Authoritative contracts

`PerceptionSnapshotV2` is the only detector/tracker/selector wire contract.
It separates raw detections, tracked identities, per-track assessments, and a
single selection tied to the frame where the decision was applied. Every
timestamp names its clock domain.

`ControlObservation` is the fixed-rate controller input. Its target component
contains complete geometry: target centre, aim reference, raw pixel error,
signed bearing error, range and range provenance, and parallax state. A valid
target cannot omit those values. The controller therefore never reconstructs
pixels from angular error.

`ControlIntent` is the bounded policy result used by simulator and replay
qualification. `ControlCmd` remains the command/display wire shared with the
gimbal bridge and host HUD.

## Ownership

1. DeepStream registers objects and NvSORT assigns track identities.
2. `DeepStreamTargetSelector` adds known-size range and policy assessments and
   chooses at most one tracked identity.
3. `SnapshotTransport` correlates optional source headers and publishes the
   immutable V2 snapshot. It has no control socket.
4. `ControlObservationAssembler` combines the latest V2 snapshot, CamState,
   manual authority, and local freshness into one atomic observation.
5. `ControlLoop` runs PID or MPC at a fixed monotonic cadence. Missing or stale
   authority is a zero-rate disarmed state.
6. `gimbal_bridge` consumes commands and publishes encoder-derived CamState;
   it is separate from perception and control policy.

The removed monolithic server, mutable detection schema, CPU YOLO path,
standalone tracker, legacy result transport, and duplicate shadow controller
sidecars are available only through git history.

## Deterministic verification

`common.synthetic_perception` produces schema-valid detections, tracks,
assessments, selection, identity switches, occlusions, delays, duplicates,
and out-of-order delivery without invoking a learned model. This is the
preferred source for transport, selection, observation, and policy tests.

Rendered OpenGL scenes are used when the detector itself is under test. The
target mesh keeps its configured physical size so known-size ranging and the
parallax projection are not made artificially easy. Model effectiveness is
measured separately from tracker, UI, or controller behavior.
