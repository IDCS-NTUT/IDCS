# Controlled Verification Strategy

## Principle

Every test must isolate the responsibility under evaluation. A failure in an
upstream learned model, network link, clock, renderer, or device must not
invalidate a test of a downstream component unless that dependency is itself
the subject of the test.

Passing evidence applies only to the boundary exercised by the test. A
TensorRT benchmark is not detector-quality evidence, a detector fixture is not
tracker-identity evidence, and a simulated control trace is not hardware
acceptance.

## Verification ladder

Run tests in this order and stop at the first failing boundary:

1. Pure unit tests with no network, GPU, wall clock, or hardware.
2. Deterministic contract replay using versioned input and golden output.
3. Synthetic component integration with bounded queues and injected faults.
4. Recorded visual or telemetry replay with fixed artifacts and labels.
5. Control-free live transport and perception canary.
6. Hardware-in-the-loop with physical command authority disabled.
7. Bounded unloaded hardware actuation with independent stop protection.
8. Loaded and operational acceptance only after every earlier gate passes.

## Visual perception

Detector effectiveness is tested only with frozen, labeled images or video.
Report recall, precision, localization error, class confusion, target pixel
size, range or scene grouping, and the exact model digest. Synthetic imagery
may supplement these fixtures but cannot replace representative real data.

Tests of tracking, target selection, control, scheduling, or transport must not
wait for a learned detector to recognize a rendered object. Supply
schema-valid synthetic detections or simulator ground-truth boxes with stable,
explicit identities. The injected stream must cover:

- guaranteed target presence and absence;
- deterministic motion and occlusion;
- identity retention, switching, loss, and reacquisition;
- confidence and class changes;
- delayed, dropped, duplicated, stale, and out-of-order observations.

An end-to-end visual simulation remains useful after these isolated gates, but
its result is a separate integration measurement and cannot replace them.

The simulated camera is nevertheless a fidelity target for perception work.
Its projection, resolution, frame cadence, crop/resize path, compression,
latency, target pixel scale, blur, noise, exposure, and occlusion distributions
must be measured against representative real-camera captures. Any known gap
must be reported. Detection conclusions require real labeled replay even when
the simulated camera passes its own camera/transport contract.

## Simulation motion modes

The established `sim.use_jetson_cam_state` item is the authoritative mode
selector:

- `false` selects `stable_substitute`. A conservative simulator-only
  controller may move the versioned simulated plant for system-operation,
  video, perception, tracking, selection, and UI evaluation. Its settings are
  independent of real motor tuning.
- `true` selects `hardware_in_loop`. The simulated camera follows fresh
  physical encoder `CamState`; only the separately authorized tuned live
  controller may move the physical mount. Stale encoder state holds the last
  pose and must never fall back to simulated motion.

The two motion sources are mutually exclusive. Hardware-in-loop must reject a
simulator command endpoint, and the stable simulator controller must reject
hardware-in-loop configuration. No simulation result may select, tune, or
qualify real controller gains. Hardware-in-loop is an integration check of a
controller already tuned and accepted from hardware evidence.

## Control and actuation

Controller tests consume deterministic `ControlObservation` records and
produce golden `ControlIntent` records. Simulations must use a versioned
plant model, explicit delay and encoder cadence, fixed seeds, and bounded
disturbances.

Motor dynamics and motion control may use a stable substitute when they are not
the boundary under test. Real controller tuning and acceptance require real
encoder/command traces and bounded hardware tests; simulated motion is never a
substitute for those gates.

Hardware tests must state the command envelope, duration, travel limits,
timeout or heartbeat behavior, emergency path, serial owner, and mechanical
load. Prefer read-only observation, shadow output, and timed commands before
granting live authority.

## Configuration and provenance

Every retained result must record:

- Git commit and dirty-worktree state;
- resolved configuration and its digest;
- model and plant-model digests;
- input artifact or scenario and random seed;
- host/device identity and relevant runtime versions;
- clock domain for every timestamp;
- output artifact locations and acceptance decision.

Runtime state, readiness files, logs, generated datasets, model exports, and
config-sync markers are artifacts, not source configuration, and must not be
written into or committed from the source tree.
