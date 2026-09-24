# Serial execution feedback rework plan

## Purpose

The serial service must distinguish these three facts:

1. a producer published a command;
2. the service admitted or queued that command; and
3. the service actually wrote that command to the RS485 bus.

Only the third fact is authoritative for actuator-state prediction. The current
`SerialUpdatePublisher.send_update()` result establishes only local ZeroMQ
publication. A `SerialCommandAck` establishes only queue admission. Neither
means the motor received a frame.

This distinction became visible in the encoder-anchored HIL renderer. The
bridge integrated commands after publication, while the serial service could
later coalesce, expire, or deliberately discard them when a critical stop was
pending. The safety behavior was correct; the predicted render pose was not.

The rework adds execution truth without weakening emergency preemption,
changing encoder authority, or putting blocking serial work in the controller.

## Implementation status

Implemented in shadow mode on 2026-09-24:

- terminal lifecycle events and per-address actuation snapshots;
- boot epochs, monotonic event/snapshot sequences, and admission/terminal
  accounting;
- distinct pre-write and write-complete timestamps while preserving the
  existing request-to-wire timing boundary;
- explicit uncertain-wire handling when a write completed but its reply failed;
- bridge correlation by `update_id`/`cmd_id`, timed-command expiry, gap/restart
  fallback, and fresh-encoder recovery;
- publication-driven rendering retained as the selected source while the
  wire-execution predictor runs in shadow;
- a standalone execution audit recorder/report; and
- deterministic coalescing, preemption, stale/write outcome, expiry,
  sequence-gap, uncertain-write, and alternating motion/stop coverage.

Offline qualification passed the complete 342-test and 12-subtest suite. The
remaining work is the bounded shadow hardware evidence in Phase 4, followed by
the explicit configuration switch from `publication` to `wire_execution` if
all gates pass.

## Existing qualified behavior to preserve

The following work is already implemented and must remain intact:

- one serial-service process exclusively owns each serial bus;
- periodic status/encoder polling and asynchronous `SerialUpdate` ingestion;
- latest-wins F6 coalescing and stale-command rejection;
- one-transaction-at-a-time emergency arbitration;
- critical F7, zero-speed F6, and disable commands preempt queued motion;
- absolute receive deadlines and bounded timeout/retry overrides;
- an acknowledged REQ/REP lane for emergency requests;
- measured same-host emergency request-to-wire latency below the 25 ms gate;
- monotonic enqueue, execution, wire, and reply timestamps for published
  data-bearing replies and emergency timing;
- 38,400-baud operation on the deployed three-motor bus;
- timed F6 commands, currently refreshed before their firmware runtime expires;
- encoder-authoritative controller limits and fault handling; and
- render-only command integration re-anchored by cumulative `0x31` encoder
  counts.

The migration journal remains the evidence record for the emergency-lane,
baud-rate, wire-timestamp, system-identification, and HIL trials. This document
defines the next change rather than superseding those results.

## Observed failure mode

In the corrected 20-second HIL trial, the controller emitted 931 intents, of
which 814 were `target_invalid`. Transitions to zero produced critical stop
commands. The serial service reported seven emergency-preemption events that
discarded 12 queued motion/enable commands.

The render predictor had already integrated some of those commands because
the bridge interpreted successful PUB delivery as execution. The maximum pan
reconciliation was `0.01915 rad`, approximately 50 encoder counts or 92 ms at
the quantized 2 RPM test rate. That is consistent with one predicted command
window that was never executed. The current correction later returned to zero,
and the mean pan correction remained `0.00147 rad`; this does not look like
persistent motor step loss. The pitch maximum, `0.00038 rad`, was one encoder
count.

There are two additional execution details the current predictor cannot know:

- a write may reach the wire but its reply may be lost; and
- a timed F6 command stops in firmware when its runtime expires if no refresh
  reaches the wire.

## Target architecture

Keep the existing request, update, reply, and emergency timing interfaces.
Add two backward-compatible outbound contracts from the serial service.

### 1. Command lifecycle events

Publish `SerialCommandEventV1` on `serial.command.<target>`. Each event contains:

```json
{
  "type": "SerialCommandEventV1",
  "version": 1,
  "service_epoch": "boot-unique-id",
  "sequence": 42,
  "target": "gimbal",
  "source": "serial_io_service",
  "update_id": "intent:1234",
  "cmd_id": "intent:yaw:1234",
  "addr": 1,
  "func": "F6",
  "payload": [0, 2, 10, 0, 0, 0, 10],
  "event": "wire_sent",
  "terminal": true,
  "reason": null,
  "related_cmd_id": null,
  "timing": {
    "ingest_monotonic_ns": 1000000000,
    "execute_start_monotonic_ns": 1002000000,
    "wire_monotonic_ns": 1003000000,
    "event_monotonic_ns": 1004000000
  }
}
```

Required terminal dispositions are:

- `wire_sent`: the frame write completed; include the authoritative wire time;
- `superseded`: a newer command replaced it during coalescing;
- `preempted`: a pending emergency removed it;
- `stale`: it exceeded the F6 age limit before dispatch;
- `write_failed`: the service knows no complete frame was written; and
- `wire_uncertain`: a transaction failed after a complete write but before
  reply confirmation; and
- `cancelled`: the service shut down before dispatching the admitted command.

Reply success or timeout is a separate follow-up event for commands that expect
a reply. A missing reply must not be mislabeled as a missing write. Every
admitted command must reach exactly one terminal disposition, and every event
must carry a monotonically increasing service sequence plus a boot-unique
epoch so consumers can detect restart and gaps.

`update_id` correlates the yaw, pitch-A, and pitch-B commands derived from one
control intent. `related_cmd_id` identifies the replacing command or the
emergency command that caused a terminal drop.

### 2. Recoverable actuation snapshot

Publish `SerialActuationStateV1` on `serial.actuation.<target>` after each F6
wire write and periodically while any timed command remains active. It contains
the latest wire-written F6 command for each motor address, its wire timestamp,
payload, firmware expiry time, confirmation state, and lifecycle sequence.

This snapshot serves two purposes:

- a subscriber that misses a PUB event can recover without waiting for an
  encoder correction; and
- a bridge or recorder can detect service restart, stale state, and firmware
  timer expiry deterministically.

The service remains actuator-semantic-neutral: it reports address, raw payload,
and timing. The gimbal bridge owns motor sign, gear ratio, camera-axis mapping,
and conversion to camera-frame rate.

## Service changes

1. Add `update_id` to `SerialUpdate` and retain it on every decoded command.
   Existing producers that omit it remain valid; the service generates a local
   correlation ID when necessary.
2. Centralize queue removal so coalescing, emergency preemption, staleness, and
   shutdown each emit terminal outcomes instead of only incrementing counters.
3. Refactor command execution to distinguish frame-write completion from reply
   validation. Preserve `last_tx_monotonic_ns`; expose whether an exception
   occurred before the write, after a possible write, or during reply wait.
4. Emit `wire_sent` immediately after a complete write when practical. If the
   current blocking driver cannot publish until reply processing finishes,
   retain the true wire timestamp and treat that delivery delay as a measured
   migration metric before splitting write/read phases.
5. Maintain per-target/per-address F6 actuation snapshots. Parse only protocol
   facts needed for expiry; do not embed camera coordinate conventions in the
   service.
6. Publish lifecycle and snapshot health counters: admitted, wire-sent,
   superseded, preempted, stale, failed, uncertain, event sequence gaps, and
   oldest queue age.
7. Preserve emergency ordering and the existing 25 ms request-to-wire gate.
   Execution telemetry must never delay an emergency write. Use non-blocking
   publication, count event-send failures, and rely on sequence-gap detection
   plus the actuation snapshot for recovery from PUB back-pressure.

## Gimbal bridge changes

1. Give every live intent a stable `update_id` and retain a bounded pending map
   from `cmd_id` to the already bounded and firmware-quantized camera rate.
2. Stop updating the render predictor from `send_update()` success.
3. Apply a rate only on the authority motor's `wire_sent` event, using the
   event's wire timestamp. Yaw follows the yaw address; pitch follows the
   configured pitch-authority address. The second pitch motor remains visible
   for divergence diagnostics but must not double-apply pitch motion.
4. Remove pending entries on `superseded`, `preempted`, `stale`, or definite
   `write_failed`. Treat `wire_uncertain`, event-sequence gaps, epoch changes,
   and stale snapshots as degraded prediction.
5. Model timed-command expiry: predicted rate becomes zero at
   `wire_monotonic_ns + runtime_ms` unless a newer wire-sent command refreshes
   that axis.
6. Continue re-anchoring from `0x31` encoder counts. Raw `CamState.pan/tilt`
   remain the only controller and safety authority.
7. On degraded execution feedback, publish no render prediction and let the
   streamer fall back to the measured pose pair. Recovery requires a fresh
   actuation snapshot plus fresh encoder anchors for both axes.

## Configuration and rollout controls

Introduce explicit settings with conservative defaults:

```yaml
serial_io:
  publish_command_events: false
  publish_actuation_state: false
  actuation_state_heartbeat_ms: 50

gimbal:
  render_prediction:
    source: "disabled"  # disabled | publication | wire_execution
    shadow_wire_execution: false
    execution_stale_ms: 100
    require_fresh_encoder_anchors: true
```

The new settings default off while the contracts are introduced. During
migration, `source: publication` preserves the existing renderer and
`shadow_wire_execution: true` computes and records the execution-backed result
without selecting it. The publication-driven mode is removed after
wire-execution qualification.

The repository's qualified control configuration now enables command events
and actuation snapshots, selects `source: publication`, and enables wire shadow
calculation. Thus telemetry is available for the bounded canary without
changing the active rendered pose.

## Implementation sequence

### Phase 1 — Contracts and deterministic service tests

- Add lifecycle/snapshot schemas and topic helpers.
- Refactor queue mutations to return explicit command outcomes.
- Add service epoch and monotonic event sequence.
- Do not change bridge behavior yet.

Exit gate: every admitted fake-bus command has exactly one terminal outcome in
coalescing, emergency, stale, normal-write, pre-write failure, and uncertain
post-write cases. Existing emergency arbitration tests remain unchanged.

### Phase 2 — Shadow execution feedback

- Publish events and snapshots from the real service behind config flags.
- Add a recorder/report that reconciles updates, outcomes, wire writes,
  replies, and encoder samples by ID and time.
- Run fake-serial and loopback saturation tests before hardware.

Exit gate: 100% command accounting, no duplicate terminal outcomes, no event
sequence gaps, and no regression in emergency request-to-wire latency.

### Phase 3 — Execution-backed render prediction

- Make the bridge consume wire events and snapshots.
- Add firmware-expiry handling and fail-closed measured-pose fallback.
- Run both predictors in shadow and compare them against encoder anchors.

Exit gate: a deterministic alternating tracking/target-loss test shows zero
phantom displacement from preempted commands. Missing-event and service-restart
tests fall back to measured pose within two 50 Hz controller periods.

### Phase 4 — Bounded hardware qualification

Use the established process/port/TTY preflight, unloaded gimbal, synthetic
target input, calibration disabled, encoder zero disabled, `0.2 rad/s` cap,
timed commands, and automatic cleanup.

Run these cases separately:

1. zero-only command lifecycle canary;
2. constant-rate yaw with no target-loss transitions;
3. deliberate rapid motion/stop alternation to exercise preemption;
4. moving synthetic target with the full 50 Hz controller and HIL renderer;
5. event-loss/service-restart injection on a fake bus only.

Acceptance gates:

- 100% of admitted commands receive a terminal disposition;
- no preempted/superseded/stale command changes predicted rate;
- each wire-sent authority-axis command changes prediction exactly once;
- timed prediction stops at firmware expiry when refresh is absent;
- no lifecycle sequence gaps during the bounded hardware trial;
- wire-event-to-`CamState` p99 is at most one 20 ms bridge period;
- the existing 25 ms emergency request-to-wire gate still passes;
- steady-rate encoder reconciliation is reported in encoder counts and radians,
  separately from acceleration/reversal transients;
- the prior approximately 50-count publication/execution mismatch does not
  recur; and
- all motor-facing processes and the TTY owner are absent after cleanup.

Do not set a new controller gain or plant parameter from this HIL render test.
If significant corrections remain after execution truth is established,
classify them against the independently qualified plant response before adding
an optional render-only acceleration model.

## Compatibility and removal

- Existing `SerialReplyData`, `SerialEmergencyTiming`, REQ/REP command ACKs,
  and `SerialUpdate` producers remain valid during migration.
- Queue-admission ACKs retain their current meaning and are never renamed to
  imply execution.
- Consumers that do not subscribe to the new topics are unaffected.
- Remove publication-driven prediction only after Phase 4 passes and the
  journal records the evidence artifact and commit.
- Do not remove emergency discarding, F6 coalescing, stale filtering, or
  encoder-authoritative safety behavior as part of this rework.
