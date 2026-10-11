# Serial I/O IPC contract

Current implementation: `common/serial_io/` and
`tools/serial_io_service.py`, reviewed 2026-10-11.

## Transport and authority

- Client REQ -> service REP: `SerialCommandRequest` and immediate
  `SerialCommandAck` queue admission.
- Client PUB -> service SUB: `SerialUpdate` containing already encoded
  `commands`; arbitrary descriptive `fields` do not generate motor commands.
- Service PUB -> clients SUB: replies, terminal command outcomes and latest
  actuation snapshots.

Endpoints come from the service CLI/unit and consumer configuration.
These are trusted-network IPC contracts, not authenticated public APIs.

A successful PUB send means only that the producer handed data to ZeroMQ.
`accepted=true` means queue admission, **not** an RS485 write, firmware ACK
or measured motion.

## Commands

Requests identify `cmd_id`, source/target, function, motor address, byte payload,
expected reply/length, priority and optional timeout/retry overrides. Updates
carry a command list and optional timing/correlation IDs. Validation checks the
command representation; it is not operational motor authority.

Legacy `sent_ts_ms` and schedule times are wall-clock milliseconds.
Execution/reply records also carry same-host monotonic nanoseconds. Preserve
field clock domains; never subtract timestamps from unrelated hosts without a
qualified mapping.

## Replies

`SerialReplyData` includes target, address/function, raw/parsed reply and timing.
F1 status, 0x31 encoder counts, 0x33 microsteps and parameter reads have parsed
data. F3 enable/disable ACKs are deliberately published for bridge readiness.
High-rate F5/F6/F7 and certain initialization-write ACKs are normally suppressed
on this reply channel; explicitly instrumented sweep F6 replies are an exception.

There is no implemented generic `serial.health` topic. Timeouts/errors are
logged and, when enabled, appear as terminal execution outcomes. Do not
subscribe to an invented health channel as a readiness or stop mechanism.

## Execution feedback

With `serial_io.publish_command_events`, each admitted command receives a
terminal `SerialCommandEventV1` on `serial.command.<target>` during the normal
service lifecycle. Outcomes distinguish `wire_sent`, `superseded`,
`preempted`, `stale`, `write_failed`, `wire_uncertain` and `cancelled`.

Events identify service epoch, sequence, update/command ID, raw payload,
write-complete time when known, and admitted/terminal/pending accounting.
`wire_uncertain` means a complete write occurred but confirmation failed;
it is neither definite non-execution nor confirmed motor response.
PUB events can be lost and abrupt process termination cannot guarantee delivery.

With `serial_io.publish_actuation_state`, `SerialActuationStateV1` on
`serial.actuation.<target>` provides recoverable latest per-address F6 payload,
write/expiry time, outcome/confirmation, activity and event sequence. It is
published on relevant changes and at the configured heartbeat.

Replies use `serial.reply.<target>`; emergency timing instrumentation uses
`serial.telemetry.<target>`. Wire state is not mechanically measured pose.
The current HIL renderer uses measured `CamState`, not the removed speculative
wire-predicted pose path.

See [scheduling](serial_io_scheduling.md), [lifecycle](serial_io_service_definition.md)
and the historical [execution-feedback plan](serial_io_execution_feedback_plan.md).
