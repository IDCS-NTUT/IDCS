# Serial I/O scheduling

Current implementation: `tools/serial_io_service.py`, reviewed 2026-10-11.

## Dispatch loop

Each iteration accepts a REQ command if present, drains currently available
PUB updates, enqueues due scheduled commands, publishes an actuation heartbeat
if due, and dispatches **one** command. It then returns to IPC.
It does not execute an entire priority-sorted round before checking new input.

`_pop_next_command` chooses the first emergency command; otherwise it preserves
FIFO/startup ordering. A normal/high label is not a general reorder guarantee.
Pending emergencies discard queued motion/enable commands before dispatch.
An in-progress blocking bus transaction is not asynchronously interrupted.

Periodic entries advance `next_due_ts_ms = now + interval_ms` when enqueued;
missed intervals are not replayed as a catch-up burst. This schedule and legacy
`sent_ts_ms` filtering use wall-clock milliseconds, distinct from monotonic
execution timestamps and controller timing contracts.

## Queue and freshness rules

Non-critical latest-wins motion updates coalesce by target/address/function.
Superseded commands receive an execution outcome. Emergency commands are not
coalesced away. Current compatibility handling includes F5 as well as F6,
although the active bridge uses timed F6.

Immediately before dispatch, non-emergency latest-wins motion is dropped when
its age exceeds `serial_io.f6_stale_threshold_ms` (default 120 ms).
Malformed/missing update timestamps fall back to ingest time with warnings.
Queue admission or successful PUB send is not an execution guarantee.

The update-draining loop runs until no update is ready; it has no fixed
per-iteration count budget. Consequently this document does not claim a hard
end-to-end emergency latency bound under arbitrary publisher overload.
Bounded normal-command timeouts and observed emergency latency are separate
from an operating-system/transport worst-case guarantee.

## Configuration and lifecycle

`serial_io.startup` is the ordered one-time list; the checked-in base contains
stop and status queries, not enable/zero commands. `serial_io.schedule` provides
named target/function/address/payload/reply/interval entries. Overlays replace
lists, so review the complete resulting axis coverage.

Base configuration uses 25 Hz step polling on yaw/pitch-A with slower encoder
cross-checks; the control publisher itself runs at 50 Hz. Serial bus capacity,
write acknowledgements and actual reply latency constrain delivered cadence.

Orderly shutdown closes the bus and cancels pending commands, rather than
sending the queue or an automatic final disable. Bridge shutdown must occur
first. See [lifecycle](serial_io_service_definition.md).

## Observability

Counters distinguish coalescing, stale drops and emergency preemption.
`SerialCommandEventV1` and `SerialActuationStateV1` add correlated terminal
outcomes and recoverable latest F6 state; they are implemented, not a future
scheduling proposal. See [IPC](serial_io_interface_design.md).
