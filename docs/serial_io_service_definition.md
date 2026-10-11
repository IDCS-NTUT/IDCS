# Serial I/O service: current lifecycle

Cross-checked against `tools/serial_io_service.py` on 2026-10-11.
This describes code, not authorization to start hardware.

## Ownership

The managed Jetson `idcs-serial.service` is the sole physical RS485 owner,
configured for `/dev/ttyCH341USB0`. `gimbal_bridge` submits serial IPC commands
and consumes replies; the production Pi `rpi.runtime_control` publishes panel
state only. Direct-bus diagnostic tools and legacy command producers still
exist, but must never run alongside managed bus/control owners.

The standing repaired-wiring/all-motors-answer gate in `AGENTS.md` applies.
Installing a unit or passing `--check` does not authorize activation.

## Startup and readiness

The service loads configuration, creates IPC sockets, opens `RS485Bus`, and
queues configured startup commands before recurring work. The checked-in base
startup list is stop/query, not automatic motor enable, calibration or zeroing.
Enable/ACK checks and separately authorized initialization belong to the bridge.

`--check` strictly parses every requested configuration layer, rejects missing
overlays and duplicate keys, lists the startup/schedule entries, then exits
without opening sockets or serial. It neither probes hardware nor confirms
all axes are ready. See the R1 follow-up in the
[review](codebase_review_20261011.md).

A command enqueue acknowledgement proves admission, not port health, a firmware
reply or physical movement. There is no implemented general `serial.health`
publication or successful ping-based motor-readiness contract.

## Runtime and shutdown

One selected command is dispatched per loop iteration. Reads/writes are
blocking, with bounded per-command handling; the loop returns to IPC between
commands. Emergency preemption, latest-wins motion and stale filtering are
described in [scheduling](serial_io_scheduling.md).

On orderly stop, the serial context closes and remaining queued commands receive
cancelled terminal events. **The service does not drain the queue to hardware
or automatically send stop/disable on shutdown.** The bridge owns its stop and
release sequence and must stop while serial IPC is still available. Firmware
timed-command expiry provides another boundary; abrupt crashes must not be
represented as a guaranteed graceful shutdown.

Reply timing, terminal execution events and latest actuation snapshots are
implemented. They distinguish admission, wire transmission and reply outcomes;
none proves mechanically correct movement. See [IPC](serial_io_interface_design.md)
and [inventory](serial_io_inventory.md).
