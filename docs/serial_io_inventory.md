# Serial I/O inventory

Cross-checked against the repository on 2026-10-17.

| Module / managed owner | Responsibility |
|---|---|
| `common/gimbal/mks_servo42_rs485.py` | Physical framing, checksums, blocking serial transport and motor command encoding |
| `tools/serial_io_service.py` / Jetson `idcs-serial` | Managed bus owner, command queue, timed replies, execution events |
| `jetson/gimbal_bridge.py` / `idcs-bridge` | Consumes `ControlIntent`, verifies authority/feedback, sends timed F6 through IPC, publishes measured `CamState` |
| `rpi/runtime_control.py` / Pi `idcs-manual` | Panel state and lamps; no serial commands |
| `rpi/manual_control.py` | Shared GPIO/ADC helpers used by the panel runtime; the legacy serial entry point was removed |
| `tools/` and `jetson/tools/` bench/identification utilities | May use serial IPC or direct bus access for isolated tests; not concurrent production owners |

`jetson/tools/test_mks_gimbal_serial.py` and
`jetson/tools/serial_send_loop.py` remain direct-serial diagnostic entry points,
not managed services. Their example ports/baud rates are not deployment
authority; never run them concurrently with the serial owner or as harmless
status probes. Do not infer installed services from their presence.

## Relevant protocol use

| Function | Active role |
|---|---|
| F1 | Motor status reply |
| F3 | Enable/disable; bridge uses the returned ACK to check enablement |
| F6 | Timed rate command; bridge supplies runtime expiry and refreshes it |
| F7 | Emergency stop |
| 0x33 | Raw microstep count; default control-position source |
| 0x31 | Encoder count; independent slower cross-check, or configured alternate pose source |
| 0x92 / parameter commands | Explicit initialization/bench operations, not normal serial-service startup |

Base scheduling polls yaw and pitch-A steps every 40 ms, pitch-B every 100 ms,
and encoders every 1000 ms; ordered bench overlays can replace this schedule
and omit uncoupled pitch-B. The bridge publishes at 50 Hz, which does **not**
mean every publication contains a new simultaneous two-axis measurement.

Per-axis measured timestamps must be retained through camera-pose interpolation.
Command write time, reply receipt time and physical sensor-sampling time are
different quantities. Raw steps are firmware position information, not proof
of no lost steps or mechanical alignment; encoder disagreement must be analyzed
against the actual coupling/load.

Read [lifecycle](serial_io_service_definition.md),
[scheduling](serial_io_scheduling.md), [IPC](serial_io_interface_design.md) and
[controller architecture](controller_architecture.md). Hardware remains subject
to the repaired-wiring/RS485 acceptance gate.
