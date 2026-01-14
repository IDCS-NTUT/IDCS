# Serial I/O service definition

## Runtime and lifecycle

**Runtime model**
- **Standalone process/service** running on the host that owns the physical serial device (Jetson for `/dev/ttyTHS0`, RPi for `/dev/ttyUSB0`).
- **Single instance per serial bus** (one process per port) to enforce exclusive ownership.
- **Launch method**: started by a **separate launch script on each device** alongside the Jetson server/gimbal process and the RPi stack (service wiring not yet implemented inside those modules).
- Launchable as:
  - `systemd` unit on Jetson/RPi, or
  - a supervised process (e.g., `tmux`/`supervisord`) in development.

**Startup order**
1. **Serial I/O service starts first** and claims the serial port.
2. **Consumer processes start after** the service is healthy (e.g., Jetson server/gimbal, RPi joystick client).
3. If consumers start early, they should retry connecting to the service until ready.

**Shutdown order**
1. **Consumer processes stop first** (stop sending new requests).
2. **Serial I/O service drains queue**, issues configured stop/disable commands, and closes the port.

**Health checks**
- **Liveness**: service responds to a lightweight `ping` IPC message.
- **Readiness**: serial port open + successful basic status query (F1) for required axes.
- **Metrics** (optional): publish counters for timeouts, retries, CRC/framing errors, and last successful command timestamp.

## Responsibilities

**Exclusive serial ownership**
- The service is the only process that opens the serial device.
- All other processes interact via IPC; no direct `pyserial` usage outside the service.

**Blocking serial I/O**
- The service uses blocking read/write semantics for serial to preserve protocol correctness and timeouts.
- IPC layer is non-blocking for clients (requests queued or dropped based on policy).

**Command scheduling**
- Maintain a **fixed schedule** for periodic commands (e.g., encoder reads, status checks).
- Maintain a **next-round queue** for commands triggered by incoming IPC data updates.
- Apply prioritization rules (e.g., estop > stop > speed > telemetry).

**Validation and parsing**
- Validate outbound command payloads (address, function, payload length, value ranges).
- Parse inbound replies, verify CRC, and map to structured data for IPC publication.

**Retries and timeouts**
- Per-command timeout and retry policy configurable (default matches existing `timeout` and `retries`).
- Retry on timeout, CRC mismatch, or framing errors; reset serial buffers between attempts.

**Logging and error handling**
- Log warnings for transient failures with command context.
- Log errors for repeated failures and mark the service as degraded.
- Emit structured error events over IPC for consumers to react (e.g., switch to safe mode).

## Interfaces (high level)

**Inbound (from local processes)**
- **Command requests**: explicit command to enqueue (manual override, estop).
- **Data updates**: new values required for scheduled commands (e.g., updated rate command).

**Outbound (to local processes)**
- **Telemetry publication**: encoder readings, status bytes, error counters.
- **Health status**: readiness/liveness info and last-error info.

## Configuration

- **Serial port**: device path, baudrate, timeout, retries.
- **Gimbal configuration**: addresses, group addresses, `respond_on_writes`.
- **Scheduling**: periodic command intervals, queue policy.
- **Logging**: log level, warning thresholds, error escalation.
