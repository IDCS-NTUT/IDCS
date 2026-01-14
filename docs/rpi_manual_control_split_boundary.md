# RPi manual control split boundary (step 2)

## Module A: Manual input / command producer

**Purpose**
- Produce high-level gimbal commands from joystick input without any direct serial access.

**Responsibilities**
- Read joystick inputs via the PCF8591 ADC (I2C) and apply deadzone, scaling, and inversion.
- Clamp rates to configured limits and encode speed payloads for F6 commands.
- Build and publish `SerialUpdate` messages (enable/zero/status/speed/stop/disable as needed).
- Optionally wait for readiness replies (e.g., F1 status) via `SerialReplySubscriber`.
- Handle ADC failure recovery by issuing stop commands and resuming on recovery.
- Parse CLI/config for joystick tuning and serial IPC endpoints/target.

**Explicitly out of scope**
- Opening `/dev/tty*` devices or performing blocking serial reads/writes.
- Retrying/timeout policies or reply parsing beyond readiness checks.

---

## Module B: RPi serial I/O service

**Purpose**
- Own the serial device exclusively and execute blocking serial I/O for all RPi gimbal commands.

**Responsibilities**
- Reuse `tools/serial_io_service.py` with RPi-specific configuration (port, baudrate, timeout, retries).
- Execute the `serial_io.startup` list at service boot (enable/zero/status readiness).
- Run the `serial_io.schedule` for periodic commands (e.g., encoder/status polling).
- Enforce retries/timeouts, validate command payloads, and log warnings/errors.
- Publish data-bearing replies to `SerialReplySubscriber` clients.

**Explicitly out of scope**
- Joystick input, rate mapping, or application-level control logic.
- Any direct UI or user interaction.

---

## Interface between modules

- **Module A → Module B**: `SerialUpdate` PUB (non-blocking), containing command payloads and metadata.
- **Module B → Module A**: `SerialReplyData` PUB (non-blocking), containing parsed reply data when available.

This split keeps serial I/O blocking behavior isolated in Module B while Module A remains non-blocking and focused on joystick logic.
