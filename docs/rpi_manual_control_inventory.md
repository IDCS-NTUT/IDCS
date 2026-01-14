# RPi manual control inventory (step 1)

## Current responsibilities in `rpi/manual_control.py`

### Joystick / ADC polling
- Owns I2C access to the PCF8591 ADC via `smbus` and reads both axes (`read_adc`).
- Performs deadzone filtering and rate mapping (`map_value_to_rate`) to convert joystick input into signed rad/s commands.

### Command encoding
- Encodes MKS speed payloads using `MksServo42Axis._encode_speed_payload`.
- Builds `SerialUpdate` payloads (commands + metadata) for enable, zero, status, speed, stop, and disable commands.

### Serial updates (IPC)
- Publishes `SerialUpdate` messages to the serial I/O service using `SerialUpdatePublisher`.
- Uses `SerialReplySubscriber` to consume `SerialReplyData` for status checks.

### Reply handling
- Waits for F1 status replies at startup and validates that each axis responds with a non-zero status.
- Logs a timeout and exits if status replies are not received in time.

### Startup sequence
- Sends enable commands for yaw/pitch.
- Sends zeroing commands for yaw/pitch.
- Sends initial F1 status queries and blocks until they are received (or timeout).

### Runtime control loop
- Continuously reads joystick inputs, converts to rad/s, applies optional inversion, and publishes speed commands.
- Logs joystick values and commanded rates periodically.
- On ADC read errors, sends stop commands and waits until ADC becomes responsive again.

### Stop / shutdown sequence
- On exit, sends stop commands and disable commands for both yaw and pitch.
- Closes the reply subscriber and update publisher.

### Config parsing
- CLI argument parsing covers serial endpoints/target and gimbal addressing parameters.
- Optional YAML config allows early exit if `gimbal.auto_control_enabled` is true.

---

## Proposed split: input/logic process vs. serial I/O process

### Input/logic process (new RPi manual input module)
- Joystick/ADC polling and rate mapping (deadzone, inversion, clamp).
- Encodes speed payloads (F6) and publishes `SerialUpdate` messages.
- Publishes startup requests (enable/zero/status) as updates **or** relies on serial I/O service startup list.
- Consumes status replies for readiness checks.
- Handles ADC error recovery (send stop commands, resume after ADC is back).
- Owns CLI/config parsing for joystick settings and serial endpoints/target.

### Serial I/O process (dedicated service)
- Owns serial device access, blocking reads/writes, and retry/timeout policy.
- Executes startup list (enable/zero/status) on service boot.
- Runs periodic schedule (e.g., encoder/status polling).
- Publishes reply data to subscribers and logs errors/timeouts.

---

## Notes for the split
- The input/logic module should no longer import or use any direct serial bus objects.
- Startup sequencing can move into the serial I/O service’s `serial_io.startup` config so the input process is non-blocking at boot.
