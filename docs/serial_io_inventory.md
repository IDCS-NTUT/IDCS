# Serial I/O inventory

## Modules and entry points

| Location | Purpose | Serial port usage |
| --- | --- | --- |
| `common/gimbal/mks_servo42_rs485.py` | Shared RS485 driver for MKS SERVO42D/57D_RS485 controllers, used by Jetson and RPi code. | Opens a `pyserial.Serial` port in `RS485Bus.__post_init__`, performs blocking reads/writes, and implements command framing/parsing and retries. |
| `jetson/gimbal_bridge.py` | Jetson-side bridge that subscribes to `ControlCmd` and emits gimbal motion commands plus telemetry. | Builds `RS485Bus`/`GimbalInterface` for runtime control, parameter writes, encoder reads, and stop/disable sequences. |
| `rpi/manual_control.py` | RPi joystick-driven manual gimbal controller. | Opens an `RS485Bus` and drives the gimbal via `GimbalInterface` while reading joystick ADC. |
| `jetson/tools/test_mks_gimbal_serial.py` | CLI tool to exercise MKS commands from the Jetson. | Opens `RS485Bus` and issues command-specific reads/writes plus raw frame send/receive. |
| `jetson/tools/serial_send_loop.py` | Minimal serial write loop utility. | Opens a `serial.Serial` port and repeatedly writes a text payload. |

## Command types and expected replies

**MKS SERVO42D/57D_RS485 command set (`common/gimbal/mks_servo42_rs485.py`)**

- **F1 (status query)** → expects **1 data byte** status reply. Raises framing error if missing. 【F:common/gimbal/mks_servo42_rs485.py†L1-L13】【F:common/gimbal/mks_servo42_rs485.py†L244-L249】
- **F3 (enable/disable)** → write payload `[0x01|0x00]`, reply optional based on `respond_on_writes`/group writes. 【F:common/gimbal/mks_servo42_rs485.py†L216-L239】
- **F6 (speed mode)** → write payload `[byte4, byte5, acc]`, reply optional based on `respond_on_writes`/group writes. 【F:common/gimbal/mks_servo42_rs485.py†L335-L373】
- **F7 (emergency stop)** → write payload `[]`, reply optional based on `respond_on_writes`/group writes. 【F:common/gimbal/mks_servo42_rs485.py†L251-L259】
- **0x92 (zero axis)** → write payload `[]`, reply optional based on `respond_on_writes`/group writes. 【F:common/gimbal/mks_servo42_rs485.py†L261-L269】
- **0x31 (read encoder counts)** → expects **6 data bytes** (signed 48-bit) reply. 【F:common/gimbal/mks_servo42_rs485.py†L269-L279】
- **0x46 (write all parameters)** → write **34 data bytes**, expects **1 status byte** reply when `respond_on_writes=True`; returns optimistic success when replies are disabled. 【F:common/gimbal/mks_servo42_rs485.py†L282-L317】
- **0x47 (read all parameters)** → expects **34 data bytes** reply. 【F:common/gimbal/mks_servo42_rs485.py†L319-L327】

**Group-addressed commands (`PitchAxisGroup`)**

- **F3/F6/F7/0x92** are sent to the shared `group_addr` with `response_expected=False` and `retries=0` for dual-pitch rigs. 【F:common/gimbal/mks_servo42_rs485.py†L392-L429】

**Jetson bridge command usage (`jetson/gimbal_bridge.py`)**

- Startup sequence: enable axes (F3), zero axes (0x92), read status (F1) for required readiness. 【F:jetson/gimbal_bridge.py†L330-L357】
- Runtime loop: apply `ControlCmd` via speed mode (F6) and read encoder angles (0x31) on telemetry ticks. 【F:jetson/gimbal_bridge.py†L372-L392】【F:common/gimbal/mks_servo42_rs485.py†L528-L569】
- Shutdown: soft stop (F6 → 0 speed), then disable axes (F3 off). 【F:jetson/gimbal_bridge.py†L436-L445】

**RPi manual control (`rpi/manual_control.py`)**

- Startup: enable axes (F3). 【F:rpi/manual_control.py†L188-L236】
- Runtime loop: apply joystick rate commands via speed mode (F6). 【F:rpi/manual_control.py†L284-L299】
- Shutdown: stop (F6 → 0 speed) and disable axes (F3 off). 【F:rpi/manual_control.py†L300-L315】

**Jetson CLI tools**

- `test_mks_gimbal_serial.py` issues the full command set (F1/F3/F6/F7/0x31/0x46/0x47) and supports raw frames with optional response lengths. 【F:jetson/tools/test_mks_gimbal_serial.py†L1-L214】
- `serial_send_loop.py` performs simple periodic writes to any serial port without parsing replies. 【F:jetson/tools/serial_send_loop.py†L1-L60】

## Timeouts, retries, and blocking behavior

- `RS485Bus` opens `serial.Serial(..., timeout=<config>, write_timeout=<config>)` and performs **blocking** reads with `read(size)` / byte-by-byte frame assembly. Timeouts raise `TimeoutError`. 【F:common/gimbal/mks_servo42_rs485.py†L55-L106】
- `RS485Bus.send_command` retries on `TimeoutError`, CRC mismatch, or framing errors; resets serial buffers between attempts. Default retry count is `max_retries` (configurable). 【F:common/gimbal/mks_servo42_rs485.py†L136-L205】
- Jetson config (`configs/dev.yaml`) sets defaults: `timeout: 0.1s` and `retries: 1` for the gimbal. 【F:configs/dev.yaml†L149-L152】
- RPi manual control defaults to `timeout: 0.05s` and `retries: 1`. 【F:rpi/manual_control.py†L68-L71】
- CLI tools allow `--timeout`/`--retries` flags (`test_mks_gimbal_serial.py`) or fixed `timeout=1` (`serial_send_loop.py`). 【F:jetson/tools/test_mks_gimbal_serial.py†L30-L45】【F:jetson/tools/serial_send_loop.py†L44-L46】

## Logging and error handling

- `RS485Bus.send_command` logs a **warning** with command details on timeout/CRC/framing errors and rethrows on final failure. 【F:common/gimbal/mks_servo42_rs485.py†L174-L205】
- `jetson/gimbal_bridge.py` logs warnings for decode failures, serial command suppression, CamState publish failures, and encoder divergence; it raises `SystemExit` on startup status failures. 【F:jetson/gimbal_bridge.py†L322-L420】
- `rpi/manual_control.py` logs warnings for ADC read failures and config parse failures, and logs errors on fatal exceptions. 【F:rpi/manual_control.py†L153-L176】【F:rpi/manual_control.py†L256-L315】
- `serial_send_loop.py` prints errors for serial exceptions and unexpected exceptions. 【F:jetson/tools/serial_send_loop.py†L47-L60】

## Processes that issue serial commands

| Process / runtime | Host | Serial responsibilities |
| --- | --- | --- |
| `jetson/gimbal_bridge.py` | Jetson | Opens `/dev/ttyTHS0` (configurable), handles auto-control speed commands, status queries, parameter writes, encoder telemetry, and stop/disable. 【F:jetson/gimbal_bridge.py†L59-L158】【F:jetson/gimbal_bridge.py†L330-L445】
| `rpi/manual_control.py` | Raspberry Pi | Opens `/dev/ttyUSB0` (configurable) to drive the gimbal via joystick input. 【F:rpi/manual_control.py†L63-L83】【F:rpi/manual_control.py†L188-L315】
| `jetson/tools/test_mks_gimbal_serial.py` | Jetson (CLI) | Manual testing tool that opens the gimbal serial port for diagnostic reads/writes. 【F:jetson/tools/test_mks_gimbal_serial.py†L1-L214】
| `jetson/tools/serial_send_loop.py` | Jetson (CLI) | Generic serial writer for ad-hoc testing. 【F:jetson/tools/serial_send_loop.py†L1-L60】

> **Note:** No PC-side serial I/O is present in the repo; serial usage is currently limited to Jetson/RPi tools and the shared gimbal driver. 【F:common/gimbal/mks_servo42_rs485.py†L55-L205】【F:rpi/manual_control.py†L188-L315】
