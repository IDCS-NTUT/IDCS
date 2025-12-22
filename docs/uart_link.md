# UART link and RS485 paths

## Existing motor-controller link

- The Jetson RS485 bridge for the MKS gimbal motors uses ``/dev/ttyTHS0`` with a 38,400 baud default. See ``gimbal.serial_port`` and ``gimbal.baudrate`` in ``configs/dev.yaml`` and the ``RS485Bus`` initialisation in ``jetson/gimbal_bridge.py`` for the live values.
- The framing for that bus is MKS-specific (``0xFA``/``0xFB`` start bytes, one-byte checksum) and remains isolated from the new Raspberry Pi link below.

## Raspberry Pi framed UART protocol

**Framing**

- Preamble: ``0xAA 0x55``
- Version: ``0x01`` (one byte)
- Message type: one byte (see below)
- Sequence: ``uint16`` big-endian
- Payload length: ``uint16`` big-endian (0–255 bytes)
- Payload: binary payload
- CRC: ``uint16`` CRC-16/X25 computed over ``[version..payload]``

**Message types**

- ``0x01`` — ``HANDSHAKE`` (payload: version ``u8``, heartbeat period ``u16`` ms, role string, capabilities string)
- ``0x02`` — ``HANDSHAKE_ACK`` (payload: version ``u8``, ok flag ``u8``, role string, info string)
- ``0x03`` — ``HEARTBEAT`` (payload: last_error ``u8``, uptime ``u32`` ms)
- ``0x10`` — ``MODE_REQUEST`` (payload: mode ``u8`` where 0=standby, 1=active, 2=diagnostic; reason string)
- ``0x11`` — ``MODE_ACK`` (payload: mode ``u8``, accepted ``u8``, info string)
- ``0x7E`` — ``ERROR_REPORT`` (payload: code ``u8``, info string)

Strings are UTF-8 with a one-byte length prefix (max 255 bytes) to keep framing small and deterministic.

**State machine**

1. **idle** — no link; transmitter schedules a handshake.
2. **sync** — handshake(s) in progress; transition to **active** after receiving ``HANDSHAKE`` or ``HANDSHAKE_ACK``.
3. **active** — periodic heartbeats and mode messages; if no frames arrive before ``heartbeat_timeout_s`` expires, increment the missed counter and fall back to **fault**.
4. **fault** — stop sending heartbeats, raise a log entry, and re-enter **sync** so handshakes can restart.

**Timeouts and retries**

- ``handshake_interval_s`` retries synchronisation without blocking the gimbal RS485 bus.
- ``heartbeat_interval_s`` and ``heartbeat_timeout_s`` gate the health counters and last-heard timestamps for metrics.
- Outbound frames are rate-limited on the Raspberry Pi side to avoid flooding the UART during error storms.

**Metrics and logging**

- Jetson: ``jetson.pi_uart_link`` logs track CRC failures, dropped frames, last-heard timestamps, missed heartbeat counts, and state transitions.
- Raspberry Pi: ``rpi.uart_service`` logs CRC failures and mode changes and applies rate limiting to protect the bus. Its CLI defaults to ``/dev/ttyUSB0`` to match ``rpi/manual_control.py`` for the USB RS485 dongle.
