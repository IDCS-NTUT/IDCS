# Serial I/O service IPC interface

## IPC mechanism

**Choice:** ZeroMQ (ZMQ), aligning with existing system usage.

- **REQ/REP**: command requests and immediate acknowledgements.
- **PUB/SUB**: telemetry updates and command reply broadcasts.
- **PUSH/PULL**: optional bulk update stream (if we need one-way fire-and-forget updates).

> Recommended baseline: **REQ/REP + PUB/SUB**, since ZMQ is already used for `ControlCmd` and `CamState`.

---

## Message schemas (JSON)

All messages are UTF-8 JSON objects. Timestamps are integer milliseconds (`*_ts_ms`).

### 1) Command request (process → I/O service)

Sent over **REQ** (client) → **REP** (service).

```json
{
  "type": "SerialCommandRequest",
  "cmd_id": "c1e4b2a0-3f2a-4e9d-8a47-1d2b7ac2f6d9",
  "source": "jetson.gimbal_bridge",
  "target": "gimbal",
  "func": "F6",
  "addr": 1,
  "payload": [0, 32, 10],
  "expect_reply": false,
  "expected_len": null,
  "priority": "high",
  "schedule_tag": "immediate",
  "timeout_ms": 100,
  "retry": 1,
  "sent_ts_ms": 1727250038
}
```

**Field notes**
- `cmd_id`: client-generated unique ID for correlation.
- `func`: function code as string (e.g., `F6`, `F1`, `0x31`), service maps to byte.
- `payload`: list of bytes (0–255).
- `priority`: `critical | high | normal | low`.
- `schedule_tag`: `immediate | next_round | periodic:<name>`.
- `timeout_ms`/`retry`: optional override; service may clamp to safe ranges.

**REP (ack)**

```json
{
  "type": "SerialCommandAck",
  "cmd_id": "c1e4b2a0-3f2a-4e9d-8a47-1d2b7ac2f6d9",
  "accepted": true,
  "queued": true,
  "queue_position": 2,
  "reason": null,
  "ack_ts_ms": 1727250039
}
```

---

### 2) Telemetry/update (process → I/O service)

Sent over **PUSH** (client) → **PULL** (service) or over **REQ/REP** with a `SerialUpdate` type if we want acknowledgements.

```json
{
  "type": "SerialUpdate",
  "source": "jetson.controller",
  "target": "gimbal",
  "fields": {
    "pan_rate_cmd": -0.35,
    "tilt_rate_cmd": 0.22,
    "yaw_accel_byte": 10,
    "pitch_accel_byte": 10
  },
  "update_ts_ms": 1727250040
}
```

**Field notes**
- `fields`: key/value set used by the I/O service to construct next-round commands.
- Service should enqueue related commands for the **next round** when updates arrive.

---

### 3) Reply publication (I/O service → processes)

Sent over **PUB** (service) → **SUB** (clients).

```json
{
  "type": "SerialCommandResult",
  "cmd_id": "c1e4b2a0-3f2a-4e9d-8a47-1d2b7ac2f6d9",
  "source": "serial_io_service",
  "target": "gimbal",
  "status": "ok",
  "addr": 1,
  "func": "F6",
  "payload": [0, 32, 10],
  "reply": {
    "bytes": [],
    "parsed": null
  },
  "timing": {
    "sent_ts_ms": 1727250038,
    "reply_ts_ms": 1727250042,
    "duration_ms": 4
  },
  "error": null
}
```

**Error example**

```json
{
  "type": "SerialCommandResult",
  "cmd_id": "...",
  "status": "timeout",
  "error": {
    "kind": "TimeoutError",
    "message": "Timeout while reading 6 bytes from RS485 port",
    "retry_count": 1
  }
}
```

---

## Topic naming (PUB/SUB)

Use topic prefixes to allow selective subscriptions:

- `serial.result.<target>` (e.g., `serial.result.gimbal`)
- `serial.telemetry.<target>` (e.g., `serial.telemetry.gimbal`)
- `serial.health` (service health pings and error counters)

---

## Compatibility notes

- Keep JSON schema close to existing `ControlCmd`/`CamState` style for ease of adoption.
- Use `zmq.CONFLATE` or low HWM for telemetry if only latest data is needed.
- Clients should treat `SerialCommandAck.accepted=false` as a soft failure and retry/backoff.
