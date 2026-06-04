# Serial I/O service IPC interface

## IPC mechanism

**Choice:** ZeroMQ (ZMQ), aligning with existing system usage.

- **REQ/REP**: command enqueue requests and immediate acknowledgements.
- **PUB/SUB**: update publishing into the service and data-bearing reply broadcasts.

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

> Ack responses must include useful queue metadata; do not send acknowledgements that only indicate success/failure.

---

### 2) Telemetry/update (process → I/O service)

Sent over **PUB** (client) → **SUB** (service). Messages are fire-and-forget and processed in a non-blocking loop.

```json
{
  "type": "SerialUpdate",
  "source": "jetson.controller",
  "target": "gimbal",
  "fields": {
    "pan_rate_cmd": -0.35,
    "tilt_rate_cmd": 0.22,
    "pan_accel_cmd": 3.5,
    "tilt_accel_cmd": 3.5,
    "pan_accel_effective_cmd": 3.5,
    "tilt_accel_effective_cmd": 3.5,
    "yaw_accel_byte": 10,
    "pitch_accel_byte": 10
  },
  "commands": [
    {
      "cmd_id": "update:rate:1727250040",
      "func": "F6",
      "addr": 1,
      "payload": [0, 32, 10],
      "expect_reply": false,
      "expected_len": null,
      "priority": "high",
      "target": "gimbal"
    }
  ],
  "update_ts_ms": 1727250040
}
```

**Field notes**
- `fields`: key/value set used by the I/O service to construct next-round commands.
- Service should enqueue `commands` entries for the **next round** when updates arrive.

---

### 3) Reply publication (I/O service → processes)

Sent over **PUB** (service) → **SUB** (clients).

Only **data-bearing replies** are published. Replies that only indicate success/failure are not published; errors are logged and tracked in health telemetry.

```json
{
  "type": "SerialReplyData",
  "cmd_id": "c1e4b2a0-3f2a-4e9d-8a47-1d2b7ac2f6d9",
  "source": "serial_io_service",
  "target": "gimbal",
  "addr": 1,
  "func": "F1",
  "reply": {
    "bytes": [3],
    "parsed": {
      "status": 3,
      "status_label": "Slowing down"
    }
  },
  "timing": {
    "sent_ts_ms": 1727250038,
    "reply_ts_ms": 1727250042,
    "duration_ms": 4
  }
}
```

**Error handling**
- Timeout/CRC/framing failures are **logged** and emitted via health/error counters.
- If consumers need error visibility, subscribe to `serial.health` events rather than receiving empty success/fail responses.

---

## Topic naming (PUB/SUB)

Use topic prefixes to allow selective subscriptions:

- `serial.reply.<target>` (e.g., `serial.reply.gimbal`)
- `serial.telemetry.<target>` (e.g., `serial.telemetry.gimbal`)
- `serial.health` (service health pings and error counters)

---

## Compatibility notes

- Keep JSON schema close to existing `ControlCmd`/`CamState` style for ease of adoption.
- Use `zmq.CONFLATE` or low HWM for telemetry if only latest data is needed.
- Clients should treat `SerialCommandAck.accepted=false` as a soft failure and retry/backoff.
