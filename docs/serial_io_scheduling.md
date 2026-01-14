# Serial I/O service scheduling behavior

## Scheduling overview

The serial I/O service maintains a **fixed periodic schedule** for recurring commands
(e.g., encoder reads, status checks) and a **next-round queue** for commands that must
run soon after new data arrives (e.g., speed updates).

**Round definition**
- A **round** is one full pass through the service’s command loop.
- A round executes **all periodic commands that are due** plus any queued “next-round”
  commands, then returns to the top of the loop.
- The loop runs continuously with a short sleep (or poll timeout) to keep latency low.

---

## Periodic schedule configuration

Define periodic commands in configuration with fixed intervals (milliseconds):

```yaml
serial_io:
  schedule:
    - name: "read_status"
      target: "gimbal"
      func: "F1"
      addr: 1
      payload: []
      expect_reply: true
      expected_len: 1
      interval_ms: 1000
      priority: "normal"

    - name: "read_encoder"
      target: "gimbal"
      func: "0x31"
      addr: 1
      payload: []
      expect_reply: true
      expected_len: 6
      interval_ms: 50
      priority: "high"
```

**Behavior**
- Each entry is tracked by `next_due_ts_ms`.
- When `now >= next_due_ts_ms`, the command is added to the **current round**.
- After execution, `next_due_ts_ms += interval_ms` (catch-up policy configurable: skip vs. run back-to-back).

---

## Next-round queue behavior

**Trigger**
- When new data arrives via the IPC update channel (e.g., new `pan_rate_cmd`), the
  service enqueues the **corresponding command** for the **next round**.

**Rules**
- **De-duplication**: if the same command is already queued, replace it with the
  newest payload (latest-wins).
- **Ordering**: next-round commands are inserted ahead of periodic commands with the
  same or lower priority.
- **Priority**: critical/safety commands (e-stop, stop, disable) always execute
  before next-round or periodic commands in the same round.

**Example**
- A new `pan_rate_cmd` update arrives at `t=105 ms`.
- The service places a `F6 speed` command into the **next round**.
- The next loop iteration begins at `t=110 ms` and executes the queued command
  before lower-priority periodic reads.

---

## Round execution order

Recommended order within a round:

1. **Critical queued commands** (e-stop, stop, disable)
2. **High-priority next-round commands** (e.g., speed updates)
3. **Due periodic commands** (status/encoder reads)
4. **Low-priority maintenance commands** (optional)

After execution, the loop publishes replies/telemetry immediately and proceeds to
the next round.
