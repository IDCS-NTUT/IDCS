# Serial I/O service scheduling behavior

## Scheduling overview

The serial I/O service maintains a **fixed periodic schedule** for recurring commands
(e.g., encoder reads, status checks) and a **next-round queue** for commands that must
run soon after new data arrives (e.g., speed updates).

**Round definition**
- A **round** is one full pass through the service’s command loop.
- A round executes **all queued commands** plus any periodic commands that are due,
  sorted by effective priority.
- The loop runs continuously with a short sleep to keep latency low.

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

## Startup command list

Define one-time startup commands that run **before** the periodic schedule starts:

```yaml
serial_io:
  startup:
    - name: "enable_yaw"
      target: "gimbal"
      func: "F3"
      addr: 1
      payload: [1]
      expect_reply: false
      priority: "critical"
```

Startup commands are processed in order, then the service begins normal
round-based scheduling.

**Behavior**
- Each entry is tracked by `next_due_ts_ms`.
- When `now >= next_due_ts_ms`, the command is added to the **current round**.
- After execution, `next_due_ts_ms = now + interval_ms`.

---

## Next-round update queue behavior

**Trigger**
- When new update data arrives via IPC (`SerialUpdate`), commands are appended to the
  **next-round** queue.

**Rules**
- **F6 coalescing (latest-wins)**: update commands with function `F6` are coalesced by
  `(target, addr, func)`. If another non-critical F6 for the same key is already queued, it is replaced
  with the newest payload before execution.
- **Critical commands are never coalesced**: `F7`, stop/disable commands, and critical-priority
  speed updates are always kept as independent queue entries.
- **Critical preemption**: execution uses effective priority where critical commands are always
  processed before speed updates and periodic reads.

## Staleness filter for speed commands

To prevent delayed execution of obsolete setpoints, `F6` commands can be dropped if too old.

```yaml
serial_io:
  f6_stale_threshold_ms: 120
```

- Applies only to **non-critical** `F6` commands; critical `F6` stop/disable-style commands are never dropped by the stale filter.
- A command is stale when `dispatch_check_ts_ms - sent_ts_ms > f6_stale_threshold_ms`, evaluated immediately before each command is sent.
- `sent_ts_ms` can be provided per-command or at update top-level; missing, `null`, booleans, or other malformed values fall back to ingest time (with warning logs).

## Runtime counters

The service logs backlog mitigation counters in debug logs:

- `coalesced_count`: number of queued `F6` commands replaced by newer updates.
- `dropped_stale_count`: number of stale `F6` commands dropped before send.

These counters are cumulative over the process lifetime.
