# Live configuration

The host overlay is `~/.config/idcs/live.yaml`, after the release's ordered
stack. `IDCS_LIVE_CONFIG` overrides its location, principally for isolated tests.
A missing overlay is empty. YAML duplicate keys, unknown paths, incorrect
types, nonfinite numbers and invalid merged combinations are rejected.
There is no cross-host config synchronization: run the CLI on each intended
host, using that host's release and agent. No service copies another host's file.

## Tiers and safe points

| Tier | Application |
|---|---|
| `live` | The owning controller tick, DeepStream metadata probe, selector worker request, or simulator frame applies the validated settings. |
| `idle` | Controller changes wait for a fresh manual panel state, automatic permission off, and no previous nonzero live intent. Missing/stale panel state holds them pending. |
| `restart` | Only affected active units on the explicit restart allow-list are `try-restart`ed, subject to the Jetson motor/engagement veto below. Inactive units stay inactive. |
| `fixed` | Named constants or hash-bound model metadata; authoring/live YAML cannot set these paths. |

The complete [key contract](live_configuration_keys.md) includes every committed
YAML field and supported optional controller overrides. Maps describe structure;
CLI changes address leaves or whole lists. The native tracker profile is
inventoried as restart-only. nvinfer profiles, model paths, labels and their
native contents remain restart-only artifacts; nvinfer handling is unchanged.
Existing typed parsers remain authoritative in addition to the field contract.

Controller gains, feedforward/prediction, idle-return policy, DeepStream shadow
gates/stitching, class-size calibration, planner selection/timing, threat zones,
laser geometry and operator loss grace are live. Controller rate, acceleration,
travel, capture-age, coast and manual caps are idle. Endpoints, GPIO, signs,
serial schedules, Argus settings and model/tracker files are restart.

Some consumers need a restart for a generally live/idle key: simulator legacy
rate caps and DeepStream planner motion caps are constructor settings. The agent
reports that unit-specific outcome. SIM source/viewer dependency changes restart
the active software session together, including its controller, panel and
recorder, as one systemd transaction. No inactive session is started.

## Commands

From the intended release on the intended host:

```sh
python -m tools.idcs_config --host pc get controller.yaw_kp
python -m tools.idcs_config --host pc --actor bench-reviewer set controller.yaw_kp 7.0
python -m tools.idcs_config --host pc set controller.rate_limit_rad_s 0.7
python -m tools.idcs_config --host pc status
python -m tools.idcs_config --host pc diff
python -m tools.idcs_config --host pc unset controller.yaw_kp
python -m tools.idcs_config --host pc revert 1
python -m tools.idcs_config --host pc clear
python -m tools.idcs_config --host pc fold
```

Use `--host jetson` on the Jetson. Use its configured Python environment
(`bin/python3` on Jetson, `venv/bin/python3` on PC). Values are YAML scalars/lists;
quote shell arguments containing spaces. `unset` restores release/default
behavior for that override. `clear` removes all overrides. `revert n` restores
the overlay before the last n completed transactions, including prior reverts.
Every mutation, including revert/clear, is validated against every local unit
stack before persistence. `get` reports effective per-stack values; `diff`
compares those values with the release. `status` includes service receipts,
applied digests, requested overlay digest, pending keys, errors and receipt age.
`awaiting_service` is an unacknowledged update, never a claim of application.

The default command channel and update publication are local IPC sockets next
to the overlay. The PC agent previously had only outbound polling; this adds a
local endpoint without adding an inbound PC network address. The Jetson also
accepts `ConfigCommand` on its existing operator endpoint (REQ clients,
deferred ROUTER replies). `--endpoint`
can select that endpoint; the CLI still validates the local intended release.
The default actor is the local account name, never an inferred email address.

Both agents use one background config worker for their host-local channel;
Jetson panel requests enter that same bounded FIFO. YAML validation, writes,
journal records, restart decisions and CLI receipt waits run there. The owning
loop sends publications and replies in completion order; an in-process worker
notification wakes the Jetson poller without changing its selection period.
REQ payloads and reply JSON are unchanged. The single-writer file lock and
expected-digest checks still apply to every request.

Validated bundles and flattened values (including controller parser defaults)
are cached by overlay digest and the ordered stack fingerprint: unit/module,
resolved file path, size and nanosecond mtime for every stack file. A file-key
change discards both caches; an overlay-key change selects its own validated
entry. The bounded cache retains the release and recent current/candidate
overlays. Unchanged `get` rebuilds neither stacks nor defaults; `status` reads
only the overlay digest and receipts. A warm mutation validates its candidate.

Actors `pi-panel` and `pi-panel:*` return mutation results without collecting or
waiting for service receipts. Affected active hot consumers initially report
`awaiting_service`; inactive/restart/veto outcomes retain their existing meaning.
CLI actors retain their configured receipt wait. While Settings root/group/result
is open, the panel polls cheap `status` every 0.8 seconds after a reply. It loads
full values on open, digest changes, mutation replies and explicit Refresh.
Only later matching, fresh receipts advance application results. Timing evidence
and reviewer commands are in the [follow-up report](panel_config_menu_followup_20261018.md).

## Validation, durability and provenance

`common.config_schema` and the explicit `common/config_fields.json` inventory
define types, finite numeric domains, selected ranges/enums and tiers. No schema
is learned from the candidate config. Full-stack validators include
`ControlRuntimeConfig`, aiming/ranging, DeepStream settings and operator modes.
Live/controller caps cannot exceed either the candidate limits or the release
bridge caps. Physical symmetric travel must fit the bridge span; at application,
a travel change is additionally checked against the controller's measured
origin. Absolute bridge hard limits remain the final authority. Software SIM's
existing 3.0-rad relative envelope remains unchanged.

Writes use a temporary file, fsync and atomic replace. A single-writer lock and
expected-overlay digest prevent lost updates. A stale lock fails closed and
requires operator investigation; it is never removed by a service automatically.
Each `LiveSettings` owns one background worker. It owns the update SUB socket
(including creation and teardown), the 200 ms durable-file fallback poll,
stack loading, YAML parsing, full and partial schema validation, tier splitting,
digesting and pure service-specific parsing. Controller policy/PID coefficients,
aiming/laser objects, DeepStream shadow/stitch policies and selector configuration
are constructed there. The selector's existing process still applies its own
selection revision asynchronously, without reloading its model.

The owning thread's `tick()` makes a nonblocking, constant-time check, then
swaps a complete prepared object and acknowledges it in memory. It performs no
file I/O, polling, parsing, hashing, schema validation, receipts or journal
writes. The measured-origin travel check remains on the controller owner before
any assignment. An engagement/force epoch and base-bundle identity prevent an
outdated candidate from being accepted. The worker invalidates older ready work
when it observes another revision and rechecks the overlay after preparation;
an edit during a slow build discards that build. File-only edits are observed
at the next worker poll; PUB/SUB notifications accelerate this, rather than
providing an atomic transaction with an external file writer.

Idle keys remain held while engaged. An engagement change requests preparation
for that state; reengagement invalidates an idle candidate before it can apply.
Invalid full, partial or typed candidates retain the running settings. `force`
requests asynchronous preparation; it does not wait or load synchronously.
Receipts describe owner-confirmed state. The worker journals applications only
after the owner confirms a successful swap, using the swap's timestamp/digest,
and retries transient persistence failures. Shutdown joins the worker and drains
confirmed applications; socket teardown stays on its worker thread.

Motivation: reviewer measurements on Jetson Orin (2026-10-17) put stack rebuild
plus validation at **64 ms median / 110 ms maximum**, previously on the 60 fps
metadata probe or 50 Hz controller tick (4–7 frames / 3–5 ticks per change).
The former probe's 200 ms overlay poll also cost about **1 ms**. Local follow-up
timing, target remeasurement commands and remaining qualifications are in the
[implementation report](live_configuration_report_20261017.md#follow-up-2026-10-18).

`live-journal.jsonl` is append-only. Each changed override records time, actor,
key, old/new, presence flags, tier, transaction, resulting overlay digest and
affected units. Old/new describe the override (absence is distinct from YAML
null). A final transaction result records per-unit applied/pending/restarted,
inactive, rejected or unavailable outcomes. Service application entries record
later idle transitions with the applied full-stack digest. The journal retains
before/after overlay snapshots for reversions. Individual appends use O_APPEND
and fsync; they never truncate earlier records.

The intent entry is written before the atomic overlay replacement. If the agent
crashes between intent, replacement and result, compare the overlay digest and
receipts with that transaction before continuing. An incomplete transaction is
not included in automatic `revert n`; retain the journal and explicitly restore
the recorded `overlay_before` through reviewed CLI changes. This is detectable
write-ahead evidence, not a multi-file atomic transaction. Directory fsync/power
loss durability needs qualification on the target filesystem.

Controller diagnostics, health and tick/manual/engagement recorder publication
include the applied digest. DeepStream ready/health and recorded V2 snapshots
include the probe digest; `selection_config_digest` separately identifies the
asynchronous selector worker revision. Old-worker results are discarded while
its new revision is pending. Config errors, stale receipts and pending worker
acknowledgements remain visible. A worker failure stops reuse of its decisions;
it cannot silently relabel an old decision with a new revision. Different units
can have different applied digests while an idle/restart change is pending.
Services with no hot consumers report their startup digest in existing logs or
check outputs; Pi PanelState additionally carries its startup digest.

## Fixed policy and model metadata

`common/config_constants.py` records the exact values taken from base commit
`245d103`. Runtime bundles materialize these for existing typed readers, so
effective maps remain identical even though authoring YAML no longer contains
these knobs. Direct algorithm/parser fixtures can still construct typed policy
values; managed startup and live overlays reject their fixed YAML paths.

The committed swarm checkpoint's embedded normalization was independently read
with `pickletools` and matches the constants. `assets/models/swarm/normalization.json`
binds these values to SHA-256 hashes of the checkpoint, ONNX and external tensor
data. Model-derived artifacts carry `<artifact>.metadata.json`. The ONNX exporter
and reviewed Jetson engine build bind normalization to their output hashes.
The engine build checks an existing engine's metadata; it does not invent
provenance for an unrecognized old engine. A missing/mismatched binding rejects
the learned backend, using only its explicitly configured planner fallback.

Immediately after a reviewed engine build from the named ONNX, bind and check:

```sh
bin/python3 -m tools.swarm_model_metadata assets/models/swarm/swarm_policy.engine \
  --built-from assets/models/swarm/swarm_policy.onnx
bin/python3 -m tools.swarm_model_metadata assets/models/swarm/swarm_policy.engine
```

Do not use `--built-from` to bless an engine of unknown origin. Preserve that
engine and its evidence; obtain a reviewed build with known provenance.

## Folding and release gate

`fold` prints an applicable unified patch against the last declaring YAML source
of each affected unit stack. It never writes the repository. Multiple files can
be necessary: for example a controller cap can require both base and
`bench/tuned.yaml` changes. Generated tuning output must be regenerated/reviewed
through its normal procedure, not edited blindly. Optional new keys may require
a structural YAML diff, including visible comment changes. Review every host
variant and compare merged stacks after folding; retain overlay/journal evidence.

Standalone `tools/deployment.py stage` refuses a nonempty target-local overlay
unless `--live-overlay folded` or `--live-overlay discard` is supplied. The stage
record retains the choice, overlay and exact file hash; verification carries the
record forward. The choice records an operator decision; staging does not erase
the overlay or claim that the patch has been folded. Before activating the next
release, clear the overlay through the agent after archiving/reviewing it, then
revalidate the intended release. The same choice is available by invoking the
standalone installer directly; `scripts/deploy.sh` has no new implicit decision.
No deployment was run for this change.

## Safety boundaries and target qualification

The physical bridge and serial service receive no overlay or update subscription.
Their startup schemas and fixed motor-format constants are local-only.
`RESTART_ALLOWED_UNITS` is a host-scoped, explicit allow-list:

- Jetson: `idcs-deepstream-camera.service`, `idcs-deepstream-video.service`,
  `idcs-recorder.service`.
- PC: `idcs-camera-ui.service`, `idcs-ui.service` (display only), `idcs-hud-relay.service`,
  `idcs-sim-streamer.service`, `idcs-sim-swarm-streamer.service`,
  `idcs-sim-ui.service`, `idcs-sim-controller.service`,
  `idcs-sim-swarm-controller.service`, `idcs-sim-panel.service`,
  `idcs-sim-recorder.service`.

Every other unit, including future units, physical/HIL owners and the command
agents, reports `cannot_restart` when an active unit needs a restart. This check
does not rely on sudoers restrictions; the Jetson reviewer has `NOPASSWD: ALL`.
The Jetson agent also vetoes perception restarts when `motors_live()` is true or
an active controller reports tracking/coasting through its existing health
report. Missing, malformed or stale active-controller health fails closed. The
operator-agent motor check refreshes unit status and treats an unsuccessful
refresh as live. Both gates are checked again immediately before restart.
The result is `cannot_restart`, reason `motors live`; the durable overlay stays
pending for a reviewed manual restart. Stopping motion later does not schedule
an automatic retry. Fold/review physical/HIL changes before a bench-authorized
restart. No automatic change modifies
E-stop, panel authority, lamp/fire-control semantics, motor signs or serial
ownership. Physical wiring/RS485 and all existing bench gates remain in force.

Windows tests use deterministic observations, fake systemctl, fake IPC and fake
DeepStream objects. They do not qualify GPU thread ownership, GStreamer, Linux
IPC permissions or real systemd ordering. See the exact target checks and test
counts in the [implementation report](live_configuration_report_20261017.md).

## Pi panel Settings

The Pi operator display adds a Settings page for the approved speed, loss,
detection persistence, engagement-rule and calibration subset. Edits use the
existing operator command channel with digest checks and service receipts;
there is no new GPIO button or cross-host synchronization. Rules and Calibration
require fresh, readable Pi-local fire-control OFF. Numeric proposals use
Cancel-first confirmation; calibration nudges preview live, with Save or one
atomic Discard when leaving. Preview writes remain durable if the display exits.

The fixed UI editing bands, navigation, authority/PC-forwarding findings and
reviewer qualification steps are in the
[panel implementation report](panel_config_menu_report_20261018.md).
Revert all only unsets managed menu leaves and preserves other tuning overrides.
Laser/threat-zone results explicitly report PC not updated; use PC-local CLI for
any separately reviewed PC change.

Atomic coordinated CLI changes can use the same validated transaction path:

```sh
python -m tools.idcs_config --host jetson --actor bench-reviewer set_many \
  '{controller.rate_limit_rad_s: 0.7, controller.manual_rate_limit_rad_s: 0.5}'
python -m tools.idcs_config --host jetson set_many '{}' --unset laser.dir_cam.x laser.offset_m.x
```

`set_many` accepts a mapping of dotted leaf keys plus optional `--unset` keys.
The sets/unsets must be disjoint. All changes validate together before any
persistence, publication or restart handling; failure leaves the existing overlay.
Per-unit rejection after persistence retains the normal pending-overlay behavior.
