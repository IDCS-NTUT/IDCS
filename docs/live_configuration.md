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
| `restart` | Only affected active software units are `try-restart`ed. Inactive units stay inactive. |
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
accepts `ConfigCommand` on its existing operator REQ/REP channel. `--endpoint`
can select that endpoint; the CLI still validates the local intended release.
The default actor is the local account name, never an inferred email address.

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
Services subscribe to `config_update` and also inspect the durable file, so an
update missed at subscription/startup is recovered. File fallback polls at most
every 200 ms. Swaps happen in the owning thread; subscriptions are not shared
between the controller tick, probe or agent threads.

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
Their startup schemas and fixed motor-format constants are local-only. Requests
affecting physical/HIL units or the current command agent report `cannot_restart`;
fold/review them before a bench-authorized restart. No automatic change modifies
E-stop, panel authority, lamp/fire-control semantics, motor signs or serial
ownership. Physical wiring/RS485 and all existing bench gates remain in force.

Windows tests use deterministic observations, fake systemctl, fake IPC and fake
DeepStream objects. They do not qualify GPU thread ownership, GStreamer, Linux
IPC permissions or real systemd ordering. See the exact target checks and test
counts in the [implementation report](live_configuration_report_20261017.md).
