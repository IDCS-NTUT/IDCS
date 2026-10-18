# Shared configuration and message safety, 2026-10-18

Base: `8caf074` (`lab/panel-config-menu-followup`); branch
`codex/config-cross-host`. Implementation and verification are local only.
No lab hosts, units, hardware interfaces, models or recordings were used or
changed. No deployment, push, tag or pull request was performed.

## Design as built

`common/shared_config.py` wraps the existing local `ConfigAgent` in both
`jetson.operator_agent` and `pc.mode_agent`. `shared_keys()` derives ownership
from `unit_stacks()` and the existing `uses_key()` consumer table, including
inactive managed units. No laser/threat key list determines ownership. Changes
to shared leaves go through Jetson; single-host leaves use the existing local
engine. Ordered release stacks and their host-specific startup settings remain
release configuration; Jetson owns the shared live-overlay layer.

`net.zmq_config_link`, declared in `configs/base/network.yaml` and the restart
schema, is `tcp://192.168.0.5:5598`. PC connects with DEALER; Jetson binds ROUTER.
There is no new inbound PC port. The dedicated link carries ConfigCommand and
ConfigReply dictionaries with phase, transaction, expected_digest, shared values,
and per-host/per-unit results. PC-originated batches also carry pc_digest,
pc_candidate and adopt when needed; prepare sends shared, expected_digest and
transaction; commit returns units/digest and carries coordinator results; report
carries the PC result/shared_digest. Forwarding checks the last reconciled
Jetson digest independently from PC's digest. Its RPC envelope has id/request or
reply_to/response. Phases are prepare, commit, abort, snapshot, report and forward.
Heartbeat packets carry no configuration. Configuration never travels inside
PcModeTask or PC_ACTIONS. PcModePoll only adds a boolean config_ready so an
unreconciled PC cannot advertise mode readiness.

Each link owns its socket on a separate I/O thread. A bounded handler queue (32)
and output queue (64) isolate validation, filesystem work and RPC waits. RPC
waits are bounded at two seconds. Heartbeats continue during slow validation,
and a full disconnected send queue cannot kill the reconnecting PC thread.
The existing AgentChannel worker and owner-only reply/publication path remain.
Forwarding releases PC's locks before waiting, allowing Jetson's prepare to
reach PC while the CLI worker waits for its coordinated reply.

For online changes, Jetson validates its candidate and PC validates its candidate
against all its managed stacks, using different expected overlay digests. PC
persists its prepared plan without changing live.yaml; local mutations are
refused while prepared. Only after both validations succeed does Jetson persist
a commit decision, publish its overlay and commit PC's prepared plan. Each host
keeps its restart allow-list, SIM dependency expansion and motor/engagement
vetoes. Mixed PC-local/shared batches preserve the PC-local candidate in the
same transaction. Validation failures and prepare timeouts write neither overlay.

A failed/lost commit response is an uncertain committed transaction, not an
abort: it is shown as commit_pending and blocks further shared mutations and
staging. Durable decisions/prepared plans survive agent restarts; reconnect
repeats commit idempotently. A crash after the atomic overlay write is recognized
by its resulting digest and does not repeat a restart. A prepared transaction
whose coordinator never decided to commit is aborted during reconciliation.
The files are published sequentially; distributed readers do not observe a
simultaneous filesystem switch. An interrupted commit converges to the durable
decision instead of falsely reporting all-or-none failure.

If PC is offline before prepare, Jetson commits and reports PC pending.
Reconnection sends Jetson's shared overlay before PC reports ready. PC validates
before applying; rejection retains PC's previous values and is mirrored into
Jetson status/panel results. The original transaction ID is retained across an
offline reconciliation. Both journals record the transaction and host/unit
results. PC service receipts add shared_config_ready/shared_config_mismatch;
readiness also checks the actual overlay against its reconciled projection and
fresh reconciliation evidence.

An existing unrecognized PC shared overlay is logged at startup and returned in
status.shared.conflicts. It is retained until an explicit CLI decision:

```sh
venv/bin/python3 -m tools.idcs_config --host pc status
venv/bin/python3 -m tools.idcs_config --host pc adopt_shared
venv/bin/python3 -m tools.idcs_config --host pc drop_shared
```

Adoption forwards the values to Jetson and validates on both hosts; dropping
removes the PC shared overrides and then normal reconciliation applies Jetson's
values. These are reviewer/operator actions, not commands executed for this task.
Internal install/phase/candidate fields are refused on local IPC. Phase handlers
are available only through the dedicated config link.

Runtime state is `config-shared.json` beside live.yaml, outside the release tree.
It contains the key inventory, reconciled projection, host results, prepared plan
and durable decision. Staging on every role now checks this mirrored evidence and
refuses missing/stale evidence, conflicts, pending decisions/prepares or either
host's nonready state. It also compares the actual local shared overlay. The
Pi staging accepts an explicit --shared-status path to freshly mirrored PC/Jetson
evidence; without it (or a local mirror), it fails closed. The
existing folded/discard review gate is still required for any nonempty overlay;
that decision cannot override the shared reconciliation gate. No staging action
was run.

The panel renders per-host results and advances each unit from its own host's
expected digest. It shows PC pending only when pending, and displays reconciliation
rejections. Menu entries, bands, steps and the display fire-control guard are
unchanged.

## Raw panel guard and restart exclusion

The controller health report now includes the latest raw control_cmd_enabled,
control_cmd_valid and control_cmd_age_s. It uses local receive time, not Pi wall
clock time, and publishes every 100 ms. Invalid manual messages invalidate this
raw evidence. Rules/Calibration mutations and calibration_save require an
explicitly readable off value with panel age plus health-report age <=350 ms.
Unavailable/stale/invalid/on input is refused. Jetson rechecks its raw authority
after remote preparation and aborts before any write if it became unreadable. A successful fresh unit query
establishing no controller active permits the edit and returns a notice that
automatic tracking cannot run. The PC guard uses the existing software-controller
health file; the Jetson guard uses its existing controller health directories.

The operator's existing busy lock now reserves config restarts. A mode job
already running yields cannot_restart / mode change in progress while retaining
the overlay. While a restart owns the reservation, mode jobs cannot start. PC
mode tasks have the same reciprocal exclusion. Jetson unit state is published as
one tuple containing known/states/active, and readers capture one snapshot for
mode decisions. Refresh failure publishes one unknown snapshot. The worker's
fresh safety queries cannot expose half-updated unit state.

## Dropped-field audit

All Pydantic message families in common were audited: schemas, perception,
control_activity, target_authority, sim_score and synthetic_perception. The last
five already forbid extra fields. ControlCmd, CamState and ManualControlState
now do too; unknown fields raise validation errors rather than disappearing.

Fields found:

| Producer | Dropped field | Resolution |
| --- | --- | --- |
| rpi/runtime_control.py | ManualControlState.control_cmd_valid | Declare bool, default false; retain the real producer value. |
| pc/streamer.py build_cam_state | CamState.origin | Remove unused metadata from this producer; it had no model consumer. Source/RTP identity headers retain their separate origin field. |

No other dropped field was found. tools.sim_panel now declares its synthetic
armed state readable; its disabled guard remains invalid by default. Tests execute
every concrete producer constructor expression using deterministic inputs without
importing/starting hardware modules. They execute the actual streamer dictionary
builder for single/swarm outputs, test dynamic decoder boundaries, and check all
common Pydantic families reject unknown fields. Existing producer integration
tests remain applicable on a machine with their optional dependencies.

## Validation and timings

Environment: Windows 10 build 19045; Python 3.14.4, pytest 9.0.3, pyzmq 27.1.0,
PyYAML 6.0.3, pydantic 2.13.3. Base was clean before work. Inputs are versioned
release stacks, temporary isolated overlays, fake unit runners, synthetic panel
health and in-process/fake transports. No learned model, GPU, physical control
or external network is exercised. Timing uses local perf_counter/monotonic.

Release YAML parsing was the added latency bottleneck. ConfigAgent now caches
immutable authored release bundles by ordered stack fingerprint, clearing them
on any file fingerprint change. Candidate field/cross-field/unit validation is
still performed. The initial shared mutation took ~1.2 s; after this change a
warm three-sample run measured 201.28 ms median / 203.72 ms maximum, including
both hosts' temporary durable files. PC prepare+commit added 120.18 ms median.
Cached Jetson get measured 9.81 ms median / 10.49 ms maximum (n=10), and a
Jetson-local hot mutation measured 75.51 ms. These are local handler measurements with fake service state, not installed
Jetson/LAN evidence; service restart duration is not qualified by the hot-key
measurement. The existing one-second fake-validation operator-loop cadence tests
remain in the focused command.

Commands/results:

- `python -X utf8 -m pytest -q tests/test_shared_config.py tests/test_live_config.py tests/test_deployment.py tests/test_shared_message_producers.py tests/test_config_safety.py tests/test_panel_config_menu.py tests/test_config_agent_responsiveness.py tests/test_operator_agent.py tests/test_target_authority.py`: **323 passed in 82.52 s**. The final additional authority-recheck test passed with its adjacent contracts (**3 passed**), and is included in the final full run.
- `python -X utf8 -m pytest -q --continue-on-collection-errors --basetemp C:/Users/Lab412/AppData/Local/Temp/idcs-config-cross-host-complete-20261018-checked --tb=short`: **1,219 passed, 49 skipped, 7 failed, 8 collection errors, 20 subtests passed in 107.13 s**. No task regression remains. Five failures need Pillow; one needs trimesh; one needs the absent external `artifacts/gimbal_fit/controller_sysid_step_timed_unloaded_20260908/fit_report_external_selected.json`. Four collection errors need gi/GStreamer and four need Pillow. Skip guards cover optional PyTorch/matplotlib, cairo/smbus/GStreamer and native Jetson VPI/CUDA facilities. The checked temp path avoids the unrelated case-sensitive Windows preflight assertions.
- Final isolated timing command: `python -X utf8 -m pytest -q -s tests/test_shared_config.py::test_warm_shared_panel_timings tests/test_config_agent_responsiveness.py::test_cached_get_keeps_parser_defaults_without_runtime_or_parser_rebuild tests/test_config_agent_responsiveness.py::test_mutation_validates_only_candidate_and_reuses_it_for_next_get`: **3 passed in 3.05 s**, printing the timings above.
- `python -X utf8 -m compileall -q common pc jetson rpi tools tests` and `git diff --check`: pass. Git's existing LF/CRLF normalization notice is informational.

Final full log: `C:/Users/Lab412/AppData/Local/Temp/idcs-config-cross-host-full-suite-complete-20261018.txt`. Earlier trial logs are retained beside it under the same `idcs-config-cross-host-` prefix. Provenance, all 21 resolved release digests and empty-overlay digest are in `C:/Users/Lab412/AppData/Local/Temp/idcs-config-cross-host-provenance-20261018.json`. Test overlays, journals and recovery evidence remain in the checked temp directory. No dependency installation or source/artifact deletion was performed.

## Reviewer checks (not run on hosts)

All commands below are local to the named host's reviewed checkout. They use
fakes or startup validation only; they do not start units, construct motor owners
or use serial/bridge/HIL. Any real panel/render check remains at the bench under
the project lead's existing no-actuation rules.

Jetson, using the existing project bin environment:

```sh
bin/python3 -m pytest -q -s tests/test_shared_config.py tests/test_config_agent_responsiveness.py
bin/python3 -m pytest -q tests/test_config_safety.py tests/test_shared_message_producers.py tests/test_live_config.py tests/test_operator_agent.py
bin/python3 -m jetson.operator_agent --config configs/base --check
```

Pass: all fake contracts pass; warm shared mutation maximum <0.8 s; PC phase
round-trip cost is printed; cached get <50 ms; one-second validation keeps
operator/PC replies and selection gaps <=0.4 s. Startup validation succeeds.
The real LAN two-host mutation timing remains a reviewer qualification item;
this task does not authorize changing installed units or running a mutation on
a lab host.

PC, using its existing venv:

```sh
venv/bin/python3 -m pytest -q tests/test_shared_config.py tests/test_shared_message_producers.py tests/test_config_safety.py tests/test_pc_mode.py tests/test_target_authority.py tests/test_streamer_frame_ids.py tests/test_streamer_gst_video_writer.py tests/test_streamer_hil_pose.py tests/test_streamer_sim_mount.py
venv/bin/python3 -m pc.mode_agent --config configs/base --check
venv/bin/python3 -m pytest -q
```

Pass: all applicable tests pass with dependencies/artifacts present; startup
validation succeeds; no strict-model regression in actual streamer integrations.
The test_streamer_hil_pose module exercises measured-pose math with fixtures,
without opening a HIL connection.

Pi, using its existing project bin environment:

```sh
bin/python3 -m pytest -q tests/test_panel_config_menu.py tests/test_shared_message_producers.py tests/test_operator_menu.py tests/test_panel_indicators.py
bin/python3 -m rpi.operator_display --config configs/base --check
```

Pass: menu steps/bands remain unchanged; differing host digests advance the right
receipts; ready PC has no obsolete host-local warning; pending/rejected PC is
shown; field validity survives serialization; existing indicator tests pass.
The --check path validates available GStreamer elements and exits before
constructing OperatorDisplay. Real Pi rendering/input usability and installed
health-file cadence still need bench review. No motors or bridge/serial checks
are required or authorized by these commands.
