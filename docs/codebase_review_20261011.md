# Codebase and documentation review — 2026-10-11

## Scope and verdict

Reviewed local branch `v3-pid-hardware-verification`, starting at
`0818341c3f42205c8820f47cf6fcefe3b39f9557` (deploy-83 receipt; product commit
`eedb303180635c6fc20ad2a0ae976ec3077499b3`). The tracked tree was initially clean.

**R2 and R6 remain open. R1, R3, R4 and R5 were fixed locally in subsequent
2026-10-11 follow-ups.** The initial review updated documentation only;
follow-ups changed the affected runtime and deployment code with regression
tests. No service, hardware, commit or deployment change was made. The last
recorded deployment was not re-audited over SSH.

Coverage: inventory/static parsing of all 310 tracked Python files (201
production/tool modules, 109 test modules), manual review of critical
perception/selection, timing/PID/feedforward, bridge/serial, panel/HUD/operator,
mode-switch and deployment paths; native parser/VPI lifetime spot checks;
configuration/template/document cross-reference and local regression tests.
This is not an exhaustive proof of every branch or a native build review.

P1 means fix before relying on the affected operational gate; P2 means a
reproducible functional/isolation/design defect needing planned correction.
No severity below implies that hardware activation is authorized.

## R1 — P1: missing required bridge/serial overlays silently disappeared (fixed locally)

Locations: [bridge loader](../jetson/gimbal_bridge.py) lines 56–61,
[serial loader](../tools/serial_io_service.py) lines 357–365,
[legacy snapshot loader](../common/config_sync.py) lines 87–109 and 131–141.

Both startup loaders use `read_snapshot`, which converts FileNotFoundError to
empty text/config. A base config plus misspelled required bench overlay therefore
continues with base values rather than rejecting the missing layer. Those base
values can have broader limits/different axis coverage. Legacy YAML parsing
also lacks the strict loader's duplicate-key rejection.

Reproduction: pass the same nonexistent file to both loaders and to
`load_config_bundle`. The first two return `{}`; the strict loader raises
`ConfigError`. No device/socket is opened.

Fix direction: use the strict required-file bundle at these startup boundaries
and add missing-file/duplicate-key tests for the actual `--check` paths.
A motor preflight must not infer config completeness from today's successful
legacy check.

**Follow-up resolution (2026-10-11):** both startup loaders now use the strict
bundle, require their service sections, and fail on missing files or duplicate
keys. The bridge also rejects missing/duplicate-key explicit motor parameter
files. Its `--check` completes validation before creating ZeroMQ sockets. The
stale base parameter-file reference became `null`; parameter writes require an
explicit reviewed overlay. These changes passed managed-stack check-only and
failure-path regression tests on the local checkout. They are not deployed or
hardware qualified. The historical reproduction above describes the pre-fix
behavior; running it against the updated tree should no longer pass its R1
assertion. Other uses of `read_snapshot` remain outside this fix.

## R2 — P2: clearing Swarm score can expire a valid operator lock

Locations: [operator agent](../jetson/operator_agent.py) `active_catalog`
lines 289–299, `expire_lock` lines 336–349, `_start_score_job` lines 534–551,
and SIM activity forwarding around lines 709–713.

Score reset sets the same `busy` flag as a source transition. Any busy job
hides the active catalog and withholds SIM activity. If the PC acknowledgement
takes longer than `lock_lost_s` (default one second), normal expiry clears a
still-present target lock despite continuing fresh visual snapshots. Operator
mode remains, so there is no planner fallback. Safety indication can also be
withheld, though score reset is not a source/motion change.

Reproduction: fake systemd and clock, fresh target42, accepted operator lock,
delayed score acknowledgement, then advance 1.1 seconds and deliver another
fresh snapshot. `expire_lock` clears the lock. No real systemd/network is used.

Fix direction: separate source-transition suppression from score/recording job
bookkeeping; preserve active catalogs/activity for non-source jobs and test
delayed acknowledgements through the main-loop lifecycle. Existing immediate-ACK
reset acceptance did not cover this interleaving.

## R3 — P2: failed selector reuses old selection/range without expiry

Locations: [async selector](../jetson/deepstream/async_target_selection.py)
`_apply_completed_snapshot` lines 108–162 and `submit_and_apply_snapshot`
lines 198–241.

The worker's last result remains in `_latest`; a worker error only records
`_error`. Matching track IDs keep old assessments and selection while
`applied_frame_id` advances to current frames. There is no maximum result age
or failed-worker invalidation. Current-frame timing therefore cannot detect
stale planner/range facts, including range used for parallax.

Reproduction: retain a completed frame10 result with range12m, mark the worker
failed, then submit matching track7 on frame1000 one minute later. Selection
source frame remains10, applied frame becomes1000, and range12m is reused.

Fix direction: bound decision/assessment age, require matching source/session,
and invalidate failed-worker results. Consider computing current geometric
range independently of the slower policy worker. Preserve legitimate short
asynchronous reuse rather than requiring every decision on the same frame.

**Follow-up resolution (2026-10-11):** completed decisions and assessments now
require a matching nonempty tracker session and source/observation clock domains,
ordered frame/source timestamps, and both source-frame and local-reuse ages
within `swarm_eval.learned_model.max_result_age_ms` (150 ms by default). Worker
errors clear the cached result. Expired results fail closed to the current
unselected snapshot; a current-frame operator lock remains independently
possible. Synthetic tests cover recent reuse, stale time, session/source change
and worker failure. Computing independent per-frame range remains a possible
future design improvement, not a prerequisite for this expiry fix.

## R4 — P2: controller broadens explicit loopback listeners to all interfaces

Locations: [video runtime](../jetson/control/video_runtime.py) lines 46–47,
162–179; [SIM endpoint profile](../configs/controller/sim_mount.yaml).

`_bind` retains only the port and returns `tcp://0.0.0.0:PORT`. Thus configured
loopback manual input, intent, diagnostics and recording listeners bind to every
IPv4 interface. The SIM units' “loopback only” description is not true for these
sockets. An unauthenticated manual-input listener is unnecessarily exposed on
reachable interfaces; actual reachability also depends on host/network rules.

Reproduction: `_bind('tcp://127.0.0.1:15559')` returns
`tcp://0.0.0.0:15559`; the same occurs for intent port5571. No socket is opened.

Fix direction: preserve explicit localhost and validate bind addresses; use
wildcards only where the deployment deliberately requests them. Add socket/bind
contract tests. The PC mode agent's outbound task channel is a different path
and does not resolve this exposure.

**Follow-up resolution (2026-10-11):** `_bind` validates TCP host/port and
returns the configured endpoint unchanged. Tests cover loopback, LAN and
explicit wildcard addresses plus malformed endpoints. This is local code
validation, not a live firewall or service audit.

## R5 — P2: deployment's config-drift gate misses new files

Location: [deployment planner](../tools/deployment.py) lines 191–217,
especially the old-tree `rglob('*.yaml')` at line196.

The planner compares only files found under the previous source's config tree.
A candidate can add a configuration file and reference it from a new unit
command without that file entering semantic drift review. Existing `.yml`
tracker config is also outside this scan. A clean commit and changed-unit diff
do not turn an empty `config_changes` into a complete config audit.

Reproduction: temporary old/new trees share an unchanged base config; the new
tree adds an overlay changing Kp5 to20 and its unit references that overlay.
`plan()` reports `config_changes={}` without a config-review policy.

Fix direction: compare the union of old/new config files and effective ordered
service stacks, including additions/removals and relevant non-`.yaml` profiles.
Until then, independently review those differences during deployment.

**Follow-up resolution (2026-10-11):** the planner now compares the union of
old/new `configs/**/*.yaml` and `configs/**/*.yml`. New files require an
explicit before/after review; removed files still fail; `.yml` native profiles
are compared by exact bytes because `%YAML:1.0` is not PyYAML. Focused tests
cover additions, modified native profiles, removals and review-hash mismatch.
Unit command/order changes remain visible in the plan's desired-unit diff and
must still be reviewed as effective ordered stacks; this fix does not parse
arbitrary unit commands into a separate config-stack approval gate.

## R6 — P2: software SIM is still coupled to hardware tuning

Locations: [Single controller unit](../deploy/systemd/pc/idcs-sim-controller.service)
lines 14–16, [Swarm controller unit](../deploy/systemd/pc/idcs-sim-swarm-controller.service)
lines 15–17; [streamer quantization](../pc/streamer.py) lines 735–749.

Both software-SIM controller stacks load `configs/bench/tuned.yaml` last
(before Swarm's endpoint-only detection overlay), inheriting hardware Kp, rate
and feedforward settings. The streamer also applies the MKS F6 quantizer.
This violates the explicit stable-substitute requirement in `AGENTS.md` and
the verification strategy: hardware retuning changes camera/UI/perception SIM
motion, even with `use_jetson_cam_state=false`.

Reproduction: resolve the actual Single service stack and assert that yaw/pitch
Kp, rate cap and feedforward equal the hardware tuning overlay. Static tracing
confirms the F6 quantizer call in the software intent path.

Fix direction: define an independent stable software-mount controller/plant
profile and ensure neither mode loads hardware tuning. Keep measured-plant
offline gain experiments and explicitly authorized physical HIL separate.

## Other reproducibility/cleanup gaps

- `tests/test_validate_gimbal_fit.py:106` requires an absent historical fit
  report under `artifacts/gimbal_fit/controller_sysid_step_timed_unloaded_20260908/`
  and its external validation data. Restore a versioned/minimal fixture or
  explicitly classify it as an artifact-dependent test; do not silently drop
  it to claim a clean full suite.
- The stale `configs/base/gimbal.yaml` motor parameter path was removed in the
  R1 follow-up. The existing reference data remains at
  `configs/bench/mks_parameters.yaml`; using it now requires an explicit overlay.
- Parsed known-size `ema_alpha` is not applied by the active DeepStream selector;
  calibration instructions no longer promise that changing it smooths range.
- Retained legacy Pi command code, config-sync helpers and MPC/ControlCmd
  compatibility mean “all legacy removed” is still inaccurate. Their removal
  needs dependency-specific work, not deletion based on old docs alone.

## Local validation

Environment: Windows, Python3.14.4, pytest. No motor/serial/GPIO/GPU owner was
started; running processes were checked before the baseline. Test output and
temporary files were kept outside the repository.

The available-platform baseline completed with **909 passed, 2 failed,
39 skipped and 20 passed subtests** in about49 seconds. The failures were:

1. `test_person_mesh_bounds_are_centered_for_world_transform`: optional
   `trimesh` dependency absent in this environment.
2. `test_existing_low_range_fit_is_not_offline_control_qualified`: missing
   historical fit artifact described above.

Four GI/GStreamer-import streamer modules were excluded from this Windows run:
`test_streamer_frame_ids.py`, `test_streamer_gst_video_writer.py`,
`test_streamer_hil_pose.py`, `test_streamer_sim_mount.py`.
The 39 skips include Pi GI/Cairo, Torch, native VPI/NVMM and opt-in GPU coverage.
Native GPU execution, Linux systemd, real camera, physical panel and hardware
acceptance were **not rerun**. Existing journal results are labeled historical.

Baseline command (from the repository, with the existing external Pillow test
dependency directory on PYTHONPATH):

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='C:/Users/Lab412/idcs-dev/ui-test-deps;C:/Users/Lab412/idcs-dev/repo'
python -m pytest -q -p no:cacheprovider --basetemp=C:/Users/Lab412/idcs-dev/review-20261011-tests-01 --junitxml=C:/Users/Lab412/idcs-dev/review-20261011-tests-01.xml --ignore=tests/test_streamer_frame_ids.py --ignore=tests/test_streamer_gst_video_writer.py --ignore=tests/test_streamer_hil_pose.py --ignore=tests/test_streamer_sim_mount.py
```

Six isolated reproduction probes passed in1.19 seconds. **These assert the
defects' present behavior, not that the product passes corrected acceptance.**
Harness:
`C:/Users/Lab412/idcs-dev/review-20261011/test_review_reproductions.py`.
Final reproduction receipt:
`C:/Users/Lab412/idcs-dev/review-20261011-repros-final.xml`.
Baseline receipt:
`C:/Users/Lab412/idcs-dev/review-20261011-tests-01.xml`.
Use fresh external temporary directories for repeat runs.

All 310 tracked Python files parsed successfully. Final documentation/link/diff
checks passed (38 Markdown files, no missing local link targets;
`git diff --check` clean). The post-edit run passed 95 existing configuration,
deployment, operator and tuning tests plus all six defect probes: 101 total in
9.86 seconds. Receipt: `C:/Users/Lab412/idcs-dev/review-20261011-postedit.xml`.
The [journal](deepstream_migration_journal.md) records the full handoff.

## Documentation reconciliation

Updated the root overview, config/perception/controller/serial/panel inventories,
operations, deployment cautions, operator/HUD guide, visual-lock install status,
calibration/tuning instructions and DeepStream readme. Added the
[documentation index](README.md); archived the old controller architecture and
marked dated handoffs/plans historical. Dated journal entries, vendor manual,
historical Word documents, recordings and external trial evidence were preserved.

Important corrections include NvDCF (not NvSORT), ControlIntent (not ControlCmd)
on the physical control path, implemented sensor timestamp integration and
later successful bounded FF trials, raw-step versus encoder pose, no automatic
serial-service shutdown disable, current panel-only Pi ownership, internal IDs
versus priority rank, and managed releases versus old trial paths.

The original documentation reconciliation did not fix runtime behavior. Later
local follow-ups fixed R1, R3, R4 and R5 with targeted tests. R2 and R6 remain
open. No part of this review qualifies real-camera or loaded-hardware operation.
