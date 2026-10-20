# Managed deployment

This document replaces source-path claims in historical handovers. A release
being present, a Git tag being fetched, and a process actually using it are
different states. Repository synchronization is separate from deployment.

## Last recorded deployment (2026-10-20, deploy-104)

Commit `70f2ddb` / tag `deploy-104` is installed on PC, Jetson and Pi. On top
of deploy-102 it fixes the camera visual-lock unit (no PrivateTmp; deploy-103
reached Jetson/PC only, the Pi stayed on 102) and adds Pi return-video
recording (`rpi.operator_display.recording`, reviewed config addition). The
Pi's policy entry had to be keyed on its installed deploy-102 source. Pi
activation needed a second attempt (overlay p95 17.3 ms on the first, likely
menu use during the check). Validation PC 370/4 skipped, Jetson 393, Pi 82;
audit: all units reference deploy-104. Physical units inactive.

## Earlier deployment (2026-10-20, deploy-102)

Commit `db21d48` / tag `deploy-102` is installed on PC, Jetson and Pi from
`~/Desktop/project/IDCS-releases/deploy-102`. On top of deploy-99 it adds the
managed visual-lock source modes (Camera / Visual lock, Simulator / Visual
lock with a single-drone PC stack), box-size presets, release/loss returning
the box, the overlay queue and CUDA input defaults, and the bounded DCF-only
hold (default 12,30). deploy-100 was tagged but never installed; deploy-101
was installed earlier the same day. No config values changed from deploy-101.
Validation PC 369 passed/4 skipped, Jetson 392, Pi 82; activation Jetson ->
Pi -> PC verified; audit clean. Physical bridge/serial/controller/HIL units
remain inactive (hardware hold). Logs: `IDCS-build/deploy102-*.log`.

## Earlier deployment (2026-10-19)

Commit `11cf19e` / tag `deploy-99` is installed on PC, Jetson and Pi from
`~/Desktop/project/IDCS-releases/deploy-99`. It merges live configuration
(panel menu, shared-config rollout) with the camera ROI line (Harris
threshold 0.5, seed diagnostics with tightening off, interleaved LK, fusion
restructuring, registered CUDA visual-lock input). No deployed unit runs the
visual lock, so the camera ROI work ships as code only. Reviewed config
removals passed under `IDCS-build/deploy-99-config-policy.json`. Detector
engine, swarm engine and YOLO26 parser are linked to the same unchanged bytes
as deploy-98. Validation: PC 294 passed/4 skipped, Jetson 317 passed, Pi 82
unittest cases. Sequential activation Jetson -> Pi -> PC, each verified, then
audit. Source mode remains Camera; physical bridge/serial/controller/HIL units
remain inactive (hardware hold). Logs: `IDCS-build/deploy99-*.log`.

## Previous deployment (2026-10-16)

Commit `362b5a1` / tag `deploy-98` is installed on PC, Jetson and Pi from
`~/Desktop/project/IDCS-releases/deploy-98`. The release includes native
GStreamer RTP/RTCP return recovery, UI ownership fixes and exact-frame ROI
acquisition diagnostics. Managed validation, sequential software activation
and initial live verification passed on all three hosts. Source mode remains
Camera; physical bridge/controller/HIL units remain inactive. The full rollout
receipt and repeat verification are recorded in the newest journal entry and
host manifests.

ROI remains an opt-in, bounded, control-free SIM VPI session, not an enabled
ordinary console or real-camera ROI mode. The previous controlled validation
passed three45s explicit locks, deliberate blank refusal and subsequent
corrected selection. Camera sensor metadata already exists, but the camera ROI
seed/cache/return-frame join is not integrated. Deploying this release does not
enable that missing feature or qualify real GPIO/motor operation.

Deploy-97 was staged but never activated: full validation caught two old tests
asserting the former one-way UDP layout. Deploy-98 corrects those assertions
for native AVPF RTP/RTCP. Both the staged candidate and failed evidence are
preserved; no tags or immutable release sources were rewritten.

## Previous deployment (2026-10-14)

Commit `0b597f9` / tag `deploy-96` is installed on Jetson and Pi from
`~/Desktop/project/IDCS-releases/deploy-96`. PC ordinary services remain on
deploy-85; its bounded ROI source uses `IDCS-build/deploy-96`. Managed native
validation passed314 Jetson tests and82 Pi tests. The final software-switch
acceptance passed two exact-frame lock/track/release cycles and Manual-OFF menu
restoration:418 matched Pi outputs, zero unmatched,303 selected display ticks.
Physical switch transitions were not qualified; no motor/HIL units were started.

ROI remains an opt-in, bounded VPI-only viewing session, not a permanent
operational console mode or NvDCF handoff qualification. Its exclusive guards
pause conflicting software owners, retain Pi GPIO, and restore the previous
software services. Native libraries are retained under
`deployments/managed/deploy-90/native-build/`; session evidence is under
`deployments/idcs-roi-acceptance-20261014-08/`. See the newest journal receipt
and host manifests for actual running state and rollback paths.

## Previous deployment (2026-10-13)

Scoped commit `6e93b0c` / tag `deploy-85` is installed on PC and Pi from
`~/Desktop/project/IDCS-releases/deploy-85`; Jetson remains on deploy-84.
The release adds a Pi manual-state mirror and makes the PC software-SIM panel
bridge obey the real panel when configured. The single config addition was
reviewed by exact before/after semantic hashes. Clean managed validation,
sequential Pi then PC software activation, repeat verification and reference
audit passed. A PC read-only canary received live Pi samples; SIM control
entered `manual` when Pi reported manual active. Physical switch-transition
and link-loss acceptance remain untested at the bench. No motor/HIL units were
started. Exact current state and rollback evidence are in each host's current
manifest and `deployments/managed/deploy-85/`; see the newest journal entry.

## Previous deployment (2026-10-11)

Commit `9ae987a` / tag `deploy-84` is installed from immutable
`~/Desktop/project/IDCS-releases/deploy-84` on PC, Jetson and Pi. Managed
stage, host validation, sequential software activation, repeat verification and
reference audit passed. The only configuration-value difference from deploy-83
was `gimbal.parameter_file` changing from the nonexistent implicit path to
`null`, covered by exact semantic review hashes on all three hosts. Jetson's
new release reuses the existing ignored model-store and hash-addressed native
parser-library links; the binary/model bytes were not replaced. Swarm remains
active; physical serial, bridge, controller and HIL units remain inactive.
Post-rollout passive health showed PC tracking with zero invalid messages,
Jetson pipeline/return about 57.7/29.3 fps, and Pi video about 29.6 fps.
This is software-source/health acceptance, not hardware, detector-quality,
timing, or long-duration tracking qualification. R2 and R6 in the
[review](codebase_review_20261011.md) remain open. Exact live source/owner
state is in each host's current manifest, not this dated receipt. See the
newest [journal](deepstream_migration_journal.md) entry for test scope.

## Previous deployment (2026-10-10)

The 2026-10-10 software rollout uses code commit `eedb303`, tag `deploy-83`, at
`~/Desktop/project/IDCS-releases/deploy-83` on PC, Jetson and Pi. Clean-release
validation, sequential activation, passive live verification and reference audit
passed. Running source is **Simulator / Swarm**; Single is inactive. The Pi menu
provides System -> Mode -> Simulator -> Single / Swarm with confirmation and
ordered source/readiness/rollback guards. Jetson visual target-lock identity is
separate from the simulator pose/HUD identity; Pi tapes/control/FF overlays were
visually confirmed. PC and Pi show Swarm HIT/BREACH/LIVE counters or Single raw
yaw/pitch error in a shared compact strip. System -> Clear Swarm score has
Cancel-first confirmation; its live command-path test cleared the score epoch
without restarting owners or changing the SIM session. Single's fresh numeric
error and both receivers were verified, then Swarm restored. Final Pi return
measured29.99fps; Jetson return29.52fps during the deploy-83 canary. This is UI/transport/reset acceptance,
not detector-quality, controller-tuning or swarm-engagement qualification.
Physical menu interaction is deferred. Target IDs are now internal; operational
labels, target picker/status and lock/engagement notices do not expose them.
Native GPU rank badges use an opaque lower-right fill with the numeral inside,
verified on guaranteed multi-target fixtures at1280x720 and640x360. Single,
unranked and unreadably small corners omit rank. Passive live PC screenshots
confirm ID-free labels and Pi screenshots confirm the HUD; a clear compressed
multi-target badge frame was not captured, so that visual gate remains separate.
See the newest journal receipt for evidence and limitations. All installed service templates,
including inactive camera/HIL units, use that source; inactive hardware units
remain inactive. No useful trials, recordings or rollback sources were deleted.

This deploy-83 status was the 2026-10-10 receipt, not the current running
release. The later code review found score-reset and config-drift edge cases;
deploy-84 corrected the file-scan gap, while R2 score-reset behavior remains
open. See [codebase review](codebase_review_20261011.md).

Authoritative live manifests on each host:

```
~/Desktop/project/deployments/managed/pc-current.json
~/Desktop/project/deployments/managed/jetson-current.json
~/Desktop/project/deployments/managed/pi-current.json
```

They record the commit, installed WorkingDirectory, actual `/proc` cwd/command,
PID/restarts, active/inactive state and health. Read only the file for that host.
`IDCS-runtime` is a retained Git object/worktree store, not the running release.
The main development repo on39.68 is not implicitly advanced or reset.

## Explicit phases

Run the coordinator from the PC Linux development checkout (not Windows):

```bash
scripts/deploy.sh stage deploy-N pc,jetson,pi
scripts/deploy.sh validate deploy-N pc,jetson,pi
scripts/deploy.sh activate deploy-N pc       # one host at a time
scripts/deploy.sh verify deploy-N pc
scripts/deploy.sh activate deploy-N jetson
scripts/deploy.sh verify deploy-N jetson
scripts/deploy.sh activate deploy-N pi
scripts/deploy.sh verify deploy-N pi
scripts/deploy.sh audit deploy-N pc,jetson,pi
```

- Stage requires a clean reviewed commit and tag; it creates a detached release
  without moving any development/runtime checkout or starting services. Run
  stage from the tagged checkout so installer templates and product code agree.
- `IDCS_BUNDLE_BASE` optionally supplies a shared ancestor for a smaller Git
  bundle. Receivers verify prerequisites. Model/DeepStream/native assets must be
  prepared before validation; failure does not trigger installation/activation.
- Unit templates use `@RELEASE@`; never copy them directly into systemd.
- Plan inventories *all* installed IDCS units, not just active ones, captures
  originals and overrides, compares configurations and inventories asset links.
  Unknown units/overrides or detected unexplained config drift fail closed.
  The config scan covers the union of old/new `configs/**/*.yaml` and `.yml`;
  new files require explicit review, removed files fail, and `.yml` profiles
  use byte hashes because native syntax need not parse as ordinary YAML.
  Review effective ordered config arguments in each desired-unit text diff;
  arbitrary unit commands are not separately parsed into a stack gate.
- Validate runs platform-appropriate deterministic tests, check-only software
  CLIs and `systemd-analyze verify`; no new video/GPU/GPIO/serial owner is started.
  Motor/HIL CLI execution and hardware qualification are deliberately excluded.
- Activate checks source cleanliness, expected owners and unit bytes, writes
  rollback information first, stops/restarts only previously active software
  units, reloads systemd, and verifies actual source/owner/cadence. Failure
  restores the backed-up units and previous software mode. Physical/HIL units
  block activation; the tool cannot start them.
- PC SIM viewer/session dependencies restart the software session together;
  mode/target identity restarts invalidate old locks by design. Plan maintenance
  pauses for recording/video and fresh GPIO samples after Pi service restart.
- Verify requires stable unique module owners, no unaccounted overrides,
  expected inactive states, clean release code and fresh live health.
  This is not hardware, detector-quality, real-camera or endurance acceptance.

The initial 2026-10-07 migration used reviewed bootstrap tools outside runtime
source. Deployment tools/templates are now committed in product release77;
each release retains its exact installer in `deployments/managed/<release>/installer/`.
The following config review and command describe the historical consolidation,
not a required policy for every later release (release77 had zero config drift).
`deploy/consolidation-20261007-config-review.json` contains one-time complete
before/after semantic hashes for historical config migration; it is not a
permanent wildcard approval. Any value mismatch fails, including gain changes.
Use the direct host command to supply this review file when planning:

```bash
~/Desktop/project/bin/python3 ~/Desktop/project/deployments/deployment.py plan \
  --role jetson --release deploy-76 \
  --templates ~/Desktop/project/deployments/consolidation-20261007/templates/deploy/systemd \
  --config-policy ~/Desktop/project/deployments/consolidation-20261007-config-review.json
```

PC uses `venv/bin/python3`. All receipts/backups/tests live outside source in
`deployments/managed/<release>/<role>/`. The initial read-only code-difference
inventory is retained in `deployments/consolidation-20261007/inventory.json`.

## Rollback and preservation

```bash
scripts/deploy.sh rollback deploy-N pc       # explicitly chosen host
```

Rollback restores exact original unit/override bytes and the prior software
active set. It refuses if an operator changed source mode during the transaction.
The previous verified manifest is archived; the current manifest is explicitly
marked `restored-unverified`, never left claiming the failed release is active.
It never restores by checking out/resetting a dirty development tree. Restored
asset links can remain canonical external paths: the old model store and bytes
are retained; no model data was deleted.

Old trial/release folders, model/native binaries, source diffs, recordings and
test evidence remain intact. The old Jetson nested-box override is removed
ONLY after its contents are backed up and its source choice is folded into the
managed main unit. Its exact bytes remain in rollback.json.

The audit command is read-only and intentionally does not delete trials. No
automatic garbage collection: recorded evidence, rollback plans, symlinks,
native-library/config paths, installed units and running processes must all be
reviewed before a separately authorized deletion. Historical AGENTS content is
preserved in `docs/archive/agents-20261007-before-consolidation.md`.

## Remaining qualification

Motor wiring/RS485, real switch operation, HIL handover/position continuity,
production timing bounds, real-camera ROI, console ROI integration, dynamic
label collisions and representative multi-target identity remain separate gates.
Consolidation must not silently remove these cautions or promote SIM evidence
into hardware acceptance. See the journal, operator display and visual-lock docs.


## Shared overlay reconciliation gate (2026-10-18)

Staging now refuses absent/stale PC/Jetson reconciliation evidence, an unknown
PC shared overlay, a pending prepare/commit, or either host's mismatch/rejection.
It also compares the actual local shared overlay with its reconciled projection.
Evidence is config-shared.json beside live.yaml; Pi staging can receive an
explicit fresh mirror via --shared-status. A folded/discard decision does not
override this gate. The first rollout must supply reviewed reconciliation
evidence; missing evidence is not treated as an empty overlay. No deployment was
performed for this change. See the [design/report](config_cross_host_20261018.md).

Review follow-up (2026-10-19): a host with neither `live.yaml` nor
`config-shared.json` passes as `bootstrap` (first rollout from a release without
live configuration); any existing overlay or shared state still requires fresh
reconciled evidence. Pi staging records the gate as not applicable unless
`--shared-status` is given (the Pi runs no config agent; PC/Jetson stages gate).
A removed config file is accepted only when the policy lists it as
`{"removed": true, "before": <semantic sha256, or byte sha256 for .yml>}`.
`scripts/deploy.sh` passes `IDCS_CONFIG_POLICY` to every host's stage/plan.
