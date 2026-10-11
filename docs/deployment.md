# Managed deployment

This document replaces source-path claims in historical handovers. A release
being present, a Git tag being fetched, and a process actually using it are
different states. Repository synchronization is separate from deployment.

## Last recorded deployment (2026-10-10)

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

This status is the last journal/manifest receipt, not a live re-audit on
2026-10-11. The local code review found unresolved score-reset and config-drift
edge cases; the bounded deployment acceptance above did not cover them. See
[codebase review](codebase_review_20261011.md).

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
