# Agent instructions

## Safety and authority

- No motor/HIL commands, bridge/serial activation, homing, steps or sweeps until
  the user confirms repaired panel/motor wiring and every RS485 motor answers.
  Pitch address2 stopped answering after wiring disturbance on 2026-09-29.
- Hardware testing is performed by the user at the bench. Camera, software SIM,
  read-only diagnostics and display work are permitted. Cleanup never starts
  inactive motor/HIL units.
- Never transmit the user's email address to any service.
- Commit or push only when explicitly requested. Required commit trailer:
  `Co-Authored-By: Codex <noreply@openai.com>`.
- Preserve user edits, useful trial fixes, recordings, reports and rollback
  sources. Audit references before deletion; deployment is not repo sync.

## Task references

- Current deployment and rollback: `docs/deployment.md`; authoritative host
  manifests: `~/Desktop/project/deployments/managed/{pc,jetson,pi}-current.json`.
- Operations: `docs/launch_procedure.md`; panel/menu:
  `docs/operator_display.md`; config: `docs/configuration_architecture.md`.
- Controller: `docs/controller_architecture.md`, `docs/tuning_procedure.md`.
- Verification: read `docs/verification_strategy.md`; record results in
  `docs/journal.md` (newest entries at the end). The closed
  `docs/deepstream_migration_journal.md` holds entries up to 2026-10-19.
- ROI/VPI tools: `docs/visual_lock.md`. Preserved pre-cleanup handover:
  `docs/archive/agents-20261007-before-consolidation.md`; this archive is
  historical, not authority for current paths or commands.

## Architecture boundaries

- PC streamer owns source-time/RTP frame identity; Jetson DeepStream owns
  YOLO/NvDCF, V2 selection/publication and GPU return-video OSD. Passive
  video runtime never constructs a motor controller.
- Perception: `common/perception.py` V2; control: `common/schemas.py`;
  operator: `common/operator_commands.py`. Core DeepStream cannot import
  legacy perception schemas/projections; only named compatibility modules and
  display sinks may do so.
- Stable software SIM control is independent of hardware tuning.
  `sim.use_jetson_cam_state=true` means HIL: fresh physical pose only,
  no simulated fallback. HIL remains prohibited above.
- PC/Pi HUD is display-only. Latest telemetry is not an exact return-frame join;
  transit age remains unqualified. Never feed SIM truth to a visual tracker.
- GPIO12 gates automatic tracking; GPIO5 independently selects manual authority;
  GPIO16 gates fire-button indication only (no effector). Only GPIO12/16 share
  lamp power. Lamp-off means unreadable input, not switch-off; preserve fresh
  sampling, hysteresis and fail-closed watchdogs.
- Safety lamp: fresh applied SIM/gimbal tracking, with manual exception;
  fire-control lamp: visual target available; yellow: motor feedback available.
  E-stop dominates. Full semantics and bench gates: `docs/operator_display.md`.
- Operator locks are source/session scoped. With Planner selection off, Operator
  mode chooses the nearest visible eligible ranged target, holds a manual or
  automatic lock until lost, then reacquires the nearest after the loss grace;
  it never falls back to Planner. Menu inversion cannot invert motion.
- ROI VPI initialize -> NvDCF remains opt-in/isolated, not an operational console
  mode. Early standalone EGL bootstrap is not an extra owner in an existing
  pipeline. Never conceal teardown faults with `_Exit` or reseed after loss.

## Working and deploying

- Check running owners before tests; do not stack GPU/video/serial owners.
  Use bounded durations and deterministic contract inputs wherever learned
  detection is not the responsibility tested.
- PC: user systemd, `venv/bin/python3`; Jetson: system systemd/passwordless sudo,
  `bin/python3`; Pi: user systemd, `bin/python3`.
- Units use `@RELEASE@` templates rendered by `tools/deployment.py`.
  `scripts/deploy.sh`: explicit stage/validate/activate/verify/rollback/audit;
  no bulk runtime checkout or implicit hardware activation.
- Back up units/overrides, explain config differences, validate before activation.
  PC viewer dependencies restart its software SIM session together.
- Keep this file short. Current deployment goes in its manifest/document;
  dated evidence and test results go in the journal.
