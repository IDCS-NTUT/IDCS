# Documentation index

Use current guides for behavior and the dated journal for evidence.
This index was added during the [2026-10-11 review](codebase_review_20261011.md).

## Current guides

| Topic | Guide |
|---|---|
| Overview and boundaries | [Repository README](../README.md), [AGENTS.md](../AGENTS.md) |
| Findings, test scope and open issues | [Codebase review](codebase_review_20261011.md) |
| Installed source, manifests, rollback | [Managed deployment](deployment.md) |
| Source switching, owners and checks | [Launch procedure](launch_procedure.md) |
| Config authority and service stacks | [Configuration architecture](configuration_architecture.md), [config inventory](../configs/README.md) |
| V2 perception and passive video | [Perception architecture](perception_architecture.md), [DeepStream validation](deepstream_cutover.md) |
| Controller, estimator, timing and plant scope | [Controller architecture](controller_architecture.md), [tuning procedure](tuning_procedure.md) |
| Serial ownership and execution | [Lifecycle](serial_io_service_definition.md), [inventory](serial_io_inventory.md), [scheduling](serial_io_scheduling.md), [IPC](serial_io_interface_design.md) |
| Console, lamps and operator authority | [Operator display](operator_display.md), [Pi inventory](rpi_manual_control_inventory.md) |
| Drawing ownership and layout | [HUD design](hud_design.md), [UI style](ui_style.md) |
| Experimental detector-independent ROI | [Visual lock](visual_lock.md) |
| Calibration and model preparation | [Camera calibration](camera_calibration.md), [TensorRT build](tensorrt_engine_build.md) |
| Evidence requirements | [Controlled verification](verification_strategy.md) |

“Current” describes the code cross-reference, not an assertion that every
requirement is implemented or every host was just inspected. Read documented
limitations. Last recorded deploy-84 software acceptance does not qualify
physical switches, loaded motors, operational ROI or optical-to-display timing.

## Historical records and proposals

- [Development journal](deepstream_migration_journal.md): chronological evidence,
  newest entries at the bottom. Old paths/results can be superseded; do not
  execute historical commands as current procedures.
- [Early V3 handoff](controller_handoff.md),
  [controller overhaul plan](controller_overhaul_plan.md),
  [estimator plan](estimator_feedforward_plan.md),
  [serial execution-feedback plan](serial_io_execution_feedback_plan.md), and
  [Pi split proposal](rpi_manual_control_split_boundary.md): preserved design and
  trial history, not current task/status lists.
- [Previous controller architecture](archive/controller-architecture-before-20261011-review.md)
  and [pre-consolidation AGENTS](archive/agents-20261007-before-consolidation.md):
  retained evidence, not active instructions.

`IDCPS.docx` and `IDCS Pipeline Architecture and MIPI Camera Integration.docx`
are historical project/design documents, not the current software/deployment
authority. They were not rewritten by this Markdown reconciliation.
The MKS RS485 PDF is a vendor protocol reference; matching installed firmware,
motor parameters and hardware behavior still requires explicit verification.

Configuration cleanup evidence: [2026-10-17 reader audit and unit-stack diffs](config_dead_removal_20261017.md).
