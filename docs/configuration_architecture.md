# Configuration architecture

Cross-checked against the repository on 2026-10-17.

## Authoring and layer order

Version-controlled YAML plus explicitly reviewed deployment overlays is the
authoring source. Processes load local files at startup; current managed runtime
paths do not run a peer-to-peer winner-by-mtime configuration handshake.

`configs/base` expands into topic files. Ordered overlays follow it.
Mappings merge recursively; lists and scalars replace previous values.
The exact service stacks are in [configs/README.md](../configs/README.md) and the
systemd templates, not old flat filenames from historical trial notes.

`common.config.load_config_bundle` provides required-file and duplicate-key
rejection, JSON-compatible string-keyed values, optional required-section checks,
immutable resolved data and source/resolved SHA-256 provenance. Process-specific
configuration validation is a separate layer.

## Migration status, not a blanket guarantee

| Path | Current behavior |
|---|---|
| DeepStream, PC streamer/UI, video controller, operator/mode agents, Pi `runtime_control` | Local strict bundle loading; no runtime config synchronization |
| Physical `gimbal_bridge`, `tools.serial_io_service` | Strict local bundle loading; bridge also validates an explicitly configured parameter file |
| `rpi.manual_control` | Shared GPIO/ADC helpers only; no standalone serial command loop |
| Auxiliary/bench utilities | May retain the local helpers in `common.config_sync`; not all use strict bundle loading |

The historical `common.config_sync` module now contains local snapshot, path,
merge and video-profile helpers only. Network server/client/startup handshakes,
markers, locks and write helpers have been removed. Existing imports and merge
semantics are preserved.

**R1 resolved in the local follow-up:** bridge and serial startup loaders use
`load_config_bundle` and reject missing overlays and duplicate YAML keys.
The bridge's configured parameter file is also checked, and its preflight now
returns before opening sockets. The base parameter-file default is null so
parameter writes require an explicit reviewed overlay. The local snapshot helper
still treats missing files as empty for legacy offline callers; managed runtimes
use strict required-file loading. A passing `--check`
proves configuration parsing, not serial connectivity or motor readiness.

## Checks and deployment provenance

Strict-loader consumers expose config provenance where implemented; it is a
requirement, not a verified universal property of all long-running processes.
A resolved digest is not motor readiness or a clock/calibration qualification.

Managed deployment compares the union of prior/candidate `configs/**/*.yaml`
and `configs/**/*.yml`. New files and changed native `.yml` profiles require
explicit review hashes; removed files fail closed. `*.yaml` review hashes are
semantic, while `.yml` review hashes cover exact bytes. R5's missed-file scan
was fixed locally; review the effective ordered config stack in each changed
unit's desired-text diff, which is not separately parsed into a stack gate.

See [deployment](deployment.md), [review findings](codebase_review_20261011.md)
and [verification strategy](verification_strategy.md). Retained results need
commit/dirty state, resolved config and model digests, host/runtime identity,
clock domains and artifact paths. Runtime artifacts belong outside source.

## Sources after the 2026-10-17 cleanup

`camera.intrinsics` owns aiming calibration. The simulated vertical-FOV override
constructs camera intrinsics directly; it does not overwrite `control` keys.
`control.sign_convention` is live through `pixel_delta` in aiming and planning.
`controller.local_clock` and `controller.source_clock` are also live: they select
clock domains and upstream receipt handling; `controller.clock.*` defines the
mapping policy. They are separate from `operator.modes.*.source_clock`.

Bridge rate caps and the configured controller acceleration cap come from the
unchanged tuning output. The required base controller rate cap remains duplicated
in that output; removing it would invalidate supported untuned shadow startup or
require changing the parser's required-key contract. See the
[cleanup report](config_dead_removal_20261017.md) for the audited exceptions,
unit-stack diffs, runtime-equivalence evidence and removed-file deployment gate.
