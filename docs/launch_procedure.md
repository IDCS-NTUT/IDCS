# Launch, status and deployment

Use immutable managed releases and their host manifests:
[deployment](deployment.md), [config stacks](../configs/README.md).
A Git tag or synchronized development tree is not proof of running source.
This guide was cross-checked against templates on 2026-10-11, not against live hosts.

**Hardware gate:** no motor/HIL, serial or bridge activation, homing or sweeps
until the user confirms repaired wiring and every required RS485 motor answers.
Ordinary camera/SIM source switching does not grant that authority.

## Managed owners

| Host / systemd scope | Units |
|---|---|
| Jetson / system | `idcs-operator-agent`, `idcs-deepstream-video`, `idcs-deepstream-camera`, `idcs-controller-camera`, `idcs-recorder`; hardware `idcs-serial`, `idcs-bridge`, `idcs-controller`; camera/HIL targets |
| PC / user | `idcs-pc-mode-agent`, `idcs-hud-relay`, `idcs-camera-ui`; Single/Swarm streamers and controllers, shared sim panel/UI/recorder/session; separate HIL streamer/UI |
| Pi / user | `idcs-manual` (`rpi.runtime_control`), `idcs-operator-display` |

PC identity need not be fixed for outbound mode-agent coordination, but configured
video destinations still matter. Common bench addresses in config are Jetson
192.168.0.5 and Pi 192.168.0.3; inspect effective config rather than assuming
a PC IP. Do not infer enabled-at-boot state from the table: inspect installed
units and manifests. Motor owners must not be enabled for unattended startup.

## Ordinary source changes

Use the Pi **Operations -> Source & scene** menu: Camera, Simulator -> Single/Swarm, or Standby.
The Jetson operator agent and PC mode agent own the ordered stop/start,
fresh-source readiness checks and rollback. Do not race their transaction by
manually starting another source.

- Camera starts local Argus DeepStream, a shadow controller and the corresponding
  PC viewer. Shadow computes/records without publishing physical intents.
- Single starts the controlled synthetic-observation software mount.
- Swarm starts the detection-fed software mount; simulator truth is separate
  from visual identities and supplies scoring.
- PC Single/Swarm targets share a required session for panel/UI/recorder. Restarting
  the viewer cycles that software session and invalidates old source-scoped locks.
- Motor/HIL activity or unknown unit state blocks menu switching. Mode switching
  does not qualify hardware zeroing, position alignment or physical handover.

Check the source stage and errors in the menu. A cold change can take seconds;
the Pi keeps black fallback and alerts alive while video is unavailable.
Physical joystick/menu acceptance remains separate from command-path tests.

## Read-only status checks

On PC/Pi add `--user`; Jetson services are system scoped.

```bash
systemctl --user status idcs-pc-mode-agent idcs-sim-swarm.target
systemctl --user cat idcs-sim-swarm-controller
journalctl --user -u idcs-sim-swarm-streamer -n 80 --no-pager
systemctl status idcs-deepstream-video idcs-controller-camera
systemctl cat idcs-deepstream-video
```

Inspect host `deployments/managed/<role>-current.json`, unit WorkingDirectory,
actual process cwd/arguments and restart counts together. Check for duplicate
owners before starting any bounded test.

Health files include Jetson `/run/idcs/deepstream-video-health.json`,
HIL controller `/run/idcs-controller/controller-health.json`, and the PC user's
runtime-directory `idcs-sim/controller-health.json`. A stale file is not a live
owner; correlate timestamps and PID. Recordings are summarized with
`python -m tools.flight_log summary RECORDING_DIRECTORY`.

Most video/controller units have configuration-only preflight checks plus
process guards. Not every unit has `--check`. Bridge and serial preflights now
reject missing overlays and duplicate keys; they still do not prove motor or
timing readiness. See the [review](codebase_review_20261011.md).

## Panel and HUD

Jetson encodes one 720p30 return stream and mirrors it to configured receivers.
PC/Pi add operational telemetry locally. E-stop, fire and authority indications
remain visible with the menu open; stale/low-video-fps warnings are separate.
Steady source and authority are not repeated as menu status rows.
See [operator display](operator_display.md) and [HUD design](hud_design.md).

Only GPIO12/16 share lamp power. Their lamps are driven by the documented
visual-target/applied-tracking policies, not permanently powered. Lamp-off means
unreadable contact; GPIO5 manual mode and E-stop are independent.
Do not use a screenshot or software-only GPIO fixture as electrical acceptance.

## Hardware and variants

The physical chain is serial -> bridge -> controller, with shutdown in reverse
while serial remains available for bridge stop/release. Serial-service shutdown
does not itself disable motors. Neither shutdown intent nor a stale screenshot
proves all motors de-energized; actual replies/bench verification are required.

Once hardware work is separately authorized, use the reviewed
[tuning procedure](tuning_procedure.md), exact effective configs and a bounded
test plan. Do not run old journal/trial wrappers as diagnostics. No motor-start
recipe is part of ordinary Camera/Simulator switching.

For software experiments, pause the corresponding owners first, use a bounded
duration and artifact directory outside source, and restore the prior software
mode. Do not install ad hoc overrides over managed units.

## Deployment and recovery

Use explicit stage -> validate -> activate one host -> verify -> audit phases
from [deployment](deployment.md). Commit/tag and activation each require the
appropriate user request. The automated drift gate now checks the union of
old/new `.yaml` and `.yml` config files; separately inspect each resolved unit's
ordered config arguments before activation.

Rollback restores backed-up units and prior software state, not a Git reset of
the development checkout. Retain trial sources, native libraries and evidence.

If Argus fails, inspect its daemon and existing camera owners before any daemon
restart; stopping the daemon disrupts camera users. Argus clients require access
to `/tmp/argus_socket` and cannot use an isolated private copy of it.
For HIL pose refusal, inspect measured feedback and timing; never enable a
simulated fallback to make the readiness check pass.
