# Controller V3 handoff (2026-09-27)

Branch: `v3-pid-hardware-verification`. The canonical host checkout at
`/home/idcs/Desktop/project/IDCS` remains on `main` and has local edits.
Do not switch or reset that checkout. Use the isolated clean checkout at
`/home/idcs/idcs-devtools/stage` for source review and host tests. The
Jetson controller candidate at
`/home/idcs/Desktop/project/IDCS-v2-controller-candidate` is also dirty;
do not overwrite it with the host branch.

## What is qualified

- Unloaded yaw plus pitch-A raw-PID control and synthetic-target video/HIL
  actuation have passed bounded safety and timing gates. Pitch-B is omitted
  and independently guarded. The trial-only pitch proportional gain is 8;
  this is not a real-camera or loaded-system tune.
- Estimator-driven feedforward is separate from PID, is applied, and its
  capture-time motion estimate is accurate in the exact-frame-pose simulator
  path. Matched and within-run FF-off/on hardware-video trials showed no
  consistent two-axis RMS benefit. Do not present FF as effective or enable
  it for production on this evidence.
- PC-to-Jetson software-clock surveys measured offset intervals and
  observed-window constant-slope drift. The `1000 ppm` value used by the
  bounded unloaded HIL trials is a user-approved **test-only assumption**,
  not a qualified future drift bound. The watchdog holds on stale, wide, or
  contradictory exchanges. Physical exposure timing, loaded operation,
  production clock policy, and real-camera control are not qualified.

See `docs/deepstream_migration_journal.md` (latest 2026-09-27 entries) and
`docs/controller_architecture.md` for the measured results and design.
The main source and analyzer entry points are
`jetson/control/video_runtime.py`,
`tools/analyze_video_hil.py`, and
`tools/analyze_video_crossover.py`.

## External trial kit and evidence

The scripts deliberately live outside the Git checkout. On the host,
`/home/idcs/idcs-devtools` contains `run_v3_video_hil_host.sh`, the original
and smooth `v3_video_hil*_fixture.yaml` files, and evidence directories.
On Jetson, the same toolkit root contains `run_v3_video_hil_jetson.sh`,
`v3_video_hil_manifest.sha256`, `v3_overlay/`, the pitch-A-only bridge and
guard, serial event capture, and enable preflight. These files were
SHA-256-compared with the Windows-side toolkit at
`C:\Users\Lab412\idcs-dev`. The manifest pins the deployed overlay and
trial dependencies. The host wrapper pins the host streamer, perception
schema, and fixture hashes. It rejects unrecognized fixtures and missing
`test_only_1000ppm` acknowledgement.

The wrappers **can move unloaded motors**. Do not run them as a status check.
Before any new trial, check controller processes, serial ownership, safety
runtime, source hashes, fixture, and unloaded bench condition. The scripts
enforce bounded duration, 0.2 rad/s rate and 0.15 rad travel limits, a
short intent lease, and a separate pitch-B guard; they do not authorize
loaded or unattended production use. Follow `docs/verification_strategy.md`.

The remote evidence is under `/home/idcs/idcs-devtools/evidence/<trial-label>`
on both host and Jetson. The Windows evidence collection is under
`C:\Users\Lab412\idcs-dev\evidence`. Keep raw traces, serial events,
source-hash checks, and analyzer output with each report. The external
toolkit's `README.md` and `idcs.ps1` document clean-commit sync and tests.

## Next work

1. Design a finer effective rate-command strategy and validate it with
   deterministic serial/plant tests before another motor trial. The F6
   interface emitted only 0 or 1 integer RPM under the trial rate cap;
   typical FF contributions were smaller than one RPM step.
2. Repeat balanced FF-off/on crossovers after that change, retaining the
   exact-frame-pose fixture and independent safety/timing gates.
3. Separately qualify an operational cross-host clock policy and real-camera
   exposure timestamps before claiming production video control readiness.
