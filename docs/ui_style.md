# Shared operational UI

Implemented on 2026-10-08; visibility-first rework and rank/safety/cadence
follow-up were introduced in `deploy-79` and are retained in current `deploy-83`
(`eedb303`) on PC, Jetson and Pi on 2026-10-10. Deploy-81 adds the scene menu
and separates Swarm visual-lock identity from SIM pose/HUD identity; deploy-82
adds receiver-owned Swarm score/Single raw-error metrics and confirmed score reset.
Deploy-83 internalizes tracker IDs and corrects native rank-corner fill and
numeral containment. Native GPU fixture acceptance is distinct from a compressed
live-return multi-target visual check, which remains unconfirmed.
Installed state is authoritative in the host manifests described in
[deployment.md](deployment.md), not the presence of a source folder or Git tag.

The 2026-10-09 visibility-first rework is described in
[hud_design.md](hud_design.md). The ownership/layout rules below describe
the layout introduced in deploy-79, retained in deploy-83 with its reserved metrics strip. The later deploy-77 receipt is historical evidence for the earlier,
more intrusive layout; deploy-78 and deploy-79 acceptance are in the journal.

The later rank-badge/mandatory-safety/menu-cleanup/cadence-warning follow-up is
implemented, offline verified and deployed. See the first sections of the HUD
design guide; range displays whole metres without an approximate marker, with
no change to estimator precision or calibration.

## One theme, separate rendering owners

`common/ui_style.py` owns RGB-byte colors, font names/sizes, status wording,
scale rules and reserved HUD regions. `common/hud_geometry.py` owns display-only
reticle and continuously translated tape geometry. Neither grants authority,
changes timing qualification, or supplies simulator truth to perception.

| Owner | Adapter | Responsibility |
|---|---|---|
| Jetson | `jetson/deepstream/pipeline.py` | Accepted target boxes/labels and target facts in GPU return-video OSD; inference diagnostics only in health/reports |
| PC | `pc/v2_hud.py`, `pc/ui_text.py` | Latest-telemetry HUD and local receiver status |
| Pi | `rpi/display/hud.py`, `rpi/display/render.py` | Latest-telemetry HUD plus local GPIO/menu/fire/emergency information |
| ROI trial | `jetson/visual_lock/live_trial.py` | Isolated opt-in visual-lock OSD using the same theme |

Consumers do not redraw Jetson target boxes. Different backends remain necessary:
the PC uses OpenCV geometry plus cached TrueType masks, Pi uses small cached Cairo
tiles, and Jetson uses native display metadata. No full-frame Pillow conversion
or additional GPU/video owner is introduced. Cairo and GPU antialiasing are not
claimed pixel-identical. GPU label backgrounds remain rectangular because of
the native metadata API; local panels use the shared rounded-corner treatment.
The opt-in engineering MPC term display is outside the operational HUD scope.

## Presentation contract

- Unmodified bundled DejaVu Sans regular/bold and Mono fonts in `assets/fonts/`,
  with their redistribution notice. PC needs the `pc` extra's Pillow dependency.
  Cairo/GPU startup uses `common/ui_fonts.configure_fonts()` before font faces
  are created: a private process-local Fontconfig and exact file-match checks.
  System fonts and configuration are never installed or modified. Missing or
  substituted fonts fail startup rather than silently selecting another style.
- Canonical colors: green tracked, amber uncertain/coasting/hold, red selected
  box/alert, cyan active ROI or feedforward, blue manual/menu highlight, gray
  inactive/unavailable. Mode color identifies tracking provenance/state, not
  motor authority. Existing panel lamp and GPIO policies are unchanged.
- Logical pixels scale from 720-high video. Thin top heading and full-height
  right pitch strips meet without overlap; no Y/P prefix. Two bottom rows hold
  compact local status/console fields and a small FF indicator. No interior mode
  badge, top performance banner or multi-line engineering stack.
- Native target-label positions are clamped away from tapes/status, with bounded
  text and free-row selection; accepted box geometry and identities are unchanged.
  Dense scenes can exhaust free rows: this is not a universal overlap guarantee.
- Jetson labels own selected range/threat/rank; receiver HUDs do not duplicate
  those facts or target geometry. Rates/ages live in reports and Pi's on-demand
  telemetry page. Normal HUD shows source/mode and controller exceptions once.
  Parallax is a geometric aim cue, never a laser/effector capability claim.
- Local receipt age and Pi sample-age data retain their original meaning.
  The HUD explicitly says latest telemetry is not an exact video-frame join.
  No synchronized timing or reduced end-to-end latency is inferred from styling.

## Reproducible offline verification

From a checkout with the relevant display dependencies:

```bash
python -m tools.verify_ui_style --role pc --tests-only --output /outside/repo/pc-tests
python -m tools.verify_ui_style --role pc --frames 300 --baseline-commit COMMIT --output /outside/repo/pc-ui
python -m tools.verify_ui_style --role pi --frames 300 --baseline-commit COMMIT --output /outside/repo/pi-ui
```

Output directories must be new. Ten guaranteed synthetic scenarios at 360p,
720p, 1080p and 800x600 cover SIM, detector, ROI, coasting, stale/missing data,
manual, menu, emergency and screen-edge targets. No learned detector, GPU pipeline,
GPIO sampling, serial owner, real video or motor is started. Mocked GPU metadata
is painted into fixture backgrounds; native GPU font/render acceptance remains
a separate deployment gate.

PC timing includes frame copy, HUD and changing receiver status; Pi timing covers
overlay construction, excluding the native compositor's video blend. Both exclude
fixture generation/background annotation. A 15 ms p95 render-cost gate is not an
end-to-end latency, detector cadence or sustained 60 Hz qualification.

Final evidence: each host's
`~/Desktop/project/deployments/ui-unification-20261008/` outside runtime source.
Contract tests: `contract-tests-04`; PC benchmark `pc-qa-02`, Pi `pi-qa-03`.
PC p95 at 720p/1080p: 9.713/15.393 ms (1080p misses the 15 ms gate).
Pi p95: 8.182/11.289 ms. Preserve earlier failures and reports.

2026-10-09 follow-up: the local regression interruption is resolved. The Pi
session-lock test used a real POSIX `os.kill(pid, 0)` probe, which sends a console
Ctrl+C event on Windows. Lock-policy tests now inject known PID states, with
separate mocked probe error-contract coverage; production lock/GPIO code is
unchanged. The in-process serial test now owns its socket entirely on a bounded
responder thread and checks cancellation cleanup, without modifying serial code.
Full Windows regression: 864 passed, 27 skipped, 2 established exclusions and
17 subtests passed, normal process exit. Four GI streamer modules remain excluded;
mesh loading and low-range plant qualification are not newly accepted here.
These tests do not qualify the Pi's POSIX runtime helper as a Windows runtime API.

Extended device contracts (`contract-tests-20261009`): PC211, Jetson215, Pi37
passed. Shared UI/layout/fonts and bounded IPC checks are now included in managed
release validation; Pi's dependency-free unittest gate also covers GPIO/lock
policies using fake inputs. Check-only PC/Pi display startup succeeded. The
resolved PC return profile remains 720p30 (30fps, port5002, pull timeout50ms).
Those offline checks did not restart deploy-76. The subsequent approved clean
deploy-77 rollout passed PC 222, Jetson 226 and Pi 37 tests, check-only startup and
systemd verification. Actual PC/Pi display captures confirm native GPU labels,
shared HUD regions, mode, feedforward, parallax and status text without clipping
or nested duplicate boxes in the inspected single-target frames. A passive
12.24s perception survey received 706 samples, all with one track. This is not
multi-target, endurance or detector-quality acceptance. Live menu/stale states
were not forced; their acceptance evidence remains deterministic fixtures.

Repeated managed verification: all owners use product `3cdfefc`, no service
restarts; Jetson cumulative pipeline 58.285fps/return 29.553fps, Pi video 30.011fps
and overlay-build p95 9.579ms. The software SIM controller reported 11 missed
periods in 14099 ticks; styling does not qualify control scheduling. No physical
controller/serial/HIL unit was started. Captures and receipts are in each host's
`deployments/managed/deploy-77/<role>/`; local visual copies are outside the repo
at `C:/Users/Lab412/idcs-dev/deploy-77-evidence-20261009/`.

Future deployment must follow [deployment.md](deployment.md): reviewed explicit commit,
clean tagged release, stage/validate/activate/verify. Do not point services at the
offline validation snapshot. After activation inspect actual GPU labels and both
return displays, cadence, stale/missing states and menu overlays. Physical switch
acceptance remains deferred; no hardware or HIL activation is part of UI work.
