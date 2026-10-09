# Shared operational UI

Implemented on 2026-10-08; release candidate `deploy-77` approved on 2026-10-09.
Installed state is authoritative in the host manifests described in
[deployment.md](deployment.md), not the presence of a source folder or Git tag.

## One theme, separate rendering owners

`common/ui_style.py` owns RGB-byte colors, font names/sizes, status wording,
scale rules and reserved HUD regions. `common/hud_geometry.py` owns display-only
reticle and continuously translated tape geometry. Neither grants authority,
changes timing qualification, or supplies simulator truth to perception.

| Owner | Adapter | Responsibility |
|---|---|---|
| Jetson | `jetson/deepstream/pipeline.py` | Accepted target boxes/labels and inference status in GPU return-video OSD |
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
- Logical pixels scale from 720-high video. Shared top heading tape, right pitch
  tape, upper-right tracking-mode badge, lower-left feedforward and status regions
  keep the center reticle clear. PC receiver status is in the top reserved area.
- Native target-label positions are clamped away from tapes/status, with bounded
  text and free-row selection; accepted box geometry and identities are unchanged.
  Dense scenes can exhaust free rows: this is not a universal overlap guarantee.
- Source/pose/selected range-threat-rank, controller rates, parallax aim offset,
  age basis and frame-join limitations use the shared wording. Parallax is only
  a geometric aim cue, never a laser/effector capability claim.
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
Native GPU/live return visual acceptance still requires the approved new release;
the existing deploy-76 services have not been restarted by these offline checks.

Deployment must follow [deployment.md](deployment.md): reviewed explicit commit,
clean tagged release, stage/validate/activate/verify. Do not point services at the
offline validation snapshot. After activation inspect actual GPU labels and both
return displays, cadence, stale/missing states and menu overlays. Physical switch
acceptance remains deferred; no hardware or HIL activation is part of UI work.
