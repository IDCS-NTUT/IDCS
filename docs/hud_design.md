# Visibility-first HUD

2026-10-09 visibility-first HUD, committed and deployed as managed release
deploy-78 on PC, Jetson and Pi. The offline verification below precedes
activation; live acceptance is recorded separately in the journal and
[deployment.md](deployment.md).
Current source/owner authority remains the host manifests, not this design guide.

## Drawing ownership

| Information | Only drawing owner |
|---|---|
| Accepted boxes, selected target, identity/class/confidence, range/threat/rank | Jetson GPU OSD, burned into return video |
| Home-relative camera attitude, aim-reference diamond/segment, tracking provenance/state, applied controller status, FF indicator, HUD freshness limitation | One receiver HUD: PC or Pi |
| Decoded return fps | Local receiver, once; not inference cadence |
| Console authority, fire-button indication, planner/operator policy, recording, menu and candidate preview | Pi console overlay |
| Inference timing/rate and detailed command/estimator traces | Reports/health, not default video |

PC and Pi are alternative consumers of the same annotated video, not rendering
layers that are combined. Neither re-creates target boxes, range markers,
person crosses, trails, selected rings, target labels or counts. The Pi may
outline an **unselected menu candidate** before acceptance; no persistent second
LOCK box. Pi console fields no longer repeat source or controller reason.
Fire status shares its console row; a fire-pressed/emergency alert replaces its
duplicate normal indicator. Panel authority and applied controller state are
different facts and are deliberately not inferred from one another.

The isolated ROI trial is a separate owner profile: its GPU OSD owns ROI reticle,
box and state. Its diagnostic/source row and state/control-disabled row use the
same bottom regions without repeating lifecycle state. It is not activated as
an operational console mode by this change.

## Edge instruments and unobstructed video

Logical dimensions scale from a 720-high frame:

- Heading occupies the top 34px; pitch occupies the rightmost 64px at full height.
  Heading spans the entire remaining width, stopping at the pitch corridor.
  Their reserved rectangles meet but never overlap. No Y/P prefix, interior
  mode badge or opaque top performance banner.
- The current heading is centered on the full image's horizontal principal
  point. Current pitch is centered on the full image's vertical principal point.
  Nearby tick labels are suppressed rather than drawn through the readout.
- Pitch ticks use `cy - fy*tan(tick-pitch)`, with `cy=height/2` and focal length
  derived from the **full frame** vertical FOV. Strip clipping does not rescale
  FOV or relocate zero. Heading uses the corresponding rectilinear horizontal
  offset projection, also centered on the image rather than the shortened strip.
- This replaces the incorrect mapping of a full vertical FOV onto the old
  shortened, asymmetric pitch strip. Geometry is verified against the actual
  OpenGL camera projection/view matrices at zero roll. Real lens distortion,
  camera roll, FOV calibration and exact return-frame pose joins are separate
  qualifications; the strip is not a full world-coordinate horizon overlay.
- Normal local status is one bottom row. Above it, Pi console/PC receive status
  stays on the left, with the small FF pair on the right, before the pitch strip.
  Backgrounds cover text only, not wide blank panels. Central video keeps only
  a small reticle and a fresh aim-reference cue (no extra center-to-target ray).
- FF upper bar is yaw and lower bar pitch; centered zero, +/-0.5rad/s display
  scale. Inactive/stale values do not paint active bars. Exact values remain in
  telemetry/reports and Pi's on-demand `HUD telemetry` page, not duplicate text.

`LATEST HUD` explicitly denotes latest telemetry, not a returned-video frame
join. The on-demand page preserves per-sample ages, age basis and command/FF
rates. Stale perception suppresses aim cues; stale pose suppresses tapes.
Styling never grants motor authority or makes a timing/laser capability claim.

## Offline verification

Deterministic checks use guaranteed boxes and frozen telemetry, not YOLO
effectiveness. `tests/test_hud_design.py` covers projection, center/horizon,
corner separation, cropped-strip invariance, ownership and visibility. Included
in `tools.verify_ui_style` and managed deployment validation (Pi unittest too).
Fixtures cover SIM, YOLO, ROI, coasting, stale/missing data, manual/menu/emergency,
edge targets, +/-20deg pitch and heading wrap at four resolutions.

Evidence outside source on each device: `deployments/hud-rework-20261009/`.
Earlier failures/reports are retained. These offline results do not establish
native GPU/live-return acceptance; that requires the approved clean managed
release. Do not install dirty verification snapshots or trial overrides as runtime.

Final evidence: PC `pc-tests-05` 242 passed/1 platform skip/5 subtests;
Jetson `jetson-tests-05` 249 passed; Pi `pi-tests-05` 51 unittest passed.
Full supported Windows regression 873 passed/32 skips/2 established exclusions/
17 subtests, normal exit. Four GI streamer modules excluded on Windows.
Final CPU render benchmarks (`pc-qa-05`, `pi-qa-05`), 300 changing frames/size:
PC p95 360p/720p/1080p/800x600 = 2.437/3.989/5.779/3.095ms;
Pi = 3.960/4.913/6.967/4.642ms. All below the 15ms render-cost gate.
PC includes frame copy/render; Pi measures overlay construction, not native video
blend. These do not establish end-to-end latency or sustained inference/control
cadence. Normal 720p synthetic receiver-affected pixel fraction: PC 1.424%,
Pi 1.692%, excluding Jetson's existing annotations. Menus/alerts intentionally
occupy more video. This measures changed pixels, not perceived readability or
all-scene coverage. Actual backend fixtures were visually inspected.
