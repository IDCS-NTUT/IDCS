# Seed tightening candidate and offline review

Seed refinement runs for every explicit operator seed through the shared camera,
SIM and replay `VisualLockSession`. A new additive `vl_seed_probe` supplies the
native Harris candidates/scores and an operator crop from VPI's own unmodified
U8 gray image **before DCF learning or reference-feature initialization**. Python
decides geometry; the existing ABI4 `VLResult`, `vl_process`, commit/discard and
input-wrapper contracts retain their layouts and semantics. Rebuild the library:
an old ABI4 library without the additive symbol refuses with an explicit error.

The deterministic policy uses in-box positive-score finite corners, with the
existing 3px feature-boundary exclusion. Connected support uses a radius of
`max(12px, 0.18 * min(operator width, height))`. One component must contain at
least the resolved `min_points` and 75% of corners. A separate component with
at least `max(min_points, 25% of the main component)` refuses as ambiguous.
Both support extents must be at least 6px. There is no preference for the centre
of the box. A corner target therefore receives the same treatment.

Each clearly loose axis (support extent <=65% of that operator axis) tightens to
its support plus a margin `min(12px, max(4px, 15% of support extent))` per side.
Other axes stay as drawn. Geometry stays inside the operator box and above the
existing 12px floor. Weak or competing support refuses rather than selecting a
small noise island. These are seed-geometry rules, not changes to the existing
tracking gates. Corners alone cannot distinguish an object from one connected
background texture; this candidate still needs the native/footage gates below.

All `Gates` defaults, Harris strength **0.5**, CLAHE **off**, ports5595-5597,
CONTROL OFF, exact-frame identity and loss latching are unchanged. For example,
28x24px support is 16% x17% of a176x145 operator box, failing the unchanged20%
spread rule. Refinement to36.4x32px makes it77% x75%. No spread threshold needs
relaxing. Replenishment, scale and reference geometry use the refined seed.

Every evidence-bearing seed writes `seeds/seed-00` through `seed-63` as `.pgm`
plus `.json` in the session output (verifier: `<scenario>/seeds/`). PGM is the
actual gray crop, with integer crop origin and full-image corner coordinates in
JSON. JSON includes all in-box detected candidates/scores, operator/refined
boxes, reason, source frame and decision. This seed-only host readback is an
intentional exception to the otherwise GPU-resident frame path; there is no
per-frame image write/readback. The ring overwrites only its own64 slots and
does not prune recordings or other artifacts. Identity/geometry failures before
native input validation cannot have a crop. Probe/native failures latch loss.
Seed records also appear in acquisition, live-frame and replay reports.

## Reviewer commands on Jetson (offline, no motors)

These commands are instructions for the reviewer; the delegated agent has not
contacted lab hosts. Work from this reviewed source checkout, using its normal
`bin/python3`. Before GPU tests check running owners and physical/HIL units as
required by `AGENTS.md`; the guard refuses an active physical stack and an
existing video owner. Only previously active software units are restored.
No camera, motor, bridge, serial, HIL, YOLO or network source is opened here.

```sh
export ART=/tmp/idcs-roi-seed-REVIEW-UNIQUE
mkdir "$ART"
export BUILD="$ART/build"
export PY=bin/python3
export PROVENANCE="$(git rev-parse HEAD)-reviewed-source"
cmake -S jetson/visual_lock/native -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DIDCS_BUILD_TEST_SURFACE=ON -DIDCS_BUILD_LIVE_SURFACE=ON
cmake --build "$BUILD" -j2
export IDCS_VPI_LIBRARY="$BUILD/libidcs_visual_lock.so"
export IDCS_VPI_TEST_SURFACE="$BUILD/libidcs_visual_lock_test_surface.so"
export IDCS_VPI_STANDALONE_EGL=1
export IDCS_VPI_CUDA_INTEROP=1
unset IDCS_VPI_FEATURE_CLAHE
"$PY" -m pytest -q tests/test_seed_refinement.py tests/test_roi_seed_review.py
"$PY" -m tools.run_visual_lock_trial_guard --host jetson \
  --output "$ART/guard-native" -- \
  "$PY" -m pytest -q tests/test_vpi_visual_lock_native.py tests/test_visual_lock_egl_lifetime.py
```

Pass: clean native build, pure/native assertions pass, no unexpected skips,
normal zero exit, no abort during teardown. The new native tests check a loose
synthetic seed, real gray crop, the two supplied hand PNGs, and wall refusal.
Their hand-tight controls are thumb486,298,62,89 and open438,299,124,91; inspect
them on the supplied1280x720 images. Loose seeds are426,281,176,145 on both.
Each image repeats for120 frames. Loose and tight seeds must both initialize,
select at least one frame, with loose selected-frame count and first-loss index
at least as good as the tight control. Wall200,180,120,100 on the thumb image
must refuse at frame0 with zero selected frames. Repeated PNGs qualify frozen
initialization/retention only; actual moving-hand tracking requires recordings.

Prepare an outside-repository `clips.json` with **exactly six** entries naming
the retained clutter inputs and reviewed frame0 seeds. Paths are absolute or
relative to the manifest. Obtain exact boxes from the existing baseline; do not
guess or use an old tracked box from another frame. Example entry shape:

```json
[
  {"name":"seg01", "video":"/review/roi/seg01.avi",
   "truth":"/review/roi/seg01.truth.jsonl", "fps":60,
   "seed_xywh":[100,100,64,48], "baseline":"seg01"}
]
```

The example is a schema illustration, not a runnable six-clip manifest or a
measured seed. `baseline` optionally names the existing report directory
relative to `--baseline`. An entry may instead contain `capture` pointing to an
original exact-frame H264 capture (with its original explicit seed), and `name`;
this uses the bounded NVDEC replay with preserved PTS. Capture replays currently
have no joined truth scoring and show error/false tracking as unavailable.
For video without `truth`, errors and absent-target false tracking are likewise
unavailable; selection coverage remains measurable. Supply labeled video/truth
for a meaningful false-tracking comparison. Match measured fps, including
60000/1001 where appropriate. Original inputs/baselines are read-only.

```sh
"$PY" -m tools.review_roi_seeds --library "$IDCS_VPI_LIBRARY" \
  --output "$ART/review" --baseline /review/retained-baseline \
  --clips /review/clips.json --hand-frames tests/fixtures/camera_roi \
  --provenance "$PROVENANCE"
```

Run the runner **directly**: it invokes ONE trial guard around all offline
replays, with serialized GPU owners. Do not wrap it in another guard. It runs
the six unchanged deterministic120-frame fixtures, loose/tight120-frame hand
repeats, the wall case, then all six clutter clips. Child replays are capped at
360s; capture internal limit is300s; the outer guard command is capped at3600s.
No per-clip40s service restart is needed. Do not pass the private inside-guard
flag manually.

`comparison.md` has baseline/candidate rows for seed accepted, refined xywh,
coverage, error p50/p95, false tracking and first-loss frame/reason.
`comparison.json` also records commands/return codes, input/gate digest matches,
and `baseline_complete`. Baselines produced by this runner use `runs/fixtures`,
`runs/hands`, `runs/clutter-NAME`; existing report roots and per-clip baseline
overrides are supported. Missing baseline values show n/a and
`baseline_complete=false`; that is incomplete comparison evidence, even if the
candidate's internal gates pass. A baseline recorded before tightening uses its
original seed as its refined box when no additive seed report exists.

Pass: the verifier's existing six-fixture acceptance remains **pass** without
gate overrides; hand and wall criteria above pass; every candidate report is
present; every child exits normally (expected capture loss may exit1); no
backend fault or timeout. Clutter is measured, not granted a new automatic
quality threshold: reviewer must require zero false tracking on labeled absent
frames, inspect geometry/coverage/error/loss versus the identical-input baseline,
and explain regressions. Inspect seed crops/corners for background capture.
Static PNG results and these short replays do not qualify live camera cadence,
moving-hand retention, operational target following or any physical actuation.
