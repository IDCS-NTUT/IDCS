# Camera ROI candidate and reviewer validation

This branch adds an opt-in **isolated VPI camera session** to
`jetson.visual_lock.live_trial --source camera`. It does not add an operational
camera mode, alter an installed unit, enable a controller, or run beside YOLO.
Nothing has been deployed or tested on a lab host by the delegated agent.

## Data flow and implemented boundaries

1. One `nvarguscamerasrc` owns 1280x720 NV12 at 60/1. Sensor ID/mode, white
   balance and flip method are explicit CLI settings. Match the reviewed camera
   configuration; defaults are ID0/mode4/WB1/flip0, not inferred host settings.
2. Before conversion/mux, Argus qdata supplies the sensor frame number and SOF
   on `jetson_monotonic`. A locked, bounded PTS index joins only the exact
   `NvDsFrameMeta.buf_pts`. Conflicting PTS are poisoned. Missing metadata,
   duplicate/backward PTS, sensor frame or SOF terminate with an explicit error.
   No PC header socket, native counter fallback or cross-host clock subtraction
   participates in camera identity.
3. The existing pipeline-owned NVMM clone cache retains up to 16 unannotated
   GPU images and 400ms of source history. Eviction closes owned images. The
   panel rectangle names session, sensor frame, SOF, clock and geometry; an
   absent/stale exact image is refused, never replaced with the nearest image.
4. The existing VPI session seeds that image and performs bounded cached-image
   catch-up, preserving exact seed/end images and the source-gap guard. Default
   ABI4 texture/flow/appearance/acquisition gates are unchanged. The first lost
   result stops acquisition and its `seed:`/`catch-up:` reason reaches the panel.
   Acquisition acceptance can mean **acquiring**; only the existing three good
   frames permit V2 selection. Loss never reseeds or falls back to YOLO.
5. V2 observations use the actual processed camera identity and isolated port
   **5597**. Even a fixed-seed camera trial refuses ordinary port5564. GPU OSD
   says CAMERA and CONTROL OFF. Encoder input retains native frame metadata;
   its actual PTS joins the encoded return RTP SSRC/timestamp after final sender
   handling. Pi joins that key to the picture actually displayed. Missing or
   conflicting joins withhold selection requests.
6. Pi's existing opt-in `--visual-roi-trial` accepts fresh control-disabled SIM
   or camera status, with matching session and the appropriate source clock.
   Operations -> **Visual ROI tracking (isolated)** opens the same display-only
   workflow. Manual ON and a fresh readable GPIO12 OFF-to-ON edge are still
   required. OFF releases. In OFF/selecting, Select (configured `menu_select`
   button or USB Enter) toggles joystick movement/resize; the cursor says ROI
   SIZE while resizing. Width grows right, height grows up, with image bounds
   and a 12px minimum. This adds no GPIO assignment or motor command.
7. A texture refusal displays its actual reason and suggests OFF, then
   Select/Enter to resize. A larger box can include background and does not
   establish correct target tracking. Release/cancel/stale panel/source/video,
   E-stop and Manual OFF retain the existing fail-closed behavior. The SIM
   follow lease remains disabled for both sources.

`--gates-json /OUTSIDE-REPO/gates.json` is available in the live trial and
`tools.verify_visual_lock`. It accepts a partial JSON object of `Gates` fields,
for example `{"min_points": 12, "min_psr": 7.5}`. Unknown fields, nonnumeric,
nonfinite and invalid gate values fail before GPU startup. Resolved gates and
their SHA256 are recorded. This is an evaluation mechanism, not evidence to
lower any gate. Native Harris/DCF algorithm parameters and the fixed 12px
geometry floor are not new configuration surfaces.

## Limits and open decisions

The ordinary detector runtime and its service configuration are unchanged.
Panel sessions are VPI-only; combining panel ROI and `--handoff-nvdcf` explicitly
refuses because native tracker reset across release is not qualified. Existing
fixed-seed handoff trials remain available. Resident YOLO switching, automatic
camera follow, depth/range, physical panel acceptance, sustained 60Hz and
transport/photon latency are not implemented/qualified by this increment.

Raw Argus has no pre-decode H264 AU stream. Its report explicitly marks input
capture unsupported; the empty `input.h264` is **not** a replay recording.
Return H264 and frame/acquisition identities are retained, but return video is
annotated and is not an unannotated tracker input. Use the existing independent
real-footage replay below for quality evaluation. Adding a bounded raw-camera
recording branch is a separate lead decision; no CPU pixel readback was added.

Expect many small, white 50-140px drones to remain texture-refused or lose on
forward/backward support with ABI4. The supplied six-clutter-segment evidence
(five refusals, one 23-frame track then loss, about25px error) does not support
relaxing gates. A decision needs identical hand-verified seeds, fixed footage,
recorded gate/library/input digests, and per-seed acceptance reason, feature
count, time to loss, coverage and localization error against held-out truth.
Evaluate clutter and sky separately and include target pixel size, background
capture/identity switches and absent-target false tracking. YOLO detection on
sky is not VPI texture evidence. Any proposed gate change must improve useful
coverage/localization without hiding background tracking or loss. The lead must
choose camera quality criteria before operational acceptance; existing SIM
95% coverage/3px p95/8px peak gates are comparison references, not proof for
small real drones.

## Reviewer commands (needs Jetson unless stated otherwise)

Use the reviewed source checkout and an unused **outside-repository** artifact
directory. These commands have not been run remotely. Before every native/GPU
run, check owners as required by `AGENTS.md`; use the existing software guard
for GPU replay/camera runs. Never start hardware/HIL/bridge/serial units.
The guard refuses an active physical stack, scopes software pauses and restores
only previously active software units. Do not run another native/GPU suite
concurrently. Record commit, dirty diff digest, SDK versions and model hashes
(no model is an input to VPI).

```sh
export ART=/tmp/idcs-camera-roi-REVIEW-UNIQUE
mkdir "$ART"
export BUILD="$ART/build"
export PROVENANCE="$(git rev-parse HEAD)-plus-reviewed-diff-digest"
cmake -S jetson/visual_lock/native -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DIDCS_BUILD_TEST_SURFACE=ON -DIDCS_BUILD_LIVE_SURFACE=ON
cmake --build "$BUILD" -j2
export IDCS_VPI_LIBRARY="$BUILD/libidcs_visual_lock.so"
export IDCS_VPI_TEST_SURFACE="$BUILD/libidcs_visual_lock_test_surface.so"
export IDCS_VPI_STANDALONE_EGL=1
export IDCS_VPI_CUDA_INTEROP=1
python -m compileall -q common jetson/visual_lock rpi/display rpi/operator_display.py tools/verify_visual_lock.py
python -m pytest -q tests/test_camera_roi.py tests/test_visual_lock.py \
  tests/test_visual_roi_seed.py tests/test_visual_lock_acquisition.py \
  tests/test_visual_lock_live_trial.py tests/test_visual_lock_return_identity.py \
  tests/test_return_frame_identity.py tests/test_visual_lock_replay_tools.py \
  tests/test_roi_editor.py tests/test_roi_panel_workflow.py \
  tests/test_deepstream_metadata_adapter.py
```

Pass: clean build/compile, all pure tests pass. Run native suites only with
exclusive GPU ownership (the following guard wraps them). Missing installed
native prerequisites or an unexpected skip is not a native pass.

```sh
python -m tools.run_visual_lock_trial_guard --host jetson --output "$ART/guard-native" -- \
  python -m pytest -q tests/test_vpi_visual_lock_native.py tests/test_visual_lock_egl_lifetime.py
python -m tools.run_visual_lock_trial_guard --host jetson --output "$ART/guard-lifetime" -- \
  python -m tools.verify_vpi_process_lifecycle --library "$IDCS_VPI_LIBRARY" \
  --surface-library "$IDCS_VPI_TEST_SURFACE" --output "$ART/lifecycle.json" \
  --repeat 5 --provenance "$PROVENANCE"
python -m tools.run_visual_lock_trial_guard --host jetson --output "$ART/guard-fixtures" -- \
  python -m tools.verify_visual_lock --library "$IDCS_VPI_LIBRARY" \
  --standalone-egl --output "$ART/fixtures" --provenance "$PROVENANCE"
```

Pass: normal process exits, all six fixture gates pass (translation, scale,
distractor localization and textureless/occlusion/disappearance loss), no
teardown abort. CUDA/EGL opt-ins and environment are documented in
`visual_lock.md`; run those separately when qualifying NVMM wrappers.

Replay the six supplied segments **without modifying originals**:

```sh
export ROI="$HOME/Desktop/project/IDCS-trials/clutter-video-2026-10-14/roi"
for video in "$ROI"/seg*.avi; do
  stem=$(basename "$video" .avi)
  python -m tools.run_visual_lock_trial_guard --host jetson --output "$ART/guard-$stem" -- \
    python -m tools.verify_visual_lock --library "$IDCS_VPI_LIBRARY" \
    --standalone-egl --video "$video" --truth "$ROI/$stem.truth.jsonl" --fps 60 \
    --output "$ART/$stem" --provenance "$PROVENANCE" || break
done
```

The first truth box supplies the one-time test-operator seed only. Verify it
matches the prior hand-verified frame0 box; otherwise provide the original
`--seed-xywh X Y W H`. Match the original measured `--fps`; use
59.94005994005994 for 60000/1001 clips. Repeat the same command on the sky clips
with their actual video/truth paths; those paths were not supplied to this
agent. Replay exit0 means **measurement completed**, not tracking acceptance.
Compare the per-frame rows/report to retained `out*/`: default outcomes must
remain explainable against baseline. Expected texture refusals are evidence,
not fixture transport failures. Repeat variants with `--gates-json` and a new
output/guard directory for every run; never overwrite baseline evidence.

For bounded camera/Pi integration after offline gates, set `RETURN_HOST` to
the intended receiver IP and use the reviewed camera orientation/WB settings:

```sh
python -m tools.run_visual_lock_trial_guard --host jetson --output "$ART/guard-camera" -- \
  python -m jetson.visual_lock.live_trial --source camera \
  --library "$IDCS_VPI_LIBRARY" --surface-library "$BUILD/libidcs_visual_lock_live_surface.so" \
  --output "$ART/camera" --duration-s 60 --source-wait-s 30 \
  --argus-sensor-id 0 --argus-sensor-mode 4 --argus-wb-mode 1 --argus-flip-method 0 \
  --return-host "$RETURN_HOST" --return-fps 30 \
  --snapshot-bind tcp://0.0.0.0:5597 --return-identity-bind tcp://0.0.0.0:5595 \
  --roi-bind tcp://0.0.0.0:5596 --provenance "$PROVENANCE"
```

On Pi, first run `python -m pytest -q tests/test_operator_menu.py
tests/test_roi_panel_workflow.py tests/test_roi_editor.py
tests/test_return_frame_identity.py`, then an exclusive bounded display under
its software guard:

```sh
python -m tools.run_visual_lock_trial_guard --host pi --output /tmp/idcs-camera-roi-pi-UNIQUE -- \
  python -m rpi.operator_display --config configs/base --visual-roi-trial \
  --duration-s 90 --report /tmp/idcs-camera-roi-pi-UNIQUE-report.json
```

Verify the config's ROI identity/seed/perception endpoints match Jetson5595/5596/5597.
Do not reuse the SIM session helper: it starts a PC streamer. Pass requires
exact Argus request/cache/acquisition/end-frame identities, return join counts
without mismatches, readable CAMERA/CONTROL OFF, a blank-box texture refusal
shown on Pi, fresh OFF + resize + explicit corrected seed, release/cancel and
Manual-OFF cleanup. Check that selected V2 output stops on release/loss and no
controller lease is enabled. No truth coordinates may drive production updates.
Exercise stale/wrong-session/missing identity via deterministic tests, not a
physical link disturbance. Archive reports and inspect screenshots; UI/native
pixel correctness cannot be qualified on Windows. Real panel operation is a
user bench check; software substitution does not qualify wiring or motors.

The generic trial exit code includes aggregate sustained tracking. Deliberate
blank refusals and selection-off periods can produce exit1; retain it and
judge the explicit workflow/acquisition audit separately. Healthy transport
alone is not tracking acceptance. Require normal teardown, no residual trial
owners and restoration only of previously active software. Do not stage,
activate or sync a managed release as part of these checks.
