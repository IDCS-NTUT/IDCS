# Detector-independent visual lock (experimental)

Implemented 2026-10-07: NVIDIA **VPI CUDA DCF + Harris + forward/backward PyrLK**,
not DeepStream NvDCF and not OpenCV tracking. No YOLO output is an input.
OpenCV in the verifier only decodes, renders and annotates pixels.

## Status and boundaries

The deploy-90 candidate adds the GPU frame cache and exact return identity
sideband for an isolated Pi panel session. Its receiver publishes fresh
`visual_roi_status_v1` status tied to the video session; this permits the ROI
menu while the normal operator agent is paused. Pi ROI observations use
port5597, separately from detector/controller observations. The exclusive
guard can replace the Pi display while retaining its GPIO runtime, then
restore the normal display. Managed installation and a bounded live session
are separate operations. Read the latest journal for their actual outcomes.

The isolated tracking engine and source/session coordinator are built and tested
on Jetson VPI 4.1.3. Accepted observations produce strict `PerceptionSnapshotV2`
tracks/selection. This is **not yet a deployed console mode**: the existing Pi
picker, DeepStream pipeline and simulator controller are unchanged.

The first committed installation was `24c824f` / `deploy-75`; its retained
Jetson ABI4 tracker, test uploader and live-adapter builds are under
`~/Desktop/project/deployments/release-75-20261007/native-build/`. Those are
historical binary/evidence paths, not current service source directories.
The last recorded consolidated deployment is deploy-84; read the host manifests
and [deployment guide](deployment.md) for installed/running sources.

The release-75 build check passed three native GPU/lifetime tests with normal
exit (three opt-in interop cases skipped). Later standalone/handoff trial
results below have their own scope. No ROI mode is integrated into the Pi
console; code/binary installation is not activation or sustained60Hz acceptance.
No live owners were inspected or changed during the 2026-10-11 documentation review.

A separate **control-free live trial** now connects actual OpenGL/H264 RTP input
to pipeline-owned NVMM/EGL VPI processing, GPU OSD, V2 publication and RTP return
to the existing PC/Pi displays. A closer-range moving 0.35m drone retained
2,010/2,010 post-acquisition observations in the bounded receiver canary; normal
exit and original software-service restoration passed. The smaller ~38px drone
tracked initially, then correctly latched lost on insufficient reliable features.
This is not broad target-quality or sustained 60Hz qualification. Neither trial
uses YOLO, simulator truth, a motion controller or motors. Exact displayed-frame
ROI editing/cache and the console menu integration remain pending.
The deploy-88 Pi release candidate later packages a display-only, opt-in ROI
panel path and exact-frame join modules. This does not change the historical
trial above: the managed display unit does not enable ROI, and no managed
Jetson ROI return owner or three-host live qualification is included. Its
GPIO12 OFF selection / readable OFF-to-ON lock never grants auto-follow.
The longer viewing run **lost after ~66 seconds / 3,248 selected observations**
when reliable feature count fell to seven, despite PSR remaining29-32. Return
video stayed healthy. Those short canary passes were NOT endurance acceptance;
the following reliability work addresses that failure without automatic reseeding.
Reliability rework now captures unannotated input and exact processed-frame
identities. Frozen ABI2 replay reproduces the subsequent 41.66s failure exactly.
ABI3 retention/reference-coordinate changes pass that 120s replay but fail a
scored 180s live trial (7.27px seed-relative p95, loss around150s). Current ABI4
adds independent flow against an immutable seed-image pyramid. It passes the
exact 180s failure replay, a 15-minute live run (36,380/36,380 selected, seed-relative
p95 0.609px), and a fresh five-minute twice-speed run (12,084/12,084, p95 2.966px).
Both live runs have zero uncertain/lost observations, normal exit and restored
software services. The fast case is close to the predeclared 3px gate; these are
bounded simulated-target results, not broad operational acceptance. Errors measure
drift of the initially selected ROI point, not target centring. Raw centre errors
and initial offsets remain reported. Full live throughput is about40.4Hz with
tracking-processing p95 about17.1ms, **not sustained60Hz qualified**. No trial owner
remains active. See the latest journal entry for artifacts and remaining gates.

**Standalone CUDA-pointer and NVMM/EGL-image validation now passes**, including
normal process exit. The tracker-free baseline still aborts when EGL is not
initialized before CUDA/VPI. `StandaloneEGL` explicitly initializes NVIDIA EGL
before GPU use and terminates it after wrapper/surface/allocation cleanup. Five
unloaded and three service-loaded process repetitions pass without `_Exit`,
device reset, driver changes or host-frame fallback. The baseline is retained as
`tools.probe_vpi_cuda_interop --without-egl-bootstrap` (expected abort on this SDK).
NVIDIA's [raw-pointer sample](https://docs.nvidia.com/vpi/sample_wrap_raw_ptr.html)
documents the same finalizer class but bypasses exit; we do not use that bypass.

`StandaloneEGL` is **only for a fresh isolated process**. Do not inject another
default-display owner into an existing DeepStream process or terminate its shared
display. Live integration must initialize its display before GPU use, retain it
until every pipeline/VPI surface is released, and qualify that shutdown ordering
separately. GPU input still requires explicit opt-in until that integration passes.
The four original camera/shadow-controller/operator/recorder services were paused
for unloaded tests and restored; all physical/HIL services remain inactive.

## Separation of responsibilities

- Native C++: one bounded CUDA stream, DCF appearance/localization, native-pixel
  Harris ROI, at most 64 feature correspondences, forward/backward flow and a
  24x24 response map. Feature points/response only are read back to CPU.
- Validated adjacent features preserve identities and are selectively
  replenished, not wholesale re-detected every frame. Coalesced points are
  removed. An immutable seed-image pyramid supplies independent reference
  forward/backward flow, rather than letting adjacent LK drift indefinitely.
- `fusion.py`: bounded deterministic similarity fitting, appearance/flow agreement,
  scale refinement, spatial support and lifecycle gates. No source-clock subtraction
  against another host's observation clock.
- `session.py`: exact-frame operator seed, source/session/geometry/order checks
  before GPU work, GPU resource cleanup, backend-fault loss latch, V2 publication.
- Replay verifier: pixels + one initial operator rectangle go to tracking.
  Labels/scene truth are only a scoring oracle after initialization.

Lifecycle: idle -> explicit seed -> acquiring (3 consecutive agreeing frames)
-> tracking. Rejection immediately removes V2 selection and template updates;
three consecutive rejected frames latch lost. Uncertain recovery needs three good
frames again. Lost requires an explicit fresh seed; no YOLO/planner fallback.
Namespace, clock-domain, image dimensions, duplicate/out-of-order timestamps or
a >150ms source gap latch loss immediately. A native/commit fault also suppresses
selection. Downstream consumers must retain their independent stale-output watchdog.

The arbitrary rectangle has class `visual`, **no invented known-size depth or
parallax correction**. Track confidence 1.0 denotes gate acceptance, not a
calibrated probability. The original seed frame remains internal provenance;
every selected V2 observation names its current source/applied frame.

Seed rectangles must tightly enclose the visible object and have enough distributed
texture. Minimum box side is 12 pixels; eight reliable features are required.
Harris works on an unscaled GPU image view, not the resampled DCF patch. Its
initial window allows ~1.5x size growth; exceeding it requires a new seed.
Uniform targets, tiny objects, long occlusions, very fast motion and indistinguishable
crossing objects are not guaranteed. PSR 6 and the other default thresholds are
controlled-fixture gates, not calibrated real-camera acceptance.

## Build and controlled tests

Use an isolated source snapshot and artifact directory. Check running test owners
and the physical/HIL units first; never run a second test owner or activate motors.

```sh
cmake -S jetson/visual_lock/native -B /tmp/idcs-vpi-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/idcs-vpi-build -j2
IDCS_VPI_LIBRARY=/tmp/idcs-vpi-build/libidcs_visual_lock.so \
  python -m pytest -q tests/test_visual_lock.py tests/test_vpi_visual_lock_native.py
python -m tools.verify_visual_lock \
  --library /tmp/idcs-vpi-build/libidcs_visual_lock.so \
  --output /tmp/idcs-vpi-fixtures --provenance COMMIT-plus-dirty-source-digest
```

The output directory must not exist. Six bounded 120-frame fixtures test translation,
scale, approaching/overlapping distractor, occlusion, disappearance and texturelessness.
JSON reports include seed, scenario/fps/resolution, gates/digest, library/input hashes,
clock domains in V2 records, versions, accuracy and processing latency. No sockets
are opened. Use `--video`, `--truth` and `--seed-xywh X Y W H` for a replay.
Without `--seed-xywh`, the first truth box simulates operator initialization ONLY.

On the PC, `tools.render_visual_lock_fixture` records the actual OpenGL drone mesh,
lossless FFV1 video, labels and GPU/scene provenance without a streamer/controller.
No CPU rendering fallback is accepted. It neither edits config nor changes services.

Build the optional **test-only** NVMM uploader and run the normal-exit gates:

```sh
cmake -S jetson/visual_lock/native -B /tmp/idcs-vpi-build \
  -DCMAKE_BUILD_TYPE=Release -DIDCS_BUILD_TEST_SURFACE=ON
cmake --build /tmp/idcs-vpi-build -j2
python -m tools.verify_vpi_process_lifecycle \
  --library /tmp/idcs-vpi-build/libidcs_visual_lock.so \
  --surface-library /tmp/idcs-vpi-build/libidcs_visual_lock_test_surface.so \
  --output /tmp/idcs-vpi-lifecycle.json --repeat 5 --provenance COMMIT-plus-source-digest
python -m tools.verify_visual_lock \
  --library /tmp/idcs-vpi-build/libidcs_visual_lock.so \
  --surface-library /tmp/idcs-vpi-build/libidcs_visual_lock_test_surface.so \
  --input-memory egl --standalone-egl --pace --preload \
  --output /tmp/idcs-vpi-egl-fixtures --provenance COMMIT-plus-source-digest
```

Every child must **exit normally with status zero**. Assertions followed by an abort
remain failure. Each child exercises host/CUDA/EGL wrappers, commit/discard, input
retention and repeated resource destruction. The uploader owns one replay surface;
CPU-to-GPU upload is intentional test input, not a production CPU-readback path.
`--input-memory cuda` exercises a CUDA allocation instead. For direct native tests,
set `IDCS_VPI_LIBRARY`, `IDCS_VPI_CUDA_INTEROP=1`, `IDCS_VPI_STANDALONE_EGL=1` and
`IDCS_VPI_TEST_SURFACE` to the built surface library.

## Evidence and remaining work

Final evidence and artifact paths are in `docs/deepstream_migration_journal.md`.
Controlled images pass; the actual 1280x720, 60fps slow drone clip passes with a
tight seed, but a tall simulator billboard truth rectangle loses spatial support.
This is tracker evidence, not detector/model, real-camera, UI or controller acceptance.
With the camera GPU workload paused, the 180-frame **ready-frame** 60Hz test passes:
CUDA processing median/p95 **8.29/10.12ms**, NVMM/EGL **11.16/12.29ms**; including
replay colour conversion, p95 **10.45/12.65ms** and zero measured frame-work budget
overruns. Both retain 177/177 selected drone frames, centre error p95 **2.322px**.
Input preloading is capped at 512MiB/600 frames; hashing and PNG writes are outside
paced delivery, single-threaded OpenCV is used for replay conversion, and seed
allocation is outside steady-state pacing. The streaming FFV1 runs are retained
and miss the cadence because of replay harness costs. No synthetic clock has been
used as a live-camera timing measurement. These short standalone tests do **not**
qualify sustained full live transport/UI/controller operation or concurrent YOLO.
The deliberate restored-service A/B run retains the same tracking accuracy but
NVMM/EGL processing rises to **27.16ms median / 32.24ms p95**, with 173/177 measured
frame-work periods over 16.67ms. Therefore a future visual-lock mode must be an
alternative to the existing detector/NvDCF workload, not an unqualified second
full GPU pipeline running alongside it. No active detector mode was changed.

Live trial lifetime/startup/shutdown are now exercised independently of the
production detector runtime. Next add a bounded GPU-frame cache for exact displayed-frame ROI
seeding, connect Pi joystick ROI editing/confirm/cancel and explicit release, expose
quality/status overlays, replace selected observation authority without YOLO fallback,
and route actual visual V2 observations to the stable simulator controller (not its
ground-truth path). Measure return-video alignment, drop/age watchdogs and sustained
cadence, then test pure-software closed-loop motion. Manual motor authority and
physical engagement interlocks remain separate and untouched.

## Control-free live return trial

Build with `-DIDCS_BUILD_LIVE_SURFACE=ON`; this adds
`libidcs_visual_lock_live_surface.so`. Tracker ABI is **4**, so rebuild the main
library too. The live helper borrows exactly one 1280x720 RGBA NvBufSurface and
references its GstBuffer until synchronous VPI processing completes. Release
the VPI input wrapper **before** unmapping an owned EGLImage/unreferencing the
buffer; internal grayscale/pyramids retain their own pixels. No pixel planes
are read back or queued on the CPU.

`tools.run_visual_lock_trial_guard` is a software-only guard intended for a
bounded transient systemd unit. On Jetson it refuses an active physical/HIL
stack, pauses only the previously active camera/shadow-controller/operator
units, and restores them afterwards. On PC it pauses/restores only the mode
agent; the existing return viewer stays active. It takes a per-host exclusive
trial lock and refuses an existing video owner after its scoped pauses. Do not
stop GPIO, SSH/remote access, bridge or serial services for this test.

Run the receiver first; **wait for its new `ready.json`** before the source.
For example, inside an isolated snapshot under that guard:

```sh
python -m jetson.visual_lock.live_trial \
  --library build-live/libidcs_visual_lock.so \
  --surface-library build-live/libidcs_visual_lock_live_surface.so \
  --output NEW-ARTIFACT-DIRECTORY --duration-s 60 \
  --seed-xywh 547 255 74 46 --return-fps 30 \
  --return-host 192.168.0.1 --return-host 192.168.0.3 \
  --provenance COMMIT-plus-code-source-digest
python -m pc.streamer --config configs/base \
  --config-extra configs/sim/v2_scene.yaml,configs/sim/visual_lock_trial.yaml \
  --source sim --verified-rtp-headers --duration-s 60
```

The overlay preserves physical mesh size and camera projection, but uses closer
range and slower motion to isolate live transport/ownership from the unfinished
displayed-frame ROI workflow. It does not edit production settings. Multiple
overlays belong in **one comma-separated `--config-extra` argument**; repeating
that flag keeps only the last occurrence.

The fixed rectangle is an explicit, one-time **trial seed**, not automatic
localization or a qualified UI selection protocol. The owning streaming thread
warms NVMM/CUDA and allocates the actual tracker payload before seeding. Eight
consecutive source intervals of at most 50ms are required before that initial
seed. Afterwards the original 150ms gap and all quality gates remain unchanged;
lost never automatically reseeds. Every accepted image is joined to its exact
PC SSRC/RTP timestamp and decoded PTS header before any tracking authority.
`--duration-s` starts at the first verified video frame, not process launch;
`--source-wait-s` separately bounds missing-source startup (default 60 seconds).
Give the transient unit enough wall time for both bounds plus service restoration.

Return OSD says `LIVE SIM | VPI: tracking | ... | NO YOLO / CONTROL OFF`.
Green is the accepted visual lock; cyan marks feature points. The return video
is display-only, 1280x720 with a 30fps ceiling, mirrored to UDP5002 on PC and Pi.
Trial UI is not the production YOLO/NvDCF mode. `frames.jsonl`, `health.json`,
`report.json`, a bounded 20MB encoded return sample and guard status/log record
acceptance, faults, exact identities, library hashes and restoration. Processing
latency excludes network/display; frame-work additionally includes NVMM borrowing
and publication. Source and observation clocks are never subtracted across hosts.
The report marks sustained tracking separately from transport/normal exit; a
brief initial lock followed by loss must not pass the sustained tracking gate.
The final 30-second live-video canary retained 1,454/1,454 selected updates,
about 49.45Hz observations, session processing p95 12.95ms and borrowing/tracking/
publication frame-work p95 13.24ms. Thus live transport/return and this target
case pass, but the complete pipeline is **not yet sustained-60Hz qualified**.

## Reliability capture and independent scoring

Every new live trial records `input.h264` BEFORE decode/OSD, plus `input.jsonl`
with exact AU index/offset/digest/PTS. `frames.jsonl` names the exact verified
processed PTS/frame/source time and seed, feature lifecycle, independent-reference
support, DCF box, scale confidence and template updates. `timing_by_state` avoids
mistaking near-zero work on lost frames for tracker performance. Input evidence
is capped at whole AUs (default256MB, explicit `--capture-max-mb` up to1024MB);
check `input_capture.stop_reason` before claiming a complete replay window.

Run replay under the same exclusive software guard, with no other GPU owner:

```sh
python -m tools.replay_visual_lock_capture --capture CAPTURE-DIRECTORY \
  --library build-live/libidcs_visual_lock.so --output NEW-REPLAY-DIRECTORY \
  --max-wall-s 360 --provenance COMMIT-plus-source-digest
```

Replay starts at a captured IDR (retaining captured parameter sets), decodes with
original Gst PTS on NVDEC, and processes only the original verified-frame sequence.
The seed must be in that decodable window. This is tracking evidence, NOT live
throughput or cross-host latency qualification. No pixels are read back in replay.

For live localization scoring, the simulator source can separately record
`--dump-truth /OUTSIDE-REPO/truth.jsonl` without copying video or publishing truth
to the tracker. After BOTH files stop changing, run:

```sh
python -m tools.score_visual_lock --frames TRIAL/frames.jsonl \
  --truth /OUTSIDE-REPO/truth.jsonl --target-id 1 --output NEW-SCORE.json
```

This compares projected scene centres, not mesh-mask IoU. It reports raw centre
error and the initial ROI offset separately, then scores drift of that initially
selected point under known projection changes. Require >=95% selection coverage,
seed-relative p95<=3px and peak<=8px; do not claim target centring from offset-
compensated scores. The faster test overlay changes only speed/acceleration;
physical target size and camera projection remain fixed. Camera movement/control,
real clutter, displayed-frame ROI editing and sustained60Hz remain separate gates.

## One-time VPI initialization, then native NvDCF

The opt-in `--handoff-nvdcf` path is a **control-free trial**, not a deployed Pi
menu item or a resident-YOLO mode manager. One decoded GPU video pipeline contains
VPI initialization and the actual DeepStream `nvtracker` plugin. No YOLO inference,
truth subscription or motion controller runs in that pipeline.

VPI must first pass its normal three-frame acquisition gates. Its current verified
frame/rectangle is registered once as an explicit external `visual` region
(component9001/class65535), not a YOLO classification. NvDCF learns a fresh filter
from that image; no private VPI filter state is transferred. VPI resources close
on the upstream owner thread immediately after this registration. On the exact
native seed output, position/size must agree with the validated VPI region; then
three consecutive confident native observations confirm the handoff. There is no
ongoing VPI/NvDCF double tracking and no repeated seed metadata.

Subsequent frames have `bInferDone=false` and no input objects. This matters:
"inference skipped" is different from "detector ran and found nothing" in the
[NvDCF input workflow](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_plugin_gst-nvtracker.html#workflow-and-core-modules-in-the-nvmultiobjecttracker-library).
The trial derives its own one-target profile from the production NvDCF profile,
changing only capacity1, probationAge0 (VPI already validated the seed), and no
shadow-track output. Existing native confidence0.2 and termination settings are
not relaxed. Confirmation times out after12 native frames. Production YAML is
never rewritten.

Nvtracker returns asynchronously. A bounded eight-frame identity queue joins
native outputs to their **exact input PTS/source frame**, never latest metadata.
Native ID is pinned; public visual-lock ID/source session remain stable across
handoff. Ambiguity, changed native ID, invalid geometry, source/session/clock/order
change or >150ms source gap suppress/latch loss. Weak confidence, missing output
or implausible position/size changes immediately suppress selection; three bad
frames latch loss, and temporary uncertainty needs three good native frames for
recovery. Lost does not switch back to VPI/YOLO or reseed. V2 publishes no invented
detection, class, depth or parallax estimate. Only accepted boxes are drawn.

Append `--handoff-nvdcf` to the live/replay commands above. A fresh explicit seed
or release/mode change is not yet wired to console inputs. Shared resident YOLO
contexts, operational switching, native scale accuracy without detector updates,
real-camera clutter/occlusion and image-driven simulator control remain separate
qualification work. The full standalone VPI quality gates do **not** keep running
after handoff; native quality and offline localization must be measured separately.

Measured on Jetson VPI4.1.3/native tracker9.1.0: a five-minute fast live case retains
17,787 selected updates without rejection/loss, about59.44Hz and tracking-processing
p95 5.76ms; seed-relative drift p95 1.419px. The final serialized coordinator also
passes a fresh fast30s case and six deterministic GPU fixtures. These are scoped
passes with competing YOLO stopped, not an exact60Hz or real-camera guarantee.
All trial services are stopped and the original camera/display services restored.

Deterministic handoff fixtures use bounded synthetic pixels encoded once to H264,
then exact NVDEC replay through the real VPI/NvDCF path:

```sh
python -m tools.verify_nvdcf_handoff --library build-live/libidcs_visual_lock.so \
  --output NEW-FIXTURE-DIRECTORY --provenance COMMIT-plus-source-digest
```

Run this under the software guard with no other GPU video owner. It checks motion,
scale and distractor localization, textureless initialization, disappearance and
occlusion with loss latching. Generated CPU pixels are fixture input, not a
production image-readback path. Truth is used for the initial test operator seed
and offline scoring only; fixture clocks are synthetic, not host timing surveys.
Expected loss children exit normally with status1, not by abort. See the newest
journal entry for measured results and current service restoration.
