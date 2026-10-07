# Detector-independent visual lock (experimental)

Implemented 2026-10-07: NVIDIA **VPI CUDA DCF + Harris + forward/backward PyrLK**,
not DeepStream NvDCF and not OpenCV tracking. No YOLO output is an input.
OpenCV in the verifier only decodes, renders and annotates pixels.

## Status and boundaries

The isolated tracking engine and source/session coordinator are built and tested
on Jetson VPI 4.1.3. Accepted observations produce strict `PerceptionSnapshotV2`
tracks/selection. This is **not yet a deployed console mode**: the existing Pi
picker, DeepStream pipeline and simulator controller are unchanged.

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

Next qualify live pipeline-owned NVMM/EGL lifetime and startup/shutdown, add a
bounded GPU-frame cache for exact displayed-frame ROI
seeding, connect Pi joystick ROI editing/confirm/cancel and explicit release, expose
quality/status overlays, replace selected observation authority without YOLO fallback,
and route actual visual V2 observations to the stable simulator controller (not its
ground-truth path). Measure return-video alignment, drop/age watchdogs and sustained
cadence, then test pure-software closed-loop motion. Manual motor authority and
physical engagement interlocks remain separate and untouched.
