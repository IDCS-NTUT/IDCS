# Detector-independent visual lock

Implemented 2026-10-07: NVIDIA **VPI CUDA DCF + Harris + forward/backward PyrLK**,
not DeepStream NvDCF and not OpenCV tracking. No YOLO output is an input.
OpenCV in the verifier only decodes, renders and annotates pixels.

## Managed console candidate (2026-10-20)

The source modes `camera_visual_lock` (Camera / Visual lock) and
`sim_visual_lock` (Simulator -> Visual lock) now use exclusive managed VPI
receivers. This is a software candidate, **not a deployment or hardware
qualification**. The dated trial evidence below remains historical.

- The camera unit replaces the Argus DeepStream owner and starts the existing
  camera **shadow** controller. The SIM unit replaces DeepStream RTP and uses
  the existing PC Swarm streamer/controller plus its shared panel/UI/recorder
  session. No detector runs alongside either receiver.
- Only `--managed-mode camera_visual_lock|sim_visual_lock` allows V2 port5564.
  Every plain trial, including a fixed-seed trial, now defaults to5597 and
  refuses5564. The managed switch requires a matching source, exact return
  identity and panel ROI, and refuses fixed seeds/NvDCF handoff.
- The normal Pi display exposes Targets -> Select box / Release / Box size.
  No `--visual-roi-trial`, guard or replacement display process is needed.
  Release and loss discard the request and cached frames, prepare a fresh
  tracker and require another explicit box. A terminal request/reason is also
  repeated in receiver status; the Pi cannot miss loss merely because a
  one-shot result was dropped. GPIO12 must return to a readable OFF before
  another OFF-to-ON lock edge. No YOLO fallback or automatic reseed occurs.
- V2 has no invented range or detector class. The existing aiming solver
  follows camera centre without parallax when range is absent. Controller
  code and physical units are unchanged.
- The physical Pi output retains the ROI auto block and joystick capture.
  Only the PC loopback SIM panel bridge can translate a fresh accepted
  **SIM** lease into simulated auto authority. It checks Pi/video/panel gates
  at the display and fresh receiver source/session/request acceptance on PC.
  Loss, release, stale panel/status/lease and E-stop withhold that authority.
  Ordinary mode switching restarts the shared PC session. Isolated trials
  still publish no enabled SIM lease.
- Mode readiness requires active expected units, three fresh unique V2
  frames with the correct clock/session, and new encoded return frames in
  `/run/idcs/visual-lock-{camera,sim}-health.json`. Ordered transactions and
  rollback remain owned by the operator/mode agents. Outgoing receiver
  shutdown publishes an empty V2 selection after its streaming owner stops.

ABI4, CUDA input, overlay queue, Harris0.5 and all tracker gates retain their
existing defaults. Exact cache/return identities remain mandatory. Camera raw
input capture remains unsupported; its empty input files are not evidence.
GPU OSD labels managed SIM as SIM MOUNT and camera as SHADOW; CONTROL OFF
continues to identify isolated trials. These labels do not prove applied motion.

Managed evidence is under `/var/lib/idcs/visual-lock/{camera,sim}/session-*`.
There are eight retained slots **per source**, with no automatic deletion.
Each run keeps the first32MB of frame JSONL, first8MB of acquisition JSONL,
first64 seed diagnostics (no overwrite), at most256MB input H264 (1805s
ceiling), the existing ~20MB whole-AU return evidence, bounded summaries,
health and final report. Log caps stop evidence writes, not tracking. Reports
include dropped-record counts. A full slot set refuses the next start; archive
reviewed sessions outside this directory before resuming. Never delete trial,
recording or rollback directories as part of retention. Abrupt process death
may leave partial evidence without a final report; retain it for review.

### Reviewer checklist (no motors)

These commands are instructions for the reviewer on each named host; none
were executed on lab hosts by the delegated agent. Use the reviewed candidate
release, the current manifests and existing deployment procedure. **Do not
start HIL, bridge, serial or the physical controller.** Preserve all prior
release/evidence directories and capture before/after unit state.

1. **All hosts, ownership/readiness:** inspect the current manifest, source
   processes and unit fragments. On Jetson, read-only
   `systemctl is-active idcs-controller idcs-bridge idcs-serial idcs-hil.target`
   must show all inactive. Check source owners before any GPU test. Record
   commit, dirty digest, JetPack/VPI/GStreamer versions and output paths.
2. **Jetson, release preparation:** in the candidate release, run
   `bash scripts/prepare_jetson_runtime.sh "$RELEASE" "$MODELS"` using the
   existing external model directory. This now runs the equivalent of:

   ```sh
   cmake -S "$RELEASE/jetson/visual_lock/native" -B "$RELEASE/build/visual-lock" \
     -DCMAKE_BUILD_TYPE=Release -DIDCS_BUILD_LIVE_SURFACE=ON -DIDCS_BUILD_TEST_SURFACE=OFF
   cmake --build "$RELEASE/build/visual-lock" -j2
   ldd "$RELEASE/build/visual-lock/libidcs_visual_lock.so"
   ldd "$RELEASE/build/visual-lock/libidcs_visual_lock_live_surface.so"
   ```

   Pass: both libraries exist with no unresolved dependencies; ABI4 native
   tests pass in an exclusive guarded GPU session as documented below.
   Building files does not activate a unit.
3. **Each host, plan/validate:** use a reviewed config policy covering the
   `operator.yaml` mode additions and stop lists. With `$ROLE` set to
   `jetson`, `pc` or `pi`, `$PY` to that host's documented Python and
   `$RELEASE_ID` to the staged candidate identifier:

   ```sh
   "$PY" tools/deployment.py plan --role "$ROLE" --release "$RELEASE_ID" \
     --config-policy "$REVIEWED_POLICY"
   "$PY" tools/deployment.py validate --role "$ROLE" --release "$RELEASE_ID"
   ```

   Pass: exact inventory, expected config review, native asset digests,
   check-only CLIs, contract suites and `systemd-analyze verify`. The planner
   accepts only the two named new Jetson fragments as absent additions;
   rollback restores absence, including failure before installation. Inspect
   their conflicts and confirm neither has an Install section. Physical
   units must contain no `configs/sim/` overlay. Review the PC panel unit's
   lease5582/status5595 subscribers. Deployment/activation remains a separate
   reviewer operation; do not infer it from a passing plan.
4. **Pi/PC software tests:** run the normal host validation plus
   `python -m pytest -q tests/test_managed_visual_lock.py tests/test_operator_agent.py tests/test_roi_editor.py tests/test_roi_panel_workflow.py tests/test_operator_menu.py tests/test_sim_panel.py`.
   Pi without pytest can run `python -m unittest -v tests.test_operator_menu`;
   run the pytest contracts on Jetson/PC. Verify actual normal display unit
   has no trial flag and the PC software controller's output is loopback.
5. **SIM demo, after separately reviewed software activation:** from Pi
   Operations -> Source & scene -> Simulator -> Visual lock -> Confirm.
   Verify only `idcs-visual-lock-sim` owns Jetson video; both DeepStream units
   and camera shadow are inactive. PC Swarm streamer/controller and shared
   session must be active, HIL absent. Wait for the ready message and fresh
   return identity. Choose Targets -> Box size (Medium96x72 by default), then
   Select box. Select Manual, move GPIO12 to readable OFF, position over a
   textured object, then OFF-to-ON to request a lock. Pass: accepted exact
   frame acquisition, selected V2 on5564 and the **software** mount follows
   the box. Verify Small64x48/Large160x120/X-large240x180, persistence after
   normal display restart, clamping, optional Select/Enter fine resizing,
   release, deliberate texture refusal and subsequent explicit re-selection.
   Loss must clear selection and offer a new box after readable OFF. No
   automatic reseed, detector owner or truth-fed tracking is allowed.
6. **Fail-closed and switch-back:** use deterministic tests for stale/missing
   source/video/session, panel and request mismatch, E-stop and lease expiry;
   do not disturb hardware wiring or network links. Switch with an active box
   to Simulator -> Single/Swarm, Camera or Standby. Pass: outgoing request
   cleared, PC source stops before Jetson changes owner, old ports released,
   readiness checked and rollback restores the exact previous mode on fault.
   Keep original source/evidence state for recovery. No physical unit starts.
7. **Camera run:** select Camera / Visual lock and verify camera receiver plus
   shadow controller and PC camera viewer; both DeepStream units and SIM/HIL
   sources remain inactive. Confirm sensor0/mode4, flip2 and WB3 match the
   reviewed local-camera orientation. Perform a bounded60s textured-box
   selection/release/loss/re-selection check at the Pi. Pass: exact Argus
   sensor/SOF/cache/return identities, selected/empty V2 transitions and
   shadow geometry without motor intents. Inspect healthy return video and
   normal teardown, then return through the Source menu. Do not claim camera
   pixel quality, sustained tracking or panel electrical acceptance from the
   Windows contract tests.

## Historical trial status and boundaries

Camera ROI candidate (2026-10-16): the isolated receiver now accepts
`--source camera` with exact Argus sensor/SOF identity, the shared GPU cache,
panel seed/release and return-frame join. It is not enabled in managed units
and needs Jetson/Pi validation. See [camera ROI validation](camera_roi_validation.md)
for data flow, limits, gate overrides and exact reviewer commands. Historical
camera-pending statements below describe their dated evidence, not acceptance
of this candidate.

The deploy-96 release includes the GPU frame cache and exact return identity
sideband for an isolated Pi panel session. Its receiver publishes fresh
`visual_roi_status_v1` status tied to the video session; this permits the ROI
menu while the normal operator agent is paused. Pi ROI observations use
port5597, separately from detector/controller observations. The exclusive
guard can replace the Pi display while retaining its GPIO runtime, then
restore the normal display. Managed installation and a bounded live session
are separate operations. Read the latest journal for their actual outcomes.

The uncommitted 2026-10-14 smearing-fix candidate adds nominal encoder cadence,
independent IDR return pictures, separate GPU quality/receiver status rows, and
bounded cached-image catch-up that preserves the exact seed/end and existing
source-gap guard. It is tested in an isolated guarded session, not yet a managed
release. Camera ROI integration remains pending. See the newest journal receipt.

The 2026-10-15 investigation supersedes the earlier unproven codec-fault
hypothesis: Pi jitterbuffer `drop-on-latency` discards damaged keyframes even
while `num-lost` remained zero. A bounded predictive A/B with that policy
disabled preserved every picture and produced clean pixels. Increasing jitter
to60ms or adding a bounded packet queue did not eliminate the discards, so
neither is retained. All-IDR remains the tested fallback, not a packet-loss fix.
Reports now separately count latency/too-late discards via the application's
existing bus watch. The subsequent native recovery implementation is described
below; installed return policy is unchanged and end-to-end delay is unqualified.

Native recovery implementation (2026-10-15, uncommitted): both ordinary and
isolated return builders use AVPF RTP/RTCP. Receivers enable `do-lost`, request
keyframes and output complete H264 access units with `wait-for-keyframe` enabled.
The installed NVIDIA encoder's `force-IDR` action is driven by a rate-coalesced
adapter on its input owner, only after its first encoded output. RTP source
identity is taken after the sender session's final SSRC handling. Pi's identity
probe skips only redundant same-key/same-PTS non-marker fragments; conflicts and
markers still reach the existing fail-closed join. No tracker is reseeded.

RTP remains UDP5002. PC/Pi feedback goes to Jetson UDP5003; sender reports go
to PC/Pi UDP5004, configurable via `net.return_rtcp_feedback_port` and
`net.return_rtcp_sender_report_port`. The isolated trial exposes matching
`--return-feedback-port`/`--return-sender-report-port` arguments. Use
`--return-keyframe-interval 15` for the tested predictive trial; interval1 remains
the default/all-IDR fallback. The implementation is not a managed deployment.
Latest journal receipts distinguish fault-replay, live video and endurance scope.

CPU-only native fault replay, using a retained IDR-first Annex-B fixture with
unique picture identities and at least180 AUs:

```sh
python -m tools.verify_return_rtp --input /OUTSIDE-REPO/fixture.h264 \
  --output /OUTSIDE-REPO/NEW-REPORT --port-base 25000
```

It uses dedicated loopback ports and no GPU, camera, GPIO or control commands.
It injects a missing packet, burst loss and a200ms downstream stall, verifies
byte-identical complete pictures and keyframe-only recovery across output gaps,
and confirms native RTCP requests reach the sender. Its file source cannot
produce a requested new IDR, so actual NVIDIA-action acceptance is a separate
live test; synthetic fixture PTS are not a transport/photon-latency qualification.

The isolated tracking engine and source/session coordinator are built and tested
on Jetson VPI 4.1.3. Accepted observations produce strict `PerceptionSnapshotV2`
tracks/selection. This is **not yet a deployed console mode**: the existing Pi
picker, DeepStream pipeline and simulator controller are unchanged.

The first committed installation was `24c824f` / `deploy-75`; its retained
Jetson ABI4 tracker, test uploader and live-adapter builds are under
`~/Desktop/project/deployments/release-75-20261007/native-build/`. Those are
historical binary/evidence paths, not current service source directories.
The last recorded Jetson/Pi deployment is deploy-96 (PC ordinary services remain
deploy-85); read the host manifests
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

Operator seeds are refined from native Harris evidence before template learning;
ambiguous or under-supported refinement keeps the operator box with a skip
reason. Only insufficient original texture refuses. See [seed tightening](roi_seed_tightening.md)
for geometry, bounded seed-only gray diagnostics and the one-guard review suite.
Minimum box side is 12 pixels; eight reliable features are required.
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
are read back or queued on the CPU during updates. Explicit seeds additionally
copy the tracker's own gray operator crop for bounded diagnostics.

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

Panel acquisition additionally writes `roi-acquisitions.jsonl`: the original
request/rectangle, Jetson receipt/processing times, gates and every processed
seed/catch-up step, with exact source identity, decoded PTS and quality metrics.
The first lost result stops acquisition; the panel receives its actual reason
prefixed `seed:` or `catch-up:`, not a generic quality-gate message. No subsequent
update may replace it with `explicit new seed required`. `frames.jsonl` retains
`result_frame` separately because a rejected cached seed can precede the current
rendered/decoded frame. Source times and Jetson times remain separate domains.

Repeated-selection workflow tests must not copy an old track rectangle across
the release/selection wait and attach it to a newer displayed image. The
isolated `tools.roi_relock_fixture` moving-texture source provides deterministic
pixels; its `seed_for` helper emulates a test operator for the exact requested
source timestamp, only on an explicit seed. It is never a production tracker
input or an ongoing truth correction. The outside-repo Pi fixture's
`--controlled-target` path checks repeated explicit locks/releases, a blank
region's expected refusal and a subsequent corrected selection. An unlabelled
live scene qualifies only one fixed selection, not automatic re-selection of a
moving object. This test does not qualify real switches, camera ROI integration,
mesh-target accuracy, NvDCF handoff or physical tracking.

Run replay under the same exclusive software guard, with no other GPU owner:

```sh
python -m tools.replay_visual_lock_capture --capture CAPTURE-DIRECTORY \
  --library build-live/libidcs_visual_lock.so --output NEW-REPLAY-DIRECTORY \
  --max-wall-s 360 --provenance COMMIT-plus-source-digest
```

Replay starts at a captured IDR (retaining captured parameter sets), decodes with
original Gst PTS on NVDEC, and processes only the original verified-frame sequence.
The seed must be in that decodable window. This is tracking evidence, NOT live
throughput or cross-host latency qualification. Only explicit seed gray crops
are read back for diagnostics; update images stay on the GPU.

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
