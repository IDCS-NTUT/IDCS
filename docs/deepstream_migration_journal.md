# DeepStream and Low-Latency Control Migration Journal

## Purpose

This journal records the evidence, decisions, implementation plan, validation
results, and brief progress of each work session that modernizes the IDCS video,
tracking, estimation, and gimbal-control pipeline.

The target outcome is a measured reduction in sensor-to-actuator latency while
preserving detection range, target identity, safety behavior, and pointing
accuracy. FPS and inference latency are supporting metrics; they are not the
success criteria by themselves.

## Working plan

1. Establish a reproducible baseline on the PC, Jetson, and Raspberry Pi.
   Measure frame age and latency at capture, decode, inference, tracking,
   command publication, serial transmission, and encoder feedback boundaries.
2. Remove avoidable buffering and measure the effect of RTP jitter-buffer,
   queue, decoder, and encoder settings.
3. Build a minimal DeepStream proof of concept that keeps frames in NVMM from
   source/decode through TensorRT inference and exports detection metadata only.
4. Benchmark capture and inference resolutions against detection recall by
   distance, target pixel size, p50/p95/p99 latency, throughput, and GPU load.
5. Add and compare DeepStream tracking configurations, beginning with NvSORT
   and NvDCF before evaluating TensorRT ReID where identity data justifies its
   cost.
6. Add a timestamp-aware line-of-sight target estimator and validate its bearing
   and bearing-rate estimates using recorded data.
7. Implement PD plus consistently defined target-rate feedforward in shadow
   mode, compare it with the existing PID/MPC outputs, then validate on hardware
   under bounded safety limits.
8. Improve RS485 feedback scheduling and/or prediction so the controller is not
   falsely treated as 50 Hz when encoder observations arrive more slowly.
9. Migrate return-video overlay and encoding to a non-blocking DeepStream branch,
   retain existing ZMQ schemas during transition, and remove the legacy pixel
   path only after regression and hardware acceptance tests pass.

## Acceptance criteria (to quantify after baseline)

- Sensor-to-ControlCmd and sensor-to-serial-send latency reported at p50, p95,
  and p99, with a target chosen from the measured baseline.
- No hidden queue may grow without a latest-only/leaky policy in the live path.
- Detection recall and identity continuity must be reported by target distance
  and apparent pixel size.
- Closed-loop pointing error, overshoot, settling time, and target reacquisition
  must be no worse than the accepted baseline.
- Emergency stop, manual authority, command gating, and hard gimbal limits must
  remain independent of the perception pipeline.

## Session log

### 2026-08-16 — Repository survey and migration hypothesis

Progress:

- Mapped the PC, Jetson, RPi, shared schemas/configuration, launch scripts, test
  suite, stored gimbal traces, and current TensorRT/BoT-SORT/MPC paths.
- Identified a 200 ms RTP jitter buffer in the Jetson network receiver.
- Confirmed that received/captured NVMM frames are converted to CPU RGBA/BGR,
  copied into NumPy, uploaded to CUDA for inference, and returned through CPU
  memory before hardware encoding.
- Confirmed the active configuration uses 1080p video, a 1920-square search
  inference engine, a 960-square tracking engine, and BoT-SORT with ReID off.
- Confirmed the existing Kalman filter estimates gimbal plant state inside MPC;
  the PID path differentiates filtered target error and does not implement a
  clean target line-of-sight rate feedforward term.
- Read stored yaw sweep evidence showing approximately 50.9 ms mean RS485 reply
  latency and approximately 150.4 ms maximum latency.
- Initial SSH probes reached both configured devices but authentication was not
  available at that time. The user has now installed the public key; live
  read-only inventory and baseline discovery are in progress.

Decisions:

- Treat DeepStream primarily as a zero-copy pipeline and optimized metadata/
  tracking framework, not as an assumption that TensorRT execution itself will
  become faster.
- Preserve the current ZMQ message contracts during migration.
- Do not select a lower inference resolution until recall-versus-range evidence
  is available.

### 2026-08-16 — Live device inventory after SSH access was restored

Progress:

- Completed read-only SSH inventory of the Jetson at `192.168.0.5` and the
  Raspberry Pi at `192.168.0.3`.
- Verified that the local and Jetson copies of the five core perception/control
  files are byte-identical: `server.py`, `receiver.py`, `yolo_engine.py`,
  `multi_target_tracker.py`, and `controller.py`. Network, perception, and
  system configuration files also match. `control.yaml` differs and must be
  reconciled before hardware experiments.
- Measured healthy wired-link reachability: zero loss in the short tests, about
  0.8 ms average PC-to-device RTT, and about 0.2--0.24 ms average Jetson-to-Pi
  RTT. These tests establish connectivity only; they do not characterize RTP
  jitter under load.
- Confirmed that no IDCS perception/control processes were running during the
  inventory, so live pipeline latency and utilization could not yet be sampled.

Jetson evidence:

- The deployed platform is Ubuntu 24.04.4, L4T 39.2, TensorRT 10.16.2, and
  DeepStream 9.1. This supersedes the repository documentation that names
  JetPack 6.2.1/L4T 36.4.4.
- The Jetson is in `MAXN_SUPER` power mode. At idle it used about 4.0 GiB of
  7.4 GiB RAM, had about 3.3 GiB available, used some swap, and reported zero
  GPU activity during the sample.
- Core GStreamer/DeepStream elements (`nvarguscamerasrc`, `nvv4l2decoder`,
  `nvvidconv`, `nvinfer`, `nvtracker`, `nvstreammux`, `nvdsosd`, and
  `nvv4l2h264enc`) are discoverable.
- The full `deepstream-app` executable does not start because
  `libgstrtspserver-1.0.so.0` is missing. Several optional plugins also report
  missing Triton, Rivermax, and OpenTelemetry libraries. The minimal pipeline
  should avoid optional components, but the package installation needs a
  dependency audit.
- A bounded `trtexec` load test proved that the existing
  `small_1280.engine` cannot be deserialized by TensorRT 10.16.2. All TensorRT
  plan files must be treated as platform-version-specific and rebuilt from ONNX
  on this Jetson before either the legacy or DeepStream inference path can run.
- A one-frame `nvarguscamerasrc` test reported `No cameras available`. The
  `nvargus-daemon` journal contains camera-provider failures, including an
  inability to open the bandwidth ioctl device. Camera hardware/device-tree and
  the L4T 39.2 upgrade must be validated before camera-based benchmarking.
- The CH340/CH341 RS485 adapter is present as `/dev/ttyCH341USB0` and the `idcs`
  account belongs to `dialout`.
- PC and Jetson are NTP synchronized, but the application primarily exchanges
  `time.monotonic()`-derived timestamps. Monotonic epochs are not comparable
  across machines. The current timestamp model therefore cannot directly
  measure PC-capture-to-Jetson latency and needs an explicit clock-domain design.

Raspberry Pi evidence:

- The device is a Raspberry Pi 4 Model B running Debian 13 and Python 3.13.
- No project virtual environment with the required OpenCV/ZMQ dependencies was
  found in the checked paths, and no IDCS services were running.
- No physical camera is detected, which is acceptable only if the Pi remains a
  manual-control and return-display node.
- Hardware video decode and KMS/Wayland sinks are available.
- `vcgencmd get_throttled` returned `0x50000`: no current low-bit throttle flag
  was set, but historical undervoltage/throttling flags are latched. Power
  integrity should be corrected or cleared/retested before performance trials.
- The Pi reports NTP as unsynchronized. This reinforces the need to avoid
  subtracting timestamps from different monotonic or unsynchronized wall-clock
  domains.

Immediate plan adjustment:

1. Restore a runnable hardware baseline before measuring or refactoring:
   reconcile Jetson configuration, repair camera availability, rebuild TensorRT
   engines for TensorRT 10.16.2, complete the minimal DeepStream dependencies,
   and create/verify the Pi runtime environment.
2. Define timestamp semantics in the schemas: clock domain, originating host,
   capture timestamp, and per-host monotonic stage durations. Use synchronized
   wall/TAI/PTP time only for cross-host latency, or carry an explicitly measured
   clock-offset estimate.
3. Add passive stage instrumentation before changing jitter-buffer or inference
   behavior, then collect an idle and loaded baseline.
4. Begin the DeepStream proof of concept only after a camera or deterministic
   recorded source and a native TensorRT 10.16 engine are known-good.

Risks and blockers:

- No meaningful inference benchmark is currently possible: the camera is
  unavailable and the serialized engines are incompatible with the installed
  TensorRT runtime.
- Installing dependencies, rebuilding engines, changing camera configuration,
  or starting hardware control processes would mutate device state and was not
  performed during this information-gathering session.

### 2026-08-16 — Provisional training and streaming resolution

Progress:

- Set a provisional data-collection and transport target of **1280x720 (16:9)
  at 60 FPS** so model training and dataset preparation can begin without
  waiting for the DeepStream implementation.

Decision rationale:

- 720p halves the pixels per frame versus the current 1080p stream, while 60 FPS
  halves nominal frame spacing from 33.3 ms to 16.7 ms.
- It preserves the native 16:9 geometry used by the UI and avoids the waste of
  training/streaming square images only to letterbox them for a detector.
- It is a practical transport resolution for an 8 GB Orin NX. The detector
  benchmark should begin at a 960-square TensorRT input (letterboxed from
  1280x720), with 640 and 1280-square variants retained as comparison points.

Training guidance:

- Keep original 1280x720 frames and labels as the dataset source of truth.
  Augmentation/export may letterbox to the deployment tensor shape; do not bake
  the letterbox padding into stored annotations.
- Include a deliberate range/pixel-size split in validation. For small drones,
  record the bounding-box width/height in source pixels and report recall for
  distant/small objects separately.
- Do not discard existing higher-resolution captures; they remain useful for
  crop-based training, super-resolution research, and a later accuracy ceiling
  comparison.

Required hardware validation before this becomes a committed runtime setting:

1. Restore camera availability on the Jetson and enumerate supported sensor
   modes.
2. Confirm sustained 1280x720 at 60 FPS with the required exposure, gain, and
   motion blur limits.
3. Measure H.264 transport loss/jitter, DeepStream p50/p95/p99 frame age, GPU
   load, and thermal behavior.
4. Compare detector recall, localization/bearing error, ID continuity, and
   closed-loop pointing error against 1080p/30 at representative target ranges.
5. Keep 1280x720/60 only if its small-target recall and pointing quality meet the
   agreed acceptance threshold; otherwise test 1440x810/45 or 1080p/45 before
   increasing model size.

### 2026-08-16 — Camera investigation status

Progress:

- Inspected the Jetson camera software and performed a bounded one-frame
  `nvarguscamerasrc` acquisition test. The plugin is installed and
  `nvargus-daemon` is running, but acquisition returns `No cameras available`.
- The Argus service log shows camera-provider initialization failures after the
  L4T 39.2 platform upgrade, including a bandwidth-ioctl failure.
- The Raspberry Pi reports no attached camera. It currently appears suitable for
  control/return-video duties rather than as the image source.

Conclusion:

- The camera has been investigated for software availability and basic
  acquisition, but its sensor model, supported modes, actual frame rate, image
  quality, and 1280x720/60 capability cannot be verified until the Jetson camera
  hardware/Argus issue is restored. The 1280x720/60 selection remains a
  training-facing provisional target, not a confirmed sensor mode.

### 2026-08-16 — Camera re-check after reported attachment

Progress:

- Re-ran Argus enumeration and bounded acquisition attempts after the camera was
  reported attached. `nvargus_nvraw --lps` cannot enumerate a camera; both a
  120-frame 1280x720/60 pipeline and a 60-frame 1920x1080/30 pipeline exit
  immediately with `No cameras available`.
- Confirmed that no competing camera client was running.
- Confirmed that the Tegra capture, VI, and NVCSI kernel modules are loaded, but
  did not find a registered camera-sensor node in the active device-tree probe.

Conclusion and next prerequisite:

- The attached camera is still not electrically/driver-visible to the Jetson.
  This prevents sensor-mode enumeration and all resolution/FPS testing.
- Before software work proceeds, identify the exact camera module/sensor and
  carrier-board CSI connector used. Then, with the Jetson powered off, verify
  cable orientation, full connector latch engagement, correct CSI port, and
  compatibility with the L4T 39.2 camera driver/device-tree configuration. A
  non-supported sensor requires its matching driver and device-tree overlay;
  changing GStreamer resolutions cannot make an unregistered sensor appear.

### 2026-08-16 — Camera recovery and sensor-mode validation

Progress:

- Investigated the new JetPack boot configuration. Jetson-IO had correctly
  created the `JetsonIO` boot entry with the `CSI Camera IMX219 Dual` overlay.
- The subsequent boot log shows the IMX219 driver loading. One expected sensor
  address reports I2C `-121` because the dual overlay expects a second camera;
  the attached camera on the other connector binds successfully. This is normal
  for a single camera under the dual overlay.
- Retested after the post-configuration reboot. Argus now enumerates the camera
  as `jakku_front_RBP194` (IMX219), `/dev/video0` exists, and capture completed
  successfully at both 1280x720/60 and 1920x1080/30.

Validated IMX219 modes:

| Mode | Maximum frame rate |
| --- | ---: |
| 3280x2464 | 21 FPS |
| 3280x1848 | 28 FPS |
| 1920x1080 | 30 FPS |
| 1640x1232 | 30 FPS |
| 1280x720 | 60 FPS |

Decision:

- **1280x720 at 60 FPS is now confirmed as the new camera and training stream
  resolution.** Store training captures at this native sensor mode. Initial
  deployment benchmarking will use 960-square letterboxed inference while
  retaining 640 and 1280 variants for comparison.
- The dual overlay need not be changed to proceed with one camera. A later
  cleanup can select the matching single-camera overlay for the connected port,
  which removes the expected missing-second-sensor boot warning but is not a
  runtime blocker.

### 2026-08-16 — Existing training-data audit

Progress:

- Audited the training directory one level above the repository:
  `/home/idcs/Desktop/project/train/dataset2`.
- Found 51,857 image/label pairs occupying 3.6 GiB, split into 42,141 training,
  6,828 validation, and 2,888 test images. Every image in each split has a
  same-stem YOLO label file.
- Identified the source as Roboflow's 2023 general People Detection dataset,
  originally containing 17,401 images and then augmented. The included `op.py`
  explicitly rewrites every label class to `1`.
- Confirmed sampled annotations use class `1` (`person`) and the source README
  describes a people-only corpus. It contains no evidence of drone annotations.
- Confirmed heterogeneous, augmented geometry rather than a native IMX219
  collection: samples include 572x572, 596x596, 640x640, 500x375, 375x500,
  and a small number of larger 16:9 frames. The README documents random crops,
  flips, brightness/exposure changes, blur, and salt-and-pepper noise.
- Found that `data.yaml` declares `../train/images`, `../valid/images`, and
  `../test/images`. Resolved relative to the YAML file, those paths do not
  exist. It must be corrected before a standard Ultralytics run can find this
  dataset.

Decision:

- Do **not** use `dataset2` to train the primary drone detector for the new
  1280x720/60 IMX219 pipeline. It can be retained as an auxiliary person-only
  dataset after repairing its YAML paths and explicitly preserving class mapping
  `1 = person`.
- Begin a new versioned native-camera dataset with unmodified 1280x720 IMX219
  images. Use it as the source of truth; apply deterministic train-time
  letterboxing and augmentation rather than saving only pre-augmented crops.
- For every native capture session, record scene, range, target pixel width and
  height, lighting, motion/blur condition, and camera controls. Split train,
  validation, and test by capture session/scene rather than by neighboring
  frames so temporal duplicates cannot leak across splits.

Next data work (not yet performed):

1. Define the drone/person class taxonomy and annotation policy.
2. Capture and label a small native IMX219 pilot set at 1280x720/60.
3. Add a dataset validation script that checks YAML path resolution, paired
   images/labels, class balance, box-size distribution, and scene-level split
   leakage before every training run.

### 2026-08-16 — External-data decision for rapid model training

Progress:

- Researched publicly available target-drone datasets appropriate to a fixed,
  ground-facing IMX219 camera.
- Confirmed that VisDrone is largely collected *from a UAV platform* and is not
  a primary match for detecting another small UAV from the ground. It may offer
  generic small-object training value but should not determine the model's
  target-drone performance.

Decision:

- Search for the original, pre-merge drone annotations first. If the images in
  `dataset2` were merged from a drone corpus, the included class-rewrite script
  has made the current labels unusable for separating drones from people.
- If that original labeled source cannot be recovered, download new drone data;
  do not train `dataset2` as if it were a valid drone/person two-class corpus.
- Prioritize data that shows UAVs as targets from fixed ground, handheld, or
  surveillance cameras. It must include small targets, sky/ground backgrounds,
  motion blur, and birds or other hard negatives.

Recommended external starting set:

1. **Anti-UAV**: large annotated RGB/IR target-UAV tracking sequences with
   challenging backgrounds and tiny targets. Convert RGB sequences to detection
   frames while preserving video-level train/validation/test grouping.
   Official repository: https://github.com/ZhaoJ9014/Anti-UAV
2. **MAV-VID**: single-UAV videos from other UAVs, ground surveillance cameras,
   and handheld devices; useful for viewpoint diversity. Access/conversion is
   documented by the UAV Detection and Tracking Benchmark:
   https://github.com/KostadinovShalon/UAVDetectionTrackingBenchmark
3. **Drone-vs-Bird**: especially valuable for learning the hard false-positive
   boundary. It requires acceptance of its data-use agreement; do not assume its
   license permits all deployment uses.
4. **DroneSwarms**: request-gated anti-UAV data with many very small drone
   instances; use it only after reviewing its access and usage terms:
   https://hiyuur.github.io/

Rapid training rule:

- Start a drone-only baseline if drone labels arrive before valid person data.
  A correct one-class drone model is preferable to a two-class model trained on
  mislabeled objects. Add `person` only after its labels and class mapping are
  independently validated.
- When merging valid sources, map classes explicitly (`drone = 0`,
  `person = 1`), retain source/sequence metadata, and deduplicate or split by
  sequence before augmentation. Never apply a global class-ID rewrite.

### 2026-08-16 — Dataset conversion contract

Progress:

- Defined the required conversion boundary for downloaded external datasets.

Decision:

- Convert only when the downloaded source is not already valid YOLO detection
  format. Many anti-UAV datasets use video/frame metadata or JSON annotations;
  some MAV-VID exports may already have YOLO labels. Inspect the source first,
  then use a source-specific, versioned converter rather than editing labels in
  place.
- The final training layout must contain `train`, `valid`, and `test` image and
  label directories, with one label file per image. Each non-empty label row is
  `class_id x_center y_center width height`, where coordinates are normalized to
  `[0, 1]` and the class mapping is explicit (`drone = 0`, `person = 1`).
- Frame-derived datasets must be split by original video/sequence before frame
  extraction or augmentation, preventing adjacent frames from leaking into
  validation/test. Run the planned validator after conversion and before any
  training run.

### 2026-08-16 — Person-dataset selection

Progress:

- Selected CrowdHuman as the recommended primary external person dataset for a
  research/prototype training run: https://www.crowdhuman.org/download.html
- It provides 15,000 training and 4,370 validation images with approximately
  470,000 human instances across train/validation. Its annotations include
  visible, full-body, and head boxes plus ignore/occlusion metadata.

Decision:

- Convert CrowdHuman's line-delimited JSON (`.odgt`) to YOLO detection labels,
  using visible-body boxes (`vbox`) for the initial detector and ignoring boxes
  marked `ignore` or tagged `mask`. Map every retained `person` annotation to
  class `1` when merged with a drone corpus (`drone = 0`, `person = 1`).
- Do not use CrowdHuman without a licensing decision: its terms restrict use to
  non-commercial research and education and prohibit redistribution of image
  data. For a commercial deployment path, select a separately reviewed source
  with appropriate image and annotation rights; COCO is broad and useful but
  much larger (118k training images) and needs filtering/conversion to a
  person-only subset.

### 2026-08-16 — CrowdHuman download fallback

Progress:

- User reported that CrowdHuman's Google Drive mirrors are unavailable. It is
  therefore removed as the immediate acquisition path.
- Identified the maintained official Open Images V7 distribution as an optional
  supplement. Its downloader supports selecting image IDs/classes instead of
  requiring the entire 1.9M-image detection corpus:
  https://storage.googleapis.com/openimages/web/download_v7.html

Decision:

- Use the existing `dataset2` as the immediate **person-only** component. It is
  already local and its labels have deliberately been mapped to class `1`.
  Repair its YAML paths when creating the merged training manifest, but do not
  mistake it for a drone/person dataset.
- If more person diversity is needed, selectively download Open Images V7
  `Person` bounding-box images and annotations through the official downloader
  or a class-filtering client. Retain the source license metadata and convert
  only the selected records to YOLO class `1`.
- Avoid a full COCO download for the first rapid iteration: the official COCO
  2017 training archive alone is roughly 19 GB and requires class filtering.
  It remains a later fallback when broader context classes are required.

### 2026-08-16 — Correction: old mixed images versus old labels

Progress:

- User clarified that the old image collection visually contains both drones and
  people. This changes the handling of `dataset2`.

Correction and decision:

- Because every existing annotation was rewritten to class `1`, `dataset2` is
  unsafe even as a person-only source unless image provenance proves that every
  retained image came from a people-only source. Any drone image labeled as a
  person would directly corrupt two-class training.
- Recover original pre-rewrite labels if possible. Otherwise, rebuild from
  independently labeled drone and person sources, then merge them with the
  explicit mapping `drone = 0`, `person = 1`.

Open Images guidance:

- Open Images V7 is the practical maintained person-source alternative. It has
  1.74M training images with detection annotations overall, and its related V6
  statistics identify 95,335 images with dense human-body-class annotation.
  The official site does not publish one stable person-only V7 image count, but
  its selective downloader can fetch a bounded `Person` subset instead of the
  full corpus: https://storage.googleapis.com/openimages/web/download_v7.html
- For the first model, target **5,000--10,000 verified person images** rather
  than maximum scale. This is ample as the secondary class beside a smaller,
  domain-matched drone corpus and is much faster to convert, validate, and train.
- Prefer Open Images V7 over unverified community mirrors. It has maintained
  download tooling and per-image license metadata. Treat COCO and WiderPerson as
  later alternatives rather than blocking the first training run.

### 2026-08-16 — Anti-UAV-RGBT acquisition audit

Progress:

- Inspected the newly downloaded `../train/Anti-UAV-RGBT` corpus (4.3 GB). It
  is a video tracking dataset, not an image/YOLO dataset: each sequence contains
  `visible.mp4`, `infrared.mp4`, and a same-length JSON annotation for each
  modality.
- The visible/RGB annotations have the expected unambiguous schema:
  `exist[i]` identifies whether the UAV is visible at frame `i`, and
  `gt_rect[i]` is `[left, top, width, height]`. A sampled sequence is 1,000
  frames at 1,920 x 1,080 and 20 FPS.
- The local download contains 136 physical training sequences (126,810 frames;
  122,835 frames with a target present). Its supplied split manifests declare
  160 training and 67 validation sequences. Twenty-four declared training
  sequences and **all 67 validation sequences are absent locally**. The 91 test
  sequences are present but will remain untouched as a held-out source split.
- Among present-target RGB annotations, 82 rectangles fall partly outside the
  recorded frame. A converter must clip such boxes to image bounds and reject
  boxes that become empty; absent-target frames must receive an empty YOLO label
  file only when deliberately selected as negative samples.

Decision and next implementation:

- Do not run the supplied `framecut.py`: it extracts every frame, creates highly
  correlated samples, and does not create labels. Keep RGB only for the first
  camera model; IR data is a later separate modality experiment.
- Build a source-specific converter that uses only the manifest-listed sequences
  actually present, splits **by sequence** (never by frame), samples positive
  frames at a configurable interval, adds a bounded number of true-negative
  frames, clips boxes, and writes YOLO labels with `drone = 0`.
- The 136 currently present train sequences can support a temporary training
  split, but a trustworthy official validation set requires retrieving the
  missing 67 validation sequences (and preferably the remaining 24 training
  sequences) before reporting model quality. Person data remains a separate,
  correctly labelled class-1 source for the final merged corpus.

### 2026-08-16 — Anti-UAV-RGBT converter implementation

Progress:

- Added `tools/convert_antiuav_rgbt_to_yolo.py`. It reads the visible/RGB video
  and its JSON directly, keeps sequences in one split, samples at a configurable
  cadence, clips valid positive boxes, writes empty labels for intentionally
  sampled target-absent frames, and emits `dataset.yaml` plus a machine-readable
  conversion report.
- Corrected for the corpus's legitimate absent-target representation:
  `exist = 0` may pair with either `gt_rect = []` or a four-value placeholder.
  A non-four-value rectangle is rejected only when the target is marked present.
- The converter refuses to silently fabricate a validation split. In the current
  incomplete download it requires an explicit `--temporary-val-fraction`, uses a
  deterministic **sequence-level** holdout, and marks that condition in the
  report. It never trains on the provided test directory.
- Full dry-run validation passed. With a 20-frame stride (1 FPS for this 20 FPS
  source), it plans 5,299 drone positives plus 54 negatives for training and
  864 positives plus 17 negatives for the temporary validation holdout.
- A real two-sequence smoke conversion passed, producing matching image/label
  pairs, valid normalized YOLO labels, and an empty label file for a sampled
  no-UAV frame. Python byte-compilation also passed.

Next execution plan:

```bash
python tools/convert_antiuav_rgbt_to_yolo.py \
  --source-root ../train/Anti-UAV-RGBT \
  --output-root ../train/yolo-antiuav-rgb \
  --temporary-val-fraction 0.15 \
  --frame-stride 20 --negative-stride 60
```

This produces an interim **drone-only** dataset. Do not merge the old relabeled
`dataset2`; merge only a separately verified person source as class `1`. Once
the official validation sequences arrive, rerun conversion without the
temporary-holdout option and use the official validation split for training
decisions.

### 2026-08-16 — Open Images person-data acquisition route

Progress:

- Rechecked the current official Open Images V7 download route. Its browser
  page is an index of raw archives rather than a class-filtering UI; manual
  navigation is needlessly error-prone for a `Person` subset.
- Selected the supported FiftyOne dataset-zoo path instead. It can request only
  `detections` for class `Person`, cap the sample count, retain source labels,
  and export the result directly in YOLO layout. Open Images documents this
  partial-download workflow, and FiftyOne documents YOLO export.

Plan:

- Acquire a deterministic 5,000-image training subset and 1,000-image
  validation subset with `classes=["Person"]`, `only_matching=True`,
  `shuffle=True`, and a fixed seed. Do not use Open Images test images for
  training.
- The temporary export is a one-class YOLO dataset where Person is class `0`.
  Remap it to `person = 1` only during the controlled final merge with Anti-UAV
  (`drone = 0`), rather than modifying downloaded labels in place.

### 2026-08-16 — Two-class dataset and training plan

Plan:

1. Finish and audit both source exports independently. Anti-UAV contributes
   RGB drone images with `drone = 0`; Open Images contributes Person images but
   temporarily encodes them as class `0`. Count images and instances, check
   image/label pairing and normalized coordinates, and visually spot-check each
   source before any merge.
2. Create a versioned final corpus, proposed as
   `../train/yolo-drone-person-v1`, with the canonical mapping
   `0 = drone`, `1 = person`. Copy or hard-link source images using prefixed
   filenames; rewrite only Open Images labels from `0` to `1`; leave source
   directories and their original labels unchanged.
3. Preserve splits at the **source-sequence/source-image** level. Final train
   is Anti-UAV train plus Open Images train; final validation is Anti-UAV's
   temporary sequence holdout plus Open Images validation. Do not use either
   source test split for training. Replace the temporary Anti-UAV validation
   component with its official validation sequences when they become available.
4. Run a post-merge validator: class IDs must be only `0` or `1`, every image
   must have the correct label companion, boxes must be non-empty/in-bounds, and
   filenames must be unique. Produce per-source/per-class counts and a sampled
   contact sheet, not just one combined total.
5. Train an initial two-class model, evaluate drone and person AP separately,
   then export ONNX/TensorRT and benchmark on the confirmed 1280 x 720 @ 60 FPS
   camera stream. Select inference size only from measured latency and small
   drone recall; it is not necessary to permanently resize the raw dataset.

Decision:

- Do not add the old `dataset2` to this merge because its labels no longer
  faithfully distinguish people from drones. The anticipated ~5.3k sampled
  Anti-UAV drone images and 5k Open Images Person images make a sensible first
  balanced image-level training baseline; final balance will be based on the
  measured instance counts, not assumed from image counts.

### 2026-08-16 — Open Images transfer status

Progress:

- Confirmed the corrected FiftyOne export remains active as background process
  PID 341184, owned by its launched shell PID 341179.
- It is still downloading the required Open Images training index; 3.0 GB has
  been cached so far, and no final YOLO files are expected until that index and
  the selected Person images finish downloading.

### 2026-08-16 — Codex configuration check

Progress:

- Inspected the local Codex CLI configuration (v0.147.0). It already trusts the
  project parent directory, enables network access for the development
  permission profile, and uses request-based approvals.

Decision:

- No `config.toml` change is needed for the active dataset download or the
  planned merge/training work. The running FiftyOne process is an ordinary
  background OS process; its lifecycle and download behaviour are not improved
  by changing Codex model, approval, or sandbox settings.
- Keep request-based approval rather than globally enabling unrestricted access.

### 2026-08-16 — Open Images transfer progress update

Progress:

- The FiftyOne Python process remains active after 15 minutes.
- Its required train-index cache is now 4.2 GB. The selected Person images and
  final YOLO export have not started yet, so the export directory is correctly
  still empty.

### 2026-08-16 — Long-running task monitoring design

Plan:

- A background shell monitor can safely observe a process, write a timestamped
  status/completion/failure marker, and optionally emit a local desktop
  notification. That makes process state durable and inspectable without
  repeatedly polling manually.
- A shell cannot by itself wake an inactive Codex conversation. For automatic
  post-completion work, use an explicitly requested durable Codex goal together
  with the monitor's marker. The continuation first verifies the result, then
  performs only the previously authorized next step (for this run: audit the
  Person export and prepare the controlled merge).
- Do not change `.codex/config.toml` for this. Use a per-task monitor and
  marker under the project/training workspace so it is visible, removable, and
  does not create an unbounded system service.

### 2026-08-16 — Open Images Person export completed and audited

Progress:

- The background FiftyOne process has finished. Its export at
  `../train/openimages-person-yolo` contains the requested 5,000 training and
  1,000 validation images, each with a matching YOLO label file.
- Structural validation passed: train has 21,309 Person instances and validation
  has 2,470; all 23,779 label rows use the expected temporary class ID `0`, have
  five numeric YOLO fields, and lie within normalized image bounds. There are no
  orphan labels, unpaired images, or invalid rows.

Next implementation gate:

- The Person source is ready for the controlled two-class merge. Before writing
  the final corpus, run the Anti-UAV converter in full, then combine the two
  verified sources with the planned `drone = 0`, `person = 1` mapping and
  preserve their train/validation partitions.

### 2026-08-16 — Two-class merge implementation started

Progress:

- Started the full Anti-UAV visible/RGB conversion with the planned temporary
  sequence-level validation holdout. It is actively extracting the verified
  1-FPS samples into `../train/yolo-antiuav-rgb`.
- Added `tools/merge_yolo_drone_person.py`. It verifies source image/label
  pairs, validates all YOLO rows, remaps the Open Images source class `0` to
  canonical `person = 1`, retains Anti-UAV as `drone = 0`, namespaces filenames,
  and hard-links images when possible to avoid a second large image copy.
- The merger's syntax check and a limited dry run passed. Full merging correctly
  remains gated on completion of Anti-UAV's validation output.

Next implementation step:

- When the converter finishes, audit its complete report and run the merger to
  create `../train/yolo-drone-person-v1`; then run the final two-class dataset
  validator before beginning training.

### 2026-08-16 — Two-class corpus completed and validated

Progress:

- Completed Anti-UAV conversion at `../train/yolo-antiuav-rgb`: 5,350 train
  images (5,296 drone instances, including 54 intentional negatives) and 881
  validation images (864 drone instances, including 17 negatives). Three
  sampled invalid target boxes were rejected safely. All remaining labels are
  paired, class-0-only, and in bounds.
- Ran the controlled merger to `../train/yolo-drone-person-v1`. All 12,231
  images are hard-linked rather than duplicated; labels are new files with the
  canonical class map `0 = drone`, `1 = person`.
- Final validation passed with no invalid rows, orphan labels, pairing failures,
  or duplicate stems across splits:
  - train: 10,350 images; 5,296 drone and 21,309 person instances
  - validation: 1,881 images; 864 drone and 2,470 person instances
- Visual spot checks confirmed a correctly framed visible-light UAV example and
  a valid Person example. The Anti-UAV source includes its camera overlay; this
  is acceptable for the first baseline but remains a deployment-domain gap to
  address with captured local footage later.

Training readiness:

- `../train/yolo-drone-person-v1/dataset.yaml` is ready for a two-class YOLO
  baseline. Image counts are near-balanced, but person instances outnumber
  drone instances about 4:1; report per-class precision/recall and drone AP,
  rather than relying on combined mAP alone.
- The Anti-UAV validation component is a temporary sequence-level holdout until
  the missing official validation sequences are obtained. Keep this limitation
  in every experimental result.

### 2026-08-16 — Initial training configuration recommendation

Environment:

- The available training GPU is an NVIDIA RTX 2070 with 8 GB VRAM. The deployed
  camera stream is confirmed at 1280 x 720 @ 60 FPS, while the current runtime
  configuration still has an unnecessarily expensive 1920-pixel search engine.

Recommended first baseline:

- Start from the standard COCO-pretrained YOLOv8-small checkpoint, not the
  repository's existing engines/checkpoints whose label provenance has not yet
  been audited. Train a two-class detector with the completed
  `../train/yolo-drone-person-v1/dataset.yaml`.
- Use `imgsz=960`, `batch=4`, `epochs=120`, `patience=25`, `device=0`,
  `workers=4`, AMP enabled, disk cache, deterministic seed, AdamW
  (`lr0=0.001`, `lrf=0.01`, `weight_decay=0.0005`), and three warm-up epochs.
  This is the practical RTX-2070 starting point for small UAVs. Do not begin at
  1280 or 1920; run that comparison only if drone recall at 960 is insufficient.
- Use conservative camera-realistic augmentation: horizontal flip 0.5, vertical
  flip 0, rotation ±5 degrees, translate 0.1, scale 0.5, HSV h/s/v
  0.01/0.5/0.3, mosaic 0.5 with the final 15 epochs mosaic-free, and mixup 0.
  Avoid arbitrary upside-down imagery or aggressive perspective transforms for
  this fixed/gimballed camera baseline.

Evaluation and deployment gate:

- Report drone precision, recall, AP50, and AP50-95 separately from Person;
  persons outnumber drones by instances about 4:1 even though image counts are
  near-balanced. Do not compensate by duplicating correlated Anti-UAV frames;
  add more distinct UAV sequences if drone recall is inadequate.
- Export the winning 960-pixel model to ONNX/TensorRT, then measure DeepStream
  throughput and end-to-end latency on 1280 x 720 input before changing the
  production inference size. Keep 1280 as a conditional accuracy experiment,
  not the default latency target.

### 2026-08-16 — Jetson FP16 YOLO26 capability verification

Device-specific environment and paths:

- Target device: Jetson Orin 8 GB at `192.168.0.5`, running in
  `MAXN_SUPER` power mode with TensorRT 10.16.2, CUDA device compute capability
  8.7, and 7,546 MiB reported usable GPU memory.
- Jetson Python environment: `/home/idcs/Desktop/project` (activate with
  `source /home/idcs/Desktop/project/bin/activate`). The deployed repository
  checkout is `/home/idcs/Desktop/project/IDCS`.
- Test assets and generated native engines are under
  `/home/idcs/Desktop/project/IDCS/assets/models/yolo/`:
  `yolo26{n,s,m}.pt`, `yolo26{n,s,m}.onnx`, and
  `yolo26{n,s,m}.engine`.

Verification state:

- Exported YOLO26 N, S, and M checkpoints to fixed-shape
  `1 x 3 x 960 x 960` ONNX. TensorRT engines are being built natively on the
  Orin using FP16 only (`--fp16`); no PC-built engine is used or considered
  deployable on the Jetson.
- All native FP16 engines built and passed TensorRT's timed inference run
  (`500 ms` warm-up, `10 s` duration, CUDA Graphs):
  - N: 7.9 MB engine; 135.60 qps; 8.16 ms mean host latency; 7.37 ms mean GPU
    compute time; 8.93 ms p95 host latency.
  - S: 22 MB engine; 65.41 qps; 16.09 ms mean host latency; 15.25 ms mean GPU
    compute time; 20.03 ms p95 host latency.
  - M: 43 MB engine; 38.13 qps; 26.93 ms mean host latency; 26.16 ms mean GPU
    compute time; 28.12 ms p95 host latency.
- The engines were serialized successfully under TensorRT 10.16.2 and are
  locally deserializable on the target. Engine generation took about 354 s (N),
  443 s (S), and 499 s (M). M's builder skipped some high-workspace tactics
  because only about 200--250 MB of temporary device memory was available, but
  still completed and passed inference.
- During the initial N build, the device was stable at approximately 20.5 W
  input and 64 C. Memory headroom was limited (roughly 1.5 GB free), therefore
  M is acceptable only if both engine construction and timed inference succeed
  under the normal runtime memory footprint.

Dataset storage audit:

- `../train/yolo-drone-person-v1` contains 12,231 JPEG images (about 2.42 GiB)
  and matching labels, but also 12,231 per-image `.npy` decoded-image disk-cache
  files totaling about 48.49 GiB. These caches explain the approximately 52 GB
  directory size; they are training-throughput optimizations only and do not
  improve accuracy or inference performance.
- The JPEGs are hard-linked to their source datasets (link count 2), not copied.
  The `.npy` files and `labels/*.cache` can be regenerated and are the only
  disposable portion of the merged dataset; retain images, labels, and
  `dataset.yaml`.

### 2026-08-16 — DeepStream 60-FPS foundation

Progress:

- Adopted 60 FPS as the runtime target. The first DeepStream performance path
  uses YOLO26n at 960 because its 135.6-FPS TensorRT-only result leaves room for
  decode, tracking, metadata, overlay, and return encode. YOLO26s at 65.4 FPS
  has insufficient full-pipeline headroom at 960; YOLO26m is a 30-FPS candidate.
- Added `jetson/deepstream/nvdsinfer_yolo26_parser.cpp`, a DeepStream custom
  bbox parser for the YOLO26 NMS output tensor `output0` shaped `1x300x6`
  (`x1, y1, x2, y2, confidence, class`). It compiled successfully against the
  Jetson's DeepStream 9.1 and CUDA 13.2 headers.
- Added the isolated FP16 `nvinfer` profile at
  `configs/deepstream/nvinfer_yolo26n_960.txt`, its generic COCO labels, a
  Jetson preflight command, and `scripts/run_deepstream_file_smoke.sh`.
- Jetson preflight passed: the parser library and engine exist; `nvinfer`,
  `nvtracker`, `nvstreammux`, `nvv4l2decoder`, and `nvdsosd` load; and the
  target-native `yolo26n.engine` deserializes under TensorRT 10.16.2.

Decisions:

- Begin with file/replay source, NVMM hardware decode, `nvinfer`, and no
  tracker; use the supplied NvSORT profile as the first tracker comparison.
  Do not enable NvDCF or ReID until their incremental FPS, latency, memory,
  ID-switch, and reacquisition impact is measured.
- The generic YOLO26 COCO engine validates DeepStream throughput and parser
  integration only. It must never publish control-driving drone/person classes.
  Preserve the current server and ZMQ/control contracts until a trained
  two-class engine and an `NvDsObjectMeta` to `DetectionMsg` adapter are ready.

### 2026-08-16 — DeepStream Python metadata binding

Progress:

- Built NVIDIA's official PyDS 1.2.3 binding from source for the Jetson's
  Python 3.12 environment and installed it only in
  `/home/idcs/Desktop/project` (the project venv). Import verification passed
  for `gst_buffer_get_nvds_batch_meta`, `NvDsFrameMeta`, and `NvDsObjectMeta`.
- Added `jetson/deepstream/verify_pipeline.py`, an isolated replay verifier
  that attaches a pad probe after `nvinfer` or NvSORT, counts DeepStream frame
  and object metadata, records tracker IDs, and writes a JSON report. It does
  not publish ZMQ or control messages.
- Staged the existing 4K H.264 60-FPS replay (`assets/videos/two.mp4`) on the
  Jetson. Hardware decode, `nvstreammux`, raw-tensor `nvinfer`, and the native
  engine all reached PLAYING. The parser-enabled metadata replay remains the
  acceptance run; do not claim 60-FPS end-to-end performance until its report
  is complete.

Interim blocker (resolved in the following session):

- Ran the bounded parser-enabled, paced replay verification. `nvinfer`
  successfully deserializes and initializes `yolo26n.engine`, but no frame
  metadata reaches the post-inference pad probe before the 90 s timeout, so no
  JSON report is produced. This is a DeepStream nvinfer/parser integration
  failure, not a successful 60-FPS result. Decoder-only and
  decoder-plus-`nvstreammux` pipelines reach PLAYING with the same source;
  continue by inspecting the fixed-output custom parser contract and
  nvinfer's first-buffer behavior before running NvSORT or ReID comparisons.

### 2026-08-16 — Detector-only DeepStream metadata and 60-FPS gate passed

Verification method:

- Checked the implementation against NVIDIA's DeepStream custom-model and
  `nvinfer` API documentation. The detector configuration uses the documented
  `parse-bbox-func-name` entry point and an absolute `custom-lib-path`; the
  exported C++ function matches `NvDsInferParseCustomFunc`.
- Added one-shot diagnostics to the custom parser. A native GStreamer run
  proved that DeepStream passes one `output0` layer with 1,800 elements and
  network size 960 x 960, the parser returns successfully, and object metadata
  is attached before the pipeline reaches PLAYING.
- Found the actual replay-harness fault: a downstream-leaky queue before the
  hardware H.264 decoder discarded compressed access units. Dropping reference
  frames can prevent decoded output even though decoder and `nvinfer` initialize
  normally. File-verification queues now block instead of dropping; the future
  live path will apply latest-only behavior only at safe decoded-frame or
  metadata boundaries.
- Ran `jetson.deepstream.verify_pipeline` from the Jetson project venv against
  the complete 4K H.264 replay. The pad probe consumed both `NvDsFrameMeta` and
  `NvDsObjectMeta`; this is a metadata-path test, not merely a TensorRT timing
  or GStreamer state check.

Results:

| Run | Frames | Objects | Measured duration | FPS |
| --- | ---: | ---: | ---: | ---: |
| Unpaced capacity | 1,840 | 1,868 | 17.222 s including startup | 106.838 |
| PTS-paced real time | 1,840 | 1,868 | 30.606 s first-to-last frame | 60.086 steady state |
| PTS-paced including engine/pipeline startup | 1,840 | 1,868 | 31.091 s | 59.181 |

- The paced run produced detections on 1,666 frames. Its source PTS span was
  30.632 s, and it completed without dropped DeepStream frame metadata or a
  pipeline error.
- Reports are retained locally under ignored runtime artifacts as
  `artifacts/deepstream/detector-{unpaced,paced}-2026-08-16.json`; their SHA-256
  hashes are `aabc6d766f28c5eca0b050b4677f8c22eda460ba75a1d48695d98030fad0e51f`
  and `b25436944d706e0d2598e3a337a3cf21b464ff16a857992a843184b319f46719`,
  respectively.

Decision:

- The detector-only DeepStream gate is passed for YOLO26n FP16 at 960 on this
  Jetson: valid frame/object metadata is demonstrated and measured capacity has
  enough headroom to sustain the 60-FPS replay.
- These are generic COCO detections and establish execution capacity only, not
  drone/person accuracy. They remain isolated from ZMQ publication and gimbal
  control.
- NvSORT, NvDCF, and ReID were deliberately excluded from this acceptance run.
  The next implementation boundary is the `NvDsObjectMeta` to existing
  `DetectionMsg` adapter; tracker and ReID cost must then be measured as
  incremental detector-side workloads.

### 2026-08-16 — DetectionMsg shadow adapter and NvSORT comparison

Implementation:

- Added `jetson/deepstream/shadow_adapter.py`, a pure conversion layer from
  `NvDsObjectMeta` fields to the existing `DetectionMsg`/`Box` schema. It
  clips and normalizes boxes, preserves tracker IDs when present, sets no
  target-selection fields, and has no GStreamer, ZMQ, or control dependency.
- Extended `jetson.deepstream.verify_pipeline` with `--shadow-jsonl`. It
  writes one schema-valid, control-disabled DetectionMsg per replay frame and
  never opens a ZMQ socket. File `src_ts_ms` is relative source PTS; `rx_ts_ms`
  and `infer_ts_ms` use Jetson monotonic time. This mode is intentionally not a
  cross-host latency measurement.
- Added decoder-to-infer-input and infer/tracker/metadata timing. Because
  `nvstreammux` rewrites GStreamer PTS, replay telemetry correlates its single
  source, batch-one buffers by preserved FIFO order; all 1,840 buffers matched
  at every measured stage.
- Added five off-device-safe unit tests for clipping, schema serialization,
  timestamp conversion, and FIFO timing. They passed in the Jetson project
  venv.
- DeepStream 9.1 rejected obsolete `nvtracker` element properties
  `enable-batch-process` and `enable-past-frame`; removed them. `nvtracker`
  then exposed a missing `libmosquitto.so.1` dependency in
  `libnvds_nvmultiobjecttracker.so`. Installed the required Ubuntu package
  `libmosquitto1` (about 215 KiB installed). Preflight now checks low-level
  tracker-library dependencies with `ldd`.

Matched PTS-paced 4K replay results (1,840 frames, 30.6 s source span):

| Metric | Detector only | NvSORT | Difference |
| --- | ---: | ---: | ---: |
| Steady pipeline FPS | 60.085 | 60.987 | sustained 60 FPS in both runs |
| Infer/tracker/metadata stage p50 | 8.622 ms | 8.661 ms | +0.039 ms |
| Infer/tracker/metadata stage p95 | 8.737 ms | 8.763 ms | +0.026 ms |
| Mean RAM (31 one-second samples) | 2,550.7 MB | 2,690.3 MB | +139.6 MB |
| Maximum RAM | 2,566 MB | 2,693 MB | +127 MB |
| Metadata frames | 1,840 | 1,840 | no frame loss |
| Boxes | 1,868 | 1,629 | tracker policy filters tentative/shadow detections |
| Unique tracker IDs | 0 | 2 | plumbing only; no identity-quality conclusion |

Validation and decision:

- Detector and NvSORT JSONL files both parse as the existing DetectionMsg
  schema. All boxes are normalized and bounded; target and control fields are
  absent. The files contain 1,840 messages each, with 1,868 detector boxes and
  1,629 tracked boxes respectively.
- NvSORT is accepted as the first DeepStream tracker baseline on this device.
  The generic COCO replay provides no ground-truth ID-switch, reacquisition, or
  drone/person quality evidence, so NvDCF and TensorRT ReID are **deferred**.
- Evidence is retained under ignored `artifacts/deepstream/`, including JSONL,
  reports, and 31-sample tegrastats logs. Shadow JSONL SHA-256 values are
  detector `841cab5f72efac6f73f3680861fd6bfbcad97e6b42950f7945448833e0fdcb46`
  and NvSORT `05115d3ce97c3adfef44e2d543bc6cee6a510cc4f0eb6a89e345cfe80dc8b677`.

Next boundary:

- Implement live PC-header matching by `frame_id` so DeepStream can carry the
  original `src_ts_ms` and true Jetson receive time into the same adapter.
  Keep output shadow-only until a trained two-class model passes accuracy and
  safety acceptance; do not couple the generic COCO engine to control.

### 2026-08-23 — Provisional two-class checkpoint deployment compatibility

Progress:

- Received `assets/models/yolo/small_960.pt` (YOLO26s detect task, classes
  `{0: drone, 1: person}`). It is explicitly a partially trained checkpoint,
  suitable for export and pipeline compatibility testing only.
- Exported it natively on the Jetson to fixed `1 x 3 x 960 x 960` ONNX with
  the end-to-end `output0` shape `1 x 300 x 6`, then built
  `small_960.engine` with TensorRT 10.16.2 FP16. Engine generation completed
  in 430.612 s and produced a 21.176 MiB plan.
- The 10-second native TensorRT run passed: 43.975 QPS, 22.688 ms mean GPU
  compute, 23.634 ms mean host latency, and 23.758 ms p95 host latency.
- Added a separate two-class DeepStream profile
  `configs/deepstream/nvinfer_small_960_drone_person.txt` and two-class labels,
  plus a verifier `--nvinfer-config` switch. The established YOLO26n profile
  remains untouched as the 60-FPS baseline.
- Preflight passed for the new engine/profile. Its PTS-paced full replay also
  passed parser/schema integration: 1,840 metadata frames and 1,840
  schema-valid control-disabled shadow messages. It emitted 160 class-1
  (person) boxes on the available replay; that replay contains no labelled
  basis for accuracy assessment.

Performance decision:

- The provisional model achieved 38.724 steady DeepStream FPS (47.490 s for a
  30.632 s source span), with a 65.811 ms p50 and 68.711 ms p95
  infer/metadata stage. It therefore **does not satisfy the 60-FPS runtime
  gate at 960**.
- Retain it for parser, metadata, and future accuracy tests while training
  completes. Do not use it for control, ZMQ shadow publication, NvSORT
  evaluation, or production sizing. Re-evaluate after a trained checkpoint is
  available, with per-class validation and a resolution/model-size decision.
- Evidence is retained under ignored `artifacts/deepstream/` as
  `small-960-{shadow,report}-2026-08-23.*` (SHA-256 shadow:
  `af634779170297a9f46db96f87aae14a070c7102418359d5b213b9aaf1e7d765`,
  report: `cccaa65624f900f7784ad8e9e1f5ab807b383bb1c029cfd8a0d2091d56132454`).

### 2026-08-23 — Balanced two-class training subset (no training run)

- Audited `yolo-drone-person-v1`: its training split has 5,296 drone and
  21,309 person instances (4.02:1), even though source image counts are close.
  The validation split remains 864 drone and 2,470 person instances and must
  not be resampled for the primary comparison.
- Added `tools/build_balanced_yolo_subset.py` and created the deterministic,
  train-only `/home/idcs/Desktop/project/train/yolo-drone-person-balanced-v1`
  subset with seed `20260823`. It keeps all 5,296 drone instances and 54
  negative images, then selects 1,844 of 5,000 person-only images for exactly
  7,944 person instances (1.5:1).
- All 1,881 validation images are retained. Images and labels in the subset
  are hardlinks to the source; it contains neither `.npy` nor `.cache` files.
  The apparent 1.6 GiB directory size is hardlink accounting, not duplicated
  image storage. The reproducibility report is `balance_report.json` in the
  subset root.

Next model experiment:

- Start from a YOLO26n two-class checkpoint and 1280x720 source capture with
  a 640-square inference tensor. Keep 960-small as a compatibility-only
  checkpoint and defer 1080p inference: it increases cost after the measured
  960-small path already missed 60 FPS. Compare 736/768 only if per-class
  drone recall at 640 is insufficient.

### 2026-08-23 — PC training hold; Jetson-only execution

- The PC power-supply surge is **not fixed**. Do not run PC-side GPU training,
  TensorRT conversion, or inference benchmarks until the operator explicitly
  confirms that the power supply has been repaired and the PC is safe to use.
- Requested one-epoch comparison work is to run on the Jetson instead:
  YOLO26s at nominal 720 (the model stride will round this to 736) and YOLO26n
  at 1080, followed by native Jetson TensorRT FP16 engine benchmarks.

### 2026-08-23/24 — Jetson one-epoch smoke run interruption

- Copied and SHA-256 verified the cache-free balanced archive to the Jetson,
  then extracted 9,075 images and 9,075 labels under
  `/home/idcs/Desktop/project/train/yolo-drone-person-balanced-v1`.
- The first YOLO26s nominal-720 run trained at stride-aligned 736 with batch 3
  for all 2,398 iterations, but its SSH-attached validation lost the connection
  at 71%; no checkpoint was written. A detached retry was started with the
  same data/settings, followed by queued YOLO26n nominal-1080 (actual 1088)
  and detached export/FP16 TensorRT benchmark stages.
- During the retry validation, Jetson remained responsive to ICMP and accepted
  TCP/22 connections, but `sshd` stopped replying with an SSH banner. This is
  recorded as an SSH/OS stall or resource-exhaustion symptom, **not** a proven
  power fault. Results are pending a local-console recovery/reboot and artifact
  inspection; do not report either model as benchmarked.

### 2026-08-24 — One-epoch two-class TensorRT throughput smoke results

- After the Jetson recovered, added `tools/train_yolo_epoch_smoke.py` for this
  constrained test only. Ultralytics always performs final-epoch validation,
  even when `val=False`; this wrapper bypasses that unstable validation/final
  evaluation path but retains normal optimizer, EMA, and checkpoint saving.
  It is not an accuracy-evaluation training workflow.
- Both models trained for exactly one epoch on the balanced corpus (2,398
  iterations, batch 3) and saved `last.pt` checkpoints: YOLO26s nominal-720
  uses a 736-square tensor and took 13m47s; YOLO26n nominal-1080 uses a
  1088-square tensor and took 15m00s.
- Native Jetson TensorRT 10.16.2 fixed-shape FP16, CUDA-graph, 10-second
  benchmarks passed:

| Checkpoint | Tensor | Engine | Throughput | GPU mean | GPU p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO26s smoke | 736x736 | 21 MiB | 152.539 QPS | 6.550 ms | 6.564 ms |
| YOLO26n smoke | 1088x1088 | 7.2 MiB | 136.658 QPS | 7.311 ms | 7.319 ms |

- Both have native-engine headroom above 60 FPS. These results do **not**
  establish detection accuracy: checkpoint training was one epoch and final
  validation was deliberately bypassed. The benchmark log and artifacts stay
  on the Jetson under
  `/home/idcs/Desktop/project/IDCS/artifacts/training/`.

### 2026-08-24 — First trained-head DeepStream development replay

- Added fixed-shape, two-class DeepStream development profiles for the smoke
  engines: `nvinfer_yolo26s_736_drone_person_smoke.txt` and
  `nvinfer_yolo26n_1088_drone_person_smoke.txt`. Both are explicitly
  shadow-only and reference their Jetson-local generated engines.
- Preflight passed for YOLO26s/736: DeepStream elements, parser, NvSORT
  dependencies, and TensorRT engine deserialization all succeeded.
- Ran the 4K, 60-FPS PTS-paced replay (`two.mp4`) through decoder, YOLO26s/736
  nvinfer, NvSORT, stage probes, and the control-disabled shadow adapter.
  All 1,840 frames matched through decode/infer/metadata by the single-source
  FIFO correlation; 1,840 schema-valid DetectionMsg lines were written, with
  no ZMQ or control output.

| Metric | YOLO26s/736 + NvSORT |
| --- | ---: |
| Steady pipeline FPS | 61.020 |
| Whole-run pipeline FPS | 58.431 |
| Decode-to-infer-input p95 | 25.708 ms |
| Infer/metadata p95 | 9.365 ms |
| Metadata frames / messages | 1,840 / 1,840 |
| Boxes / tracker IDs | 18 / 2 |

- The steady-state 60-FPS runtime gate passes narrowly with the full tracker
  path. The one-epoch model emitted only 18 boxes on this replay, so this is a
  DeepStream compatibility and timing result only, not an accuracy or tracking
  quality acceptance. Shadow schema validation found zero invalid boxes and no
  target/control fields. Jetson artifact SHA-256 values are shadow
  `d7c93598dc5c5b024e2372cdfd172ca0ec6c3fbb4475563ce2c3363cd85d0c2a`
  and report
  `8271ab8903be38d66b17cd28a2eb4c06bb8d724caa9fcf55380bb0199249a7d0`.

### 2026-08-24 — Bounded live IMX219 DeepStream verification

- Reconfirmed raw Argus capture from sensor 0, mode 4: IMX219 delivers native
  `1280x720` NVMM `NV12` at 59.999999 FPS.
- Extended `verify_pipeline.py` with `--live-argus` and a mandatory bounded
  `--duration-s` option for live use. It uses `nvarguscamerasrc` directly,
  preserves the same nvinfer/NvSORT/stage-probe/shadow path, and still never
  opens ZMQ or enables control. Live reports explicitly identify their
  same-host Argus PTS mode rather than claiming replay timestamps.
- The 10-second YOLO26s/736 + NvSORT live run passed: 540 metadata/shadow
  frames, 1280x720 source metadata, 60.993 steady pipeline FPS, source FPS
  57.808 including startup/stop boundaries, and infer/metadata p95 11.020 ms.
  Stage FIFO correlation matched 540 infer inputs to 540 metadata buffers; the
  final capture buffer was intentionally stopped at the duration boundary.
- Shadow schema parsing passed with zero invalid boxes and no target/control
  fields. No objects were emitted because this is the one-epoch smoke
  checkpoint; this verifies camera-to-DeepStream runtime plumbing, not detector
  accuracy. Jetson SHA-256: shadow
  `a7cc0fd53b86eca9537a3ef742e13e03b63ab741e5d1d881fe81fb5fef0e3e55`,
  report `d53464d3ef6ee48ed9448c39e6ca7d43080b1c8fe5422d3bdfdef5aca08218a5`.

### 2026-08-24 — GPU OSD + NVMM H.264 return-path verification

- Extended the isolated verifier with `--gpu-osd` and `--return-h264`. It
  attaches detector/tracker labels and a control-disabled status line as
  `NvDsDisplayMeta`; `nvdsosd process-mode=1` renders them on the GPU. The
  pipeline remains NVMM `NV12` through `nvv4l2h264enc`, with no OpenCV frame
  copy, CPU BGR appsrc, ZMQ socket, controller, or gimbal actuation.
- Verified the actual Jetson DeepStream 9.1 plugin contracts before the run:
  `nvdsosd` accepts NVMM `NV12`/`RGBA` and defaults to GPU mode, while
  `nvv4l2h264enc` accepts NVMM `NV12`. The verifier counts H.264 buffers after
  `h264parse` and fails a requested return path that emits none.
- Bounded 10-second IMX219 mode-4 `1280x720/60` run with YOLO26s/736 + NvSORT
  passed: 539 schema-valid metadata/shadow frames and exactly 539 post-parse
  H.264 buffers; steady pipeline FPS was 60.708. Inference/metadata p95 was
  11.069 ms and source PTS FPS was 57.590 across the bounded start/stop window.
  The startup-inclusive 51.291 FPS is not the steady operating rate. Jetson
  artifact SHA-256 values are shadow
  `4f98455c62fb76e44468908d65892f6cd805bd61c20d7a70bb99571745334347` and
  report `551326a05153cb3ec9c6b81e46148f226e9ac51cc2507cf4329e092f0d5c37c3`.
- A separate eight-second Jetson-loopback RTP test used the IDCS return
  contract (`rtph264pay pt=97` → UDP 127.0.0.1 → RTP depay → hardware decode).
  It emitted 418 H.264 access units and decoded 413 before the bounded
  shutdown boundary, with 60.913 steady pipeline FPS. Loopback report SHA-256:
  `cc407a6e22a91b5fbb59a5d661c176b9db336ed8d1ba3348e2262c06378669cd`.
- The one-epoch smoke checkpoint emitted zero objects in this scene, so the
  status overlay and full GPU OSD/encoder path are verified, but visible box
  styling still needs a scene containing detections. This does not alter the
  existing control-disabled, runtime-only qualification of the checkpoint.

### 2026-08-24 — Control-free DeepStream header/shadow transport

- Added an explicit bounded `HeaderCorrelator` and `ShadowTransport` for the
  migration runtime. The only sockets it can create are PC-header PULL and
  DetectionMsg PUB; it has no ControlCmd, controller, gimbal, or actuation
  dependency. Missing headers withhold publication rather than fabricating a
  frame ID or source timestamp.
- Live no-header verification passed: 239 DeepStream metadata frames and 238
  hardware-encoded H.264 buffers were produced, while all 239 potential
  DetectionMsg records were withheld and zero identities were published.
- A local synthetic CamState/PUB-SUB test passed with the live camera pipeline:
  190 matched records were published (188 observed by the subscriber) and
  retained supplied frame IDs/source timestamps from `2/1032` through
  `240/4840`. Fifty queued headers were explicitly dropped under the bounded
  latest-only policy and 109 frames were withheld before/after available
  headers; the report exposes both counts rather than disguising correlation
  loss. Steady pipeline FPS remained 61.244.

### 2026-08-24 — PC-contract RTP ingress verification

- Added bounded verifier support for the PC uplink contract: UDP RTP/H.264
  payload type 96 → jitter buffer → depay/parse → `nvv4l2decoder` → DeepStream.
  This remains separate from controller activation and supports the same GPU
  OSD/NVMM H.264 return tail.
- Local Jetson sender/receiver verification passed using a 1280x720/60 Argus
  sender: 294 decoded/inferred/encoded frames, 62.419 steady pipeline FPS,
  infer/metadata p95 11.898 ms. This proves the live RTP ingress topology
  rather than only Argus-direct capture; it does not substitute for a PC
  end-to-end test while the PC power supply remains unsafe.

### 2026-08-24 — Unified control-free DeepStream live path

- Ran the full bounded Jetson migration path in one process: RTP/H.264 payload
  96 ingress, hardware decode, YOLO26s/736, NvSORT, GPU `nvdsosd`, NVMM H.264
  encoding, PC-header PULL correlation, and DetectionMsg PUB. No control socket
  or controller was opened.
- The local contract-level test processed and encoded 294 frames at 62.463
  steady FPS (infer/metadata p95 11.844 ms). It published 234 matched
  DetectionMsg records; 60 frames without an available header were withheld and
  66 queued headers were explicitly discarded under the bounded latest-only
  policy. This proves the composed Jetson runtime topology while retaining
  truthful correlation accounting.

### 2026-08-24 — GPU OSD encoded visual qualification

- Added `--return-h264-file` to preserve the actual post-OSD NVMM hardware
  encoder output for local inspection, without an RTP receiver or PC.
- A paced 4K replay with the generic YOLO26n engine, NvSORT, GPU OSD, and the
  encoded-file tail completed all 1,840 frames. It produced 1,840 H.264 access
  units, 1,629 detector objects (COCO class 4), and two tracker IDs. The GPU
  status overlay was extracted after hardware decode and visually confirmed in
  the resulting JPEG. Steady replay rate was 51.313 FPS because this generic
  960 model is below the 60-FPS target; this is an OSD visual qualification,
  not the selected runtime performance model.

### 2026-08-30 — Trained two-class checkpoint replacement

- Replaced the active 736 DeepStream profile's one-epoch smoke engine with the
  supplied trained `best (1).pt`: YOLO26s detection, classes `drone` and
  `person`, native 736 input. The old smoke engine remains intact for rollback.
- Exported FP16 TensorRT on the target Jetson. Ultralytics prefixes its engine
  export with JSON metadata, which TensorRT/nvinfer cannot deserialize; added
  `tools/extract_ultralytics_engine_plan.py` and used the resulting raw plan
  `yolo26s_drone_person_best_1_raw.engine`. TensorRT 10.16.2 and DeepStream
  preflight both successfully deserialized it.
- Bounded live IMX219 1280x720/60 run passed with the trained engine, GPU OSD,
  NvSORT, and H.264 encoder: 648 metadata/shadow frames, 647 encoded buffers,
  60.502 steady FPS, and infer/metadata p95 11.204 ms. The camera scene had no
  detections, which is a scene observation, not a model-quality conclusion.
- PTS-paced replay produced 1,840 schema-valid, control-free messages, 51
  person boxes, and one tracker ID. Its 49.042 steady FPS includes the replay
  decoder/encoder setup and is not the live 720p acceptance measurement.

### 2026-08-30 — CPU-only PC streamer shadow smoke test

- Added PC `--deepstream-shadow` mode. It keeps the RTP/H.264 payload-96
  contract, suppresses the ControlCmd subscription, skips unavailable
  config-sync, and emits exactly one origin-tagged header per frame. SimCamera
  CamState now replaces rather than duplicates its bare header, avoiding false
  non-monotonic correlation drops.
- Added `--cpu-encoder` to force `x264enc`, and a trailing configuration
  override selecting the CPU renderer, simulation source, 1280x720/60 target,
  and no Jetson-pose feedback. This bounds the PC smoke test away from its
  unsafe GPU/power path.
- Bounded PC CPU-renderer/x264 → Jetson trained-DeepStream run passed:
  PC delivered approximately 58 FPS; Jetson processed 481 frames, published
  470 header-correlated DetectionMsg records, had zero non-monotonic headers,
  detected 204 persons with one tracker ID, and produced payload-97 return
  video. Jetson steady pipeline FPS was 60.954; source PTS was 58.031 FPS, so
  CPU PC delivery—not Jetson—is the present end-to-end rate limiter.

### 2026-08-30 — PC CPU pipeline investigation and pacing fix

- Direct stage timing identified the original 58-FPS limit: CPU world rendering
  averaged 15.734 ms/frame and appsrc submission another 1.213 ms, exceeding
  the 16.667-ms 60-FPS budget. CPU profiling attributed about 8.3 ms to target
  sprite warps and 3.6 ms to the decorative ground grid.
- Added `sim.renderer_opts.draw_ground_grid`; the CPU shadow profile disables
  only this decorative grid and preserves target sprites/detection value. Raw
  CPU render capacity improved to 121.01 FPS.
- Replaced relative simulator pacing with an absolute deadline scheduler. The
  old loop reset after each `sleep`, accumulating normal OS wake-up overshoot;
  the PC source now converges to 59.9 FPS standalone. Also corrected appsrc
  EOS shutdown to use the PyGObject signal API.
- Final bounded PC CPU/x264 → trained Jetson run measured 59.069 source-PTS
  FPS, 391 matched metadata publications, zero non-monotonic headers, 173
  person detections, and 62.446 steady Jetson pipeline FPS. Remaining sub-60
  PC delivery is normal CPU/x264 and scheduler variance, not the renderer or
  Jetson inference path.

### 2026-08-30 — Full CPU-fallback PC ↔ Jetson validation

- Ran the trained two-class raw FP16 plan through preflight, then a bounded
  PC CPU-renderer/x264 → Jetson DeepStream RTP test with PC-side DetectionMsg
  subscription and a payload-97 RTP H.264 return receiver. Preflight, parser,
  tracker dependencies, and plan deserialization passed on the Jetson.
- Jetson decoded/inferred/encoded 1,139 frames, emitted 1,129 matched
  control-free DetectionMsg values, detected 471 persons across 457 frames,
  and produced the GPU-OSD H.264 return stream. PTS matching was 1,139/1,139
  with one decoded-buffer startup residual; header accounting recorded zero
  invalid/non-monotonic entries, 10 startup-withheld frames, and 30 bounded
  latest-only overflows.
- The PC received 1,127 valid core-schema DetectionMsg records with strictly
  increasing frame IDs and source timestamps. `target_idx` and smoothed range
  were intentionally absent: they are optional legacy/controller fields and
  remain disabled in the control-free shadow path.
- Throughput result: Jetson steady pipeline rate was 60.240 FPS and inference
  plus metadata p95 was 10.974 ms, but source ingress was 58.480 FPS (the PC
  streamer log converged near 59.5 FPS). Therefore functional validation
  passes, while strict 60-FPS end-to-end acceptance remains blocked only by
  the CPU/x264 fallback; repeat with PC NVENC after the PSU is safe.

### 2026-08-30 — Decoupled simulation and Jetson-owned display

- Added `sim.render_fps` for simulated PC sources. SimCamera stays on its
  thread-affine render thread, produces frames independently, and the sender
  keeps only the freshest rendered frame at each configured RTP slot. Frame
  headers are created only for frames actually sent, preserving one-to-one
  header/video correlation. The DeepStream PC profile requests 120-FPS render
  production and retains a 1280x720/60 RTP contract.
- Added PC UI `--deepstream-shadow`: it displays the Jetson payload-97 video
  without drawing local status, MPC, or detection overlays and does not open a
  ControlCmd subscription. Jetson `nvdsosd` is consequently the single overlay
  authority; the PC still consumes DetectionMsg for health/correlation.
- Bounded CPU-renderer/x264 validation found that this host sustains about
  90-FPS simulation under full network load (not the requested 120 FPS), which
  is nevertheless above the 60-FPS sender cadence. The sender converged to
  about 59.5 FPS and the PC received 1,125 core-schema-valid, monotonically
  ordered DetectionMsg records.
- Jetson decoded/inferred/encoded 1,137 frames with GPU OSD and payload-97
  return enabled. Steady pipeline FPS was 60.312 and inference/metadata p95
  was 11.058 ms; header accounting had zero invalid/non-monotonic entries,
  10 startup-withheld frames, and 29 bounded latest-only overflows. Source
  ingress was 58.569 FPS, confirming that CPU/x264—not simulation or Jetson—
  remains the strict end-to-end 60-FPS blocker.

### 2026-08-30 — DeepStream learned target-selection groundwork

- Added the explicit `jetson.server --deepstream-target-selection` feature
  flag for the control-free DeepStream shadow runtime. It maps DeepStream
  class IDs to IDCS labels, estimates configured known-size range, retains
  NvSORT identities/history, then reuses the trained swarm-policy runtime to
  populate threat annotations, `target_idx`, and `target_track_id`. It never
  opens a controller or control PUB socket; GPU OSD only highlights metadata
  selected by this shadow stage.
- A first live run contained only class-1 `person` detections. `person` is an
  intentionally excluded target class, so zero selections is correct. Loading
  the separate policy TensorRT context unnecessarily reduced the Jetson path
  to about 51 FPS even with no eligible candidates.
- Made learned-runtime creation lazy: person-only frames are mapped and
  explicitly unselected without loading the policy engine. Revalidation
  processed/encoded 965 frames at 60.668 steady Jetson FPS, infer/metadata p95
  11.063 ms, and zero invalid/non-monotonic headers. The PC received 954
  records. A drone-containing stream is still required to validate actual
  learned-policy inference, target lock/hysteresis, and selected-target GPU
  OSD before controller activation.

### 2026-08-30 — Real-drone target-selection qualification (blocked)

- Added `pc.streamer --pace-file` and
  `configs/deepstream_drone_file_validation.yaml` so a PC-hosted real drone
  clip can be decoded sequentially and sent with a 60-FPS RTP/header cadence
  rather than flooding the latest-only transport. The 4K source is CPU
  decode/resize limited on this fallback PC (about 45.7 FPS), so this is a
  selection-functional test rather than a throughput acceptance test.
- Ran `assets/videos/one.mp4`, which visibly contains a large ground drone,
  through the trained DeepStream engine and target selector. The engine emitted
  794 detections, all class `1` / resolved label `person`; no class-`0` drone
  detection was produced. Because `person` is correctly excluded from the
  target policy, selection stayed at zero and the learned swarm policy was not
  loaded.
- Controller activation is blocked pending class-specific detector
  qualification/retraining. Do not invert or remap the labels merely to make
  this clip select: that would make person detections eligible as flight
  targets. The next detector test must establish drone recall and class
  separation on representative held-out drone footage.

### 2026-08-30 — Person-only learned-selection simulation

- Added `configs/deepstream_person_sim_validation.yaml`, an explicitly
  simulation-only override which makes `person` eligible and `drone` excluded.
  It retains the full nested learned-policy settings because IDCS config merge
  replaces nested mappings rather than recursively merging them. It must never
  be supplied to a real target-selection deployment.
- Bounded CPU-simulation → trained DeepStream run produced 961 PC-side
  DetectionMsg records. The learned policy selected a tracked `person` on 258
  messages, emitted learned threat scores on 508 boxes, and used NvSORT IDs
  0–4. No ControlCmd fields or controller transport were present.
- This proves the selector/model metadata wiring, range feature path, and
  target-ID propagation. It is not 60-FPS capable: the separate swarm TensorRT
  engine reduced steady Jetson throughput to 51.459 FPS (infer/metadata p95
  11.099 ms). The policy engine must be consolidated/scheduled before it can
  coexist with the detector at the 60-FPS target.

### 2026-08-30 — Learned-policy scheduling qualification

- Set the trained policy update limit to 10 Hz while preserving per-frame
  NvSORT, cached target selection, metadata, and OSD. The person simulation
  continued to select targets (238 messages) and emit learned scores (493
  boxes), but steady Jetson throughput remained 51.523 FPS. The second
  TensorRT context, rather than per-frame policy request frequency, is the
  dominant contention source.
- Tested the CPU PyTorch checkpoint next, explicitly forcing CPU device and a
  one-thread worker. It removed the CUDA compatibility warning but still
  measured only 49.895–50.006 steady FPS with learned selections active.
  Therefore neither GPU-context sharing by frequency reduction nor a spawned
  CPU worker is viable on the metadata probe's real-time path.
- Keep the 10-Hz bound as a safe semantic limit, but do not enable learned
  policy alongside the 60-FPS detector yet. The required redesign is a
  non-blocking latest-only policy service: feed it compact tracked-feature
  snapshots, let it publish cached selection results independently, and have
  the DeepStream probe consume only the latest result without model/process
  lifecycle work. Validate that architecture before controller activation.

### 2026-08-30 — Latest-only learned target-selection service (validated)

- Implemented `AsyncDeepStreamTargetSelector`: the DeepStream metadata probe
  serializes and non-blockingly submits its newest DetectionMsg snapshot to a
  bounded, separate CPU process. That process owns known-size ranging and the
  learned swarm-policy update, coalesces stale requests, and returns only its
  newest completed annotations and selected NvSORT ID. The probe never waits
  for policy work and only applies a result when its tracker ID still exists in
  the current frame.
- Re-ran the explicit person-only simulation override for 17 seconds through
  PC CPU-renderer/x264, trained DeepStream detector, NvSORT, GPU OSD, and
  payload-97 return video. Jetson processed and encoded 957 frames at
  **60.444 steady FPS**; inference plus metadata p95 was 11.105 ms. This
  restores the 60-FPS Jetson target (the baseline control-free path measured
  about 60.24 FPS) while selection is enabled.
- The service accepted 948 header-correlated snapshots and applied 242 fresh
  results. The PC received 946 DetectionMsg records; selected messages carried
  tracked person IDs 0--4, known-size range, learned threat probabilities,
  engagement rank, and planning fields. No ControlCmd transport or controller
  was enabled. The difference between submitted and applied counts is expected
  latest-only coalescing at the configured 10-Hz policy cadence, not loss of
  detector frames.
- This validates scheduler isolation and metadata integration only. It does
  not unblock real drone engagement: the detector still needs class-specific
  drone qualification before any controller or physical actuator path may use
  selected targets.

### 2026-08-30 — Person-class shadow-controller groundwork

- Added a simulation-only `jetson.deepstream.shadow_controller` sidecar. It
  subscribes to the control-free DeepStream DetectionMsg PUB stream and binds
  a separate `net.zmq_sim_control` endpoint (port 5562). It explicitly rejects
  the production `net.zmq_control` endpoint, never starts a hardware driver,
  and uses PID for the first closed-loop exercise.
- Added `target_selector: preselected` to the shared controller. This makes a
  controller honor the already-selected NvSORT ID from the learned DeepStream
  target-selection service, rather than silently selecting another highest-
  confidence box. A missing selected tracker ID is treated as no target.
- The PC streamer has an explicit `--deepstream-sim-control` opt-in, valid
  only with `--deepstream-shadow` and a simulation source. It subscribes only
  to the dedicated simulation endpoint; the normal DeepStream shadow mode
  remains controller-free.
- Initial bounded coupling confirmed the sidecar consumed 795 correlated
  DetectionMsg records and 130 selected-person records, then emitted bounded
  PID tracking commands.
- Final PC ↔ Jetson ↔ PC simulation validation passed the command-transport
  boundary: DeepStream published 576 correlated metadata messages; the
  sidecar consumed 574, accepted 79 selected tracked persons, and the PC
  SimCamera explicitly received 576 commands on port 5562. The first tracking
  command was bounded to -0.076/+0.076 rad/s. No physical-control endpoint,
  gimbal driver, or ControlCmd production transport was opened.
- This is a functional closed-loop transport test, not a 60-FPS end-to-end
  acceptance result. The safe CPU/x264 PC source delivered 55.9 FPS in this
  shorter run, so Jetson steady processing was 59.0 FPS; prior uncoupled
  measurements remain the Jetson 60-FPS evidence. The next controller task is
  to record pointing-error/settling metrics from SimCamera rather than only
  command delivery.

### 2026-08-30 — DeepStream legacy-video replacement checklist

Before default cutover, complete and accept: production runtime/service and
health checks; source-mode support; PC hardware-encoder 60-FPS validation
after PSU repair; timestamp/schema completion; tracking qualification; drone
detector qualification; target-selection qualification; GPU/PC UI parity;
return-video reconnect/loss behavior; metrics/logging; side-by-side soak;
and reversible rollout before retiring the legacy CPU/OpenCV path. Controller
and physical actuation remain separate safety gates.

Implementation started with `jetson.deepstream.runtime` and
`configs/deepstream_runtime.yaml`: a standalone, checked, control-free RTP
runtime profile. `--check` resolves and validates ports, endpoints, model
profile, return destination, and optional metadata-only selection without
opening any video, control, or gimbal socket.

Operational/UI increment:

- Added a systemd unit template with a non-mutating runtime preflight before
  start, restart-on-failure behavior, and no control/gimbal command path.
- GPU OSD now reads the nvinfer profile's configured label file, displaying
  semantic labels such as `person` and `drone` rather than raw class IDs. It
  remains NVMM/GPU-only and intentionally does not fabricate controller UI.
- Selected-target labels now also surface controller-independent range, learned
  threat level, and engagement rank when present. The status strip includes
  the correlated frame ID and Jetson-local inference-stage time; it does not
  claim cross-host end-to-end latency without a shared clock-domain solution.

### 2026-08-30 — Passive canary acceptance and rollback groundwork

- Added `jetson.deepstream.acceptance`, a machine-readable gate for final
  runtime reports. It requires DeepStream frames, GPU OSD, encoded return
  video, a control-disabled output contract, correlated metadata publication,
  and zero invalid/non-monotonic headers. An optional FPS threshold distinguishes
  functional CPU/x264 canaries from future hardware-encoder acceptance.
- Added a cutover/rollback runbook. It explicitly prevents port ownership by
  both pipelines, preserves the legacy path during canary operation, and makes
  rollback a stop/disable/restore sequence rather than an ad-hoc recovery.

### 2026-08-30 — Standalone runtime passive canary

- Ran the new `jetson.deepstream.runtime` rather than the legacy-server shadow
  dispatcher: PC CPU/x264 RTP → runtime → NvSORT/GPU OSD/FP16 detector →
  payload-97 return video and correlated DetectionMsg PUB. The runtime's own
  checked YAML profile was used.
- The acceptance tool passed: 639 DeepStream frames and 639 encoded return
  buffers, 631 correlated metadata publications, zero invalid/non-monotonic
  headers, and an explicitly control-disabled return-output contract.
- Jetson inference/metadata p95 was 11.110 ms and steady pipeline rate was
  59.429 FPS. PC CPU/x264 source rate was 56.576 FPS; this is a functional
  canary and does not replace the deferred hardware-encoder 60-FPS gate.

### 2026-08-30 — Receiver-side metadata qualification

- Added `pc.metadata_monitor`, a passive latest-only subscriber that records
  message/box/class/selection/tracker counts and detects non-monotonic frame
  IDs or source timestamps at the actual PC receiver.
- A runtime canary captured 438 receiver messages and 210 boxes with four
  tracker identities. It found a mixed raw-ID/semantic-label contract during
  asynchronous selector warm-up; fixed normalization at the DeepStream probe.
  Repeat capture had 438 messages, 210 boxes, only `person` labels, zero
  invalid/non-monotonic fields, and three tracker identities.
- The acceptance tool can now consume that PC report and fails a canary whose
  receiver sees no messages, invalid messages, or non-monotonic identity/time.

### 2026-08-30 — Runtime readiness signal

- Added `--ready-file` to the standalone runtime and pipeline. It is atomically
  created only after the first DeepStream metadata frame and removed in every
  shutdown path, preventing a merely spawned process from being treated as a
  usable video service.
- The systemd unit now requests `/run/idcs/deepstream-video.ready`. A live
  canary observed the file after its first frame, observed removal after exit,
  and passed the passive acceptance gate.

### 2026-08-30 — Person-simulation tracker baseline

- Extended the PC receiver monitor with per-tracker observation counts and
  selected-track-change reporting. It resolves a target ID from the selected
  box when a transient message has `target_idx` before `target_track_id`.
- Standalone runtime person-simulation canary passed passive transport
  acceptance: 605 PC receiver messages, 279 person boxes, 162 selections,
  zero invalid/non-monotonic fields, and three bounded header gaps. NvSORT
  emitted IDs 0--3 with 31/8/151/89 observations and the selected identity
  changed four times.
- This is the measured tracker/selection baseline, not a stability sign-off.
  Define a scene-specific identity-switch/reacquisition budget before claiming
  parity with the legacy dual tracker; real-drone qualification remains blocked
  on the detector data/model gate.

### 2026-08-30 — GPU status-panel parity increment

- The GPU OSD status strip now includes current Jetson pipeline FPS alongside
  correlated frame ID and local inference-stage time. This replaces the
  controller-independent portion of the old PC status display without a CPU
  frame copy or invalid cross-host latency subtraction.

### 2026-08-30 — Live runtime health snapshot

- Added an atomically refreshed, one-Hz `--health-file` with frame count,
  last-frame Jetson monotonic time, and pipeline FPS. It complements the
  first-frame readiness file: health detects a running-but-stalled process.
- A bounded Jetson canary observed 77 flowing frames in the health snapshot,
  then observed both health/readiness files removed on shutdown; passive
  acceptance passed. The systemd unit requests both runtime files.

### 2026-08-30 — Graceful runtime stop and restart safety

- Reconnect testing exposed that SIGTERM was not wired into the GLib main loop:
  a terminated runtime could exit before final report/cleanup. Integrated the
  shared shutdown event into the pipeline loop so SIGTERM follows EOS cleanup.
- Live test started a flowing runtime, sent SIGTERM after first-frame readiness,
  then verified final report generation, passive acceptance, and removal of
  both readiness and health files. This is the required service-stop behavior
  for systemd restart and rollback; stream-restart soak remains a separate
  longer-duration qualification.

### 2026-08-30 — Controlled runtime reconnect qualification

- Kept the PC CPU/x264 sender and PC metadata subscriber alive while running
  two consecutive standalone Jetson runtime instances. Both reached first-frame
  readiness, produced a passing passive acceptance report, and shut down with
  readiness/health cleanup after SIGTERM.
- The uninterrupted PC subscriber received 942 valid messages and 442 boxes;
  frame IDs and source timestamps remained strictly increasing across the
  restart. The deliberate service gap created 423 latest-only frame gaps, which
  is expected loss during a restart and is distinct from reordering/corruption.
- This qualifies bounded restart/reconnect semantics. A longer soak with real
  PC hardware encoding remains deferred with the PSU gate.

### 2026-08-30 — Configured Jetson-local Argus runtime

- Extended the standalone, control-free runtime to support either the existing
  PC-originated RTP/header-correlated source or Jetson-local Argus capture. The
  new complete `configs/deepstream_argus_runtime.yaml` overlay uses IMX219
  sensor 0, mode 4, at 1280x720/60. It publishes DetectionMsg records without
  inventing PC headers; frame identity and source timing are explicitly
  same-host values.
- Bounded live canary passed with Argus → FP16 detector → NvSORT/GPU OSD →
  payload-97 H.264 return: 540 frames, 539 encoded buffers, 540 published
  metadata records, 60.609 steady pipeline FPS, and 538 valid PC receiver
  messages with no invalid/non-monotonic fields. The passive acceptance gate
  passed at 60 FPS. The scene yielded only four person boxes, which validates
  transport/runtime behavior, not drone detector quality.
- Acceptance now distinguishes header-correlated RTP reports from headerless
  local-camera reports, while retaining strict header checks for RTP and strict
  receiver schema/order checks for both modes.

### 2026-09-01 — Trained-checkpoint drone regression diagnosis

- Confirmed that the downloaded 20,312,965-byte `best.pt` and the Jetson's
  60,207,843-byte `yolo26s_drone_person_best_1.pt` contain identical learned
  weights. The size difference is checkpoint stripping (not different model
  weights), so replacing one with the other cannot change inference.
- Compared the two YOLO26s checkpoints on the Jetson. The earlier provisional
  960 checkpoint is the best state from epoch 77 (13,902 optimizer updates,
  validation fitness 0.44631); the later 736 checkpoint is the best state from
  epoch 66 (11,826 updates, fitness 0.44241). Both specify the same two-class
  dataset, Ultralytics 8.4.126, seed, initialization name, and training options;
  material argument differences are `imgsz` 960 versus 736 and batch 2 versus
  4. A completed training run does not make its selected `best.pt` a late-epoch
  model: the deployed later checkpoint is its epoch-66 validation peak.
- Ran a controlled checkpoint-by-inference-size matrix on the unchanged 1,881
  image validation split. Overall mAP50-95 was: old weights at 736 = 0.4478,
  old at 960 = 0.4459, new weights at 736 = 0.4425, and new at 960 = 0.4295.
  Drone-only mAP50-95 was respectively 0.6701, 0.6750, 0.6716, and 0.6587.
  Therefore the 224-pixel inference-size decrease does not explain the observed
  deployment collapse; raising the new weights to 960 does not recover it.
- On the identical CPU-rendered simulator drone frame, direct PyTorch scores
  were old@736 0.1044, old@960 0.4011, new@736 0.0060, and new@960 0.0117.
  This isolates a learned-weight/domain-generalization difference before
  TensorRT, DeepStream, the parser, or the configured 0.30 threshold. It is not
  a confidence-unit conversion such as 0.55 becoming 0.0055.
- The primary validation set is not deployment-representative: every drone
  validation image comes from the sampled Anti-UAV corpus, while the real 4K
  replay contains a much smaller dark drone over an urban background and the
  simulator uses a synthetic billboard. Typical validation drone width at a
  736 tensor is about 44.5 pixels (10th percentile 28.0); the replay drone is
  roughly 15--20 pixels after scaling. The full earlier 960 DeepStream replay
  emitted no drone boxes, and direct low-threshold tests show that both
  checkpoints effectively fail this real-drone domain. The older weights'
  billboard success is real but is not evidence of real-camera readiness.

Decision:

- Treat the later 736 checkpoint as a regression on the simulator probe and
  treat both checkpoints as unqualified for real-drone control. Do not attempt
  to repair this with only a lower runtime threshold or a TensorRT re-export.
  Before retraining, freeze a cross-domain acceptance set containing labelled
  local-camera drone frames, simulator frames across target sizes, and the
  existing Anti-UAV holdout; compare checkpoints at a common inference size
  with per-domain/per-size recall, precision, AP, and fixed-threshold recall.

### 2026-09-01 — Independent Dataset2 drone validation

- Located the independent `dataset2` on the Jetson and validated its untouched
  2,888-image `test` split (2,202 drone and 2,784 person instances). Visual
  spot checks confirmed that class 0 is an unadorned drone and class 1 is a
  person; its bundled README is stale and describes only a People Detection
  source, so the README must not be used as class provenance.
- All four checkpoint-by-size combinations fail the independent drone domain:
  drone mAP50-95 / recall were old@736 0.0148 / 0.0127, old@960 0.0154 /
  0.0149, new@736 0.0135 / 0.0117, and new@960 0.0191 / 0.0172. Person
  mAP50-95 remained about 0.27--0.28, confirming that this is principally a
  drone-domain failure rather than a broken two-class output mapping.
- This qualifies the preceding result: the older 960 weights are stronger on
  the particular CPU billboard probe, but are not meaningfully better on an
  independent real-image drone set. The Anti-UAV-only validation score is the
  misleading metric; its persistent HUD/reticle/text and camera distribution
  are a credible shortcut/domain cue. Both models remain unqualified for
  real-drone use.

### 2026-09-01 — Three-motor RS485 connectivity check

- Performed read-only F1 status and 0x31 encoder queries on the idle Jetson
  serial bus at `/dev/ttyCH341USB0`, 38,400 baud. No enable, speed, position,
  zero, or stop command was issued.
- All configured motor addresses replied with valid status byte `1` and a
  six-byte encoder value: yaw address 1 = 0 counts; pitch-A address 2 = -4;
  pitch-B address 3 = -5. This confirms end-to-end serial connectivity to all
  three motors and the configured address mapping.

### 2026-09-01 — Unloaded controller identification and bounded 3D hardware run

- The initial `0.1 rad/s` unloaded sweep exposed the one-RPM MKS command
  quantisation: pitch did not move at that setting.  The command dither is now
  covered by unit tests, and the subsequent repeatable identification used
  both directions, three repeats, `0.2` and `0.4 rad/s`, 35 Hz sampling, and
  no automatic encoder-zero command.  It recorded 868 samples in
  `logs/controller_sysid_steps_0p2_0p4_20260901.csv` on the Jetson.
- A held-out fit of that data selected the continuous derivative model.  Its
  pitch parameters are `a_u=27.134`, `a_f=26.614`, with validation omega/theta
  RMSE `0.03820 / 0.00731 rad`; yaw is `a_u=30.781`, `a_f=30.497`, with
  `0.02710 / 0.00615 rad`.  The full fitted report is
  `artifacts/gimbal_fit/controller_sysid_steps_0p2_0p4_20260901/fit_report.json`.
  These bounded step data are evidence for the current low-speed plant, not
  yet authority to replace live MPC plant parameters: a persistently exciting
  delay/acceleration identification remains required.
- Fixed two serial-launch safety paths.  The step tuning runner now propagates
  its caller's complete config chain; the live 3D trajectory runner derives
  startup commands from the merged config and explicitly excludes MKS `0x92`
  encoder-zero commands.  Calibration is therefore not an implicit side effect
  of a trajectory run.  The persistent `control.yaml` startup list has not
  been edited in this milestone, so other launchers still need the same audit.
- With the gimbal unloaded and the bus exclusive, ran the six-second 3D spline
  plus one-second hold at a guarded maximum of `0.30 rad/s` and `0.35 rad`
  reference offset.  The service reported nine safe startup commands, the run
  completed with 140 samples, and cleanup stopped all axes.  Encoder-measured
  pointing error was RMS `0.06279 rad`, p95 `0.12074 rad`, maximum `0.12475
  rad`, final `0.00651 rad`; yaw dominates (p95 `0.12357 rad`) while pitch p95
  is `0.02663 rad`.  Artifacts:
  `logs/controller_trajectory_3d_unloaded_20260901.csv` and matching JSON
  manifest on the Jetson.  This qualifies the bounded encoder/PID trajectory
  path only; it does not validate vision-to-target control or loaded dynamics.

### 2026-09-01 — Serial critical-command arbitration investigation

- The current serial service gives a command a `critical` sort key, but this is
  not a real-time stop guarantee.  It accepts at most one REQ message at the
  top of each outer loop, snapshots and sorts the current queue, then performs
  blocking serial commands without polling IPC again.  A stop/F7 that arrives
  after that snapshot waits for the whole current round; it cannot preempt an
  in-flight serial read or a batch.
- At the active 38,400 baud configuration, the normal 20-ms schedule contains
  three encoder requests.  Runtime F6 writes are also individually
  reply-waited because `respond_on_writes: true`, so they do not take the
  nonblocking multi-frame path.  With the configured 7.5-ms timeout and one
  retry, timeouts or framing faults can delay a critical command by multiple
  serial operations.  More seriously, the driver has no absolute receive
  deadline while discarding unexpected bytes, so a noisy stream has no proved
  finite worst-case service time.
- Treat the present priority order as best effort only.  The next controller
  safety milestone is a dedicated emergency lane: bound each transaction with
  an absolute deadline, poll/drain critical IPC between transactions, coalesce
  or discard pending noncritical F6 traffic on estop, send no-reply F7/zero
  speed frames immediately, and publish/record request-to-wire latency.  Set a
  tested acceptance bound before allowing live automated control.

### 2026-09-01 — Emergency-first serial arbitration implemented and validated

- Replaced whole-round blocking with one-transaction-at-a-time arbitration.
  F7, critical zero-speed F6, and F3-disable now occupy an emergency rank above
  ordinary `critical` commands.  When any is pending, queued F6/FD motion and
  F3-enable commands are discarded before transmission.  Configured startup
  order remains stable, but can be interrupted between transactions.
- Added an absolute receive deadline to the MKS driver, including the case of
  a continuous stream of malformed/unexpected bytes.  The service clamps
  client timeout and retry overrides so one non-emergency transaction receives
  at most 20 ms of reply-wait budget.  Emergency writes never wait for replies
  or retry, then publish `SerialEmergencyTiming` with the actual pre-write
  monotonic timestamp and budget result.
- Added an acknowledged `SerialCommandClient`.  The Raspberry Pi manual
  emergency path now sends each F7 over REQ/REP and checks its enqueue ACK;
  ordinary high-rate commands remain latest-only PUB/SUB.  The old update path
  is retained only as a fallback when the acknowledged lane is unavailable.
- Unloaded Jetson validation on `/dev/ttyCH341USB0` at 38,400 baud passed a
  25-ms same-host request-to-wire acceptance target.  With the production
  three-encoder schedule, 300 F7 writes had p99 `2.938 ms`, maximum `3.065 ms`,
  and zero misses.  An adversarial test requested a 10-second timeout and 100
  retries ahead of every emergency; clamping reduced it to two bounded
  attempts, and the acknowledged lane delivered 600 F7 writes with p99
  `21.831 ms`, maximum `23.017 ms`, zero budget misses, and zero status errors.
  Artifact: `logs/serial_emergency_latency_req_acceptance_20260901.json` on the
  Jetson.
- This establishes a measured software acceptance bound on the tested Jetson,
  kernel, USB-RS485 adapter, and unloaded bus.  Linux scheduling, USB transport,
  and PUB/SUB fallback are not certified hard real-time; a safety-rated or
  independently wired physical emergency stop remains necessary for a true
  hardware safety guarantee.

### 2026-09-02 — RS485 baud-rate / timeout characterization

- Read the live 0x47 configuration from all three motors before testing.  Each
  reported UART selector `04` (38,400 baud); the repository's selector-`07`
  / 256,000 template has not been applied to this hardware.  No motor enable,
  motion, zero, or disable command was used in this exercise.
- Characterized F1 status and 0x31 encoder queries at 38,400, 57,600, 115,200,
  and 256,000 baud.  Each point contains 600 single-attempt queries of each
  type (200 samples × 3 addresses), with a 10-ms timeout and no retry.  The
  rates were changed with documented 0x8A UART commands, then all three motors
  were restored and rediscovered at 38,400 baud.

| Baud | F1 failures | F1 p99 | 0x31 failures | 0x31 p99 |
| ---: | ---: | ---: | ---: | ---: |
| 38,400 | 2/600 (0.33%) | 5.13 ms | 5/600 (0.83%) | 6.39 ms |
| 57,600 | 2/600 (0.33%) | 4.27 ms | 1/600 (0.17%) | 5.00 ms |
| 115,200 | 12/600 (2.00%) | 3.30 ms | 16/600 (2.67%) | 3.60 ms |
| 256,000 | 21/600 (3.50%) | 2.92 ms | 22/600 (3.67%) | 3.18 ms |

- The failures were serial response-start timeouts, not malformed successful
  frames.  On this CH341 USB-RS485 link, 57,600 is the only higher rate that
  improves both latency and observed raw failure rate.  Keep the deployment at
  38,400 until a longer 57,600 runtime soak includes scheduled encoders and
  live command traffic; do not choose 115,200 or 256,000 without fixing the
  physical link or adapter reliability.  Artifact on the Jetson:
  `logs/rs485_baud_sweep_20260902.json`.

### 2026-09-02 — Fixed-rate controller boundary (shadow-only groundwork)

- Added `jetson.fixed_rate_controller.FixedRateController`, a small monotonic
  scheduling boundary around the existing controller.  It accepts detections
  with an explicit *local receipt timestamp* and advances at a configured
  cadence.  If delayed, it emits one current command and records skipped
  periods instead of issuing a burst of stale catch-up commands.
- `ControlLoop.update_detection` now accepts that optional local monotonic
  receipt timestamp.  Existing live callers keep the previous behaviour by
  omitting it.  This makes target-age and lost-target behaviour replayable
  without relying on synchronized PC/Jetson wall clocks.
- Added `jetson/tools/shadow_fixed_rate_controller.py`.  It consumes
  latest-only detection/CamState metadata and may bind only an explicit shadow
  ControlCmd endpoint; it rejects `net.zmq_control`, has no serial import, and
  therefore cannot actuate the physical gimbal.  This is the correct first
  parity stage before integrating the fixed-rate path into `jetson/server.py`.
- Added unit coverage for receipt-time forwarding, fixed cadence, delayed-tick
  coalescing, and CamState forwarding.  `58` focused controller tests passed
  in the project venv.  The three shadow files were also syntax/import checked
  in `/home/idcs/Desktop/project/IDCS` on the Jetson; no shadow controller,
  gimbal bridge, or serial I/O service was running afterward.  No live
  controller configuration, gains, serial rate, or hardware state was changed.

### 2026-09-02 — Controller-overhaul inventory and dependency plan

- Recorded `docs/controller_overhaul_plan.md` as the authoritative breakdown
  for the controller redesign.  It distinguishes the existing legacy
  PID/MPC-based `ControlLoop` from the new controller that is still to be
  designed, and assigns an evidence-backed status to sixteen work items.
- Current completed foundations are serial emergency arbitration evidence,
  three-motor connectivity, low-speed unloaded identification/trajectory
  evidence, and the shadow fixed-rate scheduler.  The critical missing work is
  an atomic observation/intent contract, deterministic controller replay,
  operating-range/delay identification, controller specification and
  implementation, shadow parity, and cutover validation.
- The next implementation milestone is schema-first and shadow-only: create
  validated `ControlObservation` and `ControlIntent` contracts and an adapter
  from the existing metadata.  This prevents a new control law from silently
  inheriting frame-coupled timing or unsafe implicit defaults.

### 2026-09-02 — Controller observation/intent contract groundwork

- Added strict immutable `ControlObservation v1` and `ControlIntent v1`
  schemas.  They reject unknown keys and non-finite values; observations carry
  target, gimbal, transport, and safety validity/age independently.  Intents
  bind to an observation sequence and have a local-monotonic expiry, explicit
  shadow/live mode, saturation flags, and a reason.
- Added a shadow-only `ControlObservationAssembler` that derives the currently
  selected target bearing/rate, latest CamState encoder sample, and manual
  authority from existing metadata.  It uses only Jetson-local receipt times;
  missing or over-age inputs become invalid rather than zero-filled.
- Added targeted schema/adapter tests.  `61` controller-focused tests passed
  locally.  The gimbal bridge is intentionally unchanged: no controller emits
  an intent and no hardware path consumes one yet.  The immediate next task is
  trace recording and deterministic replay of this new boundary.

### 2026-09-02 — Controller protocol trace and replay groundwork

- Added `tools/record_control_protocol_trace.py`, a passive fixed-rate JSONL
  recorder for atomic `ControlObservation` snapshots.  It subscribes to legacy
  detection, CamState, and manual-state metadata only; it has no control PUB,
  serial import, or gimbal access.
- Added `jetson.control_replay` and `tools/replay_control_protocol_trace.py`.
  They validate and replay the versioned observation records in recorded order,
  rejecting out-of-order sequence/time records.  The sole current policy is a
  deterministic zero-rate, shadow-mode hold policy: it validates the replay
  boundary without pretending to be a tracking controller.
- Added replay coverage; `62` focused controller tests passed locally.  The
  next increment is to connect the recorder to shadow scheduling health and
  add a real redesigned policy plus golden/parity comparison—still with no
  hardware command authority.

### 2026-09-02 — Jetson controller-protocol smoke run

- Ran the passive recorder for three seconds on Jetson loopback ZMQ endpoints
  with 25 synthetic, schema-valid detection/CamState/manual-state updates.
  It wrote 60 fixed-rate `ControlObservation` records with zero decode errors:
  `logs/control_protocol_smoke_20260902.jsonl`.
- Replayed that file through the deterministic shadow hold policy.  All 60
  records were accepted in order, produced 60 shadow intents, and had zero
  invalid or rejected records; output is
  `logs/control_protocol_smoke_20260902_replay.jsonl`.
- During the publisher interval, 26 observations had fresh target and gimbal
  data and 38 had automatic authority.  The later records correctly became
  invalid/disallowed after publishers stopped, demonstrating that the adapter
  does not retain stale inputs as valid.  This is a synthetic transport smoke
  test only, not live perception, control-law, bridge, or hardware validation.
  No recorder, replay, server, gimbal bridge, or serial service remained
  running afterward.

### 2026-09-02 — Legacy controller evidence retired as acceptance baseline

- The prior PID/MPC, frame-coupled latency, gain, tracking, and unloaded
  trajectory results are now classified as historical context only.  They can
  inform experiment design but must not be represented as current performance
  or used as acceptance thresholds for the controller overhaul.
- New evidence must identify exact code/config/model versions, hardware state,
  monotonic timing boundaries, and recorded artifact, and must pass through the
  new observation → policy → intent path.  Acceptance thresholds will be set
  from that reproducible redesign evidence, not copied from legacy runs.

### 2026-09-02 — Real-endpoint observation trace probe

- Ran the new passive recorder for five seconds at 20 Hz against the configured
  Jetson endpoints: DetectionMsg `:5556`, gimbal CamState `:5558`, and manual
  state `:5559`.  Artifact:
  `logs/control_protocol_live_probe_20260902.jsonl` on Jetson.
- It produced 100 schema-valid *empty/stale* observations, with zero decode
  errors and zero valid target, gimbal, safety, or automatic-authority samples.
  Process inventory confirmed that no server/DeepStream producer, gimbal
  bridge, serial I/O service, or manual-control producer was running.  This is
  a useful fail-safe/endpoint-availability result, but is not perception or
  plant-identification data.
- No serial device was opened and no control command was emitted.  A usable
  real-input trace requires the PC/Jetson perception source plus encoder-state
  and manual-state producers to be deliberately started; plant fitting remains
  deferred until that trace and a bounded excitation trace exist.

### 2026-09-02 — New unloaded chirp response sweep and plant fit

- Ran a new bounded response sweep on the unloaded gimbal, independent of the
  retired legacy evidence.  It used three balanced 0.2-rad/s chirps per axis
  (0.10–0.80 Hz, 10 s each) at 35 Hz encoder sampling.  A temporary
  `controller_sysid_safety.yaml` overlay replaced normal serial startup with
  F6 stop and F3 enable only; it explicitly excluded MKS 0x92 encoder zeroing.
- The sweep completed cleanly with 2,634 valid encoder samples (1,317 yaw,
  1,317 pitch), no blocked limits, dropped sends, malformed/missing/stale
  replies, or settle timeouts.  Mean encoder-reply latency was 28.62 ms
  (20.37–28.99 ms), and mean encoder interval was 28.55 ms.  Raw artifacts:
  `logs/controller_sysid_chirp_unloaded_20260902.csv` and matching JSON
  manifest on Jetson.
- `tools.fit_gimbal_response` compared continuous, discrete, deadband, and
  asymmetric first-order candidates using held-out complete traces.  The
  continuous derivative model won for both axes: yaw `a_u=22.203`,
  `a_f=21.936`, tau `45.59 ms`, validation omega/theta RMSE
  `0.01545/0.00300 rad`; pitch `a_u=21.778`, `a_f=21.318`, tau `46.91 ms`,
  validation RMSE `0.02148/0.00419 rad`.  Fit report:
  `artifacts/gimbal_fit/controller_sysid_chirp_unloaded_20260902/fit_report.json`.
- Both selected delay-grid values were 0 ms.  That means no extra delay was
  identifiable at this test's ~28.6-ms encoder cadence; it does **not** prove
  zero command-to-motion latency.  The fitter also flags one-direction
  metadata coverage.  This is an initial low-amplitude unloaded simulator
  input only—not a live-controller retune or acceptance result.  The sweep
  stopped its child serial service; no bridge or service remains running.

### 2026-09-02 — Independent hardware validation of frozen chirp fit

- Collected a separate unloaded sine sweep, not used to fit the model: two
  12-second, zero-mean 0.2-rad/s trials per axis with 0.10/0.25/0.50/0.80-Hz
  blocks.  The same F6/F3-only safety overlay was used.  Transport was healthy
  (no missing/stale/malformed replies, dropped sends, or settle timeouts), but
  137 samples were position-limit blocked.  Raw data:
  `logs/controller_sysid_sine_validation_unloaded_20260902.csv` on Jetson.
- Added `tools/validate_gimbal_fit.py`, which scores a *frozen* fit against a
  separate sweep with the fitter's quality exclusions retained in its report.
  Its focused test/tool suite passed (21 tests); the Jetson report is
  `artifacts/gimbal_fit/controller_sysid_chirp_unloaded_20260902/sine_validation_report.json`.
- The frozen chirp model predicts yaw well enough for this low-amplitude
  validation: omega RMSE `0.01675 rad/s`, theta RMSE `0.00817 rad` over 1,017
  accepted samples.  Pitch is **not qualified**: its 880 accepted samples give
  omega RMSE `0.02359 rad/s` but theta RMSE `0.21713 rad`, after the validation
  run encountered position-limit blocking.  Do not use the pitch fit for a
  controller/simulator acceptance claim until rerun away from the limit with
  bidirectional coverage and no blocked samples.  No sweep, serial service, or
  bridge remains running.

### 2026-09-02 — Pitch validation limit investigation

- Read the pitch-A authority encoder after the first sine validation: raw
  angle `-1.0086 rad`, which maps through `camstate_pitch_sign=-1` to about
  `+1.009 rad`, effectively the configured `pitch_max_rad=+1.0` limit.  A
  guarded 0.1-rad/s negative recenter produced an encoded `-0.0` command and
  no motion; the MKS low-speed quantization is therefore material here.
- A guarded representable 0.2-rad/s negative step moved pitch safely to about
  `+0.685 rad`.  A subsequent pitch-only sine attempt nevertheless drifted
  back into the upper limit, producing 483 blocked samples and ending near
  `+1.007 rad`.  All transport/settling checks remained healthy and every run
  used the zero-free F6/F3 startup overlay.
- The frozen validator now includes explicit qualification.  The resulting
  pitch report is `qualified: false`, reason
  `not_qualified_limit_blocked`; artifact:
  `artifacts/gimbal_fit/controller_sysid_chirp_unloaded_20260902/pitch_sine_validation_clear_report.json`.
  Do not collect further pitch model data or tune control from this operating
  point.  First resolve pitch command/encoder sign and limit reference, then
  choose a profile whose integrated position stays inside a verified central
  travel interval.

### 2026-09-04 — Shadow rate-policy implementation

- Restored the strict immutable `ControlObservation v1` / `ControlIntent v1`
  schema boundary required by the existing assembler and replay tooling.
- Added `jetson.shadow_rate_policy.ShadowRatePolicy`: a deterministic,
  hardware-free PD plus camera-relative bearing-rate feedforward policy. It
  emits only short-lived `mode="shadow"` intents and imports no ZMQ, serial,
  gimbal, or clock source.
- Invalid/stale safety, manual or emergency authority, invalid target/gimbal
  state, target identity changes, and out-of-order snapshots all emit an
  immediate zero-rate hold and reset limiter state. Tracking output is bounded
  by explicit rate and acceleration limits; a measured outward travel-bound
  command is hard-held at zero.
- Added focused coverage for feedforward convention, authority/loss recovery,
  identity change, rate/acceleration saturation, position bounds, and ordering.
  The observation, replay, fixed-rate, and new policy tests passed: `10 passed`.
  No control publisher, gimbal bridge, serial service, or hardware state was
  changed. This is not a gain-selection, plant-validation, or live-controller
  acceptance result.

### 2026-09-04 — Deterministic shadow-policy replay fixture

- Extended `tools/replay_control_protocol_trace.py` with an explicit
  `--policy shadow-rate` option and fully explicit PD/limit parameters. The
  default remains the zero-command hold policy. Both choices are replay-only;
  the tool never imports a transport or opens a hardware device.
- Added a versioned synthetic JSONL trace and exact golden intent output. It
  exercises acceleration limiting, manual takeover/reset, target identity
  switch/reset, and replay-level out-of-order rejection. The CLI test also
  asserts its summary counts and `physical_control_disabled: true` marker.
- This is reproducibility infrastructure only. It does not make the synthetic
  gains suitable for the Jetson plant, and it does not add any command
  authority to the gimbal bridge.

### 2026-09-04 — Jetson reconnect inventory

- Restored key-only SSH access to `idcs@192.168.0.5` from the development PC.
  The Jetson LAN interface is `192.168.0.5/24`; its Wi-Fi and Docker bridge
  remain separate interfaces. This access change does not alter any controller
  or network-routing service on the Jetson.
- A read-only process inventory found no running gimbal bridge, observation
  assembler, fixed-rate controller, DeepStream, or serial-control process.
  `/dev/ttyTHS1` and `/dev/ttyTHS2` exist but were not opened. Therefore the
  new replay code has no live producer trace to consume yet, and no command
  authority was introduced during reconnect.

### 2026-09-04 — Bounded live shadow-sidecar health check

- With explicit authorization for live testing, ran the existing Jetson
  fixed-rate shadow sidecar for 10 seconds at 50 Hz. It subscribed to the
  configured local metadata (`:5556`) and CamState (`:5558`) endpoints, but
  published only to loopback `tcp://127.0.0.1:5562`; it could not address the
  production control endpoint (`:5557`) or open serial hardware.
- It completed 479 ticks with zero missed periods, zero detections, and zero
  CamState samples. Its final status included `physical_control_disabled:
  true`. The legacy-loop diagnostic output remained zero-rate/target-invalid
  throughout. No gimbal motion, serial ownership, or bridge process occurred.
- The absence of both publishers means this is scheduler and safe-no-input
  evidence only, not a vision/controller tracking test. The next live test
  requires a deliberately started metadata producer and encoder/CamState
  publisher, then passive observation recording before shadow comparison.

### 2026-09-05/06 — Shadow integration and encoder telemetry

- Exported the current training checkpoint to ONNX and built a Jetson-native
  FP16 TensorRT candidate at 736 px. TensorRT build/benchmark passed; this is
  perception integration support, not controller acceptance evidence.
- Added hold-vs-shadow parity comparison and capture qualification tooling.
  The synthetic Jetson input trace had 240 observations, 239 fully valid
  snapshots, 239 tracking intents, one startup safety hold, and no ordering
  rejections. This proves protocol integration only; synthetic CamState and
  manual inputs are not hardware evidence.
- Verified read-only MKS encoder command `0x31` over `/dev/ttyCH341USB0` at
  38,400 baud for addresses 1/2/3. Added `publish_encoder_camstate.py`, which
  uses only those encoder reads and publishes CamState on `:5558`; it neither
  enables, zeroes, nor commands a motor. A 10-second capture contained 477
  valid encoder-backed gimbal observations out of 478.
- The live capture remains unqualified: no selected target and no fresh manual
  state were present, so every intent correctly held with `safety_invalid`.
  The RPi manual endpoint is PULL-only, so the Jetson server now has a
  trace-only manual-state PUB mirror on `:5563`. This changes no control or
  authority path.
- Legacy `ControlLoop` sidecar runs and the inference-server/model work are
  transport/perception smoke tests only. They must not be described as
  redesigned-controller parity or acceptance results. No gimbal bridge or
  redesigned command authority was enabled.

### 2026-09-07 — Hardware-readiness recovery and live fail-safe baseline

- Reconciled the development checkout with its safety tests before touching
  live hardware. Restored MKS speed dithering and representable-rate reporting,
  an absolute RS485 receive deadline, pre-write monotonic timing, serial reply
  timing, and emergency-first arbitration. Emergency F7, critical zero-speed
  F6, and F3-disable writes now bypass reply waits/retries; queued motion/enable
  writes are discarded when an emergency is pending; ordinary overrides are
  clamped to a 20-ms transaction budget. The focused development-host suite
  passed: `25 passed`.
- Restored the committed gimbal-bridge encoder timing path from `c15e413` and
  added a trace-only manual-state PUB mirror at
  `net.zmq_manual_state_trace` (`:5563`). The mirror changes no authority or
  command routing.
- On the Jetson, confirmed that the encoder-only publisher retained exclusive
  ownership of `/dev/ttyCH341USB0` at 38,400 baud. The available deployed
  controller and serial-timing tests passed: `57 passed`. No second process
  opened the adapter.
- Diagnosed the inference server as stalled before pipeline startup: the
  `shadow_yolo26s_best_current_736.yaml` overlay selected `source: sim`, which
  required an absent PC sync peer and repeatedly restarted the required sync
  round. A first restart also exposed that service PATH omitted the installed
  `/usr/local/cuda/bin/nvcc`.
- Relaunched with explicit CUDA PATH, `--source-override webcam`, and
  `--config-sync-timeout=0`. The CSI camera enumerated, the 1280x720/60 sensor
  mode fed the configured 1920x1080/30 processing path, TensorRT loaded, and
  endpoints `:5555`, `:5556`, `:5557`, `:5559`, and `:5563` bound. Logs show
  the inference/control loop advancing at approximately 29 FPS with zero-rate
  hold commands.
- Recorded passive shadow-only artifacts
  `logs/control_protocol_hardware_preflight_20260907.jsonl` (470 observations)
  and `logs/control_protocol_hardware_live_20260907.jsonl` (483 observations).
  The live trace had 482 valid encoder-backed observations, zero decode errors,
  and `physical_control_disabled: true`.
- The live trace is intentionally **not qualified**: no selectable target was
  visible (`target_valid=0`) and no fresh Raspberry Pi safety/manual state
  reached the server (`safety_valid=0`, `auto_allowed=0`). All 483 shadow
  intents correctly held with `safety_invalid`.

Safety boundary and next operator prerequisites:

1. Restore key access to `192.168.0.3`, start/verify the real Pi manual-state
   producer, and exercise normal, manual-takeover, and emergency transitions
   while recording through `:5563`.
2. Place a supported person/drone target in view and rerun the passive
   three-input trace until target, encoder, and safety validity satisfy the
   capture qualifier.
3. Resolve the documented pitch command/encoder sign and central travel
   reference before further pitch excitation. Do not run a gimbal bridge,
   serial service, baud soak, or trajectory concurrently with the encoder-only
   publisher.
4. Run shadow parity and set pass/fail thresholds from the qualified trace
   before granting the redesigned controller physical command authority.

No gimbal bridge or serial command service was started, no motor enable/zero/
motion command was issued, and no redesigned intent was connected to hardware.

### 2026-09-08 - Raspberry Pi recovery and unloaded pitch-sign probe

- Replaced only the stale development-host entries for `192.168.0.3` and
  accepted replacement ED25519 host fingerprint
  `SHA256:e37LTimbmQfL6mHSAQUumSIaLn2h8U4k09xSlPNjmA0`. Installed the
  development public key (fingerprint
  `SHA256:ZHEIDRd3w0m5uwYKzlmMgjOeMwNmpuOHXmXB5YfeGQw`) and verified
  passwordless access to `idcs@192.168.0.3` (`idcs-pi`).
- Created a Pi-local `.venv` with system site packages and installed only the
  missing declared runtime dependencies, `pydantic` and `pyzmq`. Fixed
  `rpi.runtime_control --config-sync-timeout=0` to implement the documented
  local-config bypass. The focused development test passed (`3 passed`), and
  the isolated Pi regression case passed. This avoided synchronizing the
  Jetson's older GPIO map over the Pi panel map.
- A bounded eight-second Pi hardware smoke run read stable PCF8591 values near
  `(120, 127)`, resolved the intended GPIO layout, and published normal state
  with `active=false`, `emergency=false`, and command input enabled. The
  real producer now runs as `idcs-rpi-runtime-control.service`.
- Passive artifact
  `logs/control_protocol_hardware_manual_baseline_20260908.jsonl` contains
  480 observations with zero decode errors and
  `physical_control_disabled: true`; 478 samples had fresh Pi safety state
  and automatic authority. The capture remains unqualified only because no
  supported target was visible.
- A software-only Pi-to-Jetson transition probe exercised normal, manual,
  normal, emergency, and recovery states. The Jetson server log recorded
  `auto -> manual -> auto` plus emergency entry/release. Because the real
  20-Hz Pi feed was concurrently authoritative, the passive transition trace
  did not retain those short states and is diagnostic evidence only.
- With the gimbal unloaded, stopped encoder publisher PID 2297387 and confirmed
  exclusive access before one direct address-2 probe. A `-0.2 rad/s`
  command for 0.5 s changed raw encoder angle from `-0.000767 rad`
  (count -2) to `+0.053306 rad` (count 139), so raw encoder response is
  opposite the commanded sign. The motor reported stopped after an additional
  explicit F6 zero. This probes motor A only; it does not validate the complete
  dual-motor pitch-axis convention.
- Restored encoder-only telemetry as `idcs-encoder-camstate.service`; it is
  again the sole owner of `/dev/ttyCH341USB0`. The inference server and Pi
  producer remain active. No serial command service or gimbal bridge remains
  running, and redesigned physical command authority remains disabled.

### 2026-09-08 - Timed-F6 system identification and plant verification

- The local MKS RS485 v1.0.9 manual identifies timed F6 speed commands and
  heartbeat command 0x98. Hardware query 0x40 reported calibrated firmware
  v1.0.8 on all three motors: address 1 is S57D RS485 and addresses 2/3 are
  S42D RS485. Timed F6 is supported; heartbeat protection was added after this
  firmware, so no persistent 0x98 configuration was written.
- A one-shot 1 RPM yaw probe with a 200 ms F6 runtime acknowledged in 7.16 ms,
  began decelerating near 200 ms, and reported stopped at 229.2 ms. Three
  300 ms commands refreshed at 100 ms intervals extended the stop to 514 ms,
  confirming that a newer timed F6 replaces/extends the active timer.
- Repeated 300/100 ms timed-F6 chirp and sine captures were transport-clean but
  showed materially noisier velocity than legacy F6. They are retained as
  diagnostic artifacts and their fitted model was rejected for control use.
- The accepted training capture
  `logs/controller_sysid_step_timed_unloaded_20260908.csv` uses balanced
  +/- one-shot 0.5 s steps at nominal 0.2/0.4 rad/s (1/3 RPM quantized), two
  repeats per direction and axis. All 432 nonzero payloads contain runtime
  `00000032`. It recorded 1,296 valid encoder samples with zero limits,
  drops, missing/malformed/stale replies, or settle timeouts.
- Independent interpolation capture
  `logs/controller_sysid_step_timed_unloaded_20260908_validation.csv` uses
  balanced +/- 0.7 s steps at nominal 0.3 rad/s (2 RPM quantized), two repeats
  per direction and axis. All 300 nonzero payloads contain runtime
  `00000046`. It recorded 704 valid samples with zero limits or transport
  failures. Net displacement was below 0.00154 rad in both captures.
- Automatic yaw discrete-model selection was rejected after external
  validation. The frozen externally selected continuous-derivative report is
  `artifacts/gimbal_fit/controller_sysid_step_timed_unloaded_20260908/fit_report_external_selected.json`.
  Yaw: a_u=28.398995, a_f=28.980305, bias=-0.002962, tau=34.51 ms,
  DC gain=0.97994. Pitch: a_u=25.492311, a_f=26.233458, bias=-0.005512,
  tau=38.12 ms, DC gain=0.97175.
- Independent 2 RPM validation is qualified for transport and limits. Yaw
  omega/angle RMSE is 0.01545 rad/s and 0.00433 rad; pitch is 0.02181 rad/s
  and 0.00517 rad. Both beat the prior legacy model on the same capture.
  Live control configuration was intentionally not changed.
- Reconciled the development-host acquisition/fitting tools with the exact
  Jetson-side versions used for the captures. The focused suite passes 22
  tests plus 9 subtests, and development-host validation reproduces
  `step_validation_external_selected_report.json` byte-for-byte.
- Final direct status checks reported all three motors stopped before and after
  explicit F6-zero plus F7 cleanup. `idcs-encoder-camstate.service` is active
  and is the sole owner of `/dev/ttyCH341USB0`.

### 2026-09-08 - Passive three-input shadow capture and qualification hardening

- Recorded a 30-second, 50-Hz passive trace from live selected-detection,
  encoder CamState, and real Pi manual-state endpoints. It contained 1,439
  observations with zero decode errors and retained
  `physical_control_disabled: true`; 1,438 gimbal and 1,436 automatic-safety
  snapshots were valid, but no supported target was selected.
- Found that `validate_control_capture.py --require-auto` incorrectly used
  only the automatic-authority count and could qualify a targetless capture.
  It now evaluates the per-observation intersection of target, gimbal, safety,
  and automatic authority. Added explicit complete/tracking-ready counts and
  a regression case for targetless automatic-authority input.
- Extended the passive recorder with a final scheduler-health record and added
  an opt-in qualification gate for missing summaries, decode errors, and
  missed periods. The focused observation/policy/replay/scheduler suite passed
  14 tests.
- A fresh 20-second capture,
  `logs/control_protocol_hardware_three_input_scheduler_20260908.jsonl`,
  recorded 963 observations, zero decode errors, zero missed periods, 0.78-ms
  mean deadline lateness, and 3.65-ms maximum lateness. Encoder and automatic
  safety were fresh for 962 observations. SHA-256 is
  `5af53575319a88d086ebf2e400327410c51ba1626f370aad63f7e76702ef5cd5`.
- Qualification now correctly fails only because target validity is zero. No
  parity comparison or physical authority is permitted until a real supported
  target is visible for a qualifying fraction of the trace. The encoder
  publisher remained the sole serial owner; no bridge or serial command
  service was started and no motion command was emitted.

### 2026-09-09 - Structural DeepStream and perception V2 migration

- Added a controlled verification policy and versioned synthetic perception
  source. Tracker, selector, controller, scheduling, and transport tests now
  use schema-valid guaranteed detections/tracks instead of depending on a
  learned detector recognizing rendered targets. Commit: `fefd944`.
- Added a passive DeepStream migration foundation, a reproducible gimbal plant
  analysis workbench, and reproducible training/simulation tools. These remain
  control-free development infrastructure. Commits: `685d7f1`, `07f6155`,
  and `582a0e1`.
- Replaced mutable cross-module configuration merging with immutable recursive
  configuration bundles, duplicate-key rejection, source hashes, and a resolved
  digest. Commit: `7219b62`.
- Introduced strict `PerceptionSnapshotV2` contracts separating raw detections,
  tracker identities, selector assessments, and selection decisions. Added a
  deterministic synthetic scenario covering presence, absence, occlusion,
  identity change, delay, duplication, drops, staleness, and out-of-order
  delivery. Commits: `bae4d4a` and `ea07f9a`.
- Separated the shared DeepStream pipeline from its verification CLI and made
  the runtime config-only check avoid socket or GStreamer construction.
  DeepStream metadata now enters the V2 boundary before any legacy adaptation.
  Commits: `ca5a088`, `fec264b`, and `85a80a0`.
- Migrated the latest-only asynchronous selector to exchange V2 snapshots.
  Decisions explicitly retain their evaluated source frame and identify the
  later frame where they are applied; results are dropped when the NvSORT
  identity is no longer present. Commit: `c695cdf`.
- Removed the DeepStream selector's mutable `DetectionMsg` API. Class
  normalization, known-size ranging, planner input, assessments, and selection
  now remain immutable. The swarm planner exposes a V2 method and contains the
  remaining legacy conversion needed by the independently migrating
  controller. Non-finite legacy diagnostics are translated to absent optional
  V2 fields. Commit: `b0ae6ea`.
- The final focused V2/ranging/config suite passed 35 tests. The complete suite
  at `b0ae6ea` plus the journal-only working change passed 267 tests and had
  four failures. All four were reproduced unchanged against pristine
  pre-migration `HEAD`: two legacy controller rate-limit/lead tests and two
  swarm-planner expectation tests. They are tracked baseline defects, not V2
  regressions.
- `jetson.deepstream.runtime --check` and the verification CLI help path passed
  without constructing sockets, starting GStreamer, loading a camera, or
  granting control authority. No hardware or service state was changed during
  this structural work.

Next structural boundary:

1. Replace the swarm planner V2 adapter's internal `DetectionMsg` materialization
   with immutable planner observations and assessment results.
2. Preserve the legacy controller method as a narrow adapter until the
   controller consumes the new observation protocol.
3. Verify parity with versioned synthetic tracks before any live canary or
   hardware test.

### 2026-09-09 - Immutable planner observation boundary

- Added immutable `PlannerFrameObservation` and `PlannerTrackObservation`
  inputs plus immutable decision/assessment results. The V2 planner path no
  longer imports or materializes `DetectionMsg` or legacy `Box` schemas.
- Both the V2 snapshot method and the legacy controller-facing method now
  delegate to one shared planner implementation. Mutable annotations are
  confined to private working objects and cannot escape the planner boundary.
- Added direct parity coverage using the same versioned synthetic track and
  range assessment against separate immutable and legacy planner instances.
  Their decision and selected diagnostics match exactly.
- Focused planner/perception/protocol tests passed. The complete suite passed
  268 tests with only the same four previously reproduced baseline failures.
  No live transport, camera, model inference, service, serial port, or motor
  was used. Commit: `899bc45`.

Next structural boundary:

1. Move the controller's target-selection input onto immutable planner or
   perception observations while retaining legacy publication at the network
   edge.
2. Add deterministic controller-adapter parity before removing any remaining
   legacy selection mutation.

### 2026-09-09 - V2 display/simulation boundary verification

- Confirmed that the new V2 structure is display/simulation-ready through the
  explicit `detection_msg_from_snapshot` compatibility adapter. DeepStream
  keeps V2 snapshots internally, then emits the existing `DetectionMsg` shape
  at the external transport boundary consumed by `pc.ui` and the simulator.
- The PC simulator's planner-evaluation feedback remains deliberately
  `DetectionMsg`-shaped; there is not yet a V2-native UI or simulator transport.
- Focused display/simulation/shadow acceptance tests passed: 32 passed, 3
  subtests passed, 1 existing GI deprecation warning. No camera, GStreamer
  stream, socket, serial port, motor, or control service was started.
- Readiness status: suitable for a controlled synthetic display/simulation
  canary via the adapter; not yet an end-to-end V2-native display interface.

### 2026-09-13 - Controller observation V2 ingress

- Extended `ControlObservationAssembler` with an immutable
  `update_perception_snapshot()` input. It reads the validated V2 selection and
  track geometry directly and does not mutate the source snapshot.
- Added deterministic parity coverage against the equivalent legacy
  `DetectionMsg` target. Target identity, class, confidence, and bearing error
  match; V2 correctly reports no velocity when the snapshot has no velocity
  field.
- Focused V2/control-observation tests passed: 21 passed. The full suite passed
  269 tests with the same four pre-existing baseline failures; collection also
  requires `PYTHONPATH=.` for the repository's `tools` package.
- This is an ingress boundary only. The live `ControlLoop` and server still
  consume legacy `DetectionMsg`; no control, transport, serial, or hardware
  path was changed.

### 2026-09-13 - Immutable ControlLoop ingress

- Added `ControlLoop.update_control_observation()` as a V2-compatible ingress.
  It consumes the validated immutable `ControlObservation` target directly,
  reconstructs internal pixel coordinates from signed bearing error, and does
  not mutate the observation.
- Added a dry unit test covering the ingress and track identity preservation.
  Focused observation tests passed 5 tests; observation plus controller tests
  passed 52 tests with the same two known controller baseline failures.
- Legacy `update_detection()` remains the network compatibility path. No
  publisher, serial, hardware, or live control service was changed.

### 2026-09-13 - Complete perception V2 pipeline transition

- Audited the preceding controller-ingress work against
  `docs/verification_strategy.md`. The immutable observation assembler is a
  valid V2 boundary; the `ControlLoop` bridge is retained only for the
  simulation-only compatibility sidecar and is not treated as the redesigned
  observation-to-intent controller.
- Made `common.perception` a pure V2 contract/JSON module. All conversion to
  mutable `Box`/`DetectionMsg` now lives in the explicitly named
  `common.perception_compat` and `jetson.deepstream.shadow_compat` modules.
- DeepStream metadata now normalizes directly into V2 objects without
  materializing a legacy box. The pipeline, GPU OSD, async selector, and target
  selection remain V2 through their complete in-process path.
- Added `net.zmq_perception_v2` on port 5564. The runtime publishes lossless
  `PerceptionSnapshotV2` there and independently projects the same snapshot to
  the existing `net.zmq_results` legacy display endpoint. Publication counters
  and passive acceptance require both paths.
- Migrated the simulation-only shadow controller ingress from `DetectionMsg`
  to V2 snapshots via immutable `ControlObservation`, preserving source frame,
  source time, and clock-domain provenance. The existing PC UI and SimCamera
  feedback remain behind the explicit legacy display adapter.
- Migrated both passive control trace recorders, the trace analyzer, and the
  older fixed-rate shadow utility to the V2 snapshot endpoint. The analyzer
  reads nested V2 frame provenance and compares timestamps only within matching
  clock domains.
- Changed the fixed-rate scheduler boundary from `DetectionMsg` to atomic
  `ControlObservation`. The simulation sidecar now handles selected and
  no-selection snapshots and advances at configured `control.loop_hz` rather
  than its former polling-loop rate.
- Added structural tests forbidding legacy perception imports in the shared
  pipeline and metadata adapter, a lossless V2 JSON round trip, dual-output
  transport coverage, provenance checks, and a deterministic
  V2-to-simulation-command test.
- The final focused V2 pipeline suite passed 63 tests, including the transport,
  trace timing, fixed-rate observation, and V2 simulation command boundaries.
  The complete suite passed 282
  tests with the same four previously documented legacy baseline failures and
  12 passing subtests. Runtime and simulation-sidecar `--check` paths
  passed without opening sockets, video, serial, or hardware.

Resulting boundary:

1. The replacement DeepStream perception pipeline is V2-native end to end.
2. In tracked runtime code, `DetectionMsg` is confined to named compatibility
   adapters/outputs, legacy display consumers, and the separate
   `jetson.server` rollback implementation. Controller and trace consumers use
   only the V2 endpoint.
3. Physical controller redesign/cutover remains governed by
   `docs/controller_overhaul_plan.md`; this transition grants no hardware
   command authority.

### 2026-09-13 - V2-native host video pipeline

- Migrated `pc.streamer`, `pc.ui`, `pc.metadata_monitor`, and SimCamera
  planner feedback from the legacy display schema to strict
  `PerceptionSnapshotV2` on `net.zmq_perception_v2`.
- Replaced host startup config synchronization with one immutable recursive
  local load and recorded source/digest provenance. Added non-mutating
  `--check` modes and bounded `--duration-s` runtime options.
- Corrected simulator/header ingress to emit exactly one correlation header
  for every encoded frame. Simulated `CamState` is now the frame header rather
  than an additional message.
- Removed implicit subscriptions to the production control endpoint from both
  streamer and UI. Optional command/debug inputs require explicit CLI
  endpoints and reject `net.zmq_control`.
- Made legacy `net.zmq_results` publication opt-in with
  `deepstream.legacy_display_output`; the passive runtime and acceptance gate
  now require V2 publication by default. GPU OSD and RTP97 return video remain
  mandatory.
- Retained the qualified CPU sprite renderer for the PC 720p60 detector
  validation profile. OpenGL remains available for visual-only simulation,
  but its current scene did not register with the trained model and therefore
  cannot serve as the controlled detector canary.
- Focused deterministic checks passed 47 tests plus 3 subtests. All streamer,
  UI, and DeepStream `--check` paths resolved the expected 720p60, ports 5555,
  5564, 5000, and 5002 without opening sockets. The complete suite passed 283
  tests and 12 subtests with the same four documented baseline failures (two
  legacy controller and two swarm-planner expectations).

Live validation result:

- Deployed commit `d94afc7` as branch `v2-video-d94afc7` in the isolated clean
  Jetson worktree `/home/idcs/Desktop/project/IDCS-v2-video-d94afc7`. The dirty
  legacy/hardware checkout remained unchanged.
- The first OpenGL run proved transport and performance (2,992 V2 snapshots,
  zero invalid/non-monotonic headers, 59.829 steady Jetson FPS, and 3,004
  encoded return buffers) but correctly failed the detector-specific goal:
  the OpenGL 3D person mesh produced no model registration.
- A dynamic CPU-sprite run restored the qualified target and produced 765
  person objects, five NvSORT identities, and 321 applied selections. Running
  rendering and display simultaneously reduced host throughput, so the
  validation profile now freezes one qualified synthetic scene frame while
  keeping unique headers and the complete RTP/inference/tracking/selection/UI
  pipeline live at 60 Hz. This mode is scoped to the canary profile.
- The final frozen-target run delivered approximately 59.6 host FPS and
  61.153 steady Jetson pipeline FPS. DeepStream processed 767 frames, detected
  344 person objects, retained one NvSORT identity, applied 213 selections,
  published 754 native V2 snapshots, and encoded 766 RTP97 H.264 buffers. It
  published zero legacy records and constructed no control path.
- The PC V2 receiver accepted 753 snapshots with zero invalid records, zero
  non-monotonic frame IDs, and zero non-monotonic source timestamps. It saw
  331 tracked-person observations on one identity and 213 selections.
- The on-screen V2 UI used NVDEC, decoded 645 return-video frames, consumed 507
  V2 metadata records, and ended with one displayed tracked object. An X11
  `Detections` window was present throughout the bounded check. The formal
  passive-video acceptance evaluator returned zero failures and zero warnings
  at the 55-FPS floor.

The full simulator -> RTP96 -> DeepStream detector -> NvSORT -> V2 selector ->
V2 PUB -> GPU OSD/RTP97 -> NVDEC host UI path is therefore accepted for
control-free display operation. Physical controller/hardware authority remains
out of scope and disabled.

### 2026-09-14 - Varied rendered-simulator detector qualification

- Replaced reliance on a single frozen canary with a deterministic 1,350-frame
  qualification replay. Each class has nine positive cases spanning three
  distances, three lateral positions, and three background arrangements; each
  case lasts 45 frames and is preceded by a 30-frame blank negative control.
  Expected boxes are derived directly from target-versus-blank simulator
  renders, not hand-entered coordinates.
- Added paced file replay to `pc.streamer` and a strict analyzer that accepts
  both raw detector shadow records and authoritative `PerceptionSnapshotV2`.
  It reports per-case IoU/recall, class confusion, blank false positives,
  coverage, invalid messages, and non-monotonic frame IDs. Detector-only
  qualification disables selection and never creates control or hardware paths.
- The first pass exposed a deployment-lineage error: the nvinfer profile still
  loaded the older August 30 `yolo26s_drone_person_best_1_raw.engine`, not the
  requested `small_736.engine`. Its 0/405 rendered-drone result is retained as
  diagnostic evidence but is invalid as a qualification of `small_736`.
  SHA-256 verification showed that the training run's `weights/best.pt`
  exactly matches packaged `yolo26s_dataset2_e100_736.pt`, and that the Jetson
  `small_736.engine` exactly matches `yolo26s_dataset2_e100_736_raw.engine`.
- The corrected raw `small_736` pass processed all 1,350 frames at 60.150
  steady FPS. It detected people in eight of nine distinct billboard views
  (360/405 frames, 88.9% recall) and drones in five of nine views (225/405
  frames, 55.6% recall), with zero detections in all 540 blank frames and no
  cross-class confusion. All three 8 m drone views and two of three 5 m views
  passed; the three very-near/high-in-frame views and right-side 5 m view did
  not. The sweep uses CPU-rendered camera-facing PNG billboards for both
  classes; OpenGL separately substitutes `person.obj` and `drone.stl` meshes.
- The corrected networked V2/NvSORT/GPU-OSD/RTP-return pass received all 18
  positive cases, published 1,320 native V2 snapshots with zero invalid
  headers, zero non-monotonic drops, and zero legacy records, ran at 60.916
  steady pipeline FPS, and encoded 1,335 return-video buffers. NvSORT emitted
  106 person and six drone tracked observations; the deliberately
  discontinuous blank/teleport transitions make this a reacquisition stress
  result, not a raw-model recall measurement. Tracker continuity remains a
  separate smooth-motion qualification target.
- The persistent 60 Hz display canary no longer freezes the arbitrary first
  frame of the simulator's default moving scene. Its config now names the
  exact person billboard, pose, and background from a passing `small_736` plus
  NvSORT sweep case. After resetting frame-header correlation, a five-second
  live probe received 299/299 valid snapshots with the same person track and
  an applied selection in every snapshot.
- The analyzer unit tests passed 2/2. The complete repository suite passed 285
  tests and 12 subtests with the same four documented legacy baseline failures
  (two controller expectations and two swarm-planner expectations).

Qualification decision: the video transport and V2 publication contracts pass.
The correct `small_736` model demonstrably recognizes both billboard classes,
but does not pass the default 80% aggregate recall gate for the tested drone
views. Tracker continuity must be measured separately with a smooth-motion
ground-truth replay rather than this detector variation sweep.

Follow-up inference-boundary diagnosis:

- PyTorch and ONNX agree on the exact H.264 near-center drone frame at 0.713
  and 0.783 confidence respectively. Direct execution of the stripped
  TensorRT engine on an explicitly normalized, black-letterboxed RGB tensor
  also produces class 0 at 0.398, above the 0.30 nvinfer threshold. The same
  frame is absent only through the nvinfer video-input path. The remaining
  discrepancy is therefore in DeepStream preprocessing/input integration,
  not the dataset checkpoint, ONNX graph, TensorRT engine output, custom
  parser layout, or billboard renderer.
- Black instead of Ultralytics-gray letterbox padding reduces that case's
  checkpoint confidence from 0.783 to 0.396 but does not push it below the
  configured gate. A channel-swap experiment also does not reproduce the full
  DeepStream case pattern.
- Disabling aspect preservation is rejected: it improves raw DeepStream drone
  recall to 360/405 (eight of nine views) but collapses person recall to 0/405
  because 16:9 content is vertically stretched into the square network input.
  Production remains aspect preserving pending exact nvinfer tensor capture
  or an embedded/prevalidated preprocessing implementation.
- Updated the no-argument DeepStream preflight to validate `small_736.engine`
  instead of silently checking the obsolete `best_1` engine.

### 2026-09-14 - Plant-model qualification gate for controller redesign

The controller work is now explicitly gated in this order:

1. qualify a saved plant model on an independent hardware capture;
2. search and validate a basic PID baseline in offline simulation;
3. verify or replace the Kalman/feedforward estimator against the same
   scenarios; and
4. add estimator/feedforward only after the raw PID baseline is preserved for
   an A/B comparison.

Plant acquisition and tooling corrections:

- Reclassified the earlier unloaded capture as low-range evidence only.  Its
  archived nonzero F6 payloads contain only speed and acceleration bytes, so
  the previous journal statement that they contained a `00000032` run timer
  was incorrect.  The MKS SERVO42/57D RS485 V1.0.9 manual, page 76, confirms
  that timed F6 mode appends a four-byte big-endian runtime in 10 ms units.
- Added manual-backed timed F6 sweep commands with a shorter refresh interval,
  while retaining ordinary three-byte F6 for the final zero command.  A 300 ms
  canary emitted `00010A0000001E` and completed with no limits, missing replies,
  or dropped commands before the full sweeps were allowed.
- Found that acquisition had labeled samples with ZMQ publication/receipt time
  rather than the actual serial boundary.  The serial service now reports both
  speed-command and encoder-query wire monotonic timestamps plus encoder reply
  time.  The corrected canary measured about 6 ms wire-to-reply latency and no
  missing speed-wire timestamps.
- Recollected a bidirectional training sweep at encoded rates 0.209, 0.419,
  and 0.524 rad/s (1,620 samples) and a separate validation sweep at 0.314 and
  0.524 rad/s (848 samples).  Both completed with zero limit blocks, dropped
  queries, missing replies, or missing speed-wire timestamps.
- Fixed the deterministic group split so every repeated command magnitude is
  represented in both internal partitions.  The old sorted split could place
  every high-rate group in the holdout set and silently fit only low-rate data.
- Replaced the former transport-only fit acceptance with explicit independent
  accuracy gates: at least 200 samples per axis, at least 0.45 rad/s command
  coverage in fit and validation data, omega RMSE at most 0.05 rad/s, theta
  RMSE at most 0.01 rad, and direction bias at most 0.02 rad/s.
- Made frozen-fit validation use the exact selected discrete coefficients.
  Previously a selected 6 ms yaw discrete model was evaluated by an unstable
  continuous Euler step.  Also made the discrete position transition honor the
  acquisition definition: measured omega is a backward encoder difference, so
  `theta[k] = theta[k-1] + dt*omega[k]`.

Qualification result:

- Both axes selected a discrete first-order asymmetric rate model with zero
  resolved delay on the 5 ms search grid.  The unresolved sub-sample delay is
  bounded by the acquisition cadence; it is not a claim of zero physical
  latency.
- On the untouched validation sweep, yaw achieved 0.02413 rad/s omega RMSE,
  0.00520 rad theta RMSE, and 0.00362 rad/s worst direction bias.  Pitch
  achieved 0.03275 rad/s omega RMSE, 0.00614 rad theta RMSE, and 0.00328 rad/s
  worst direction bias.  Both covered 0.524 rad/s and passed every gate.
- The qualified evidence is
  `artifacts/gimbal_fit/controller_sysid_pid_range_wire_20260914/fit_report.json`
  plus `independent_validation_report.json`, sourced from the corresponding
  `controller_sysid_pid_range_wire_{train,validation}_20260914` logs.
- Focused fitting, validation, sweep-safety, and serial-timing tests passed 36
  tests plus nine subtests; the final fit/validation subset passed 14 tests plus
  two subtests.

Decision: the saved unloaded plant is qualified for bounded offline PID design
over the tested command range.  It is not yet a loaded-model or hardware-live
controller qualification.  PID gain search may proceed; Kalman/feedforward
work remains blocked until a raw PID simulation baseline is recorded.

### 2026-09-14 - Raw PID and LOS Kalman/feedforward offline qualification

Raw PID baseline:

- Added `tools/offline_pid_sim.py`, which refuses an unqualified or mismatched
  fit/validation pair and converts the selected asymmetric discrete model to a
  stable continuous equivalent for exact 50 Hz propagation.  The simulation
  applies the production 0.5 rad/s rate cap, 3.5 rad/s² slew cap, derivative on
  measured gimbal rate, and conditional integral anti-windup.
- Searched 462 gain tuples per axis on positive/negative 0.10/0.25 rad steps
  and a reversal trajectory, then evaluated the winner on unseen 0.15/-0.20
  rad steps, sine motion, and piecewise motion.  Selection never used the
  holdout scores.
- Selected yaw `Kp=8, Ki=0, Kd=0.1` and pitch `Kp=6, Ki=0, Kd=0`.  Unseen
  steps settled within 0.66 s yaw and 0.68 s pitch, with effectively zero
  overshoot and final error below 0.00015 rad.  Held-out moving RMS was
  0.0145/0.0235 rad for yaw sine/piecewise and 0.0179/0.0290 rad for pitch.
- Both axes passed explicit holdout gates: step final error <=0.005 rad,
  overshoot <=0.01 rad, settle <=1.0 s, moving RMS <=0.03 rad, and moving
  maximum error <=0.06 rad.  The versioned report, every scenario trace, gain
  table, and plots are under
  `artifacts/controller_sim/pid_baseline_wire_20260914/`.

Estimator audit and rebuild:

- Rejected the earlier estimator benchmark as controller-design evidence.  It
  used the hard-coded synthetic plant `omega_dot=-0.2*omega+0.6*u`, not either
  qualified axis.  In addition, native V2 observations currently carry no
  bearing rate, so the existing external-rate feedforward path cannot operate
  on the V2 pipeline.
- Added `jetson.los_kalman.py`: a timestamp-aware two-state absolute target-LOS
  Kalman filter with irregular-time process covariance, innovation gating,
  Joseph covariance update, non-monotonic timestamp rejection, gap and target
  identity reset, query-time prediction, and bounded recovery after two
  consecutive gated measurements.  The latter rejects isolated spikes but
  prevents a real maneuver from locking the filter to an obsolete trajectory.
- Added `tools/offline_los_estimator_sim.py`.  It compares the frozen raw PID
  with Kalman position/rate feedforward using identical deterministic 30 Hz
  measurements, 50 ms delivery latency, 3 mrad noise, 10% dropout, and the
  qualified plant.  Estimator process noise and feedforward gain are selected
  on separate sine, piecewise, and ramp/hold trajectories before the untouched
  holdout is evaluated.
- Selected yaw acceleration spectral density `0.001` and pitch `0.005`, with
  feedforward gain `0.5` on both axes.  Held-out mean RMS error improved 11.6%
  on yaw and 18.2% on pitch versus raw PID.  No scenario regressed over 10%,
  command variation stayed within 1.5x baseline, innovation rejection stayed
  below 25%, and measured estimator update/predict p95 remained below the 1 ms
  gate.  Both axes passed.
- Integrated the rebuilt estimator as an opt-in, still shadow-only mode of
  `ShadowRatePolicy`.  The default compatibility path is unchanged.  The new
  path reconstructs absolute target bearing from camera error and timestamp-
  aligned gimbal pose, deduplicates repeated source frames, predicts to the
  current controller tick, applies selected gimbal-rate damping plus absolute
  LOS rate feedforward, and atomically resets both axes on track/time changes.
- Estimator convergence, spike/reacquisition, ordering, identity reset,
  deterministic A/B, qualification, and policy integration tests pass.  The
  validation report/traces/plots are under
  `artifacts/controller_sim/los_kalman_feedforward_wire_20260914/`.

Decision: the unloaded offline chain now has three accepted gates—plant, raw
PID, and PID plus LOS Kalman/feedforward.  The estimator is not enabled in the
production configuration and no live command authority was added.  Next gate
is deterministic policy replay from V2 observations with these frozen values,
followed by shadow parity and only then bounded hardware verification.

### 2026-09-15 - Frozen controller profile and native-V2 replay gate

- Added `jetson/qualified_controller_profile.py` as the single loader from the
  qualified LOS-estimator report into `ShadowRatePolicyConfig`.  It requires
  the overall and both per-axis qualifications, validates the recorded 50 Hz
  controller/30 Hz vision cadence, preserves the selected PID, Kalman, and
  feedforward values, and rejects nonzero integral gains because the current
  shadow policy intentionally has no integral state.
- Regenerated
  `artifacts/controller_sim/los_kalman_feedforward_wire_20260914/` with the
  controller cadence recorded in the report.  The result remained yaw
  `q=0.001`, pitch `q=0.005`, feedforward gain `0.5` on both axes, and qualified
  holdout improvements of 11.6% yaw and 18.2% pitch.
- Extended `tools/replay_control_protocol_trace.py` with the explicit
  `--qualified-controller-report` option.  It is valid only with the
  non-actuating `shadow-rate` policy, loads all controller values from the
  passing artifact, and emits the resolved artifact path plus SHA-256 in the
  replay summary for provenance.  The existing explicit-gain replay mode is
  unchanged and its golden test remains exact.
- Added a six-frame native-V2 fixture with varying target error, gimbal pose,
  gimbal rate, measurement age, and complete source-frame provenance.  The
  target intentionally contains no `bearing_rate_rad_s`, exercising the new
  internal LOS estimator rather than the legacy external-rate input.  Two
  independent replays matched the same fixed six-intent golden output exactly;
  all records were accepted, outputs remained bounded by rate/acceleration
  limits, and physical control stayed disabled.
- Focused loader, replay, policy, Kalman, and estimator coverage passed 18
  tests.  The legacy replay fixture still matches byte-for-byte.

Decision: deterministic V2 replay with the frozen qualified controller is now
accepted.  The next controllable gate is time-aligned shadow parity against a
running simulated V2 observation stream, before any command-producing hardware
test.  The Jetson encoder-only publisher was found stopped after a serial write
timeout; no serial or gimbal command service was left active during this gate.

Live fixed-rate shadow qualification:

- Extended the passive `record_control_protocol_trace.py` path to accept the
  same qualified report, require its recorded 50 Hz cadence, and store the
  resolved report path/SHA-256 in trace metadata and the final summary.  It now
  reports tracking, hold, and limited-intent counts while retaining no command
  publisher, serial import, or physical authority.
- Added `publish_control_shadow_fixture.py`, a bounded three-socket source that
  publishes a guaranteed selected native-V2 target moving in both image axes
  at 60 Hz, time-varying synthetic gimbal pose/rate, and an explicit safe
  manual-authority state.  This deliberately isolates controller behavior from
  detector/model effectiveness; V2 observations contain no external target
  bearing rate.
- Added an independent trace validator.  It reloads the qualified artifact into
  a fresh policy, reproduces every recorded intent, checks exact equality,
  source provenance on every tracking decision, target-error variation,
  controller bounds, scheduler health, report hash, and the explicit
  physical-control-disabled marker.
- The preserved 3.5 s bounded socket/scheduler run recorded 169 observations
  and 169 intents.  All 169 independently replayed intents matched exactly;
  149 were tracking decisions and all 149 carried complete V2 source
  provenance.  The
  remaining 20 were correct target-invalid holds after the bounded publisher
  stopped.  There were zero decode errors, zero missed 50 Hz periods, 2.314 ms
  maximum deadline lateness, 0.723 rad yaw-error span, 0.222 rad pitch-error
  span, and no external bearing-rate observations.  Both command axes remained
  at or below the frozen 0.5 rad/s limit.  The validator returned qualified.

Decision: the redesigned qualified observation-to-intent path now passes its
first live fixed-rate shadow gate with a controlled moving target.  This is
self-parity of the redesign, not yet the item-13 legacy-versus-redesign parity
comparison and not hardware qualification.  The next gate is to preserve the
same evidence from a real selected V2 stream plus read-only encoder and real
manual state, then evaluate legacy/redesign decisions from identical snapshots.

### 2026-09-16 - Persistent moving-simulator V2 tracking display

- Added `configs/deepstream_pc_moving_tracking.yaml`, a control-free 720p60
  CPU-sprite profile.  A person billboard follows a bounded 0.25 m/s path near
  the previously qualified `[2, -6]` view.  `freeze_frame` is disabled, no
  simulator-control endpoint is configured, and the physical serial path is
  absent.
- After the host/Jetson package upgrade, the September 14 DeepStream process
  remained alive but its health timestamp was stale.  Restarted only the
  passive runtime from the isolated `IDCS-v2-video-5f80885` worktree.  It
  loaded the `small_736` TensorRT engine, NvSORT, target selection, GPU OSD,
  native V2 publication, and RTP97 return video successfully.
- A bounded 15 s moving-target canary delivered 849 valid V2 snapshots, 818
  person tracker observations, one stable NvSORT identity, and 773 applied
  selections.  There were zero invalid records, zero nonmonotonic source or
  frame timestamps, and zero selected-track changes.  The tracked box moved
  from normalized x about 0.325 to 0.275 during the sample, confirming actual
  image-plane motion rather than a repeated frame.  Host NVENC streaming held
  about 59.6 FPS.
- Updated the launch environment to use the active Mutter Xwayland authority
  created by the upgraded desktop session.  The host UI then opened on the
  desktop with NVDEC and consumed both return video and V2 metadata.  During
  the persistent run it reported one object, one track, selected ID 0, and
  continuously increasing decoded/metadata counters.
- A separate five-second live audit of the persistent run received 298 V2
  snapshots and 293 tracker observations on the same selected identity, with
  zero invalid records, frame gaps, nonmonotonic timestamps, or selection
  changes.

Decision: the simulator -> RTP96 -> small_736 -> NvSORT -> V2 selection -> GPU
OSD/RTP97 -> host NVDEC UI path is running persistently with a genuinely moving
target and stable tracking.  Host streamer/UI and the Jetson runtime are
control-free; no gimbal, serial, controller, or hardware authority was started.

### 2026-09-16 - Qualified gray-box simulation camera integration

- Replaced the simulator camera's ideal-only rate integration with an opt-in
  `qualified_gray_box` path while preserving `ideal` as the compatibility
  default.  The stateful plant applies fitted asymmetric positive/negative
  gains, disturbance, variable integration intervals, and command delay, and
  reports realized pose/rate rather than echoing the requested rate.
- Extracted the plant realization and frozen-fit loader from
  `tools/offline_pid_sim.py` into `common/gimbal/gray_box.py`, so offline PID,
  estimator, and live camera simulation now use one implementation.  Runtime
  loading fails closed unless the independent validation report qualifies both
  axes, names the supported selected model, and references the supplied fit.
  External `CamState` updates reset the plant state and delay history; tilt
  saturation also prevents simulated outward velocity from accumulating at a
  limit.
- Enabled the model in `configs/deepstream_pc_moving_tracking.yaml` using the
  independently accepted September 14 fit and validation artifacts.  The
  streamer logs resolved plant provenance at startup and publishes the plant's
  realized rates in simulated camera headers.  Plant state advances only when
  an explicit simulator-control socket is attached; without one, the passive
  video profile holds camera pose instead of interpreting socket absence as a
  zero-rate motor command and accumulating the fitted zero-command bias.  No
  `--sim-control-sub` is configured in this persistent display profile, so it
  remains control-free.
- Added deterministic coverage for directional gain, delay, realized state,
  external-pose reset, and repository artifact qualification.  The focused
  camera/offline PID set passed 30 tests plus 3 subtests; downstream estimator,
  replay, and trace-validation coverage passed 7 tests.  The full repository
  run passed 321 tests and 12 subtests; four unrelated existing controller and
  swarm-planner expectation failures remained in untouched files.
- A controlled 60 Hz step using the real qualified artifacts distinguished the
  plant from the ideal integrator on the first frame: for commands of yaw
  `+0.5` and pitch `-0.3` rad/s, the qualified camera realized `+0.4699` and
  `-0.1310` rad/s versus immediate ideal rates of `+0.5` and `-0.3`.  After one
  second it reached `+0.5038` and `-0.2966` rad/s, then decayed under zero
  command to the fitted near-zero bias.  Evidence is in
  `artifacts/deepstream/graybox_camera_20260916/plant_runtime_validation_report.json`.
- The Jetson working checkout was found stale and heavily dirty; its runtime
  lacked the V2 pipeline module and bound legacy result port 5556.  It was not
  modified.  The host-tested `common`, `jetson`, and `configs` trees were copied
  to an isolated `/tmp/idcs_v2_overlay`, verified with Python safe-path mode,
  and launched from there while continuing to use the Jetson checkout only for
  existing TensorRT assets.  The resulting runtime owns ports 5555 and 5564,
  loaded the `small_736` engine, and restored native V2 metadata to the UI.
- The persistent host streamer is running at about 59.6 FPS with logged
  qualified-fit provenance.  A separate 10 s audit received 597 consecutive
  V2 snapshots: all 597 carried person track 0 and the same selection, with
  zero invalid records, frame gaps, nonmonotonic frame/time values, or target
  changes.  The box moved across normalized spans of about 0.088 in x and
  0.030 in y.  The raw audit and report are preserved under
  `artifacts/deepstream/graybox_camera_20260916/`.
- Extended passive observation exposed the fitted pitch bias accumulating when
  no simulator actuator was attached, eventually moving the target out of the
  field of view.  The streamer now advances plant state only when an explicit
  `--sim-control-sub` exists; a passive display holds its pose.  This preserves
  the qualified disturbance model for controller experiments without treating
  an absent command transport as a continuous zero-rate motor command.
- The restart also confirmed that header-correlation generations must be
  ordered: starting a fresh correlator on an old streamer and then resetting
  the streamer frame counter correctly causes subsequent headers to be rejected
  as nonmonotonic.  The accepted sequence is to keep/start the desired streamer
  generation and then start a fresh DeepStream correlator.  With that ordering
  and passive pose hold, the final 5 s audit received 298 snapshots, all with
  selected person track 0, and again had zero gaps, invalid records,
  nonmonotonic values, or selection changes.  The final report is
  `artifacts/deepstream/graybox_camera_20260916/final_pose_hold_audit_report.json`.

Decision: the live simulation camera now uses the independently qualified
gray-box hardware plant and the full moving-target detection/tracking/UI path is
healthy.  Dynamic plant response has been verified with a deterministic local
step; the persistent video run remains passive.  The next controller gate can
attach only an explicit non-production simulator endpoint and evaluate the
qualified PID against this same plant before any hardware command authority is
introduced.

### 2026-09-16 - V2 closed-loop moving-target simulation qualification

- Added `tools/run_sim_tracking_controller.py`, a host-only V2 controller
  sidecar that consumes native `PerceptionSnapshotV2` and the simulator's
  realized `CamState`, builds the canonical `ControlObservation`, and runs the
  frozen qualified PID plus LOS Kalman/feedforward policy at its qualified
  50 Hz cadence.  It adapts the resulting `ControlIntent` to `ControlCmd` only
  for the simulator.  The tool imports no serial/gimbal driver, requires an
  explicit `--enable-sim-control`, rejects non-loopback command/state endpoints,
  and refuses the production `net.zmq_control` endpoint.
- Extended `pc.streamer` with an explicit loopback-only simulator CamState PUB
  and matching loopback-only simulator command SUB.  These endpoints are legal
  only for `source: sim` and must be distinct.  Passive runs retain pose-hold;
  the qualified gray-box plant advances only when this explicit simulator
  command path is attached.
- Added unit coverage for the loopback safety gate and V2 observation/intent to
  simulator-command adaptation.  The focused plant, camera, observation,
  qualified-profile, policy, and new sidecar set passed 43 tests plus 3
  parameterized subtests.  Both runtime entry points also passed configuration
  checks using the selected fit, independent validation report, and frozen
  controller report.
- Restarted in generation-safe order: stopped the old Jetson correlator, stopped
  the passive streamer, started the simulator controller and controlled
  streamer, then launched a fresh isolated Jetson V2 runtime.  This avoided
  carrying old frame-correlation state across a streamer frame-ID reset.  The
  existing host UI reconnected without restart.
- The first bounded 45 s run deliberately included startup and established the
  command sign: yaw pose moved 0.329 rad, pitch moved 0.026 rad, and mean bearing
  error fell from 0.299 rad over the first 50 valid samples to 0.058 rad over
  the last 50.  Its target-invalid holds are dominated by the interval before
  the fresh detector became ready, so it is preserved as startup evidence and
  not used as the steady-state acceptance result.
- A second controlled 30 s run began only after video, detector, V2 publication,
  and UI were stable.  It produced 1,429 tracking commands out of 1,430 total,
  with one initial hold, zero command drops, and zero missed control periods.
  It received 1,784 V2 snapshots and 1,785 realized camera states.  The moving
  target remained selected as track 0 while the gray-box camera moved through
  0.221 rad yaw and 0.051 rad pitch.  Mean bearing-error norm remained bounded
  and improved from 0.0543 to 0.0514 rad despite continued target motion.
  Evidence is under
  `artifacts/deepstream/graybox_closed_loop_20260916/`.
- After bounded acceptance, launched the same controller persistently against
  loopback ports 5571/5572.  Its first live five-second status interval reported
  238 tracking commands out of 239 with zero drops.  The host streamer remained
  near 59.5 FPS, and the UI continued to report one object, one stable track,
  and selected track 0.  Hardware control remained disabled throughout.
- Work was interrupted by an unrelated SSH transport fault after the host
  OpenSSH upgrade.  Packet capture showed TCP setup and the client banner
  arriving, but server SSH packets marked DSCP EF (`0xb8`) were not acknowledged.
  Setting `IPQoS none` restored reliable access; no simulation, controller, or
  serial process caused the network failure.

Decision: the V2 video/detection/tracking/UI pipeline now drives the qualified
controller against the independently qualified gray-box plant, and the
simulated camera visibly follows the moving selected target.  The accepted
path is isolated from production control and hardware serial.  Further
controller/estimator changes should be compared against the preserved bounded
trace before any hardware authority is introduced.

### 2026-09-16 - Simulator baseline separated from hardware tuning

- Visual review rejected the preceding hardware-profile controller run despite
  its short bounded report.  The longer 438 s result confirmed the concern:
  mean bearing-error norm increased from 0.0276 rad over its first 50 valid
  samples to 0.0460 rad over its last 50.  The prior section's decision is
  therefore superseded.  Its reports and short trace are retained only as
  rejected evidence under
  `artifacts/deepstream/graybox_closed_loop_20260916/rejected_hardware_profile/`.
- Found an end-to-end camera contract error.  `configs/control.yaml` describes
  the real camera as 135 by 73 degrees, while `SimCamera` rendered at a fixed
  60-degree vertical FOV (91.49 degrees horizontal at 1280 by 720).  A
  14-pixel horizontal error was consequently interpreted as 0.053 rad instead
  of about 0.023 rad, inflating the controller response by roughly 2.35 times.
- Added `configs/control_sim.yaml` as an explicit simulator-only camera and
  controller profile.  `sim.camera.fov_y_deg` is now consumed by the renderer,
  and the simulator controller derives matching horizontal FOV and focal
  lengths from that same value and the active frame aspect.  Startup evidence
  reports 91.4928 by 60 degrees and `fx = fy = 623.538 px`.  The real camera
  configuration remains unchanged.
- Replaced the simulator tool's hardware-qualified PID/Kalman/feedforward
  loader with a conservative simulator baseline: bounded proportional feedback
  at 50 Hz, gains of 3.0 per axis, yaw/pitch rate limits of 0.30/0.20 rad/s,
  and acceleration limits of 1.0/0.8 rad/s squared.  It loads no hardware
  controller report, no LOS Kalman parameters, no feedforward gains, and no
  real-gimbal position limits.  The tool reports
  `hardware_controller_tuning_loaded: false` and retains the loopback-only,
  no-serial safety boundary.
- Added explicit end-to-end simulator gates for acquisition, steady-state
  centering, target retention, command drops, saturation, duration, and actual
  camera movement.  Acquisition is measured separately from steady state so a
  large initial target offset cannot be hidden or incorrectly charged to the
  post-acquisition tracking distribution.  Focused camera/plant/controller
  coverage passed 44 tests plus 3 parameterized subtests.
- The final unchanged-controller 45 s acceptance run acquired and held a
  22-pixel p95 window in 0.91 s.  It issued 2,173 tracking commands out of
  2,176 (99.86 percent), with zero command drops and zero missed periods.
  After the declared five-second acquisition window, tracking error measured
  6.45 px RMS and 9.26 px p95, below the 14/22 px gates.  Rate limiting occurred
  on 1.70 percent of commands, below the five-percent gate, while the camera
  moved through 0.142 rad yaw and 0.174 rad pitch.  The last 50-sample angular
  error averaged 0.00551 rad.  The report qualified with no failures and is
  preserved under `graybox_closed_loop_20260916/sim_baseline/` with its trace.
- Launched the accepted baseline persistently after the bounded run.  Its first
  ten seconds delivered 482 tracking commands out of 483 with zero drops; the
  UI continued to show one person, stable track 0, and an active selection,
  while the streamer held approximately 59.6 FPS.

Decision: the simulator is an integration environment for system operation,
video transport, V2 detection/tracking, UI, and baseline closed-loop behavior.
Its controller settings are independent of real-hardware tuning.  The
gray-box rate model may provide plausible actuator dynamics, but simulator
results must not qualify real PID or estimator gains; those require separate
hardware traces and acceptance.

### 2026-09-16 - Restore the existing simulator/hardware-in-loop selector

- Revisited the earlier architecture after confirming that the mode already
  existed as `sim.use_jetson_cam_state`. It is now the authoritative selector
  instead of adding a competing command-line controller mode. `false` means
  the stable simulated-motion substitute; `true` means hardware-in-loop, where
  the simulated camera follows physical encoder `CamState` and the separately
  authorized tuned controller is responsible for mount motion.
- Added a pure `common.sim_mode` resolver with strict boolean validation and
  explicit `stable_substitute`/`hardware_in_loop` provenance. Streamer
  `--check` now reports both the resolved mode and whether it can move the
  physical mount.
- Removed the legacy stale-CamState fallback from hardware-in-loop behavior.
  Missing or stale encoder telemetry now holds the last physical pose; it can
  no longer silently switch to simulated ControlCmd integration. The streamer
  rejects a simulator command endpoint whenever
  `sim.use_jetson_cam_state: true`, and the baseline simulator controller
  refuses to start in that mode.
- Config-only verification exercised both states without opening video,
  sockets, serial, or motor authority. Stable mode reported
  `moves_physical_mount: false`; hardware-in-loop reported true. Both mixed
  mode checks failed closed with the expected diagnostics. The focused suite
  passed 48 tests plus 3 parameterized subtests.
- The V2 limitation remains explicit: DeepStream perception publishes V2, but
  the live `ControlIntent` to gimbal bridge is still non-actuating and marked
  partial in `docs/controller_overhaul_plan.md`. Therefore the
  hardware-in-loop configuration contract is restored, but live V2 mount
  motion is not yet declared operational and was not started during this work.
- Updated `docs/verification_strategy.md` so detection/camera fidelity and
  motion substitution have separate evidence rules. The simulated camera must
  be measured against real captures for projection, scale, frame/codec path,
  latency, blur/noise/exposure, and occlusion. Real labeled replay remains
  mandatory for detector conclusions. Simulation may exercise an already
  tuned controller in hardware-in-loop, but may never tune or qualify its gains.

Decision: continue using the accepted stable substitute for ordinary simulated
operation. Rebuild the V2 live intent/actuation bridge before re-enabling the
existing hardware-in-loop mode, then validate it with command authority
disabled before any bounded unloaded motion.

### 2026-09-16 - Native OpenGL 3D pipeline qualified on V2

- Added `configs/deepstream_pc_moving_tracking_opengl.yaml` as the explicit
  GPU-rendered counterpart to the CPU moving-target profile. It remains in
  `stable_substitute` mode, keeps simulator control/CamState on loopback, and
  explicitly leaves `sim.use_jetson_cam_state: false`; it cannot command the
  physical mount.
- Confirmed the renderer contract directly. In native mesh mode, `person`
  resolves to `assets/meshes/person.obj` and `drone` resolves to
  `assets/meshes/drone.stl`. A deterministic three-frame capture reported
  `OpenGLRenderer`, cached `person.obj`, and an EGL context whose vendor and
  renderer were `NVIDIA Corporation` and `NVIDIA GeForce GTX 1080 Ti`. The
  retained montage contains the moving shaded 3D person and GPU-rendered
  building, ground, sky, and shadows; no billboard override is active in the
  qualified profile.
- The first full-path attempt correctly rejected the old rendering behavior:
  H.264/RTP and Jetson inference sustained about 59/57 FPS, but the small dark
  mesh produced no reliable target. Daylight procedural lighting restored
  contrast, then deterministic captures exposed two geometry bugs. The mesh
  loader used `trimesh.rezero()` despite promising recentering, placing the
  mesh minimum rather than its centre at the origin, and target transforms
  multiplied by requested height without accounting for native mesh extent.
  Mesh bounds are now centred explicitly. Person targets are uniformly scaled
  from vertical extent and drone targets from horizontal extent, preserving
  model proportions while honoring their semantic size axis.
- The controlled 3D validation target remains a moving `person.obj`, but is
  deliberately placed and sized above DeepStream's 16-pixel minimum object
  dimension across its path. This isolates V2 video/detection/tracking/UI and
  simulated-motion operation from an avoidable synthetic-scene disturbance;
  detector thresholds and controller gains were not changed.
- Focused OpenGL and simulator regression coverage passed 42 tests plus 3
  parameterized subtests. The live host streamer held approximately 59.4 FPS,
  Jetson DeepStream held approximately 57.8 FPS, and the UI displayed one
  detected object, stable NvSORT track 0, and active selection.
- The final unchanged-controller 45 s acceptance run acquired the 22-pixel
  window in 1.31 s and issued 2,145 tracking commands out of 2,187 total
  (98.08 percent). It dropped zero commands and missed zero control periods.
  After warmup, tracking error was 9.80 px RMS and 13.17 px p95, below the
  14/22 px gates. Rate limiting was 2.15 percent, and the simulated camera
  moved through 0.385 rad yaw and 0.192 rad pitch. The report qualified with
  no failures.
- Accepted evidence is under
  `artifacts/deepstream/opengl_mesh_v2_20260916/`: the full acceptance report,
  control trace, and native-mesh montage.

Decision: the host GPU 3D renderer is operational through the complete V2
pipeline—native mesh render, NVENC H.264/RTP, Jetson small_736 inference,
NvSORT tracking, V2 metadata, UI display, and stable simulated camera motion.
This qualifies controlled system integration only; real labeled replay remains
the authority for real-world detector accuracy, and simulator results do not
qualify physical-mount controller tuning.

### 2026-09-16 - Return video reduced to 30 FPS and GPU OSD smearing removed

- Added an independently resolved `video.active_return_profile` to the
  immutable V2 configuration path and selected `720p30`: 1280x720, 30 FPS,
  and 7000 kbit/s. The post-OSD return branch alone is rate-limited; simulator
  input, inference, tracking, and V2 metadata remain on their approximately
  60 FPS path.
- Replaced hard-coded DeepStream return dimensions/rate/bitrate with validated
  runtime arguments. Live health now reports inference and post-OSD return
  rates separately, preventing a configured rate from being mistaken for a
  measured one.
- Reduced the PC return jitter buffer from 120 ms to a configured 20 ms and
  the downstream leaky queue from five frames to one. The display therefore
  consumes the newest available frame rather than accumulating visual-only
  latency. Accepted both historical `pc.monotonic` and canonical
  `pc_monotonic` provenance so the UI end-to-end latency readout works again.
- Direct decoded-frame sampling proved the observed trails were present before
  the X11/OpenCV display stage and affected only OSD text and rectangles. The
  GPU-mode `nvdsosd` element had been fed tracker NV12 surfaces directly. The
  return path now converts NVMM to RGBA before GPU OSD, then converts to NV12
  only for the hardware encoder. Five one-second-spaced direct decoded frames
  and a final live UI capture showed clean moving labels and boxes with no
  retained glyphs, broken edges, or red trails.
- The final live health sample measured the post-OSD return at approximately
  30 FPS while DeepStream inference remained approximately 60 FPS. The
  simulator streamer and loopback-only stable controller stayed running;
  physical motor/serial authority remained disabled throughout.
- Focused V2 runtime and DeepStream pipeline coverage passed 21 tests. Both
  updated processes reached readiness, the UI continued to show the moving 3D
  person, NvSORT track 0, and active selection, and the final on-screen status
  reported a finite end-to-end latency.
- A live follow-up exposed intermittent black flashes after the return-rate
  reduction. The PC UI still derived its 25 ms pull timeout from the 60 FPS
  input profile, shorter than the 30 FPS return period, and explicitly cleared
  the retained frame on each timeout. It now resolves `active_return_profile`
  independently (50 ms pull timeout for `720p30`), retains the newest decoded
  frame between arrivals, and draws local UI state on a fresh copy. The
  expanded focused suite passed 25 tests. Eight live window samples across
  several return periods all retained the scene: mean luminance stayed between
  141.968 and 143.316 with only 2.10-2.15 percent black pixels, so no sampled
  frame was a black fallback.

Decision: retain 720p30 as the visual-only return profile with 20 ms jitter and
newest-frame queuing. Preserve the roughly 60 FPS inference/tracking path and
the RGBA input contract for GPU-mode OSD; simulator display settings remain
independent of any real-mount tuning.

### 2026-09-16 - Native OpenGL drone target qualified in the V2 simulator

- Added `configs/deepstream_pc_moving_drone_opengl.yaml`, loaded after the
  qualified OpenGL profile, to replace only the scene target list with a
  moving 1.2 m native `assets/meshes/drone.stl`. Added the separate
  simulation-only `configs/deepstream_drone_sim_validation.yaml` so drone
  selection can be exercised without weakening the real-target policy or
  enabling a physical control endpoint.
- A pre-test lineage audit found that the live Jetson runtime resolved the
  relative nvinfer profile against its stale repository checkout instead of
  the active V2 overlay. It was therefore still loading the obsolete
  `yolo26s_drone_person_best_1_raw.engine`. Runtime asset resolution now follows
  the active `<root>/configs` tree. The corrected live log loaded
  `/tmp/idcs_v2_overlay/configs/deepstream/nvinfer_yolo26s_736_drone_person_smoke.txt`
  and the distinct dataset2 100-epoch
  `yolo26s_dataset2_e100_736_raw.engine` (SHA-256
  `c9ea7dfcbf8b05002a584cc3b02dd751f922b2a42ecc2e6ef8309e8a12fa9f73`).
- Passive visual verification showed the shaded 3D drone mesh, classified it
  as `drone` at 0.72 confidence, assigned NvSORT track 0, and applied the
  simulation-only rank-1 target selection. The UI return remained clean at
  30 FPS with GPU OSD active.
- The unchanged stable-substitute controller then completed a 58.33 s bounded
  run. It acquired the drone in 2.06 s, issued 2,799 tracking commands out of
  2,821 (99.22 percent), dropped zero commands, and missed zero periods.
  Steady tracking error was 8.91 px RMS and 11.96 px p95, below the 14/22 px
  gates. Rate limiting was 3.12 percent, and the simulated camera moved through
  0.371 rad yaw and 0.352 rad pitch. The simulation-integration acceptance
  report qualified with no failures.
- After the runtime path correction, live DeepStream and return rates measured
  58.74 and 29.76 FPS respectively with one drone, one stable track, and an
  applied selection. Focused runtime/pipeline/UI coverage passed 27 tests.
  Evidence is retained under
  `artifacts/deepstream/opengl_drone_v2_20260916/`.
- The bounded controller was stopped after report finalization. The drone
  streamer, control-free Jetson detector, and UI remain live; all simulated
  motion endpoints are loopback-only and physical motor/serial authority was
  disabled throughout.

Decision: the native drone mesh passes controlled V2 system-integration and
stable simulated-camera tracking. This does not qualify real-world drone
detector accuracy or any physical-mount controller tuning; those remain bound
to representative labeled captures and hardware evidence respectively.

### 2026-09-16 - Stable simulator idle drift isolated

- A persistent drone-tracking launch began with no valid selected target while
  the simulated camera was already displaced. Although the controller emitted
  only zero-rate `target_invalid` commands, pitch continued downward.
- The controller was stopped immediately. Its trace proved zero requested yaw
  and pitch rates while CamState retained a small negative rate. The
  stable-substitute stack was still inheriting the identified hardware
  gray-box plant, whose fitted disturbance term makes zero input a nonzero
  steady-rate condition.
- `configs/control_sim.yaml` now explicitly overrides `sim.plant_model.mode`
  to `ideal`. This keeps ordinary simulated-camera motion deterministic and
  command-following, as required by the simulation/real-control separation.
  Hardware-in-loop profiles must opt into the fitted plant independently.
- Focused simulator coverage passed 32 tests (plus three subtests). After the
  streamer reset and control-free Jetson runtime restart, V2 metadata resumed
  with one drone, one NvSORT track, and selected track 0.
- A bounded 30.03 s closed-loop rerun passed every simulation gate: 0.95 s
  acquisition, 98.90 percent tracking commands, zero command drops and missed
  periods, 9.05 px steady RMS error, 12.20 px steady p95 error, and 2.34
  percent rate limiting. Persistent loopback-only drone tracking was then
  enabled; live status continued to report `tracking`, selected track 0, and
  zero command drops.

Decision: zero command must be an idle equilibrium for the stable simulator.
Do not use the hardware gray-box disturbance model in visual pipeline and
system-operation simulation. The corrected drone simulation stack is approved
to remain active for V2 video-pipeline evaluation only.

### 2026-09-16 - Operational HUD restored on the V2 host display

- Audited the live V2 return against the legacy operator-overlay inventory.
  DeepStream already supplied GPU detector/tracker boxes, selected-target
  colour, labels, and pipeline health, while the host supplied only a compact
  footer. Heading/elevation, aim cues, assessment details, controller state,
  and laser status had not crossed the V2 migration boundary.
- Added a V2-native host HUD that consumes only `PerceptionSnapshotV2`,
  `CamState`, and `ControlCmd`; it does not materialize legacy `DetectionMsg`.
  The operational inventory now includes the camera-centre reticle, heading
  tape, elevation scale, track histories, person cross marks, selected-target
  vector/reticle, range dimension and value, threat/rank cues, controller mode
  and rate commands, laser state/geometry when supplied, telemetry freshness,
  and the existing frame/object/track/selection/latency footer.
- MPC cost-term bars are excluded from the operational HUD. They now require
  the separate explicit `--mpc-overlay` flag in addition to a control socket;
  subscribing to controller state alone cannot enable them.
- A deterministic two-target fixture, guaranteed selection, range/threat/rank
  assessment, CamState, controller command, and laser geometry verified the
  exact rendered inventory and asserted that no MPC element was present. The
  focused HUD/DeepStream suite passed 30 tests.
- Live OpenGL drone validation used loopback CamState and controller telemetry.
  The log repeatedly reported reticle, attitude, tracks/history, selection,
  range, threat, controller, laser-status, and freshness elements with
  `mpc_terms=off`; the UI remained near 30 FPS with one object, one NvSORT
  track, and selected track 0. The simulator honestly reports `laser n/a`
  because its baseline controller publishes no laser geometry. A final visual
  capture is stored as
  `artifacts/deepstream/opengl_drone_v2_20260916/v2_hud_live.png`.

Decision: the V2 display owns operational overlays on the host using typed V2
telemetry. GPU OSD remains responsible for detector/tracker annotations. MPC
term visualization stays an explicit diagnostic, never part of the default
operator display.

### 2026-09-16 - HUD overlap, repetition, and tape-motion audit

- The attitude tapes were not limited by OpenCV or the 30 FPS return. Their
  tick positions were fixed relative to the screen while only the numeric
  labels changed with CamState, which inherently produced stepped motion.
  Heading and elevation ticks are now anchored to fixed world-angle multiples
  and projected from the fractional live yaw/pitch value every UI frame. Two
  deterministic sub-step tests prove that a 0.25 degree pose change translates
  every shared tick continuously and uniformly rather than relabeling a fixed
  slot. Tape lines use OpenCV fixed-point subpixel coordinates with
  anti-aliasing, while current bearing/elevation readouts retain one decimal
  place. The remaining temporal limit is the intentional approximately 30 FPS
  return/display cadence, not a five-degree UI quantizer.
- Removed repeated range, threat, and rank text from the host HUD because the
  GPU target label already owns those facts. The host retains only distinct
  visual semantics: threat-coloured aim reticle, range-dimension mark, and aim
  vector. The separate rank badge was removed.
- Removed the repeated frame number from the bottom footer and named its FPS
  explicitly as return/display FPS. The GPU health banner remains the source
  for inference frame, inference latency, and pipeline FPS; the footer now
  owns objects, tracks, selection, end-to-end latency, and return FPS.
- Removed command age from the controller line because the freshness line
  already reports CamState and command ages. Made the controller panel width
  stable so changing signed rate text cannot leave clipped or residual glyphs.
- Expanded the host authority label from `SIM` to `SIM-ONLY`. This disambiguates
  the active loopback camera controller from the Jetson banner's intentionally
  disabled physical-control path.
- The final live frame has separate non-overlapping health, attitude, target,
  control/freshness, and return-status regions. The focused suite passes 32
  tests, the live inventory remains complete with `mpc_terms=off`, and drone
  track 0 remains selected under loopback-only control.

Decision: tape stepping was an implementation defect, not a UI limitation.
Keep world-anchored tape geometry and single ownership for each textual fact;
use host overlays only for cues that add information beyond the GPU label.

### 2026-09-16 - Laser cue redesignated as parallax indication

- Confirmed that the existing laser fields describe projected image-plane
  geometry only. They do not represent an emitter command, physical output,
  interlock state, or hit confirmation.
- Replaced the operational HUD states `laser ON`, `laser ADJUST`, and
  `laser n/a` with `parallax ALIGNED`, `parallax OFFSET`, and
  `parallax n/a` so the display states exactly what the calculation proves.
- Replaced the solid dot and beam-like line with a diamond aim marker and a
  compensation arrow. Existing wire-field names remain unchanged for schema
  compatibility, but the V2 UI inventory now reports `parallax_status` and
  `parallax_cue`.
- The focused HUD suite passes five tests; the broader V2 HUD/DeepStream
  regression selection passes 28 tests. The live host UI was reloaded on the
  existing 720p30 simulation stream and reports `parallax_status`, one drone,
  one NvSORT track, selected track 0, and `mpc_terms=off` at about 31 FPS.

Decision: present the projection only as a parallax-compensation cue. Do not
imply firing, emitter state, or realistic terminal capability in the UI.

### 2026-09-16 - Default parallax projection and size-consistent drone ranging

- Activated the parallax projection by default in the loopback-only V2
  simulator controller. The current `laser_*` wire names remain for
  compatibility, but they carry only the image-plane parallax cue.
- The projection now loads the mount geometry from configuration. The source
  `laser.offset_m` value `{x: 0.0, y: -0.4, z: 0.0}` becomes
  `[0.0, 0.4, 0.0]` in the internal CV frame; the configured forward direction
  is projected with the simulator camera intrinsics. The control tolerance is
  20 px.
- Selected V2 `known_size:width` assessments supply projection depth. Before a
  valid selection is available, the cue uses the explicit 10 m configured
  fallback and labels that source `config_default`; it never presents the
  fallback as a measured range.
- Restored the native drone mesh to its predetermined ranging width of 0.35 m,
  matching `camera.known_size_ranging.class_sizes_m.drone`. Corrected the
  simulator-only Jetson intrinsics from the real camera's 135-by-73 degree FOV
  to the OpenGL camera's 91.4928445-by-60 degree FOV. Real-camera configuration
  remains unchanged.
- Kept the physical mesh scale fixed and moved only the controlled validation
  path from about 4 m to 2.5-2.9 m depth so the trained detector receives a
  reliably observable target footprint. The target remains moving and range
  continues to vary.
- The combined target-selection, controller, and HUD suite passes 24 tests. A
  final 20 s live sample contained 1,192 V2 snapshots and 968 commands: track
  present 95.39 percent, selection 91.78 percent, tracking commands 91.22
  percent, parallax active 100 percent, and selected known-size range on 91.22
  percent. Estimated range varied from 1.93 to 2.45 m; this remains a detector
  box/known-size estimate rather than simulator ground truth. The live HUD ran
  near 30-35 FPS with one drone, one NvSORT track, selection, range, and the
  parallax cue present.
- Rebuilt the approved temporary Jetson V2 overlay after the prior `/tmp`
  overlay disappeared, and relaunched control-free DeepStream runtime PID
  1076140. The temporary deployment remains ephemeral across cleanup/reboot.

### 2026-09-16 - Parallax point made the controller aim reference

- Corrected the control semantics after operator clarification: the projected
  parallax point is not an independent overlay. `control_sim.yaml` now selects
  `aim_mode: laser_point`, and the bounded simulator policy measures target
  error from the projected aim point rather than the optical centre.
- `ControlCmd.err_uv`, `err_rad`, and the policy observation now share the same
  parallax-relative reference. The target is expected to remain offset from
  camera centre while converging to the diamond aim marker; `parallax ALIGNED`
  means the target-to-marker error is within the configured 20 px tolerance.
- In the final live 20 s trace, 969 commands were observed: parallax active
  100 percent, tracking 98.76 percent, alignment 96.24 percent of tracking
  commands, aim-error RMS 13.38 px, and p95 19.59 px. The target was 117.62 px
  from optical centre on average while the projected aim point was 117.41 px
  from centre, directly confirming off-centre target placement and aim-point
  alignment. Known-size range varied from 1.92 to 2.39 m.

Decision: parallax is both the default UI indication and the simulator
controller's aim reference. Keep rendered target scale equal to the configured
known-size assumption; improve detector/ranging calibration separately rather
than falsifying mesh size.

### 2026-09-16 - Repository-wide V2 structural cutover

- Replaced the partial controller ingress with complete immutable target
  geometry. `ControlTargetObservation` now carries target centre, aim
  reference, pixel error, bearing error, range/provenance, parallax state, and
  alignment. Valid observations cannot omit geometry.
- Added pure `common.aiming` and made both simulator and production control
  use it. The simulator no longer applies a second parallax correction after
  assembly, and the production controller no longer reconstructs image pixels
  from bearing.
- Added `jetson.control_runtime`, a fixed-rate V2 production controller that
  consumes snapshots, CamState, and manual authority. It owns metadata sockets
  only, requires `--enable-control-publish`, and emits zero rate when authority
  is absent or stale. The serial service remains the only serial owner.
- Promoted the stable simulator controller to
  `jetson.sim_control_runtime`. It retains loopback-only endpoint enforcement,
  separate baseline gains, explicit simulation acceptance, and the rule that
  simulator results cannot tune physical hardware.
- Replaced the dual legacy/V2 metadata publisher with V2-only
  `SnapshotTransport`; removed the legacy result endpoint, JSONL projection,
  rollback switch, and compatibility adapters. Renamed the metadata adapter
  to state its production role.
- Removed the monolithic server, CPU YOLO engine, standalone legacy tracker,
  duplicate controller sidecars, mutable detection schema and serializers,
  and their obsolete tests. The RPi return-video consumer now reads V2
  snapshots for frame and latency status.
- Reworked launch ownership: `run_jetson.sh` starts passive video only;
  `run_jetson_with_gimbal.sh` explicitly starts video, V2 controller, bridge,
  and serial service as separate processes. Added a persistent
  `idcs-v2-controller.service` definition without enabling it.
- Closed two controller safety gaps exposed by the migration: post-solver MPC
  output is clamped to configured rate limits, and target-loss predictive
  output is acceleration-limited. Predictive overlay/compatibility code was
  then removed from the V2 controller pending the separately planned estimator
  rebuild.
- Updated replay fixtures to the complete V2 geometry contract and retained
  deterministic policy parity. Focused gates passed: 19 observation/simulator
  tests, 14 snapshot transport/runtime tests, 59 controller regressions before
  compatibility removal, 36 V2 controller/MPC tests after removal, 28 launch
  and selection tests, and 19 upgraded replay/planner/capture tests.
- Removed the RPi manual runtime's startup configuration handshake with the
  deleted monolithic server. RPi manual control and return-video now load the
  same explicit immutable local bundle as the V2 runtimes; the unused
  `net.config_sync` endpoint was removed.
- The post-removal full suite passed 265 tests and 12 subtests. The only warning
  is the existing PyGObject `GLib.unix_signal_add_full` deprecation.
- A first 20 s live OpenGL run exercised the new module and exposed normal
  detector/selection variation: 97.32 percent tracking and 14.19 px steady RMS
  narrowly missed the 98 percent and 14 px gates. No threshold or tuning was
  changed. A longer 30 s confirmation passed every gate with 1,787 V2
  snapshots, 1,456 commands, 98.76 percent tracking, 0.31 s acquisition,
  13.45 px steady RMS, 19.69 px steady p95, zero command drops, zero missed
  periods, and zero rate-limited commands. The verified V2 simulator runtime
  was then restored as the UI's long-running loopback controller.

Decision: the branch now has one production perception schema, one passive
video runtime, one production fixed-rate controller runtime, and one explicit
simulator baseline runtime. Legacy behavior is available through git history,
not through parallel in-tree launch or transport paths.

### 2026-09-21 - Checkout-independent DeepStream artifact resolution

- Removed the legacy checkout prefix from every tracked nvinfer profile.
  Engine, label, and custom-parser entries are now repository-relative.
- The passive runtime materializes those entries into a temporary nvinfer
  profile rooted at the active immutable config tree. The generated profile
  remains alive for the complete DeepStream process and is removed on exit.
- This prevents a persistent isolated V2 deployment from silently loading
  parser or label artifacts from the dirty legacy Jetson checkout while still
  allowing target-specific TensorRT plans to remain outside git.
- Added a focused path-materialization regression. The DeepStream-focused set
  passed 33 tests; the complete repository passed 266 tests and 12 subtests.
  The only warning remains the existing PyGObject signal API deprecation.

Decision: deployment location is no longer part of the nvinfer source
contract. Persistent V2 and legacy checkouts may coexist without sharing
runtime code artifacts.

### 2026-09-21 - Persistent Jetson V2 deployment staged; GPU gate held

- Preserved the Jetson's dirty legacy checkout and trained artifacts without
  modification. Created the isolated persistent V2 worktree
  `/home/idcs/Desktop/project/IDCS-v2-runtime` at commit `c40d998` from a
  verified 4.8 MiB thin bundle based on shared commit `5f80885`.
- Linked only ignored model/video/training artifacts into the V2 tree and
  built its YOLO26 parser locally. Source, labels, parser, and configuration
  otherwise come from the V2 worktree.
- Preflight passed file presence, parser build, all required GStreamer
  elements, and NvMultiObjectTracker dependencies. It intentionally stopped
  before launch because CUDA could not open the Jetson GPU; no DeepStream,
  controller, serial, or motor process was started.
- The failure is platform-level rather than model-specific. `nvidia-smi` and
  `trtexec` both report no CUDA device, `nvpmodel` cannot find the GPU devfreq
  table, and the kernel repeatedly reports `invalid mem
  acr_falcon2_sysmem_desc` followed by `ACR bootstrap failed`. The device is
  running L4T R39.2.0 after a September 18 boot.
- Improved preflight so GPU-runtime loss is reported separately from TensorRT
  engine incompatibility. Deployment symlinks and the native parser build
  product are now ignored explicitly. Focused validation passed 10 tests; the
  complete repository passed 268 tests and 12 subtests with only the existing
  PyGObject deprecation warning.

Decision: keep the passive video runtime stopped until CUDA itself passes.
After GPU recovery, rerun preflight, then a bounded video-only canary before
starting the persistent passive pipeline. Do not bypass the gate by swapping
engines or enabling control.

### 2026-09-21 - GPU recovery and single-owner runtime enforcement

- Audited the Jetson before reboot: no DeepStream, TensorRT, pytest, simulator,
  or controller workload remained active. The prior ACR failure was therefore
  not caused by abandoned test processes. Only normal platform daemons were
  present.
- Rebooting the Jetson recovered NVGPU initialization. `nvidia-smi` identified
  the Orin with CUDA 13.2, the GPU devfreq table returned, the ACR error did not
  recur, and the trained 736 TensorRT engine passed preflight deserialization.
  The prior failure was a transient bad boot/driver state.
- Found a separate four-day-old host streamer that continued incrementing its
  render counter but emitted no UDP packets. A five-datagram synthetic probe
  proved the LAN path and firewall were healthy. The stale process was stopped
  once and replaced by one named user service; real RTP payload type 96 then
  arrived at the Jetson.
- The controlled 30 s passive-video canary passed repository acceptance with
  zero failures or warnings: 1,709 frames, 59.11 source FPS, 60.20 steady
  pipeline FPS, 28.81 return FPS, 1,694 V2 snapshots, active detector/NvSORT,
  and zero non-monotonic publications.
- Added `tools/runtime_process_guard.py`. It identifies exact Python `-m`
  module owners from `/proc` and rejects launch when an owner already exists;
  shell text and grep commands cannot create false matches. Added named systemd
  definitions for passive DeepStream and the host simulator streamer, UI, and
  controller. Unit identity supplies the second single-instance boundary.
- Live guard verification correctly blocked duplicate streamer and UI starts
  and allowed the absent controller. Unit syntax passed on the host target.
  The full repository passed 272 tests and 12 subtests; the existing PyGObject
  deprecation remains the only warning.
- A clean 30 s simulator-controller qualification was deliberately run only
  after stopping and verifying removal of the previous controller owner. It
  failed acceptance (81.43 percent tracking, 16.96 px steady RMS, 24.73 px
  steady p95), so no persistent controller was restarted. Video, detection,
  NvSORT, and UI remain active; serial and physical control remain absent.

Decision: every runtime launch must pass both an exact-module owner audit and
named-service ownership. Never layer a test or replacement process over an
existing owner. Keep the simulator controller disabled until a fresh bounded
qualification passes all gates.

### 2026-09-21 - Persistent V2 services and restart-safe video lifecycle

- Installed and enabled the tracked `idcs-deepstream-video.service` from the
  isolated Jetson V2 worktree. A delayed audit found one runtime owner, zero
  service restarts, fresh health data, and unique ownership of UDP 5000 and
  TCP 5555/5564.
- Replaced the host's transient simulation streamer and UI jobs with the
  tracked persistent user units. Each transition stopped the named owner,
  verified that its exact module and port were absent, and only then started
  the persistent unit. The simulator-controller unit is installed but remains
  disabled and inactive because its bounded qualification failed.
- Found a lifecycle defect when the host streamer restarted independently of
  DeepStream: its frame ID returned to one, so the Jetson's still-running
  monotonic header correlator rejected every new header and withheld all V2
  snapshots. `pc.streamer` now gives each process a Unix-microsecond source-ID
  epoch plus a sequential local count, while retaining a separate sent-frame
  counter for rate reporting. A live independent streamer restart resumed V2
  metadata without restarting the Jetson; the UI observed detections, NvSORT
  tracks, and selection from the new epoch.
- Removed the UI unit's stop-propagating `Requires=` edge while retaining
  startup ordering. The UI can now survive or recover independently from an
  uplink transition instead of being left stopped by a dependency job.
- Hardened `GstReturnVideo.release()` to wait for the hardware pipeline's NULL
  transition and made report persistence non-fatal. The persistent report now
  uses the repository log directory rather than `/tmp`. A live patched UI
  cycle exited with status zero, wrote a valid report, released UDP 5002, then
  restarted as exactly one owner and resumed V2 metadata.
- The report failure exposed 12,807,454,537 bytes of closed, untracked stale
  live-test traces in `/tmp`, including an 11.18 GB cutover trace. After exact
  path and open-handle checks, only those four temporary traces were removed;
  `/tmp` usage fell from 81 percent to 2 percent. Source, model, accepted
  reports, and the preserved dirty Jetson checkout were untouched.
- Focused restart/transport/UI tests passed 24 cases. The complete repository
  passed 278 tests and 12 subtests; the existing PyGObject deprecation remains
  the only warning.

Decision: the passive V2 video deployment is persistent and restart-safe on
both machines. Keep the simulator controller, serial, and physical actuation
disabled until their independent acceptance gates pass. Continue to audit
exact module owners, named services, and required ports before every launch.

### 2026-09-21 - Stale configuration housekeeping

- Removed the unreferenced `dev_validation_file.yaml` duplicate and obsolete
  `shadow_yolo26s_best_current_736.yaml` legacy profile. Retained the frozen
  DeepStream PC-shadow canary, detector sweep, and target validation profiles
  because they still serve bounded verification workflows.
- Removed configuration with no active V2 consumer: the legacy `yolo`
  detector/search/tracker tree, orphaned `camera.argus` settings, null manual
  libcamera exposure overrides, unused logging/performance knobs, empty HDMI
  connector overrides, and superseded network aliases/placeholders.
- Moved the only still-authoritative perception data, class labels, to
  `perception.class_labels`. The DeepStream target selector now consumes that
  key directly. The return runtime now requires `net.return_ip`; a stale
  `net.pc_ip` can no longer silently become the destination.
- Kept `common/config_sync.py` despite its historical name because current
  hardware and tool modules use its YAML loading/merge helpers. Renaming that
  active shared utility is a separate refactor, not dead-config removal.
- All PC streamer, UI, and stable simulator-controller check modes accepted
  the cleaned configuration. Focused checks passed 47 cases, the explicit
  old-key regression passed with the housekeeping/runtime set, and the full
  repository passed 282 tests and 12 subtests. The existing PyGObject
  deprecation remains the only warning.

Decision: V2 runtime configuration is the sole active contract. Keep explicit
machine endpoints and verification profiles, but reject compatibility aliases
and remove settings only reachable from orphaned legacy modules. No live
service was restarted for this source-only cleanup; coordinated Jetson
deployment remains required before switching its class-label key.

### 2026-09-21 - V2 controller recovery and fail-closed actuation boundary

- Confirmed the simulator controller had no target-loss recovery: the base
  policy emitted a zero-rate hold, leaving a displaced simulated camera unable
  to deliberately return to a reacquisition pose. Added a simulator-only
  bounded home-recovery state after a configurable loss delay. It uses neither
  the hardware controller artifact nor physical endpoints, retains
  `target_ok=false`, honors safety holds, and resets immediately on reacquisition.
- A controlled 45 s live drone run exercised recovery twice and reacquired in
  4.11 s with zero command drops and zero missed periods. It did not qualify:
  target availability was 95.43 percent, steady RMS was 16.75 px, steady p95
  was 24.75 px, and rate limiting was 6.72 percent. The failed report and trace
  are preserved under `artifacts/deepstream/sim_home_recovery_20260921/`.
  This separates working recovery behavior from remaining detector/selection
  intermittency and baseline tracking error; the service remains disabled.
- Replaced the production controller runtime's legacy `ControlLoop`/MPC path
  with the frozen qualified PID plus LOS Kalman/feedforward policy. The runtime
  consumes only V2 snapshots, encoder CamState, and real manual state; emits
  explicit short-lived live `ControlIntent`; imports no serial code; records
  health, reports, and optional atomic observation/intent traces; and uses
  timestamp-derived observation/intent sequence epochs across restarts.
- Converted the source gimbal bridge from legacy `ControlCmd` ingestion to a
  fail-closed live-intent gate. It rejects shadow authority, expired/future or
  out-of-order intent/observation sequences, non-finite rates, and motion under
  a safety-hold reason. A local 100 ms watchdog emits zero rates, while every
  accepted motor-rate write uses the manual-backed timed F6 format so firmware
  also expires motion if the bridge process disappears.
- Removed automatic motor enable and encoder-zero commands from serial-service
  startup. Bridge startup calibration and encoder zeroing now default off and
  require tracked configuration plus separate command-line acknowledgements;
  live rate actuation has its own acknowledgement. Read-only bridge startup
  suppresses parameter writes, motor enable, shutdown disable, and motion.
- No controller, gimbal bridge, serial service, or physical motor process was
  started. Focused simulator, policy, replay, bridge, serial timing, emergency,
  and restart tests passed. The complete repository passed 299 tests and 12
  subtests; the existing PyGObject deprecation remains the only warning.

Decision: source implementation of the V2 intent path and fail-closed bridge
is ready for deployment-independent review, not hardware authority. Next,
deploy only the controller runtime to the isolated Jetson V2 tree with the
bridge absent, capture real selected-target/encoder/manual shadow evidence,
then perform parity. Timed-command hardware canaries follow only after those
gates; simulator recovery must separately pass its integration acceptance.

### 2026-09-21 - Isolated Jetson controller candidate and hold canary

- After explicit authorization, transferred a 35 KiB bounded Git bundle for
  commit `92a3f86` to Jetson with matching SHA-256 and created the detached
  `/home/idcs/Desktop/project/IDCS-v2-controller-candidate` worktree. The
  active DeepStream worktree remained at `db1fdfb` and was not modified or
  restarted. Both temporary bundle files were removed after the worktree was
  established; the commit and worktree retain the content.
- Candidate check mode resolved the frozen qualified controller report and
  live-intent contract without serial access. Thirty-two focused tests passed
  natively on Jetson, and the controller unit verified; only unrelated vendor
  unit warnings about obsolete syslog output were reported.
- A preflight confirmed no controller, bridge, or serial-service owner and no
  listeners on 5557/5559. A five-second controller-only canary then consumed
  298 real V2 snapshots and emitted 233 unconsumed zero-rate intents, all for
  `safety_invalid`, with zero decode errors and zero missed periods. No bridge
  or serial process existed, and both ports were free again afterward.
- The initial conventional-name probe found no `/dev/ttyUSB*` or
  `/dev/ttyACM*`, but a follow-up exact-name audit confirmed the adapter at
  `/dev/ttyCH341USB0` (CH341 `1a86:7523`, `root:dialout`, user `idcs` in
  `dialout`, and no process owner). The tracked gimbal serial path was corrected
  from `/dev/ttyUSB0`. The reachable RPi at `192.168.0.3` had no running
  manual-state process. No encoder or manual state was fabricated, and no
  hardware process was started during this Jetson check.
- Added a hardware-free same-snapshot parity comparator. It feeds each atomic
  observation to the legacy controller and qualified V2 policy, records rate
  deltas and safety-decision mismatches, and applies explicit pass/fail
  thresholds. Focused comparator/bridge coverage passed eight tests. After the
  parity additions, the complete host repository passed 300 tests and 12
  subtests; the existing PyGObject deprecation remains the only warning.

Decision: isolated deployment and fail-safe hold behavior pass. Restore the
RPi manual-state source before the real shadow/parity gate. Do not install the
controller service or enable bridge actuation yet.

### 2026-09-21 - RPi manual-state environment and isolated canary

- A broader RPi audit found `rpi/runtime_control.py` and its launcher in
  `/home/idcs/Desktop/project/repo`; no runtime process, service, listener, or
  session lock existed. The checkout has pre-existing uncommitted changes in
  `configs/system.yaml` and `rpi/runtime_control.py`, which were preserved.
- System Python had the hardware modules (`smbus` and `RPi.GPIO`) but lacked
  the declared Pydantic 2 and pyzmq dependencies. Created the isolated
  system-site-aware `/home/idcs/Desktop/project/.venv` and installed only
  Pydantic 2.13.5 and pyzmq 27.2.0; no repository or system-Python package was
  changed.
- After confirming no runtime process, I2C/GPIO owner, session lock, or
  loopback listener, ran a six-second manual-state canary connected only to
  unused `tcp://127.0.0.1:15559`. Real ADC and GPIO initialization succeeded;
  the observed state was `active=false`, `emergency=false`,
  `control_cmd_enabled=true`, with joystick sample `(123,163)`. Timeout caused
  an orderly shutdown, GPIO cleanup, and session-lock removal. No controller,
  serial bridge, or motor path was present.

Decision: the RPi manual source is runnable, but remains stopped until the
coordinated three-input shadow capture. The next deployment must first bring
the latest host commits into the isolated Jetson candidate; actuation stays
disabled.

### 2026-09-21 - Real three-input V2 shadow capture and parity

- After explicit authorization, transferred commits `2b6e9b5`, `0d2f8fe`,
  and `2ef6f38` by hash-matched incremental Git bundle into only the detached
  Jetson controller candidate. The active DeepStream worktree remained at
  `db1fdfb`, running and untouched. Candidate check mode and 35 focused tests
  passed before hardware access.
- The first fail-closed attempt exposed a stale baud setting: the service
  opened `/dev/ttyCH341USB0` at 256,000 baud and every address timed out. The
  controller never started and all owners/ports were released. Prior measured
  evidence showed the three motors were restored to UART selector `04`
  (38,400 baud), not selector `07`. Commit `a241e0c` restored 38,400 across the
  deployment config, motor parameter template, serial library/service, and
  gimbal launch/tool defaults; it also added regression tests that pin the
  device, baud, and Byte14 selector. The full host suite passed 302 tests and
  12 subtests.
- A five-second encoder-only probe at 38,400 baud produced 89 valid real
  CamState messages. The first full bridge startup then received valid status
  from addresses 1 and 2 but lost address 3 at the ZeroMQ startup boundary,
  while the serial service logged no wire failure. Commit `19b3349` replaced
  the one-shot gate with three bounded attempts that resend F1 only to missing
  axes and still fail closed if any axis never responds. Eight focused tests
  and the complete 304-test plus 12-subtest host suite passed; native Jetson
  focused tests also passed. On the next run, address 3 replied on attempt two.
- The successful 15.03-second capture consumed 893 live V2 perception
  snapshots, 724 real encoder CamState updates, and 300 real RPi manual-state
  updates. It emitted and recorded 733 intents with zero invalid messages and
  zero missed periods: two startup `safety_invalid` holds followed by 731
  `tracking` decisions. All 733 intent sequences in the trace are unique and
  strictly monotonic. The manual input remained `active=false`,
  `emergency=false`, `control_cmd_enabled=true`.
- Bridge actuation, calibration, and encoder-zero acknowledgements were all
  absent. Although the controller produced 731 nonzero shadow intents, the
  bridge forwarded none; encoder yaw and pitch spans were both exactly zero
  across the trace. The serial service was the sole CH341 owner at 38,400 baud
  and performed only configured zero-speed startup stops plus status/encoder
  queries. All Jetson and RPi processes, locks, device owners, and ports were
  clean afterward.
- Offline same-snapshot legacy/V2 parity qualified all 733 observations with
  zero invalid records, zero safety-decision mismatches, p95 rate delta
  `0.3206947 rad/s` against the `0.5 rad/s` limit, and maximum single-sample
  delta `0.9724594 rad/s`. Evidence is under ignored candidate directory
  `logs/v2_controller_shadow_19b3349_ch341_3/`.
- Two bridge `intent_out_of_order` warnings appeared at shutdown even though
  the persisted trace has no duplicate or regressed sequence. Encoder-IMU
  horizon alignment also failed initialization with I2C remote-I/O error.
  Neither affected this non-actuating capture, but both remain explicit gates
  before enabling live bridge authority.

Decision: the real three-input shadow and parity milestone passes. Do not feed
this tracking stream into an actuating bridge: it contains nonzero commands.
Next, resolve the two remaining transport/sensor findings, then use a dedicated
zero-only intent source for the timed-command/watchdog canary with calibration
and encoder zero still disabled.

### 2026-09-21 - Shutdown warning diagnosis and encoder-IMU disable

- The two `intent_out_of_order` bridge warnings from the qualified shadow run
  were deterministic shutdown duplicates, not reordered tracking traffic. The
  controller constructed one zero-rate `controller_shutdown` intent and sent
  that identical sequence three times for stop redundancy. The bridge accepted
  the first copy and correctly rejected copies two and three as duplicate
  sequences. Shutdown still sends three zero-rate intents, but each now has a
  unique consecutive sequence while retaining the same observation sequence.
- Encoder-mode IMU horizon alignment is now explicitly opt-in through
  `gimbal.encoder_imu_horizon_enabled`. It defaults to and is deployed as
  `false`, so encoder CamState performs no IMU construction, initialization,
  or I2C access. Device-sourced CamState remains available through the separate
  `gimbal.camstate_source: devices` mode.
- Added regression coverage for both shutdown sequence epochs and the tracked
  fail-closed IMU setting. Python compilation and 14 focused controller,
  deployment-config, and bridge tests passed. The complete repository passed
  306 tests and 12 subtests; the existing PyGObject signal API deprecation is
  the only warning. No controller, bridge, serial service, or hardware-facing
  process was started for this validation.

Decision: the prior bridge warnings are explained and removed at their source,
and the unavailable encoder-horizon IMU path is disabled without weakening
encoder telemetry. Hardware authority remains disabled; the next hardware gate
is still the dedicated zero-only timed-command/watchdog canary.

### 2026-09-21 - Shutdown delivery acknowledgement and readiness gate

- Completed the shutdown fix beyond sequence de-duplication. The bridge now
  marks a zero-rate intent stopped only after its serial update is successfully
  published. A successful shutdown zero therefore suppresses an unnecessary
  later watchdog zero; a dropped accepted zero leaves the watchdog armed.
- Corrected the same acknowledgement ordering in rejection and watchdog paths.
  `watchdog_stop_required` no longer clears its own state before the serial
  zero is handed off, so publication failure is retried rather than silently
  treated as stopped. Successfully delivered moving intents remain watchdog
  protected.
- Controller traces now retain all three uniquely sequenced shutdown intents.
  Bridge rejection logs include intent and observation sequence numbers, and
  successfully forwarded controller-shutdown intents are logged by sequence.
  Offline parity readers ignore the new non-observation trace record by design.
- Added regressions for retry-until-success watchdog behavior and suppression
  of an extra watchdog stop after successful zero delivery. Sixteen focused
  tests passed, followed by the complete 308-test and 12-subtest suite; the
  unrelated PyGObject deprecation remains the only warning. No controller,
  bridge, serial service, or motor-facing process was started.

Readiness decision: the controller, intent protocol, and bridge safety logic
are source- and shadow-ready for the dedicated zero-only timed-command/watchdog
hardware canary. They are not yet approved for nonzero live tracking authority:
the zero-only canary, watchdog timing evidence under encoder bus load, and
limit/fault behavior remain hardware gates. Simulator home recovery is separate
and does not qualify physical lost-target return motion.

### 2026-09-21 - Timed-command canaries and bounded live tracking trial

- The first zero-only canaries exposed a priority error rather than a serial
  bandwidth limit. At 50, 20, and 10 Hz, every all-zero intent was forwarded as
  critical traffic; the serial arbiter repeatedly preempted encoder work and
  produced retry storms, especially on pitch-B. A read-only post-canary probe
  had no warnings, isolating continuous critical zero traffic as the cause.
- Commit `f6f1967` separates motion and emergency semantics. Moving batches use
  high priority and can be coalesced/acknowledged normally; full all-zero stops
  remain critical. Repeated zero intents still advance freshness and sequence
  state but are not republished after the bridge has confirmed a stop. The
  corrected 50 Hz canary processed 103 intents with no ordering rejection, no
  continuous RS485 warning storm, and only one recoverable startup retry.
- Deterministic uncoupled motion canaries then verified reversal on yaw and
  pitch-A. At `+/-0.2 rad/s`, yaw spanned `0.1599 rad` and primary pitch spanned
  `0.1603 rad`, with no RS485 warnings, bridge rejections, or dropped updates.
  Pitch-B reported no encoder movement and produced command-health warnings.
  The `+/-0.1 rad/s` canary also exposed an integer-motor-RPM deadband: at the
  current ratio, that requested rate rounds below one RPM and becomes zero.
- An eight-second target-driven live trial used the real V2 perception stream,
  real encoder CamState, and real RPi manual state. The controller processed 477
  perception snapshots and 380 gimbal states, received 160 manual states, and
  emitted 390 intents with zero invalid messages or missed periods. Decision
  reasons were 346 `tracking`, 41 `position_limit_hold`, and three startup
  `safety_invalid`; RPi state remained manual-inactive, non-emergency, and
  command-enabled throughout.
- Authority was deliberately bounded by a temporary derived config limiting
  both axes to `0.2 rad/s`; tracked configuration was not changed. Yaw and
  pitch-A moved, while pitch-B again remained at zero counts. Pitch divergence
  crossed the configured warning threshold and reached approximately
  `1.014 rad`; the controller ultimately entered `position_limit_hold` rather
  than continuing motion. Shutdown was forwarded once and its two redundant
  copies were suppressed; one recoverable post-shutdown address-3 encoder retry
  was observed.
- This was a live transport/controller/safety and motor-path trial, not a true
  visual closed-loop qualification. The selected target came from the
  independent simulator video, so physical gimbal movement could not move the
  target in that image. The controller therefore chased an unaffected target
  until the physical pitch limit correctly held it. A valid visual-tracking
  trial requires either the physical camera coupled to the mount or a harness
  whose rendered camera pose follows the measured real encoders.
- After the trial, the controller, bridge, and serial service were stopped;
  trial ports were free, `/dev/ttyCH341USB0` had no owner, the RPi lock was
  absent, and temporary host/Jetson/RPi trial files were removed. The host and
  isolated Jetson candidate were both clean at `f6f1967`; the active DeepStream
  runtime remained untouched at its separately qualified revision.

Readiness decision: the V2 live data path, controller loop, bounded command
delivery, shutdown arbitration, yaw actuation, and pitch-A actuation have live
evidence. The controller is not approved for coupled or production tracking:
pitch-B must be repaired and encoder-verified, fine-rate command quantization
must be addressed, and visual feedback must be coupled to the commanded mount
before tracking efficacy can be measured. Do not infer hardware tuning quality
from the independent simulator camera.

### 2026-09-21 - Higher command-rate bus budget and 120/60 candidate

- Confirmed the active DeepStream inference profile uses `interval=0` on a
  60 fps source, so "twice detection rate" means a 120 Hz controller/command
  target. The currently qualified profile is 50 Hz control with a synthetic
  30 Hz vision scenario; changing its cadence without requalification would
  invalidate its recorded evidence.
- Made the offline LOS estimator qualification cadence explicit through
  `--controller-hz` and `--vision-hz`. A new 120 Hz control / 60 Hz vision run
  against the same qualified plant and PID sources passed every gate. The
  selected yaw estimator remained `q=0.001`, feedforward `0.5`, and improved
  held-out mean RMS by 19.4%; pitch selected `q=0.001`, feedforward `0.5`, and
  improved held-out mean RMS by 21.7%. Evidence is in
  `artifacts/controller_sim/los_kalman_feedforward_120hz_20260921/`.
- Reduced each firmware encoder query from 50 Hz to 10 Hz and marked encoder
  and five-second status polls low priority. Motor firmware maintains position
  internally; these reads now serve limit, state, divergence, and command-health
  supervision without consuming most of the half-duplex bus. Cached CamState
  publication remains independent of physical query cadence.
- The existing UART cannot physically carry three timed F6 frames at 120 Hz.
  Each timed command is 11 serial bytes, so three axes require 39,600 bit/s at
  8N1 before acknowledgements, encoder reads, scheduling margin, or retries;
  the deployed bus is 38,400 baud with write replies enabled. Activating the
  120 Hz report would therefore saturate the service and was intentionally not
  done. Earlier measured baud sweeps also showed that 115,200 and 256,000 had
  worse query failure rates on the present CH341/physical link; 57,600 was the
  only improved candidate but still lacks sufficient margin for 120 Hz with
  three acknowledged timed writes.
- Focused cadence/config tests passed, followed by the complete 313-test and
  12-subtest suite; the unrelated PyGObject deprecation remains the only
  warning. No motor-facing process was started for this offline gate.

Decision: the controller/estimator is qualified at the requested 120/60
cadence, and sensor polling no longer dominates the bus. Keep the active
controller on its prior qualified report until transport capacity is raised.
Achieving a real 120 Hz three-axis command rate requires either a reliable
higher-baud physical link or a verified grouped pitch command that reduces each
tick to two frames; the latter cannot be qualified while pitch-B has no encoder
response.

Jetson no-motion verification:

- Deployed commit `51edf9b` only to the isolated controller candidate; the
  active DeepStream runtime remained on its separate qualified tree and was not
  restarted. Native focused tests passed 9/9 with the existing project Python.
- After confirming no controller, bridge, serial service, trial port, or CH341
  owner, ran the serial service for seven seconds. Startup issued only its
  configured zero-speed stops and status reads. Each of addresses 1, 2, and 3
  returned 63 encoder samples, approximately 9 Hz after startup overhead and
  consistent with the 10 Hz schedule. There were zero warnings or errors.
- The service terminated on the bounded timeout. Ports 5570-5572 were free,
  `/dev/ttyCH341USB0` had no owner, and no hardware-facing process remained.
  Ignored evidence is under
  `logs/serial_poll_10hz_51edf9b_20260921/` in the isolated candidate.

### 2026-09-23 - Measured 50 Hz simulator-driven tracking

- Started the loopback-only stable-substitute simulator controller alongside
  the existing OpenGL streamer, live DeepStream detection/tracking stream, and
  return-video UI. Preflight confirmed `hardware_control_disabled=true`; no
  physical controller, gimbal bridge, serial service, or CH341 owner was
  present. The UI restored control-status and parallax overlays while retaining
  the detection/tracking display.
- The first observation exposed scheduler drift: although configured for
  50 Hz, the runtime delivered about 47 commands/s because every tick assigned
  its next deadline from the slightly late wake-up time. The 1-2 ms polling
  overhead therefore accumulated indefinitely.
- Replaced the drifting relative deadline with an absolute cadence. Late ticks
  execute once using current data, count and skip whole overdue periods, and
  preserve the original phase; they never replay a burst of stale commands.
  Added focused coverage for small repeated wake-up delays, multi-period skips,
  and invalid periods.
- After restart, the service delivered 251, 501, 751, 1001, 1251, and 1501
  commands at successive five-second reports through 30.006 seconds: measured
  cadence was effectively 50.0 Hz. Command drops remained zero, decisions were
  predominantly `tracking`, and simulated yaw/pitch pose changed continuously.
- Fourteen focused scheduler/simulator tests passed, followed by the complete
  316-test and 12-subtest suite. The unrelated PyGObject deprecation remains
  the only warning. The corrected simulation controller was left active; the
  physical motor path remained absent.

Decision: 50 Hz simulator-driven detection, tracking, camera motion, and UI are
running with measured cadence rather than config-only intent. This validates
the simulator integration surface only and does not tune or qualify the real
gimbal controller.

### 2026-09-24 - Encoder-coupled simulator tracking

- Corrected the HIL topology after initially running the stable simulated
  actuator: the host renders the moving drone, the production V2 controller
  and serial bridge actuate the unloaded physical gimbal, and measured Jetson
  `CamState` drives the rendered camera. Each motor-facing trial was bounded
  to 20 seconds and `0.2 rad/s`; startup calibration and encoder zero remained
  disabled. Exact process, port, and `/dev/ttyCH341USB0` ownership checks ran
  before every start.
- The first two attempts failed closed. One began before the RPi safety
  publisher was valid and emitted only `safety_invalid`; the next had valid
  safety but emitted `target_invalid` because the simulator applied absolute
  encoder angles. The physical mount had stopped around pan `-0.543 rad` and
  pitch `+1.014 rad`, placing the rendered target outside the new view.
- HIL now requires and applies the gimbal bridge's startup `home_pan` and
  `home_tilt`, using wrapped relative yaw and relative pitch. Missing or stale
  home-referenced state holds the last rendered pose instead of falling back
  to an absolute or simulated actuator. Focused tests cover home subtraction,
  yaw wrapping, and missing-reference refusal.
- A subsequent live run acquired the drone and moved physical yaw from
  approximately `-0.543` to `-1.372 rad`, proving that real encoder feedback
  changed the simulator view. It also exposed a separate geometry contract
  bug: the controller loaded the network config's 1920x1080 active profile
  while snapshots described the actual 1280x720 stream. Its aim reference was
  therefore `(960, ~890)` and it steered the target toward the wrong point.
- Snapshot frame dimensions are now authoritative at the shared aiming
  boundary. Camera intrinsics, principal point, deadband, parallax projection,
  and laser tolerance are scaled from configured calibration coordinates into
  the snapshot frame before target error is computed. Regression coverage
  verifies that a 1920x1080 controller profile produces a `(640, 360)` camera
  center for a 1280x720 snapshot without changing angular geometry.
- The corrected bounded run processed 1,191 snapshots, 907 gimbal states, 400
  valid manual states, and 936 intents at 50 Hz with zero invalid messages or
  missed periods. The parallax aim x coordinate was `640`, target x moved from
  about `505` to oscillate close to aim (`631`, `662`, `667` in the last valid
  samples), and the yaw encoder moved from `-1.372` to `-1.799 rad`. This is
  direct closed-loop evidence that physical yaw tracks the simulated drone.
- Pitch remained deliberately held at zero command: pitch-A reports about
  `+1.014 rad`, already outside the configured `+1.0 rad` hard limit, while
  pitch-B still reports around zero and triggers the known divergence warning.
  The controller continued yaw correction but labeled valid-target decisions
  `position_limit_hold`; no limit was widened and no faulty-axis condition was
  hidden. Detection remained intermittent (190 valid-target ticks and 744
  `target_invalid` ticks), so detector/model efficacy remains a separate
  limitation from the now-proven HIL motion coupling.

Decision: HIL yaw tracking and encoder-to-render feedback are operational with
correct snapshot geometry. Do not qualify pitch or coupled hardware until the
pitch-B encoder/control chain is repaired and the physical pitch position is
returned inside its configured range. Repeated HIL restarts re-latch the
current physical pose as simulator home and therefore must not be used as a
substitute for a deliberate physical homing procedure.

### 2026-09-24 - Encoder-anchored low-latency HIL rendering

- The encoder-coupled renderer originally applied a zero-order hold to the
  approximately 10 Hz encoder polls. Although `CamState` was published at
  50 Hz and OpenGL rendered at 60 Hz, five or six consecutive frames reused
  each measured pose. This imposed roughly 100-150 ms of visible motion delay
  and stepping, with occasional delay near 200 ms, and is not sustainable for
  simulator-video evaluation.
- The MKS protocol distinguishes cumulative encoder coordinates (`0x31`), raw
  cumulative encoder coordinates (`0x35`), and received pulse count (`0x33`).
  The pulse count does not represent the serial speed commands used here. The
  firmware also supports interval-based unsolicited parameter reports, but
  enabling three asynchronous encoder streams on the shared 38,400-baud
  half-duplex bus would contend with acknowledged multi-axis rate commands and
  could introduce frame collisions. It was therefore not selected as the
  primary latency remedy.
- Added a render-only predicted pose in `CamState`. The bridge integrates the
  exact rate accepted for serial publication after configured rate limiting
  and firmware integer-RPM quantization, then re-anchors each axis whenever a
  raw encoder count arrives. Both raw counts and the prediction age/correction
  are published for diagnostics. A zero command stops prediction immediately.
- The production controller still consumes only measured `pan`, `tilt`, and
  measured rates. Position limits, divergence handling, and all safety/fault
  decisions therefore remain encoder-authoritative. Only the host renderer
  prefers the predicted pose; missing either predicted axis falls back to the
  measured pair. This explicitly assumes no motor step loss between encoder
  anchors while retaining bounded correction and fault visibility.
- The first bounded trial exposed that prediction used the requested rate
  before applying the temporary bridge's `0.2 rad/s` cap. Prediction now uses
  the bounded and quantized camera-frame rate, matching the bytes sent to the
  firmware. Regression tests cover integration, re-anchoring, immediate stop,
  correction statistics, complete-pair fallback, and rate quantization.
- In the corrected 20.03-second hardware trial, the 50 Hz controller processed
  902 gimbal states, 1,189 snapshots, and emitted 931 intents with zero invalid
  messages or missed periods. Mean absolute encoder reconciliation was
  0.00147 rad (0.084 deg) for pan and 0.00015 rad (0.009 deg) for pitch; maxima
  were 0.01915 rad (1.10 deg) and 0.00038 rad (0.022 deg), respectively. Later
  heartbeat samples returned to zero correction. The pitch divergence fault
  remained visible and pitch remained inhibited.
- The predicted pose is published on the next 50 Hz bridge cycle and consumed
  by a 60 Hz renderer. Excluding the upstream image/detection/controller path,
  command-to-render-pose latency is therefore nominally about 10-20 ms and
  bounded near 37 ms by one publication interval plus one render interval,
  rather than waiting for the next 100 ms encoder poll. This is a cadence-based
  bound, not a synchronized end-to-end latency measurement.

Decision: retain raw encoder coordinates as control and safety authority, but
use command-integrated, encoder-anchored pose for HIL image generation. The
measured mean correction is small enough to support the no-step-loss assumption
for unloaded, properly tuned operation. Keep the correction telemetry and fall
back to measured pose when prediction is incomplete; a future excessive
correction threshold can invalidate prediction without affecting motor safety.

### 2026-09-24 - Serial execution-feedback diagnosis and rework plan

- Audited the serial documentation and prior evidence. The existing record
  already covers exclusive bus ownership, periodic polling, latest-wins F6
  coalescing, stale rejection, emergency-first arbitration, absolute receive
  deadlines, the acknowledged emergency lane, the measured 25 ms
  request-to-wire gate, 38,400-baud characterization, wire timestamps, timed
  F6 commands, and encoder-anchored render prediction. Those behaviors remain
  requirements, not candidates for removal.
- Diagnosed the `0.01915 rad` pan correction from the latest HIL trial. Of 931
  controller intents, 814 were `target_invalid`; their critical zero-speed
  transitions caused seven logged emergency-preemption events that discarded
  12 queued motion/enable commands. The bridge had already integrated some of
  those commands after successful ZeroMQ publication even though the service
  never wrote them to RS485.
- The maximum error is approximately 50 encoder counts, equivalent to about
  92 ms at the trial's quantized 2 RPM rate. It matches one missing command
  window and later returned to zero, so it is not evidence of persistent motor
  step loss. The `0.00038 rad` pitch maximum is one encoder count. Firmware
  acceleration remains a possible source of smaller transition residuals.
- Added `docs/serial_io_execution_feedback_plan.md`. It defines command
  lifecycle events with terminal `wire_sent`, `superseded`, `preempted`,
  `stale`, `write_failed`, and `wire_uncertain` outcomes; a recoverable
  per-address actuation snapshot; boot epochs and monotonic event sequences;
  intent/update correlation; timed-command expiry; bridge fallback semantics;
  phased rollout; and deterministic, saturation, and bounded-hardware gates.
- Updated the IPC and scheduling documents to state explicitly that PUB success
  and queue ACK are admission facts, not actuator-execution facts.

Decision: preserve emergency preemption and move render prediction from
publication-driven rates to wire-execution-driven rates. No controller or
safety decision may consume the prediction. Implement lifecycle accounting in
shadow first, prove complete command disposition and unchanged emergency
latency, then enable it for HIL rendering behind an explicit config mode.

### 2026-09-24 - Serial execution feedback implemented in shadow

- Added `SerialCommandEventV1` terminal outcomes for every admitted command:
  `wire_sent`, `superseded`, `preempted`, `stale`, `write_failed`,
  `wire_uncertain`, and service-shutdown `cancelled`. Events carry boot epoch,
  monotonic sequence, `update_id`/`cmd_id`, raw payload, execution/write times,
  reply confirmation, related replacement/emergency ID, and cumulative
  admission accounting. Queue coalescing and emergency discard now report the
  affected commands instead of exposing only aggregate log counters.
- Added `SerialActuationStateV1` after F6 outcomes and on a 50 ms heartbeat.
  It records the latest raw F6 state per address, timed-command expiry,
  definite versus uncertain wire outcome, active state, and lifecycle sequence.
  Event publication is non-blocking so telemetry cannot delay emergency writes;
  sequence gaps plus snapshots provide loss detection and recovery.
- Preserved the prior pre-write `last_tx_monotonic_ns` used by emergency and
  plant evidence, and added `last_tx_complete_monotonic_ns`. A failure before
  completion is `write_failed`; a reply/transaction failure after completion
  is `wire_uncertain`. Active uncertain timed writes keep render prediction
  degraded until expiry or replacement plus fresh encoder anchors.
- Reworked the encoder-anchored predictor around timestamped per-axis command
  histories. It applies delayed wire events at their actual timestamps,
  integrates across command changes, and automatically stops at the firmware
  runtime deadline. The bridge keeps bounded pending command correlation,
  ignores known dropped commands, applies pitch only from the configured
  authority motor, detects event gaps/service restarts, and falls back to the
  measured pose pair until a fresh snapshot and both encoder anchors recover.
- Added `tools/serial_execution_monitor.py` to record lifecycle/snapshot traces
  and report outcomes, sequence gaps, duplicate terminal IDs, accounting
  completeness, and same-host delivery latency. Heartbeat logs now include
  wire-feedback health/gaps and separate publication/wire correction metrics.
- Persistent control config enables lifecycle events and snapshots but retains
  `render_prediction.source: publication`; the wire predictor runs in shadow.
  Thus this milestone changes observability and offline behavior only, not the
  selected HIL rendered pose.
- Deterministic coverage includes latest-wins replacement, emergency
  preemption, definite/uncertain writes, timed expiry, duplicate/gap handling,
  snapshot recovery, and 100 alternating motion/critical-stop cycles with 200
  of 200 admitted commands reaching terminal outcomes and no preempted motion
  entering an actuation snapshot. The complete host suite passed 342 tests and
  12 subtests; the existing PyGObject deprecation remains the only warning.

Decision: offline implementation and regression qualification pass. Keep
publication rendering selected until the bounded unloaded shadow canary proves
complete command accounting, no sequence gaps, unchanged emergency latency,
and improved wire-predictor encoder reconciliation. No motor-facing process
was started for this milestone.

Performance status and pending evidence:

- No runtime performance increase is claimed yet. The selected renderer is
  still the publication-driven predictor, so deployed frame cadence and
  command-to-render behavior are intentionally unchanged by this commit.
- Relative to the original 10 Hz encoder-only zero-order hold, wire-execution
  prediction should retain the large latency reduction. Earlier serial timing
  measured approximately 6 ms from wire write through reply; adding at most
  one 20 ms bridge publication interval and one 16.7 ms 60 Hz render interval
  gives a provisional approximately 6-43 ms wire-to-render envelope instead
  of the observed 100-150 ms encoder wait. This is a component cadence bound,
  not a synchronized end-to-end measurement.
- Relative to the current optimistic publication predictor, wire execution is
  expected to be several milliseconds later because it waits for serial
  execution feedback. Its benefit is eliminating motion from commands that
  were superseded, preempted, stale, or never written. The expected reduction
  of the prior approximately 50-count correction remains unverified on
  hardware.
- The bounded shadow canary must report event-delivery p95/p99/max, admitted
  versus terminal accounting, sequence gaps, emergency request-to-wire
  latency, publication-versus-wire encoder corrections, controller cadence,
  and CPU/message overhead. Only those measurements can justify switching
  `render_prediction.source` to `wire_execution` or claiming a net performance
  improvement.

### 2026-09-24 - Bounded serial-execution hardware qualification

- Ran the Phase-4 serial-execution cases on the unloaded, uncoupled gimbal at
  38,400 baud. Preflight found no controller, bridge, serial-service, or CH341
  owner. The existing passive DeepStream runtime and host HIL display remained
  active and were not restarted. A complete temporary gimbal mapping capped
  both axes at `0.2 rad/s`, kept IMU, calibration, and encoder zero disabled,
  and set `parameter_file: null` so the test did not rewrite motor parameters.
- The zero-only canary completed 217 of 217 admitted commands with zero
  pending, failed, uncertain, stale, preempted, or superseded outcomes. It had
  no sequence gaps or duplicate terminal IDs. Event/snapshot delivery p99 was
  `2.381 ms` and maximum was `2.483 ms`; all ports and the TTY were released.
- Hardware startup exposed that the status wait consumed new
  `SerialCommandEventV1` records as though they were `SerialReplyData`, then
  aborted on their absent parsed status. The bridge now filters by reply type
  and treats an explicit zero/missing status as pending for the existing
  three-attempt retry rather than weakening the final abort gate. Host commits
  `b1ff01e` and `cf1d452`; isolated candidate commits `7c55513` and `0f61038`.
- The emergency benchmark also exposed a missing IPC client export and a
  compatibility mismatch: `SerialEmergencyTiming` carried an authoritative
  wire timestamp but omitted the `status` field the benchmark required.
  Restored a timeout-recovering acknowledged `SerialCommandClient` and emitted
  `status: complete` only after the emergency write timestamp exists. Host
  commits `acc3822` and `cc88a41`; isolated candidate commits `bdc9e24` and
  `b4ba4df`.
- The consolidated motion canary sent 279 constant/reversing and 295 rapid
  alternating intents at an absolute 50 Hz cadence with zero missed periods.
  Nonzero intents used the production-authorized `tracking` reason and zero
  transitions used `target_invalid`; the bridge policy itself was not relaxed.
  Constant yaw spanned 1,361 encoder counts (`0.52194 rad`) and alternating yaw
  spanned 155 counts (`0.05944 rad`). Pitch was commanded zero and varied by at
  most one encoder count.
- Serial accounting closed at 1,777 admitted, 1,777 terminal, and zero pending:
  1,578 `wire_sent`, 190 `superseded`, and 9 `preempted`, with no stale,
  cancelled, failed, or uncertain outcomes. The recorder joined after the first
  11 startup events but its final snapshot reconciled the complete accounting;
  it observed zero sequence gaps and zero duplicate terminals. Delivery mean
  was `1.333 ms`, p95 `2.308 ms`, p99 `2.437 ms`, and maximum `8.737 ms`.
- Twenty three-axis emergency trials produced 60 complete timing records with
  zero budget misses and zero status errors. Same-host request-to-wire latency
  was median `1.516 ms`, p95 `6.954 ms`, p99 `24.465 ms`, and maximum
  `24.675 ms`. The existing `<25 ms` gate passes, but the maximum has only
  `0.325 ms` margin and should not be described as comfortably qualified.
- The selected publication predictor and shadow wire predictor reconciled
  almost identically. Final aggregate pan mean absolute correction was
  `0.00772 rad` for publication versus `0.00770 rad` for wire execution; both
  reached `0.02021 rad` (approximately 53 encoder counts). The constant-yaw
  capture reported publication p99/max `0.02021 rad`; rapid alternation reported
  p99 `0.01760 rad` and max `0.01798 rad`. Because preempted and superseded
  commands were excluded correctly yet the correction remained, the dominant
  residual is consistent with unmodelled acceleration/encoder sampling rather
  than publication-versus-wire execution ambiguity.
- One runtime snapshot measured approximately 10.0 percent CPU and 53.7 MiB RSS
  for the serial service, 9.1 percent and 54.4 MiB for the bridge, and 7.9
  percent and 53.9 MiB for the instrumentation monitor. These are absolute
  canary footprints, not an isolated before/after telemetry-overhead result.
  Three encoder reads needed their configured first retry and recovered; no
  write failure or uncertain write occurred.
- Evidence is retained in the isolated candidate under
  `logs/serial_execution_hw_bdc9e24_20260924_final/motion/`. The complete host
  suite passed 345 tests and 12 subtests; the existing PyGObject deprecation is
  the only warning. Automatic cleanup left no motor-facing process, TTY owner,
  or trial-port listener.

Decision: the wire-execution lifecycle path passes bounded hardware correctness
and observability qualification, including real preemption and emergency
timing. It does not demonstrate a runtime speed increase or lower encoder
reconciliation than the publication predictor. Keep
`render_prediction.source: publication`. Before selection can change, measure
wire-event-to-`CamState` p99 directly, isolate telemetry CPU overhead against a
matched baseline, classify the approximately 53-count acceleration/sampling
residual, repeat the emergency gate with more margin, and run the separate full
moving-target controller/HIL case. The known pitch-B divergence remains outside
this yaw-only canary and still blocks production coupled tracking.

### 2026-09-24 - Encoder-coupled simulated-camera tracking rerun

- Reconfirmed the intended HIL topology after the user clarified that an
  all-software camera was not the requested validation. The host rendered the
  moving native drone from measured gimbal `CamState`; Jetson small_736 and
  NvSORT produced V2 perception; the production 50 Hz controller drove the
  unloaded real yaw motor; and the resulting encoder motion moved the rendered
  view. The stable-substitute simulator controller was not used.
- Retained the persistent HIL streamer and UI and the active DeepStream runtime.
  The bounded motor-facing side used candidate `ced485d`, real RPi manual/safety
  state, a `0.2 rad/s` yaw cap, an explicit zero pitch transport cap, IMU
  disabled, and both startup calibration and encoder zero disabled. The pitch
  lock is required because pitch-A still reports about `+1.014 rad` while
  pitch-B reports approximately zero.
- The first orchestration attempt never opened the serial device because stdin
  was accidentally disabled for the RPi startup script. A second attempt opened
  only the serial service and failed before controller startup because the
  configuration loader replaces whole top-level mappings and the minimal
  `gimbal` overlay omitted required motor addresses. It executed only configured
  stop/status/encoder traffic, closed 232 of 232 admitted serial commands, and
  released the TTY. The accepted rerun used a full current control snapshot,
  SHA-256 `f94b10ab6d4f2fffc8e90c378114bdc580a6992364a6b8c595f5e3638f99d544`,
  differing from tracked `configs/control.yaml` only at the yaw and pitch rate
  caps.
- The final 30.03-second run processed 1,786 V2 snapshots, 1,354 real gimbal
  states, 599 valid RPi states, and 1,407 intents with zero invalid messages and
  zero missed periods. It produced 242 target-valid `position_limit_hold`
  decisions, 1,160 `target_invalid` holds, and five startup `safety_invalid`
  holds. Pitch intents remained zero, pitch varied by one encoder count
  (`0.0003835 rad`), and yaw moved through `0.43258 rad`, from `-2.52301` to
  `-2.92338 rad`.
- Horizontal visual error closed across the real encoder-coupled motion. Mean
  absolute x error fell from `95.23 px` over the first 25 valid samples to
  `8.95 px` over the last 25; the final valid sample was `9.57 px` from the
  `x=640` aim coordinate and the minimum was `0.217 px`. The detector produced
  41 valid bursts and reacquired after 40 intervening losses, including a
  longest loss of 261 controller ticks (about 5.22 seconds).
- Drone retention remains inadequate for full tracking qualification: only
  242 of 1,407 controller ticks (17.20 percent) carried a valid target. The
  locked pitch axis also leaves the parallax vertical aim unresolved. The
  detected drone stayed at approximately `y=258-274 px`, while the range-based
  parallax aim was `y=540-576 px`; vertical error averaged about `300 px`.
  Therefore the scalar two-axis error is not an acceptance metric for this
  deliberately yaw-only safety case.
- Serial accounting closed at 2,002 admitted and 2,002 terminal commands:
  1,863 `wire_sent`, 114 `superseded`, and 25 `preempted`, with zero stale,
  cancelled, failed, or uncertain outcomes. Three first-attempt yaw encoder
  reads timed out and recovered on retry. Wire feedback remained healthy with
  zero event gaps. Final DeepStream health remained about `59.56 FPS` inference
  and `30 FPS` return; the HIL streamer and operational UI remained active.
- Raw and derived evidence is retained in the isolated candidate under
  `logs/hil_tracking_ced485d_20260924_run7/`, including the controller report,
  trace, bridge/serial logs, and `tracking-analysis.json`. Automatic cleanup
  left no controller, bridge, serial service, RPi runtime, session lock, trial
  port listener, or CH341 owner.

Decision: the required hardware-motion-to-simulated-camera feedback topology
and closed-loop yaw correction are validated. Full hardware visual tracking is
not yet qualified. Repair and encoder-verify pitch-B before enabling coupled
parallax tracking, and separately improve or replace the 3D drone validation
target so detector retention no longer dominates the test. A controlled
high-retention synthetic target may qualify the feedback topology, but it must
not be presented as evidence of real-world drone detector accuracy.

### 2026-09-24 - Pitch-B firmware encoder recalibration

- Distinguished the bridge's IMU-based startup positioning routine from the
  motor firmware's encoder self-calibration. With IMU disabled, only the MKS
  manual's address-specific `0x80` encoder calibration was relevant. The manual
  requires an unloaded motor and states that the driver resets automatically
  after calibration; the user had confirmed that the motors were uncoupled and
  authorized the operation.
- Pre-calibration read-only evidence showed all three motors stopped and
  reachable. Pitch-B at address 3 reported the calibrated firmware flag
  (`0x40` data `11 01 00 08`) but its cumulative encoder remained exactly zero,
  while pitch-A reported `-2643` counts and yaw reported `7622` counts. No
  controller, bridge, serial service, trial-port listener, or CH341 owner was
  present.
- Sent `0x80 0x00` only to address 3. The immediate reply was status `0`,
  calibration in progress. The driver stopped answering during the operation,
  then restarted after approximately nine seconds with calibration flag `1`
  and stopped status. Its first encoder transaction immediately after restart
  timed out twice; after a five-second settling interval, three consecutive
  encoder reads on every axis succeeded.
- A bounded pitch-B-only motion check enabled address 3, commanded `+0.2 rad/s`
  for two seconds with an automatic stop, then applied the symmetric reverse
  pulse. The encoder moved from `0` to `-555` counts (`-0.21284 rad`) and
  returned to `-6` counts (`-0.00230 rad`). All three reads at each endpoint
  agreed, and the motor reported stopped after each pulse. This restores direct
  evidence that pitch-B motion and encoder feedback are functional.
- Final checks retained calibration flag `1`, address 3, 38,400-baud selector,
  response mode, and the expected operating parameters. Cleanup left the motor
  stopped with no hardware-facing process, TTY owner, or trial-port listener.
  Evidence is retained under
  `logs/pitch_b_encoder_calibration_c2991c1_20260924/` in the isolated candidate.

Decision: the prior pitch-B symptom is no longer a demonstrated dead motor or
dead encoder. The calibration cycle or its driver reset restored measurable
bidirectional encoder response. Do not yet remove the coupled-pitch gate: the
uncoupled axes still have different position origins (pitch-A approximately
`-2643` counts versus pitch-B `-6`), so bridge divergence remains expected until
both axes are mechanically aligned and given a deliberate common reference.

### 2026-09-24 - Post-calibration two-axis encoder-coupled tracking

- Ran two bounded 45-second moving-drone HIL trials after pitch-B
  recalibration. The host OpenGL renderer consumed measured encoder `CamState`,
  Jetson DeepStream/small_736/NvSORT supplied V2 target snapshots, the 50 Hz V2
  controller emitted live intents, and the real unloaded yaw and mirrored pitch
  motors moved the simulated camera. The software-only simulator controller was
  not used. IMU, startup calibration, and startup encoder zero remained
  disabled; transport rates were capped at `0.2 rad/s` on both axes.
- The first full-pitch run used a temporary pitch-A maximum of `1.3 rad`. It
  retained a target for 2,118 of 2,187 ticks (96.84 percent), but reached the
  ceiling and spent 2,050 ticks in `position_limit_hold`. Mean norm error over
  the first and last 50 valid samples was `274.11 px` and `90.63 px`; after the
  five-second warm-up it remained `110.64 px` RMS and `133.33 px` p95. This run
  confirms paired pitch response but is not a tracking-performance result
  because the artificial position ceiling dominated it.
- Repeated with a bounded `1.95 rad` temporary pitch-A maximum, config SHA-256
  `d12866f5e7cb129131316e1544a14192b8c047bada5116f8d5c9106c48e8b2c8`.
  The accepted run processed 2,680 snapshots, 2,145 measured gimbal states, 901
  RPi states, and 2,177 intents in 45.03 seconds with zero invalid messages and
  zero missed periods. It produced 2,025 `tracking`, 151 `target_invalid`, and
  one startup `safety_invalid` decision, with no position-limit hold.
- Target retention was 2,025 of 2,177 ticks (93.02 percent). There were nine
  loss events, the longest ten ticks (about 0.2 seconds), and one dominant track
  ID for 2,021 of the 2,025 valid samples. Detection began at tick 104. The
  persistent DeepStream runtime remained healthy at `59.557 FPS` inference and
  `30 FPS` return after the run.
- Mean norm error fell from `269.06 px` over the first 50 valid samples to
  `19.41 px` over the last 50; the final sample was `12.83 px`. After five
  seconds, norm error was `22.20 px` mean, `24.80 px` RMS, and `42.01 px` p95.
  During the last ten seconds, 495 of 500 ticks were target-valid; norm error
  was `21.49 px` mean, `23.74 px` RMS, and `39.43 px` p95. All valid samples in
  that interval were within 50 px, 63.23 percent were within 25 px, and 30.91
  percent were within 15 px. Only 24.24 percent were classified on-target.
- Last-ten-second signed bias was small (`-1.79 px` horizontal and `-1.71 px`
  vertical), while horizontal and vertical RMS were `20.55 px` and `11.87 px`.
  The residual is therefore moving-target lag/oscillation rather than a fixed
  pointing bias. Target source age averaged `7.47 ms` (p95 `15.24 ms`) and
  measured-gimbal sample age averaged `16.81 ms` (p95 `19.14 ms`). The bridge's
  encoder-to-render correction averaged about `10.5 mrad` yaw and `5.2 mrad`
  pitch, reaching `22.9 mrad` and `17.6 mrad` respectively.
- Controller limit telemetry was active for much of the trial: acceleration
  limited 1,543 of 2,177 ticks, yaw rate limited 1,124, and pitch rate limited
  133. The controller requested its `0.5 rad/s` maximum on 24.21 percent of yaw
  ticks and 3.40 percent of pitch ticks, while the safety transport cap clipped
  both axes to `0.2 rad/s`. This controlled mismatch materially limits the
  result and is consistent with the dynamic horizontal residual; it must not be
  treated as a real-hardware tuning result.
- Pitch-A moved from `-3472` counts while pitch-B began at `+799`. Near the end,
  they reported approximately `-4740` and `+2022`, changes of `-1268` and
  `+1223` counts whose magnitudes agree within about 3.6 percent. However,
  configured camera signs are `(-1, -1)`: pitch-A camera angle increased while
  pitch-B camera angle decreased. The reported divergence consequently grew
  from `1.638 rad` to about `2.59 rad`. A different origin could explain a
  constant offset, but not this motion-dependent growth; pitch-B's secondary
  encoder-to-camera sign is wrong for the mirrored pair. Pitch-A remained the
  authoritative `CamState`, so this defect did not invert the rendered camera
  or the controller feedback in this trial.
- Serial execution closed 7,702 of 7,702 admitted commands: 6,552 `wire_sent`,
  1,141 `superseded`, and nine shutdown/preemption outcomes, with zero stale,
  cancelled, failed, or uncertain writes. Cleanup released the controller,
  bridge, serial service, RPi runtime, trial ports, and CH341 device. Evidence
  is retained under
  `logs/hil_tracking_d6732f1_postcal_20260924_run2/`, including the expanded
  `tracking-analysis.json`; the ceiling-limited comparison is under
  `logs/hil_tracking_d6732f1_postcal_20260924/`.

Decision: two-axis hardware-motion-to-sim-camera closed-loop operation is now
demonstrated, and pitch-B responds consistently in magnitude after calibration.
Tracking performance is improved but not qualified against the existing
simulation acceptance levels: post-warm-up RMS `24.80 px`, p95 `42.01 px`, and
93.02 percent retention miss the nominal `14 px`, `22 px`, and 98 percent
targets. Before another coupled trial, correct the pitch-B secondary camera
sign and establish a common mechanical/reference zero. For system evaluation,
retain this HIL mode but use stable safety caps and report their clipping; do
not tune the real controller against simulated camera dynamics.

### 2026-09-24 - Estimator-driven feedforward planning baseline

- Audited the active V2 estimator path rather than assuming feedforward was
  absent. The controller already loads the qualified two-state absolute-LOS
  Kalman filters and `0.5` yaw/pitch feedforward gains from the 50 Hz control /
  30 Hz vision artifact. A separate 120/60 artifact remains undeployable on the
  current 38,400-baud three-axis acknowledged bus.
- Found that the estimator's claimed target sample time currently comes from
  controller-local snapshot age. It does not include the preceding stream,
  decode, inference, or tracker latency even though `PerceptionFrameV2` already
  carries Jetson-local frame-receive and inference-observation timestamps. This
  timing-contract defect must be corrected before further estimator tuning.
- Found that every target-invalid hold immediately resets estimator state. That
  is safe for commands but discards rate knowledge across the short detector
  losses observed in the HIL trace. Planned an explicitly bounded coast-memory
  state that still commands immediate zero and is reusable only for the same
  track within covariance, timestamp, and loss-duration gates.
- The recent HIL trace cannot prove feedforward efficacy because it has no
  estimator-off counterfactual and its `0.5 rad/s` controller output was clipped
  by the `0.2 rad/s` test bridge. Identical-input replay will be used only to
  attribute command terms and verify safety; causal performance comparison
  requires truth-based closed-loop holdouts with identical caps.
- Added `docs/estimator_feedforward_plan.md`. The staged plan begins with
  timing and diagnostics that leave intents byte-for-byte unchanged, followed
  by replay ablation, 50/60 hardware-plant qualification, bounded loss coasting,
  common-horizon delay compensation, controlled renderer-truth validation,
  shadow mode, and finally matched encoder-coupled HIL. Simulation-selected
  controller parameters remain separate from real-hardware parameters.

Decision: do not tune feedforward from the current simulated-camera tracking
error. First make measurement/effect timing observable and correct, expose
estimator uncertainty and command contributions, and establish an estimator-
off baseline. Keep the active qualified report and command authority unchanged
until those non-actuating gates pass.

### 2026-09-24 - Estimator timing diagnostics and replay ablation

- Extended `ControlObservation` with optional frame receive and inference-
  observation timestamps plus their clock domains. The assembler copies the
  existing `PerceptionFrameV2` values without interpreting remote source time.
  Old observations remain valid. The command policy still uses its prior
  timing calculation, so this slice does not alter active intents.
- Extended the LOS Kalman estimate with predicted angle/rate covariance, last
  innovation, innovation variance, NIS, acceptance, rejection streak, and an
  explicit reinitialization flag. These values are observational only.
- Added immutable `ControlDiagnostics v1`. It associates observation and
  intent sequence IDs with local timing, raw/estimated errors, estimator state
  and uncertainty, update disposition, feedback/damping/feedforward terms,
  pre/post-limit rates, and final rates. `jetson.control_runtime` writes a
  separate JSONL stream only with the explicit `--diagnostics-trace` option;
  default runtime output and authority are unchanged.
- Added an offline-only raw-PD baseline with gimbal-rate damping and
  `tools/analyze_estimator_feedforward_trace.py`. The tool replays raw PD,
  estimated error without feedforward, and the qualified estimator/feedforward
  policy from identical observations. It labels its report non-causal and
  checks exact parity against recorded intents. The new raw baseline flag
  defaults off and is not loaded by the qualified runtime profile.
- Replayed the retained 45-second encoder-coupled HIL trace. The instrumented
  qualified policy reproduced all 2,177 of 2,177 recorded intents exactly,
  including reasons and limits. This proves the diagnostics did not perturb
  the numeric command path.
- The replay exposed frequent estimator discontinuity. Across 2,017 new-frame
  updates, yaw rejected 416 measurements and reinitialized 325 times; pitch
  rejected 216 and reinitialized 156 times. The current qualified feedforward
  component reached `0.692 rad/s` yaw and `0.569 rad/s` pitch before final
  limiting, with RMS `0.198` and `0.104 rad/s` respectively. The qualified
  policy was acceleration-limited on 1,543 ticks; identical estimated-error
  replay with feedforward disabled was limited on 1,307 ticks.
- These are command-attribution results, not evidence that disabling
  feedforward improves tracking: the observation sequence was generated by the
  qualified active policy. The report is retained beside the HIL evidence as
  `estimator-feedforward-ablation.json`.
- Focused tests pass 31/31 locally and 31/31 natively on the isolated Jetson
  candidate. The complete host suite passes 350 tests and 12 subtests; the
  existing PyGObject deprecation is the only warning. A first full-suite launch
  from the wrong working directory produced repository-relative path failures;
  rerunning from the repository root passed completely. No motor-facing process
  or serial device was opened.

Decision: keep the active qualified estimator/feedforward profile unchanged
for now, but do not treat it as ready for further gain tuning. Its high
rejection/reinitialization rate and substantial pre-limit feedforward require a
truth-based timing capture and controlled 50/60 qualification. The next step is
a bounded diagnostics-enabled capture with unchanged command authority, then
offline candidate selection against exact LOS truth and held-out trajectories.

### 2026-09-24 - Live feedforward HUD attribution

- Added a separate read-only controller diagnostics PUB endpoint at
  `net.zmq_control_diagnostics` (`tcp://192.168.0.5:5565`). The qualified
  controller publishes its existing immutable `ControlDiagnostics v1` once per
  decision tick. This transport has no command authority and does not alter the
  `ControlIntent` path.
- The host V2 UI subscribes to that endpoint independently of the legacy
  `ControlCmd`/MPC overlay path. The operational HUD now always reserves a
  compact feedforward indicator showing exact signed yaw/pitch contributions
  in rad/s and centered direction bars.
- Indicator states are explicit: cyan `ACTIVE`, amber `LIMITED`, `INHIBITED`,
  or `STALE`, and white `UNAVAILABLE`. A 250 ms receive-age threshold marks
  stale telemetry while preserving its last numeric value for diagnosis.
- The old MPC cost-term display remains opt-in and excluded from the
  operational HUD. The feedforward indicator attributes the active estimator
  contribution only; it does not imply that feedforward has command authority
  or has passed efficacy qualification.
- Focused host validation passes 10/10 tests, including active, stale, and
  unavailable rendering plus runtime endpoint validation. The complete host
  suite passes 351 tests and 12 subtests with only the known PyGObject
  deprecation warning. The isolated Jetson candidate passes its 4/4 runtime
  tests and resolves the diagnostics publisher to `tcp://0.0.0.0:5565`.
- After confirming the existing host UI and streamer process inventory, the
  single old UI was stopped and relaunched with the same HIL arguments. Its
  live 30 FPS log reports `feedforward_indicator` in the HUD inventory. The
  streamer and persistent DeepStream process were left untouched. With no
  controller publisher running, the indicator intentionally reports
  `UNAVAILABLE`; no controller, bridge, or serial-facing process was started.

Decision: expose feedforward contribution continuously in the V2 HUD through
a non-authoritative telemetry channel. Keep the currently qualified gains and
control authority unchanged; use this indicator as attribution evidence, not
as proof of tracking improvement.

### 2026-09-24 - Isolated estimator simulation environments

- Extended the loopback-only simulator controller to support an explicit
  `qualified_estimator_feedforward` controller-under-test. Loading the saved
  qualified report requires `sim.controller.study_only: true`; the runtime
  still rejects non-loopback command/state/diagnostics endpoints, imports no
  serial driver, and refuses `net.zmq_control`.
- Added `control_sim_estimator_ideal.yaml` and
  `control_sim_estimator_graybox.yaml`. Both run the real V2 observation and
  `ShadowRatePolicy` estimator path. The ideal profile isolates estimator and
  video-pipeline effects; the gray-box profile uses the independently
  validated asymmetric hardware-derived plant as a robustness gate. The
  estimator-free stable baseline remains unchanged.
- Added a loopback `ControlDiagnostics` publisher on port 5573. The simulator
  UI service subscribes to it, so the operational feedforward indicator now
  displays live estimator contribution in simulation. Per-tick traces include
  observation, intent, command, CamState, and diagnostics; final reports add
  estimator update/rejection/reinitialization counters and feedforward RMS and
  maximum magnitude.
- Added guarded `ideal`/`graybox` systemd user-service templates for streamer,
  controller, and UI. The process guard prevents layering and study services
  do not restart after failure. Usage and safety boundaries are documented in
  `docs/estimator_simulation_environment.md`.
- Both profile preflights resolved correctly, focused tests passed 21/21, and
  the complete host suite passed 354 tests plus 12 subtests with only the known
  PyGObject warning. Static unit verification reported only an unrelated
  existing `spice-vdagent` unit warning.
- A 142.6 s ideal capture produced 7,127 command and diagnostics messages with
  zero drops. Yaw accepted 386 updates with no rejection/reinitialization;
  pitch accepted 377, rejected 30, and reinitialized 15. Feedforward RMS was
  0.0345 rad/s yaw and 0.0354 rad/s pitch. The profile failed the existing
  baseline acceptance gate on tracking fraction (0.898) and rate-limit
  fraction (0.073).
- A 77.2 s gray-box capture produced 3,858 command and diagnostics messages
  with zero drops. Yaw accepted 362 updates, rejected 2, and reinitialized 1;
  pitch accepted 354, rejected 19, and reinitialized 9. Feedforward RMS was
  0.0347 rad/s yaw and 0.0349 rad/s pitch. It also failed tracking fraction
  (0.906) and rate-limit fraction (0.075). Both failures are retained as study
  evidence, not converted into hardware tuning changes.
- Identical-input ablation reports were retained as
  `logs/estimator-sim-{ideal,graybox}-ablation.json`. Exact replay fractions
  were 0.964 and 0.968 because simulator-only target-loss recovery modifies
  the recorded post-policy intent; estimator tracking decisions remain
  replayable, and the reports make no causal performance claim.

Decision: the separate estimator laboratory and both full-stack motion profiles
are implemented. Neither current estimator profile passes the existing
simulation integration gate, so they remain study-only. Investigate detection
loss/reacquisition and limiter interaction from the paired traces before any
estimator or controller parameter proposal; do not tune physical hardware from
these results.

### 2026-09-24 - Simulator observation isolation and passing controller studies

- Root-caused the failed estimator captures instead of changing controller
  gains. The first controlled person scene produced stable tracks but no
  selection because production `swarm_eval.excluded_target_classes` excludes
  `person`. Replacing it with an eligible CPU-rendered drone exposed a second,
  independent limitation: a 20 s stationary-camera passive capture delivered
  tracks on only 492/1,179 frames (41.7%) and selection on 311/1,179 (26.4%).
  Controller motion therefore was not the source of the target losses.
- Found a camera-contract mismatch in the same capture. The simulator derived
  a 91.49 degree horizontal FOV from a 60 degree vertical FOV, while Jetson
  known-size ranging used the configured physical 135 x 73 degree camera. A
  target actually 3.5-3.9 m away was reported at a mean 1.86 m. Added explicit
  independent horizontal and vertical FOV support to `SimCamera`, CPU and
  OpenGL projection, planner projection/spawn, and simulator control. The
  stable simulator now uses the same 135 x 73 degree contract as Jetson.
- Moved the correct-width 0.35 m drone to a 0.9-1.1 m controlled path to test
  the trained model at a larger image scale. Range then measured 0.864-1.160 m
  with mean 1.040 m, confirming the intrinsics fix, but track coverage fell to
  177/1,174 frames (15.1%) with four IDs. This is retained as detector/model
  evidence; it is not hidden or presented as controller failure.
- Verified the persistent Jetson process rather than assuming another model
  regression. PID 12488 runs from `IDCS-v2-runtime`; its active profile names
  `nvinfer_yolo26s_736_drone_person_smoke.txt`, and `small_736.engine` resolves
  to `yolo26s_dataset2_e100_736_raw.engine` with SHA-256
  `c9ea7dfcbf8b05002a584cc3b02dd751f922b2a42ecc2e6ef8309e8a12fa9f73`.
  The remaining scale/view sensitivity is consistent with the already
  documented nvinfer video-preprocessing boundary, not accidental use of the
  obsolete engine.
- Added a loopback-only deterministic perception channel on port 5574. The
  streamer projects the exact rendered billboard and camera pose into a strict
  selected `PerceptionSnapshotV2`; the estimator controller consumes that
  channel. The identical rendered frames still traverse the persistent
  DeepStream detector on production port 5564, so detector efficacy can be
  evaluated separately without corrupting controller/estimator experiments.
- The corrected ideal-plant run passed every unchanged integration gate over
  77.81 s: 3,889 commands, 99.974% tracking, 0.72 s acquisition, 0.795 px
  steady RMS, 1.470 px steady p95, 0.849% rate-limited, 0 command/diagnostics
  drops, and no estimator rejection or reinitialization on either axis.
- The qualified gray-box robustness run also passed every unchanged gate over
  101.19 s: 5,058 commands, 99.980% tracking, 0.78 s acquisition, 0.693 px
  steady RMS, 1.406 px steady p95, 0.712% rate-limited, 0 command/diagnostics
  drops, and no estimator rejection or reinitialization on either axis.
- Focused projection, CPU/OpenGL renderer, streamer, and simulator-controller
  validation passes 57 tests and 3 subtests. All experiment services were
  stopped after report flush; the persistent Jetson DeepStream process was not
  restarted or modified.

Decision: the simulator controller/estimator path is now structurally isolated
from detector instability and passes both ideal and independently qualified
gray-box integration. Treat these as simulator-integration results only. Keep
physical controller tuning unchanged and treat the low detector/tracker
coverage as a separate model/perception work item requiring its own controlled
sweep and acceptance evidence.

### 2026-09-24 - Causal hardware feedforward ABBA validation

- Added explicit controller-runtime study overrides for a perception endpoint,
  feedforward scale, and lower yaw/pitch rate caps. They require a separate
  hardware-validation acknowledgement, are written into every report/trace,
  cannot increase the qualified rate limits, and never mutate the qualified
  controller artifact. Added a read-only simulator-truth exception limited to
  the configured PC LAN bind address; simulated actuation endpoints remain
  loopback-only.
- Used an exact CPU-rendered 0.35 m drone on a deterministic horizontal path.
  Real encoder-backed `CamState` drove the simulated camera, exact rendered
  `PerceptionSnapshotV2` drove the controller, and the same video continued to
  traverse the persistent DeepStream process independently. This isolated the
  estimator/controller result from detector variation without simulating the
  gimbal motion.
- The unloaded hardware ran at 50 Hz with matched `0.2 rad/s` yaw and
  `0.01 rad/s` pitch caps, camera-center aim, IMU/calibration/encoder-zero and
  parameter writes disabled, and the real Pi safety state present. The known
  pitch-B origin/sign discrepancy was kept out of the efficacy measurement;
  pitch remained fixed while yaw followed the moving target.
- The first sequential launch expired its prerequisite timeouts during SSH
  setup. Its controller received no target, encoder, or Pi state and emitted
  933 `safety_invalid` zero intents; it is not a result. Replaced it with one
  coordinated, timeout-bounded launcher and repeated an off/on/on/off ABBA
  sequence. Every accepted 20 s run had about 962 tracking decisions, 828-834
  exact target snapshots, 948-950 encoder states, 399 Pi states, zero invalid
  messages, and zero missed periods.
- In the first pair, enabling the qualified `0.5` feedforward gains worsened
  post-warm-up yaw RMS from 17.91 to 23.96 px (+33.8%) and p95 from 37.59 to
  44.69 px (+18.9%). Yaw rate limiting increased from 63.7% to 77.1%; command
  total variation was unchanged. The reverse-order pair confirmed no benefit:
  RMS changed from 18.64 to 19.74 px (+5.9%), p95 from 36.70 to 38.68 px
  (+5.4%), and acceleration limiting rose by 6.5 percentage points.
- Across both pairs, mean yaw RMS was 18.28 px with feedforward disabled and
  21.85 px enabled (+19.5%); mean p95 was 37.15 versus 41.69 px (+12.2%). The
  enabled estimator produced 0.152-0.174 rad/s feedforward RMS and peaks up to
  0.586 rad/s before the 0.2 rad/s final cap. It also continued to reject and
  reinitialize frequently, consistent with the earlier timing/replay audit.
- Serial accounting closed completely. Confirmation runs each admitted about
  3,760 commands; three queued motion writes were intentionally preempted by
  shutdown, with zero failed, uncertain, stale, or pending writes. Final audit
  found no controller, bridge, serial service, Pi runtime, streamer, or CH341
  owner. The persistent DeepStream service was neither restarted nor modified.
  Evidence is retained in the isolated candidate under
  `logs/feedforward_hil_20260924/`.

Decision: hardware execution of estimator feedforward is verified, but the
current qualified gains are rejected for live tracking efficacy. Keep the
feedforward path disabled for hardware use until source-time mapping is fixed,
estimator rejection/reinitialization is reduced, and a new candidate passes
the same causal ABBA gate without increasing limiter dependence. Do not tune
the real controller against simulated camera dynamics.

### 2026-09-26 - Source-time and camera-pose alignment hardware check

- Closed the known cross-host timing gap with a four-timestamp PC/Jetson
  monotonic-clock exchange. The PC streamer now serves the exchange on the
  configured `net.zmq_source_clock_sync` endpoint, and the Jetson controller
  maps each host frame timestamp into its local monotonic clock. It accepts a
  mapping only within five seconds of calibration and at no more than 5 ms
  uncertainty; unavailable/uncertain mappings suppress estimated velocity
  while retaining bounded position feedback. The controller records mapping
  source, frame age, uncertainty, and camera-pose age in its diagnostics.
- The PC capture thread now stamps frames when retrieved, before its latest-
  only queue; the simulator stamps the frame at the render boundary. This
  removes queue/encode/decode/inference time from the source timestamp.
  Sensor exposure-to-retrieval latency remains unmeasured for a real webcam.
- The bridge publishes its Jetson-local camera-state sample time. The
  controller retains a short history of the same render pose consumed by the
  simulated camera and interpolates it at the mapped frame time when forming
  absolute target LOS. This avoids combining an old image bearing with the
  newest gimbal angle. Recalibrating clock offset does not cause a duplicate
  Kalman update of the same source frame.
- Read-only LAN probes measured clock uncertainty around 1.6-2.5 ms. After
  normal streamer integration, the delayed exact-truth fixture had mean
  mapped frame age about 91 ms; the persistent DeepStream snapshot path had
  mean age about 124 ms in the same probe. The simulator fixture therefore
  adds a bounded 60 ms publication delay and keeps detector variation out of
  the controller comparison while real encoder motion drives the camera.
- The first corrected hardware attempt had only one Pi safety update because
  the prior Pi trial process had ignored soft timeout and held its session
  lock. The controller issued 927 `safety_invalid` holds; that run is excluded.
  The exact orphaned Pi process was stopped, and the coordinator now waits for
  Pi shutdown with a hard kill fallback. Subsequent trials had about 400 Pi
  safety states each, zero invalid controller messages, and zero missed ticks.
- In the initial old/mapped/mapped/old sequence, mapped timing lowered yaw RMS
  by 15.2% and 8.7%, and p95 by 22.3% and 14.5% respectively. It increased
  command total variation by 25-28% and acceleration limiting by about 6-8
  percentage points. After a same-frame offset-refresh fix, a further mapped
  run lowered RMS by 9.0% versus the old-timing baseline but remained below
  the 10% acceptance threshold.
- The final code, including host capture timestamping, was tested with matched
  20-second mapped feedforward-off/on trials. Post-warm-up yaw RMS was 31.70
  px off and 30.31 px on (4.4% improvement); p95 was 59.38 and 56.30 px
  (5.2% improvement). Feedforward-on increased yaw rate-limit fraction from
  81.4% to 83.7% and command variation by 5.1%. It still rejected 181 yaw
  updates and reinitialized 73 times. The on trial used mapped source time on
  all 965 tracking ticks; mapped frame age averaged 119.0 ms and clock
  uncertainty averaged 1.75 ms. Historical camera-pose interpolation was on
  average 5.79 ms from a recorded pose.
- Every accepted trial completed serial accounting. The final off/on runs
  closed 3,747/3,747 and 3,753/3,753 commands respectively, with zero failed
  or uncertain writes. The unloaded motors remained yaw-focused with the same
  `0.2 rad/s` yaw and `0.01 rad/s` pitch caps, no IMU/calibration/encoder-zero
  or parameter writes. The persistent DeepStream service remained active.
  Evidence and per-run diagnostics are under the isolated Jetson candidate's
  `logs/feedforward_clock_hil_20260926/`.
- The complete host suite passes 370 tests and 12 subtests; the focused
  Jetson suite passes 22 tests. Only the pre-existing PyGObject deprecation
  warning remains.

Decision: frame-time mapping and matching camera pose now work on the LAN and
were exercised with real gimbal motion. The current `0.5` feedforward gains
still fail the 10% RMS improvement gate and depend heavily on rate limiting;
they remain unqualified for live hardware tracking. The remaining work is to
reduce estimator innovation failures and qualify a lower-saturation policy
against real capture timing and held-out motion, without treating this
simulated target as detector-accuracy evidence.
Routine V2 hardware startup now scales feedforward to zero by default;
explicit `--study-feedforward-scale` with the hardware-validation
acknowledgement is required to exercise a nonzero gain in another trial.
Read-only `--check` confirmed effective gains of zero and the configured
source-clock endpoint. The final trace comparator confirms that the 4.4%
RMS improvement fails the 10% gate. Final process audit found no controller,
serial, or trial helpers active; the serial device was unowned and the
persistent DeepStream service was active.

### 2026-09-26 - Controller V3 foundation and timing audit

- Audited the active V2 control path. Its feedforward-off mode still uses a
  Kalman-predicted position error, not a raw PID baseline; the policy has no
  integral term and combines estimation, feedback, and limiting in one class.
- Found that PC source headers and RTP video are correlated by queue order,
  not exact in-band frame identity. DeepStream publishes timing rounded to
  milliseconds. Neither is acceptable as proof of capture time for a new
  feedforward controller. Host and Jetson Ethernet NICs expose no hardware
  PTP timestamps; software clock exchange must retain an offset interval and
  a separate drift bound rather than claim exact synchronization.
- Added isolated `jetson.control_v3.timing` for four-timestamp offset bounds,
  conservative source-age intervals, verified-frame gating, and local stage
  timings. Added `jetson.control_v3.pid`, a raw-bearing P/I plus D-on-gimbal-
  rate baseline with explicit antiwindup, limits, and loss/reset behavior.
  These modules have no network, serial, or motor command authority.
- Deterministic timing/PID tests pass 15/15 locally, including a minimal
  sign-check integrator fixture that does not qualify hardware gains. The remaining identity,
  LAN timing, replay, shadow, and hardware gates are documented in
  `docs/controller_v3_architecture.md`. V2 live behavior was not changed by
  this foundation slice.
- A read-only 50 Hz software-clock survey across the PC/Jetson Ethernet link
  collected 250 exchanges in five seconds. The median offset-interval width
  was 3.863 ms and the narrowest was 3.475 ms (at least +/-1.737 ms even in
  that sample). All observed intervals intersected, but this does not prove
  future drift or exact one-way latency. The bounded responder exited and
  released its port. Raw exchanges are retained on the Jetson candidate at
  `logs/controller_v3_timing_20260926/clock-exchanges.jsonl`.
- The complete canonical host suite passes 385 tests and 12 subtests after
  this isolated addition, with only the existing PyGObject deprecation
  warning. No V3 module was connected to a live controller or serial path.

### 2026-09-26 - Controller V3 exact frame identity canary

- Added an opt-in sender/receiver identity path, separate from deployed V2
  control. The host attaches frame ID and monotonic source nanoseconds to
  video buffers; the encoded RTP marker packet yields its actual SSRC and
  RTP timestamp. A dedicated ZMQ sender carries that mapping promptly over
  Ethernet. On Jetson, the jitterbuffer marker key and decoded DeepStream
  PTS are joined to the mapping by a bounded fail-closed correlator. The
  published V2 perception/control-observation schemas can carry verified
  source identity and nanosecond source time. Default FIFO behavior remains
  unchanged unless `--verified-rtp-headers` is selected at both endpoints.
- Isolated NVENC/RTP and Jetson DeepStream probes established reference-
  metadata propagation through the encoder, PTS matching through decode,
  and no false joins in a synthetic drop test. NVENC adds a large PTS offset,
  so appsrc PTS itself cannot be used as the cross-process identity.
- The first full canary found that `rtph264pay` emits most frame-ending
  packets in `GstBufferList`; a single-buffer probe found zero markers. A
  second canary exposed a one-render-tick header race. Both were corrected
  without changing the production runtime: the sender observes buffers and
  buffer lists, and a dedicated socket/thread transmits headers directly
  from the payloader queue.
- Final bounded canary: 240 host render frames at 30 fps, 241 marker headers
  sent without drops, 181 decoded DeepStream metadata frames, 181 verified
  snapshots, zero withheld/ambiguous joins, and all 181 subscriber samples
  retaining sub-millisecond source timestamps. The candidate ran from an
  isolated source tree and ports; the existing Jetson video service was
  restored and active afterward. No serial or motor control was used.
- This establishes frame association in the tested clean-LAN configuration,
  not clock synchronization or live controller readiness. Remaining gates:
  packet/header loss and reorder stress, measured drift and one-way timing
  bounds, versioned V3 PID replay/shadow, then bounded unloaded hardware PID
  validation. Kalman target-rate feedforward remains a later separate gate.
- The canonical host suite passes 395 tests and 12 subtests; only the
  pre-existing PyGObject deprecation warning remains. The candidate source
  tree does not carry the host test directory; the actual Jetson DeepStream
  canary above is its runtime validation for this slice.

### 2026-09-26 - Controller V3 raw-PID shadow replay

- Added a serial-free `ControlObservation` to `ControlIntent` adapter around
  the pure V3 PID. It accepts raw selected-target bearing and measured gimbal
  rate, but not target-rate prediction. Verified RTP identity, explicit PC and
  Jetson clock domains, conservative capture-age bounds, monotonic observation
  sequence/frame progression, and fresh safety/gimbal state are enforced.
  Every output is `mode=shadow`, expires at its issue time, and is never
  published to an actuator or motor process.
- Added a versioned synthetic eight-observation fixture with golden intents.
  It covers warmup, raw P/I/D terms, unverified-frame hold, recovery, target
  switch, and stale-frame hold. Altering the deliberately extreme target-rate
  field leaves raw-PID output unchanged. Additional tests cover missing clock
  drift bound, wrong clock domain, stale gimbal/safety input, sequence and
  frame regression. Focused V3 tests pass 31/31.
- Added a deterministic 200-frame identity stress test with independently
  lost and reordered headers/markers; it produced no false association.
  Actual transport loss/reorder and a defensible ongoing LAN drift bound are
  still pending. The fixture's zero drift is explicitly synthetic, not a
  measured property of the hardware clocks. No V3 live authority or gain
  qualification follows from this replay.
- The complete canonical host suite passes 403 tests and 12 subtests, with
  only the existing PyGObject deprecation warning.

### 2026-09-26 - V3 actual transport loss and longer clock survey

- Added a bounded, video-only sender probe that reuses the verified
  `GstVideoWriter`, on isolated high-numbered ports. It injects independent
  whole-frame RTP and header losses, sends no control command, and requires an
  explicit canary flag. Fixed the writer's appsrc EOS call to use the actual
  GObject signal and added a no-network regression test for clean shutdown.
- A severe 16/180 RTP-frame plus 13/180 header-loss run produced only
  109 decoded frames in one trial and none in a repeat. This exposed H.264
  keyframe/dependency and receiver-startup sensitivity; the zero-decoded run
  is not an identity validation. The service was restored after each trial.
- For the controlled acceptance run, started the Jetson candidate first and
  confirmed its UDP and header sockets were listening before video started.
  Of 180 generated source frames, four whole RTP frames (45, 90, 135, 180)
  and thirteen headers were deliberately dropped. The sender emitted 180
  keyed markers, sent 167 headers, and reported zero header backpressure.
  DeepStream decoded 175 frames, withheld 13 lacking source headers, matched
  and published 162 verified snapshots, with zero ambiguous joins. The
  subscriber received all 162, IDs increased from 1 to 179, and every
  snapshot retained sub-millisecond source time. There were no detector
  objects in these synthetic images; detector effectiveness was not tested.
  Report: candidate `/tmp/v3_rtp_synchronized_loss_report.json`.
- A separate 55-second PC/Jetson 50 Hz software-clock survey yielded 2,451
  valid four-timestamp exchanges. Offset interval widths were 3.332504 ms
  best, 3.711412 ms median, and 5.972871 ms maximum; all sample intervals
  intersected. That is window-specific consistency, not a certified future
  clock-drift bound. The V3 timing gate still refuses unbounded
  extrapolation. Raw exchanges: candidate
  `logs/controller_v3_timing_20260926/clock-exchanges-55s.jsonl`.
- Both hosts reported NTP synchronized at audit time. The PC chrony status
  reported an estimated +15.207 ppm frequency and 0.044 ppm skew; the Jetson
  timesync status reported +23.099 ppm frequency but no comparable skew
  bound. These daemon estimates are not a conservative cross-host bound on
  future monotonic-clock drift and were not fed into V3 timing.
- Production DeepStream was restored active; serial stayed unowned. No V3
  controller or hardware command process was started. Remaining gate:
  defensible oscillator-drift/clock-age policy, then live-data shadow PID
  with all intent output non-authoritative.
- The complete canonical host suite passes 404 tests and 12 subtests; the
  only warning is the existing PyGObject deprecation notice.

### 2026-09-26 - Controller V3 drift analysis and passive clock watchdog

- Clarified the earlier 55-second survey: its 2,451 valid exchanges spanned
  48.9996 seconds. Regression through all offset-interval midpoints suggests
  +4.338 ppm PC-minus-Jetson drift, while the lower-latency half suggests
  +0.869 ppm. Under the explicit constant-slope assumption, every four-
  timestamp interval permits slopes from -72.115 to +73.010 ppm. The
  disparity shows why midpoint trend alone is not a defensible limit.
- Extended the bounded read-only survey to at most ten minutes and ran it for
  five minutes at 50 Hz while the persistent video service stayed active.
  It retained 15,000 valid exchanges across 299.979 seconds. Interval width
  was 3.362741 ms best and 3.7754425 ms median. Midpoint regression gave
  +0.646 ppm; the lower-latency half gave +0.606 ppm. Constraints from all
  15,000 intervals admit constant slopes from -10.912 to +12.104 ppm.
  These are measured-window results, not a guarantee under future load,
  temperature, or NTP adjustments. Raw exchanges: candidate
  `logs/controller_v3_timing_20260926/clock-exchanges-300s.jsonl`.
- Added `tools.analyze_clock_drift` to reproduce both midpoint trends and
  interval-compatible constant-slope ranges, explicitly marking any subset
  used and reporting `future_drift_bound_established=false`. Updated the
  survey to report actual valid span, attempts/failures, and a 95% span-
  completeness check. The earlier truncated span would now be flagged.
- Added a pure clock watchdog with a missing-bound default that returns no
  controller mapping. An explicitly supplied bound is age-gated; exchange
  order or interval contradictions latch a fault until reset. This catches
  stale or grossly inconsistent timing but cannot prove future oscillator
  behavior from wide software-timestamp intervals. No configured limit was
  connected to V3 shadow or live control, and no motor process was started.
- The complete canonical host suite passes 415 tests and 12 subtests, with
  only the existing PyGObject deprecation warning.

### 2026-09-26 - Verified-video V3 raw-PID shadow canary

- Added `tools.shadow_v3_verified_video`, a bounded Jetson subscriber with no
  serial, motor, laser, or control-publish socket. It carries actual verified
  RTP frame identity and PC/Jetson timestamps into `ControlObservation`, but
  deliberately supplies a guaranteed moving synthetic target, zero gimbal
  state, and synthetic safety permission. The output is only an immediately
  expired `mode=shadow` intent. These synthetic inputs cannot qualify hardware
  gains or prove real closed-loop tracking.
- Ran isolated 720p30 CPU-sim video through the Jetson candidate DeepStream
  pipeline for 28 seconds while the production video service was paused.
  Sender emitted 360 RTP markers and 360 headers; DeepStream matched and
  published 300/300 decoded frames with verified source identity and no
  ambiguous or withheld joins. The selected YOLO engine produced 207 objects
  on 205 frames; those detections were intentionally not used by this PID
  canary. Pipeline steady rate was 30.147 fps after startup.
- Two simultaneous shadow subscribers each saw all 300 verified snapshots.
  With the default missing drift policy, all 300 decisions held
  (`clock_unavailable`) and no nonzero intent was produced. With an explicitly
  shadow-only, empirical 20 ppm test policy, 299 decisions reached raw-PID
  `tracking` and 298 produced nonzero shadow intents; one warmup decision held.
  Conservative source-age upper bounds were 44.166 ms median and 49.462 ms
  p95. The 20 ppm value exceeds the earlier five-minute constant-slope
  interval, but is *not* a certified future drift limit. Passing this canary
  does not authorize live motor output.
- Reports: candidate `/tmp/v3_shadow_video_pipeline.json`,
  `/tmp/v3_shadow_nopolicy.json`, and `/tmp/v3_shadow_empirical20.json`.
  Production service was restored active and serial stayed unowned. Focused
  canary/shadow/watchdog tests passed 15/15; the complete canonical suite
  passed 417 tests and 12 subtests, with the existing PyGObject warning.

### 2026-09-26 - V3 bounded clock-quality shadow policy

- A second read-only 50 Hz PC/Jetson clock survey ran for ten minutes while
  the production DeepStream video service remained active. It collected
  29,905 valid four-timestamp exchanges out of 29,915 attempts over 598.079
  seconds (99.68% span coverage). Offset-interval widths were 3.146 ms best,
  3.827 ms median, and 12.041 ms maximum; all intervals intersected in this
  window. Midpoint slopes were +0.093 ppm over all exchanges and +0.132 ppm
  in the narrow-latency half. A 3,000-sample subset admitted constant slopes
  from -5.471 to +5.886 ppm. None of these is a future drift guarantee.
  Raw evidence is in candidate
  `logs/controller_v3_timing_20260926/clock-exchanges-600s-active-video.jsonl`.
- Tightened the *shadow-only* capture-age cap from 500 to 80 ms and added a
  latching watchdog gate for an over-wide clock exchange. The initial 8 ms
  exchange-width cap was rejected because the active-video survey reached
  12.041 ms. The revised study policy uses 15 ms maximum exchange width,
  100 ms maximum clock-sample age, 80 ms maximum capture age, and a 20 ms
  mapping-uncertainty budget. With a deliberately conservative but still
  unqualified 1000 ppm drift assumption, the checked worst-case mapping
  interval width is 15.390392 ms. An over-budget combination fails at
  configuration time; stale, wide, or inconsistent exchanges cause a hold.
- In a bounded 720p30 verified-video re-test, 300/300 decoded frames retained
  source identity. The no-drift-policy shadow subscriber held all 300. The
  1000 ppm shadow subscriber tracked 298, held one during clock warmup, and
  correctly rejected one frame whose conservative age reached 86.050 ms.
  Upper capture-age p50/p95/p99 were 43.941/48.767/52.872 ms. No motor
  command was published; target, gimbal, and safety inputs remained
  synthetic. Reports are in candidate
  `logs/controller_v3_timing_20260926/v3_policy_*.json`.
- The 80 ms cap is a controlled study criterion, not a real-world target
  angular-speed requirement; physical camera exposure-to-retrieval delay is
  still unmeasured. The 1000 ppm parameter is a provisional stress envelope,
  not qualified across temperature or clock-service changes. Do not enable
  V3 motor authority based on this canary. Production video was restored
  active, serial was unowned, and the full host suite passed 422 tests plus
  12 subtests (existing PyGObject warning only).

### 2026-09-26 - Real IMX219 camera timing survey begun

- The Pi at `192.168.0.3` reports no attached camera. The Jetson has the
  physical IMX219 at `/dev/video0`, mode 4 (1280x720 at 60 fps). Production
  DeepStream remains RTP-input and control-free; it was paused only during
  bounded camera probes and restored afterward.
- Added a source-only `nvarguscamerasrc` probe that records GStreamer buffer
  PTS, pipeline clock, and source-pad arrival. A clean 12-second run delivered
  698 frames at 60.06 fps; steady PTS-to-pad latency was 0.103 ms median and
  0.119 ms p99. A separate four-second post-recovery check delivered 218
  frames at 60.10 fps. These values characterize the GStreamer timestamp
  boundary, **not** sensor exposure-to-delivery latency. The GStreamer PTS
  resolved as pipeline running time; its relation to sensor start-of-frame
  was not independently established by this probe.
- A direct Argus metadata experiment captured 321 valid frames over 10.665
  seconds at 30.01 fps before an abnormal shutdown left a truncated final
  record. Its preliminary, steady sensor-start-to-frame-acquire latency was
  21.796 ms median, 21.952 ms p99, 22.075 ms maximum; the Argus-frame-time
  to acquisition segment was 0.081 ms median. Argus reported 4.683 ms median
  exposure duration after warmup. This is **incomplete evidence**, and the
  frame rate did not match the deployed 60 fps mode. NVIDIA defines the
  metadata sensor timestamp as first sensor data arrival, not exposure start
  or exposure midpoint. See the
  [Argus API reference](https://docs.nvidia.com/jetson/archives/r35.3.1/ApiReference/classArgus_1_1ICaptureMetadata.html).
- The direct experiment stalled on cleanup. It was terminated by exact PID;
  a subsequent retry failed to open the sensor. The Argus daemon was restarted,
  and the supported GStreamer path then passed the four-second health check.
  The failed direct probe source and binary were removed; raw partial samples
  remain in candidate
  `logs/controller_v3_timing_20260926/argus_sensor_timing_12s.jsonl` and
  `tools.analyze_argus_sensor_timing` labels the result incomplete. Next:
  obtain sensor-start metadata from the working GStreamer path or a cleanly
  shutting-down Argus consumer, then repeat at 60 fps and characterize
  exposure timing before changing the V3 capture-age policy. The canonical
  suite passed 425 tests and 12 subtests (existing PyGObject warning only).

### 2026-09-26 - Real IMX219 60 fps sensor-to-pipeline survey completed

- Replaced the unstable direct-Argus experiment with a bounded, source-pad
  `nvarguscamerasrc` probe that reads the plugin's per-buffer sensor frame
  number and sensor timestamp metadata. The metadata timestamp denotes first
  data arrival from the sensor, **not** optical exposure start or midpoint.
  Each sample also records GStreamer PTS and local monotonic source-pad time;
  the analyzer rejects non-monotonic or implausible clock pairings. This
  survey does not measure optical exposure timing.
- The 60-second auto-exposure run exited cleanly: 3,583/3,583 frames had
  plausible sensor timestamps, with zero sensor frame-number gaps and 60.046
  sensor fps. After a two-second warmup, sensor-start to source pad was
  6.822 ms median, 7.470 ms p95, 7.530 ms p99, and 7.671 ms maximum.
  Raw samples are in candidate
  `logs/controller_v3_timing_20260926/gst_argus_sensor_meta_60s_auto.jsonl`.
- Two clean 15-second fixed-exposure checks commanded 5 ms and 15 ms.
  Each delivered 879 frames at 60.05 fps with no sensor frame-number gaps.
  Sensor-start to source-pad median/p99/max were 6.722/6.999/7.103 ms
  at 5 ms and 6.793/7.024/7.363 ms at 15 ms. Thus the observed source-pad
  segment barely changed under these shutter commands; this is **not** an
  estimate of exposure-to-pad latency. Raw samples are in candidate
  `logs/controller_v3_timing_20260926/gst_argus_sensor_meta_15s_{5,15}ms.jsonl`.
- A separate 25-second live-camera DeepStream run processed 1,466 frames at
  60.048 source fps; all 1,466 source/decode, infer-input, and infer-metadata
  PTS records matched, with no unmatched records. Its steady pipeline rate
  was 60.108 fps. Source-pad to inference input was 0.166 ms median and
  0.182 ms p95; inference and metadata took 10.750 ms median and 10.848 ms
  p95. These are separate stage distributions, not a measured joint
  sensor-to-detection percentile. Report is in candidate
  `logs/controller_v3_timing_20260926/argus_deepstream_25s.json`.
- No motor or serial process was started. The production video service was
  restored active after each bounded run. Current Argus-mode perception still
  publishes a GStreamer-relative source clock and no verified source
  identity/sensor start timestamp, so this measurement does **not** make
  real-camera V3 capture age valid. Propagating sensor frame identity and its
  timestamp into the controller's verified timing contract remains separate
  implementation work. A physical optical stimulus would be needed to
  measure exposure-to-sensor-data timing directly.

## 2026-09-26 — isolated V3 PID yaw hardware verification

- The Pi manual-state stream was observed at the Jetson controller endpoint
  before live work: `active=false`, `emergency=false`, and
  `control_cmd_enabled=true`. The bounded trial kept the bridge as sole serial
  owner, 100-ms firmware-timed F6 stop, 50-ms intent validity, 0.15-rad yaw
  travel guard, and zero pitch command. All hardware processes released the
  serial port afterward.
- Full tests on the isolated host stage passed: 432 tests, 12 subtests at
  `29aecabd55c32efc8e7ef95bb68642e704ce2a98`.
- Initial `first18` run was rejected: 667/900 ticks held on absent optional
  encoder rate, with no measured motion. Fixed the local trial to accept a
  fresh encoder angle when rate is unavailable; zero derivative is recorded
  explicitly on those samples.
- Corrected `pose18` run tracked 899/900 ticks and physically followed
  `0 → +0.06 → -0.06 → 0` rad yaw references. Final encoder error from home
  was 0.00422 rad; serial feedback recorded zero write failures and zero
  uncertain writes. Settled errors were bounded but nonzero because the
  integer-RPM F6 command format at 1:1 gearing rounds every requested rate
  below 0.10472 rad/s to zero. This is a motor-command resolution boundary,
  not a full two-axis or live-video V3 acceptance. Details and remaining
  gates are in `docs/controller_v3_architecture.md`.
- A matched P-only run retained 899/900 tracking ticks and ended 0.00153 rad
  from home; its settled errors were slightly lower than with intermittent
  derivative feedback. A subsequent bounded P-only plus existing
  `SpeedCommandDither` trial was **rejected**: settled yaw oscillated across
  0.066–0.069 rad, and mean settled error rose to about 0.018 rad. Both runs
  had zero serial write failures. Integer-RPM quantization is real, but the
  naive workaround exposed a software timing/priority/coalescing problem;
  hardware-only bottleneck closure has **not** been reached. Pitch remains
  untested due persistent ~2.52-rad disagreement between its two encoder
  readings; live-video V3 timing remains a separate gate.
- A 20-ms firmware-timer dither repeat reduced but did not remove the limit
  cycle. The corrected run tracked 899/900 ticks, but settled mean errors
  remained 0.0138/0.0118/0.0110 rad, worse than plain P-only, with 217
  superseded and 45 preempted serial commands. A first launch with a partial
  YAML override failed before motion because config sections are replaced,
  not recursively merged. The second launch used the full original hardware
  override with only `intent_command_runtime_ms` changed to 20. No write
  failures or uncertain writes were reported. This confirms that changing the
  firmware timer alone is insufficient; no dither variant is accepted.

## 2026-09-26 — bounded two-axis video/HIL controller run

- The historical September 24 paired-pitch test showed opposite-signed raw
  pitch counts with magnitudes within 3.6% over its run. The absolute
  pitch-A/B angle warning was therefore not used to keep pitch disabled. An
  isolated HIL override enabled yaw and pitch at 0.2 rad/s, retained 100-ms
  firmware-timed F6 commands and Pi manual safety, and imposed hard yaw
  `[-2.9, -1.9]` and pitch `[0.3, 1.95]` rad limits. Its SHA-256 is
  `c0f2428264d28f8e40a0746c6e1037e0632d48428c65203dab720bfc7f30e571`.
- A CPU-rendered 0.35-m drone followed a deterministic horizontal and vertical
  path with camera pose driven only by measured encoder `CamState`. Exact
  simulator truth, not detector output, drove the V2 controller; video still
  traversed passive DeepStream. Fixture SHA-256 is
  `db4051b87d840024c100379edef9d9ba225402b907f28f0a6f3c6fe39e4a64d4`.
  Feedforward was zero. This does not qualify detector recall, return UI, or
  V3 video timing.
- The first 12-second canary had 581 tracking and two startup safety-hold
  ticks, zero invalid messages, and zero missed periods. Yaw and pitch-A
  moved, but pitch-B held at 1924–1925 raw counts. The longer run was withheld.
- A direct unloaded B-only check then used repeated 100-ms-timed F6 commands
  at representable 0.2 rad/s. B moved 1925 → 1675 counts and returned to
  1928; A stayed at `-4632`. The final zero-rate F6 returned status zero,
  but the `finally` F7 stop was sent. Three subsequent read-only probes found
  stable stopped counts. No calibration or persistent parameter write occurred.
  A permanently dead B encoder/magnet is not supported by this result; an
  intermittent enable/command-path or driver state remains possible.
- A second 10-second canary restored paired motion: A changed `-4632 → -4730`
  and B `+1928 → +2028`, net magnitudes 98 and 100 counts. A passive serial
  observer captured 77 nonzero B F6 commands reaching the wire, with zero
  failed or uncertain writes. The exact recovery mechanism is unproven.
- The final 20-second bounded video/HIL run completed 972 tracking ticks and
  six startup safety holds, 864 gimbal states, 399 Pi states, zero invalid
  messages, and zero missed periods. Pitch motors moved in mirrored raw
  directions during every sampled five-second interval: A/B magnitude changes
  were 228/238, 169/167, and 234/242 counts. These interval differences of
  roughly 1–7% are not a guarantee of instantaneous synchrony. Serial
  accounting closed 3,807/3,807 admitted commands: 3,252 wire-sent, 553
  superseded, two preempted, zero failed or uncertain. Exact-target norm
  error fell from 66.41 px mean over the first 50 tracking ticks to 20.58 px
  over the last 50; whole-run RMS was 34.31 px. This is bounded system
  operation, not detector-accuracy or real-camera tracking qualification.
- Afterward, all three motors reported status `1` and stable counts across
  three reads. The CH341 had no owner, controller/streamer/Pi trial processes
  exited, and passive DeepStream remained active. Evidence resides under
  `/home/idcs/idcs-devtools/evidence/two_axis_{canary12,canary10_events,verify20}/`
  on Jetson, matching host `idcs-devtools/evidence/` directories, and
  mirrored trace/log files in the local external toolkit.

Decision: the requested bounded two-axis video/HIL run succeeded after a
pitch-B dropout and direct timed B-only recovery. Do not assume the first
dropout is permanently fixed. Before a longer unattended or production run,
verify B enable acknowledgement and enforce a relative count-change watchdog.
The existing absolute secondary-pitch warning has an origin/sign error and
does not measure paired-motion synchrony.

### Guarded follow-up (same date)

- Added trial-only tools outside the repo: `preflight_pitch_enable.py` checks
  F3 enable ACK byte `1` for yaw/pitch-A/pitch-B before the serial service
  starts; `guard_pitch_pair.py` observes 0x31 replies without opening the bus
  and fails a 0.8-s interval if relative raw pitch count changes cease to
  mirror. The launcher now aborts the controller and bridge when the guard
  fails. The first launch failed at import before opening serial; corrected
  `PYTHONPATH` in the external launcher and reran.
- The guarded 20-second run **failed closed** after 11.52 s of controller
  operation (553 tracking ticks, six startup safety holds, zero invalid
  messages or missed periods). Preflight F3 ACK was `[1]` and F1 status `1`
  for all three axes. The guard observed 21 valid windows before pitch-A
  changed `-54` counts while B changed only `-1` in a 0.8-s window, exceeding
  the 21-count paired-motion tolerance. The preceding windows included
  `+86/-82`, `+134/-141`, `+115/-116`, and `-84/+87` raw counts.
- Serial accounting was 2,323 admitted, 2,006 wire-sent, 317 superseded,
  zero write-failed or wire-uncertain; the B F6 commands near abort reported
  reply confirmation. However, the serial service validates F6 reply length,
  not its status byte, so this does **not** prove command acceptance. A single
  B encoder-query deadline retry occurred much earlier. Startup enable
  failure is ruled out; F6 status rejection, driver/firmware, encoder, or
  motor-side behavior remain unisolated. After abort, read-only counts were
  stable (yaw 5700, A about -4543, B 1854), all statuses `1`, no CH341 owner,
  and no trial Python process remained. Evidence is in Jetson
  `/home/idcs/idcs-devtools/evidence/two_axis_guarded20b/`.

Decision: the earlier 20-second success is not repeatable enough for
unattended paired-pitch operation. The guard correctly rejected a real
mid-run mismatch. The V2 bridge currently requires and commands both pitch
motors; simply changing YAML cannot omit B. A yaw+pitch-A fallback therefore
needs an explicit single-pitch control path and separate bounded validation.

### Pitch-A-only fallback verification

- Added an explicit `--pitch-a-only` bridge mode (also represented by
  `gimbal.pitch_motor_b_enabled: false`) with pitch-A authority required. It
  omits B enable, rate, and bridge shutdown commands and rejects calibration,
  encoder-zero, and parameter writes in this mode. The bridge was deployed as
  an isolated Jetson devtool copy; the dirty candidate repository was not
  overwritten. The preflight ACKed yaw/A only, stopped and disabled B, and
  the passive guard rejected nonzero B F6/F3-enable events or B movement.
- Local tests: 413 passed, five skipped, two deselected, 12 subtests passed;
  focused bridge tests: 24 passed. The 10-second unloaded video/HIL canary
  had 479 tracking ticks, zero invalid messages or missed periods. Pitch-A
  spanned 397 raw counts while B held exactly at 1854. The guard reported
  20 valid windows and no failure.
- The 20-second verification completed 971 tracking ticks, three startup
  safety holds, 934 gimbal states, 400 Pi states, zero invalid messages or
  missed periods. Pitch-A spanned 480 raw counts; B remained exactly at 1854
  across 32 guard windows. Serial accounting closed 2,862/2,862 admitted:
  2,749 wire-sent, 113 superseded, zero failed/uncertain. Exact synthetic
  target-error RMS was 27.42 px (horizontal 26.14, vertical 8.30). This is
  encoder-driven simulated-camera operation with simulator truth driving the
  controller, not learned-detection or real-camera qualification.
- After the run, all motor statuses were `1`; yaw counts were stable at 5735,
  A at about -4548, and B at 1854 across three read-only checks. No trial
  process owned the CH341. Host streamer and Pi runtime exited; passive
  DeepStream remained active. Jetson evidence is at
  `/home/idcs/idcs-devtools/evidence/two_axis_pitch_a_{canary10,verify20}/`;
  the host has matching streamer/Pi evidence. The local external toolkit
  mirrors `pitch_a_verify20_trace.jsonl`.

Decision: the requested bounded yaw+pitch-A fallback test passes, with B
omitted from motion. Do not treat this as validation of paired-pitch hardware
or deployment of the isolated fallback to the canonical Jetson candidate.

## 2026-09-27 — V3 isolated PID and Kalman-rate feedforward hardware study

- Extended the bounded Jetson-local V3 PID trial to select yaw or pitch-A,
  preserve independent 0.15-rad active-axis / 0.03-rad idle-axis travel
  guards, and expose Kp while keeping Ki=0. The pitch-A-only bridge, F3
  preflight, B-motion guard, Pi safety stream, 50-ms intent validity, and
  100-ms firmware-timed F6 stop remained in place. V3 trial, PID, timing,
  and estimator code ran from an isolated `idcs-devtools` overlay; the dirty
  Jetson candidate repository was not overwritten.
- P-only 18-second steps: yaw Kp=8 had 892/900 tracking ticks, 0.01793-rad
  whole-run RMS, and 0.00690-rad final home offset. Pitch-A Kp=8 had
  897/900 tracking ticks, 0.02131-rad RMS, and 0.01227-rad final offset;
  pitch-A Kp=4 had 896/900 tracking ticks, 0.01801-rad RMS, and 0.00345-rad
  final offset. All three B guards passed with B fixed at 1854 counts.
  Kp=8 yaw and Kp=4 pitch-A are provisional P-only baselines, not claims
  that the integer-RPM near-zero command resolution is solved.
- Added a separate timestamped constant-velocity Kalman target-rate estimator.
  It resets on target switches or discontinuities, rejects stale/future
  samples, and requires warmup. The PID receives its rate contribution as an
  explicit feedforward input; P/I/D/FF and combined command are logged
  separately, with the same rate and slew limits applied to the sum. The
  deterministic local target is a 0.06-rad, 3-second sine with an explicit
  60-ms synthetic observation delay. This does not assert real camera timing.
- Matched 20-second unloaded moving-target trials, scored at 4–19 s against
  exact *current* synthetic target truth (750 samples each):

  | Axis / P gain | FF scale | Truth RMS | p95 absolute error | Tracking ticks |
  | --- | ---: | ---: | ---: | ---: |
  | Pitch-A / 4 | 0 | 0.02711 rad | 0.04273 rad | 992/1000 |
  | Pitch-A / 4 | 0.5 | 0.01769 rad | 0.03034 rad | 996/1000 |
  | Yaw / 8 | 0 | 0.01671 rad | 0.02842 rad | 993/1000 |
  | Yaw / 8 | 0.5 | 0.01228 rad | 0.02358 rad | 996/1000 |

  Pitch-A improved about 35% RMS and yaw about 26% RMS in these matched
  trials. The estimator was valid in all scored samples; FF RMS was about
  0.0444 rad/s for each on run. B stayed at 1854 counts, guards passed, and
  both on runs had zero failed or uncertain serial writes. The yaw on run
  increased rate-limited fraction from 2.1% to 3.6%, which should be watched
  in broader trajectories. Traces and logs are under Jetson
  `/home/idcs/idcs-devtools/evidence/v3_single_{pitch,yaw}_*`; local mirrored
  traces and `analyze_v3_sine.py` are in the external toolkit.
- Local suite after implementation: 417 passed, five skipped, two deselected,
  12 subtests passed. The V3 estimator and feedforward limiter have direct
  deterministic tests. These trials validate isolated hardware response, not
  the V3 video path: observation timestamps were generated on the Jetson,
  not mapped from PC frames or camera exposure. V3 video control remains
  shadow-only; next is fail-closed integration with verified frame identity,
  bounded clock mapping, live observation assembly, and matched video/HIL
  off/on tests. Do not infer that the 0.5 FF gain transfers to detection noise
  or a different target-motion spectrum.
- A separate simultaneous yaw+pitch-A V3 trial then exercised shared serial
  scheduling. The 10-second canary tracked 497/500 ticks. Matched 20-second
  feedback-only and FF=0.5 trials each tracked 996/1000 ticks. Over the same
  4–19 s exact-truth window, yaw RMS changed 0.01648 → 0.01306 rad and
  pitch-A RMS 0.02684 → 0.01721 rad; p95 absolute errors changed
  0.02968 → 0.02340 and 0.04258 → 0.02977 rad respectively. B remained
  fixed at 1854 raw counts, the guard passed 32 windows, and the on run's
  serial service closed 2,579/2,579 commands with zero failed/uncertain.
  The combined controller ran from `jetson/control_v3/dual_pid_trial.py` in
  the isolated overlay; local suite passed 419 tests, five skips, two
  deselections, and 12 subtests.

The V3 *hardware-control* PID and 0.5 estimator-rate FF baseline is now
verified for this controlled unloaded sine and step family, including shared
bus operation. It is not yet V3 video-driven control: the full video path
requires a causal PC→Jetson clock bound, Jetson receipt/observation stamps,
and camera pose aligned to the frame's source time before target-world-rate
feedforward is valid. Do not reuse the Jetson-local 60-ms timestamp as a
substitute for those measurements.

A held-out slower four-second, 0.06-rad sine repeated the simultaneous
off/on comparison without retuning PID or FF. Over 750 scored samples at
4–19 s, yaw exact-truth RMS fell from 0.01373 to 0.01095 rad and pitch-A
from 0.02337 to 0.01681 rad; p95 errors also decreased on both axes. The
FF-on run tracked 994/1000 ticks, B stayed at 1854, the guard passed 32
windows, and serial accounting had zero failed or uncertain writes. This
strengthens the bounded local-hardware conclusion across two motion
frequencies, but does not qualify real camera timestamps or learned detection.

For the V3 video boundary, `CameraPoseHistory` now has a pure, tested
frame-time alignment primitive. It refuses captures not wholly bracketed by
encoder samples, over-wide mapped clock intervals, and excessive angular
uncertainty; it does not extrapolate a camera pose. The local suite passes
422 tests, five skips, two deselections, and 12 subtests after this addition.
No V3 video motor authority was enabled. A bounded V3 video/HIL actuation
trial still needs an operationally justified PC-to-Jetson clock-drift bound
or an explicit decision to use an empirical, test-only bound; the latter
would not qualify production real-camera timing.

2026-09-27 V3 rendered-video shadow integration: added measured Jetson
receipt/observation stamps on exact-frame simulator snapshots, fail-closed
capture-time pose bracketing, and a separate target-world-rate Kalman path.
The host simulator copy initially lacked the already-local
`source_identity_verified=True` field; all ten initial frames were correctly
rejected. A file comparison showed this was the only host/local difference,
so that one-field source update was deployed to the host. In the subsequent
15-second shadow run, 300/300 snapshots passed identity verification,
298/300 PID timing decisions tracked, and feedforward was ready on 242/300.
In a second 15-second shadow comparison, 370/370 snapshots passed, 369/370
PID decisions tracked, feedforward was ready on 271 and held on 92 stale
samples (plus startup/bracketing warmup). All 92 stale samples applied zero
feedforward. All 271 ready samples populated a nonzero explicit FF term;
only 15 changed the final command by >0.001 rad/s because this static-camera
fixture saturates the 0.2-rad/s rate cap for most frames. This demonstrates
the timing/estimation/fallback wiring, not closed-loop efficacy or video
motor readiness. The pose and safety inputs in this shadow utility are
synthetic; it never publishes an intent or opens serial. Reports and traces
are under Jetson `/home/idcs/idcs-devtools/evidence/v3_video_shadow*`, with
mirrors in local external `idcs-dev/evidence`. The local suite passed 427
tests, five skips, two deselections, and 12 subtests. The host stream and
shadow process stopped, serial was unowned, and DeepStream remained active.

Next gate remains a bounded, measured-encoder V3 video HIL run. Replace the
synthetic pose/safety shadow inputs with live sampled data, retain the
source-clock and encoder-bracketing holds, and qualify the clock drift bound
before granting video motor authority. Then compare matched FF-off/on
trials against exact simulator target truth. The present shadow result must
not be called successful tracking or used to tune the hardware PID.

The measured-encoder shadow then subscribed to the read-only gimbal bridge
while the rendered simulator consumed the same CamState stream. The first
15-second run received 325 verified frames and 636 CamState updates but
only 47 PID decisions tracked: the bridge legitimately omitted yaw/pitch
rate fields on many 50-Hz publications when no new encoder slope was
available. V3 P-only incorrectly treated those optional rates as mandatory.
The boundary now requires measured rate only when either D gain is nonzero;
it never synthesizes a nonzero rate. On repeat, 383/391 PID decisions tracked
despite 335 frames missing at least one rate; maximum measured gimbal age
was 20.48 ms. Estimator readiness was 210/391 because its 120-ms stale
cutoff was stricter than the video PID's 150-ms capture-age gate. Matching
those gates while preserving a hard stale hold produced 393/396 PID tracking
and 389/396 FF-ready decisions in the final 15-second shadow. Of the 389
ready frames, 164 had a final command difference >0.001 rad/s between
PID-only and PID+0.5 FF; commands remained immediately expired shadows.
The final run received 736 CamState updates, had zero invalid states,
and maximum measured gimbal age 20.43 ms. The captured serial-event stream
had only 555 encoder (`0x31`) and 11 F1 query sends, no F3/F6/FD motion
commands after subscription. The bridge was explicitly read-only, and the
serial service's safety startup stop is not counted as motor authority.
The final local platform-compatible suite passed 430 tests, five skips,
two deselections, and 12 subtests; `git diff --check` reported no whitespace
errors. After the run, the streamer, bridge, serial service, and shadow
runner were stopped; the TTY was unowned and passive DeepStream remained
active.
Evidence is under Jetson
`/home/idcs/idcs-devtools/evidence/v3_video_pose_shadow*`, mirrored in
the local external toolkit. This qualifies timing and controller *wiring*
under a static mount, not moving-camera closed-loop performance.

The live-video gate still needs an operationally justified inter-machine
drift bound (the shadow's 1000-ppm value was an empirical study assumption),
real manual/safety input, and a bounded V3 motor-authority runtime with
matched moving-camera FF-off/on trials. The host simulator uses the bridge's
render-prediction pose when available; measured encoder pose is not always
identical to the rendered pose at the source frame. That alignment error must
be characterized during moving-camera HIL before claiming FF efficacy.

2026-09-27 V3 fixed-rate video runtime preparation: added a 50-Hz controller
process, asynchronous 20-Hz four-timestamp clock poller, strict Jetson
receipt/observation stamping, real Pi manual-state and measured CamState
assembly, separate Kalman target-rate FF, explicit FF-off/0.5 selection,
and short-lived live-intent candidates. Shadow is the default and publishes
no intent. The live path requires separate empirical-test-clock and unloaded
hardware acknowledgements; `--check` opens no socket. A pure controller
test verifies zero-rate live candidates on frame-identity/safety failure and
a projected 0.15-rad trial travel envelope. Live HIL requires render-pose
alignment, because the host sim camera uses the bridge's `render_pan/tilt`
when present; the real-camera mode retains encoder-pose alignment. Render
predictions older than 100 ms are rejected for FF.

The first 15-second fixed-rate FF-on shadow, with real Pi and measured
encoder input, ran 749 ticks with one missed period but held 629 for the
150-ms capture-age policy. The trace's mapped capture-age upper bound was
169 ms median, 197 ms p95, 214 ms p99, one 262-ms outlier. This was a real
timing failure, not a clock-exchange or PID success. An explicit test-only
250-ms capture-age gate (also used as the Kalman sample-age cap) then tracked
738/750 ticks, FF ready on 726, zero missed periods; capture age p95 was
190 ms and max 243 ms. This is a shadow policy study, not production timing
qualification. Captured serial events were only encoder/F1 queries.

A dedicated guaranteed-selected moving drone fixture was then constrained
from observed rendered geometry, not assumed world coordinates. Its first
0.08-m horizontal path projected to about +/-0.21 rad, beyond the 0.15-rad
trial travel envelope. Narrowing to +/-0.035 m produced about +/-0.10 rad.
Its initial 0.12-m/s path could exceed the 0.2-rad/s rate cap at the actual
scene depth, so speed was reduced to 0.045 m/s. The final shadow had yaw
bearing range -0.077 to +0.073 rad, pitch -0.023 to +0.032 rad, and
estimated yaw target speed p95 0.056, max 0.061 rad/s. The render-aligned
FF-on fixed-rate run tracked 747/750 ticks, FF ready on 739, with no missed
periods and no motor publisher. An FF-off shadow on the same feasible fixture
tracked 747/750 ticks; neither shadow can establish closed-loop efficacy.
The latest local platform-compatible suite passed 441 tests, five skips,
two deselections, and 12 subtests.

Guarded external HIL wrappers now exist under `idcs-dev/` for a 10-second
canary and matched 20-second FF-off/on runs. They require an explicit
`test_only_1000ppm` argument, check process/TTY ownership, use the exact
fixture and pitch-A-only override, start the Pi safety runtime and F6-timed
serial service, omit B motion, watch B with the independent guard, and
terminate with zero-rate/shutdown cleanup. Their shell syntax and rejection
of an unacknowledged clock policy were checked; they have **not** been run
with motor authority. The empirical 1000-ppm drift assumption requires an
explicit user choice for unloaded test-only HIL and would not qualify a
production real-camera controller. The moving-camera render/encoder timing
error and matched frame-unique bearing RMS must be measured in those live
off/on trials before the V3 hardware-video goal can be claimed complete.

2026-09-27 stationary-target camera-motion check: a guaranteed-selected
stationary drone was rendered while synthetic CamState yaw oscillated
by +/-0.04 rad. The first non-actuating run inferred a spurious target
yaw speed of 0.080 rad/s median (0.115 p95). Investigation found that
the simulator's 60-degree vertical FOV (91.49-degree horizontal at this
aspect ratio, fx 935.31 px) disagreed with the real-camera control
configuration's 73/135-degree FOV. The V3 video runtime now requires
an explicit simulator vertical FOV in render-pose mode and derives the
in-memory intrinsics from the active resolution, without changing the
persistent real-camera control configuration. With that calibration,
the same 15-second shadow ran 750 ticks, 878 snapshots, 741 FF-ready
decisions, zero missed periods, and no motor authority. Absolute
spurious yaw-rate estimate fell to 0.00188 rad/s median and 0.00352
p95 (one 0.0249 startup outlier; after 4 s max 0.00568). This validates camera-motion subtraction
under the stationary fixture, but not moving-camera hardware FF
efficacy. The live matched FF-off/on HIL gate remains pending the
explicit empirical-clock test choice and actual motor-authority run.

The calibrated moving-drone fixture was then rerun without motor authority:
749/749 fixed-rate ticks (one scheduler miss), 746 tracking decisions,
739 FF-ready decisions, 487 verified snapshots, and 299 Pi manual/safety
states. Absolute target yaw-rate estimate was 0.0163 rad/s median,
0.0237 p95, max 0.0258; raw yaw bearing error peaked at 0.0327 rad.
The reduced rate versus the pre-calibration shadow is consistent with
the corrected pixel-to-angle scale. This remains an open-loop shadow, not
evidence that FF improves closed-loop tracking.

The matched-trial analyzer now also rejects a live trace if calibrated
simulator intrinsics or the explicit test-only clock policy are absent,
the cadence misses more than 1% of ticks, fewer than 90% of ticks track,
any tracking tick lacks a verified capture-time interval within the 250-ms
gate, any live intent lacks a <=50-ms lease, FF-off applies nonzero FF, or
FF-on fails to apply nonzero FF on at least 100 tracking ticks. It reports
clock-verified tracking ticks, capture-age p95, and actually applied FF
ticks alongside unique-source-frame RMS. These gates prevent a matched
score from being mistaken for a timing- or FF-qualified trial.

2026-09-27 user-approved unloaded test-only 1000-ppm V3 video/HIL: the first
10-second FF-off canary exposed one over-wide software-clock exchange; the
watchdog latched it, holding 199/500 ticks. It also exposed pitch command
quantization: Kp 4 gave sub-1-RPM pitch demands that encoded as zero RPM.
The clock poller now discards an over-wide exchange and requires two fresh
clean exchanges before resuming, while drift contradictions remain latched.
The trial-only pitch Kp is explicit and was raised to 8 under the unchanged
0.2-rad/s rate and 0.15-rad travel caps. A second 10-second canary tracked
491/500 ticks with zero missed periods and max exchange width 6.33 ms;
pitch-A spanned 320 encoder counts, pitch-B held at 1855, and serial had
no failed or uncertain writes. The independent pitch guard and analyzer
were corrected to decode the full 12-bit F6 RPM field rather than just
its low byte.

Two order-reversed 20-second FF-off/on hardware-video comparisons then
passed the safety/timing gates with the same fixture, config digest and
Kp 8/8. First pair: off yaw/pitch unique-frame RMS 0.02592/0.02306 rad
(431 frames), on 0.02712/0.02330 (438), +4.65%/+1.05% worse. Reverse
pair: off 0.02639/0.02211 (439), on 0.02763/0.02462 (429),
+4.69%/+11.31% worse. FF was actually applied on 987 and 990 ticks;
the off runs applied none. All had zero missed control periods, 992-998
tracking ticks per ~1000, capture-age upper p95 148-155 ms, hundreds
of confirmed yaw and pitch-A motion writes, pitch-A span 301-331 counts,
pitch-B unchanged at 1855, and no failed/uncertain serial writes. This
validates motor authority, separation and fail-closed timing, but **does
not validate FF efficacy**. On runs saturated the 0.2-rad/s rate cap more
often, so estimator/actuator interaction remains under investigation.

A separate non-actuating moving-drone plus sinusoidally moving-camera
study logged capture-time world-angle measurements. Against a centered
six-frame finite-difference reference over 529 unique frames, Kalman
yaw/pitch rate mean absolute error was 0.00305/0.00313 rad/s with few
opposite-sign samples. This makes a gross estimator sign reversal unlikely,
but cannot establish performance with hardware-driven camera pose.

The first live-video pairs above subsequently revealed a simulator pose
transport error, not merely Kalman tuning: the PC rendered each frame from
the last CamState it had *received*, while Jetson interpolated a newer
bridge pose at the nominal source timestamp. With hardware-driven camera
motion, the capture-time target-world-angle finite-difference comparison
showed yaw/pitch Kalman-rate errors of 0.0274/0.0236 rad/s mean absolute
and 118 opposite-sign samples per axis out of 551. The host simulator now
attaches the exact relative camera pose it used to render each guaranteed-
truth frame plus the applied CamState's Jetson timestamp. V3 peels these
HIL-only fields before validating against the untouched dirty Jetson V2
schema. Live HIL requires the exact pose, a <=100-ms applied-state age,
verified source frame identity, and the existing mapped capture-time gate;
missing/stale metadata sends a zero-rate hold. The host's prior source
files were preserved under `idcs-devtools/evidence`; the Jetson candidate
`common/perception.py` was **not overwritten** after safety review blocked
that deployment. The isolated V3 overlay and hash manifest were updated.

Non-actuating smooth-camera/shadow and a 10-second unloaded motor canary
verified the exact-frame path. In the motor canary, 318 unique frames gave
Kalman-rate mean absolute errors 0.00098/0.00197 rad/s yaw/pitch against
capture-time finite differences (zero yaw sign errors); 487/500 ticks
tracked, ten explicitly held for pose freshness, pitch-A spanned 277
counts, and pitch-B stayed fixed. The formerly false motion estimate is
therefore corrected in the hardware-driven video path.

Two matched exact-pose 20-second FF-off/on pairs passed source identity,
clock, short-lease, serial, pitch-B, and two-axis motion gates. One pair
improved unique-frame yaw/pitch RMS by 12.8%/19.5%; the reverse-order pair
worsened it by 4.5%/16.3%. The contradiction prompted within-run
crossovers rather than a claim of efficacy. Each of two 30-second runs
switched FF in three 10-second blocks (off/on/off or on/off/on), discarded
the first two seconds after each switch, and pooled one unique error per
source frame. On the original cornered path, balanced three-on/three-off
blocks changed yaw RMS 0.02765 -> 0.02742 rad (-0.84%) and pitch
0.02343 -> 0.02356 (+0.55%); 1,299 frames were scored. All 2,894
tracking ticks across both runs used verified clock bounds, pitch-B held
at 1855, pitch-A moved 296-317 counts, and no failed/uncertain writes
occurred. A separate smooth eight-waypoint 3D loop was verified in shadow
(target-rate magnitude <0.025 rad/s on both axes), then with a 10-second
two-axis motor canary. Its reverse-order 30-second crossovers scored
1,241 frames: yaw RMS 0.02725 -> 0.02757 (+1.15%), pitch
0.02064 -> 0.01959 (-5.09%). They tracked 1,401 and 1,420 of 1,499/1,500
ticks, had at most one missed period, passed B guards, and released the
serial port. These controlled video/HIL tests verify that FF is applied
and the estimator is accurate, but show **no consistent net two-axis
tracking improvement** at the current bounded operating point.

Serial evidence identifies a concrete resolution limit: under the 0.2-rad/s
trial rate cap and actual 1:1 motor ratio, integer-RPM F6 commands were
only 0 or 1 RPM. In the cornered crossover's first run yaw had 1,074
one-RPM and 396 zero-RPM writes; pitch-A had 757/466. The typical FF
contribution (~0.01 rad/s) is much smaller than the 1-RPM camera-rate
quantum (~0.105 rad/s), so it often cannot change a wire command except
near a threshold. The rate cap is a *trial safety limit*, not a claimed
motor hardware maximum; higher commanded speeds would not remove the
integer-RPM quantum. An actuator-resolution strategy would be a separate
design and safety task, not a reason to mislabel this FF result as a win.

## 2026-09-27 — F5 absolute-axis command path (software only)

- Chose F5 absolute motion by axis as the candidate actuator command for
  V3. F5 targets are multi-turn encoder counts (16384 per motor turn,
  ~0.00038 rad at 1:1), and the manual (V1.0.9, 11.4) states a new F5 may
  replace speed and target while a move is running. F5 speed and
  acceleration remain integer fields, so this changes *position* resolution,
  not speed resolution.
- `common/gimbal/mks_servo42_rs485.py` gains F5 move/stop encoding and 0x98
  heartbeat configuration. Out-of-range fields raise instead of clamping;
  speed 0 is refused for moves because the firmware reads it as stop. Tests
  reproduce the manual's four F5 example frames byte-for-byte, CRC included.
- `jetson/control_v3/position_target.py` integrates a camera-axis rate into
  an F5 target each tick. It bounds the target's lead over the measured
  position, clamps to the travel limit around home, re-anchors to the
  measured pose after a caller gap instead of catching up, and picks the
  smallest integer RPM that covers the remaining distance in one nominal
  tick (capped). Against a simulated F5 plant, a 0.01 rad/s command (F6
  encodes it as 0 RPM) reached 0.05 rad in 5 s within two counts; average
  rates from 0.003 to 0.15 rad/s held without drift.
- Not yet verified on hardware: that 0x31 counts and the F5 coordinate frame
  agree without 0x92 zeroing, how F5 updates behave at 50 Hz, and serial
  arbitration/emergency classification of F5 in `serial_io_service`. The
  bridge and trial runtime still send F6 only.
- Follow-up (same day): `serial_io_service` now treats critical zero-speed
  F5 as an emergency, discards queued F5 moves behind an emergency,
  coalesces F5 latest-wins per motor (reason `latest_wins_f5`), and drops
  stale F5 like stale F6. F5 is kept out of multi-command frames and the
  F6-only actuation snapshot.
- `gimbal_bridge` gains `gimbal.actuation_mode: f5_position` (default
  `f6_speed`, unchanged). `jetson/f5_actuation.py` plans each accepted
  intent: moving rates become F5 targets (priority high); zero rates,
  non-finite rates, a missing/stale/timing-rejected encoder reading, or a
  home outside the hard limits become an immediate F5 stop (acc 0,
  critical) on yaw and pitch-A, and the next motion re-anchors at the
  measured pose. Travel is the tighter of `f5_position.travel_limit_rad`
  around the first fresh encoder reading and the bridge hard limits,
  converted through both motor and CamState signs. F5 mode refuses pitch-B
  enabled (independent targets could make the coupled pair fight), pitch
  authority B, and wire-execution render prediction. Heartbeat (0x98) is
  not configured automatically because MKS parameter writes persist.
- Replacing timed-F6 expiry: if the bridge stops sending, the motor ends at
  its last target, at most `max_lead_rad` (default 0.01 rad) past the last
  measured position. Whether F6 zero-speed halts a running F5 move is
  unverified, so F5 mode stops with F5, not F6. The shutdown path still
  sends F6 zero and F3 disable.

## 2026-09-27 — F5 yaw bench probes: coordinate offset and step-frame plan

Unloaded bench, yaw (addr 1) only, direct bus access with no serial service,
bridge, or Pi safety stream; pitch motors received no commands. Scripts and
outputs: `/home/idcs/idcs-devtools/evidence/f5_step{1,2,2b,2c}_*` on Jetson.
Driver code: the Jetson candidate checkout (`d6732f1`); F5/F4 frames were
built inline and checked against the tested encoder.

- Step 1 (read-only): yaw status 1, 0x31 counts steady at 5627.
- Step 2: F5 to start+26 at 1 RPM moved the right way at the expected speed
  (~21 counts in 75 ms) but held 5-6 counts short; the 3-count settle
  tolerance aborted the run. Cleanup (F5 stop, F7, F3 disable) ran.
- Step 2b: after the disable/enable cycle yaw first jumped 7 counts *away*
  from the target on enable, then held 12 counts short.
- Step 2c logged 0x39 angle error: disabled 5651 (-7.4 counts error);
  enable jumped +8 counts; F5 to 5685 held at 5666 (**miss 19**) while the
  motor's own angle error read ~0.6 counts; F4 relative -26 then moved 23
  counts (miss 3, angle error 1-2).
- Interpretation: F5 absolute coordinates carry an offset from the 0x31
  reading. 0x31 and 0x39 come from the same encoder, so this is a
  zero-point difference, not two disagreeing sensors. The offset grew
  5 -> 12 -> 19 counts, about one enable jump per disable/enable cycle.
  Disabled yaw also drifts 4-5 counts between runs. F4 relative motion does
  not see the offset; its ~3-count residual is the physical floor.
- Consequence for `f5_position` mode as committed: its lead clamp is in the
  0x31 frame, so an offset near `max_lead_rad` (26 counts) would stop the
  axis short of the reference, and the first hold command would jump by
  the offset. Do not run it on hardware until this is addressed.
- Agreed direction (not implemented): command in the motor's step frame.
  Learn the offset once per enable session (candidate: 0x33 "pulses
  received" read, else an F4 zero move or settled hold), keep it fixed for
  the session, use the 0x39 angle error as a lost-step watchdog that stops
  and forces relearning, and use 0x31 only for home and travel limits. Any
  disable ends the session. Neither frame observes shaft-to-camera
  compliance; only video feedback does.
- Priority remains closed-loop tracking; this F5 work is parked behind it.

## 2026-09-27 — Latency-aware PID gain sweep (simulation only)

- `tools/latency_gain_sweep.py` runs the real V3 `BasicPID` against the
  qualified 2026-09-14 plant: 60 fps frames whose bearing error arrives after
  a sampled latency (level + 0-2 ms seeded jitter), 50 Hz decisions with D on
  the fresh encoder rate, 0.2 rad/s cap and 3.5 rad/s^2 slew, then F6
  integer-RPM truncation. The fit's bias term is disabled while the command
  is exactly zero (a stepper at 0 RPM holds). Objective: mean true-pointing
  RMS over five search scenarios; five held-out scenarios reported
  separately. Reports: `artifacts/controller_sim/latency_gain_sweep_20260927/`.
- P-only optimum Kp (F6 quantised / ideal actuator): yaw 60 ms 21/17,
  120 ms 15/8.8, 200 ms 11/5.7; pitch 60 ms 17/14, 120 ms 17/8.8,
  200 ms 14/5.1. Optimal gain falls with latency; the F6 quantum raises it
  ~1.5-2x because the settled deadband is 0.105/Kp rad.
- Not yet meaningful: 0-30 ms optima sit at the Kp grid edge because the
  simulation has no measurement noise; Nelder-Mead (Ki, Kd) refinements are
  ill-posed for the same reason (e.g. Ki=710 absorbed by the integral
  clamp). Do not use them.
- Model disagrees with hardware on pitch: it ranks pitch Kp=4 far worse than
  higher gains, while the 2026-09-27 hardware study found pitch-A Kp=8 worse
  than Kp=4. Yaw's simulated Kp=8 settled error (9.5 mrad, bound 0.105/Kp)
  is consistent with measured 4-10 mrad.
- Next: closed-loop replay of recorded yaw and pitch-A trials to validate
  or extend the plant; estimate bearing jitter from recorded static-target
  DeepStream snapshots and add it; only then confirm optima on hardware.

## 2026-09-27 — Hardware replay: F6 1-RPM overspeed and corrected gain sweep

- `tools/replay_hardware_trials.py` replays recorded `local_pid_trial`
  traces (Jetson `evidence/v3_pid_{pose18,p_only18}`,
  `v3_single_{yaw,pitch}_{p4,p8}_18`, `v3_single_{yaw,pitch}_sine_off20`)
  open- and closed-loop against the qualified plant.
- Open loop, the nominal model explains only 0.41-0.48 of hardware travel
  on every trial. Aligning `serial-events.jsonl` wire F6 commands with
  encoder motion: only 0 and 1 RPM reached the wire (0.2 rad/s cap), and a
  1-RPM F6 moves **2.0-2.5x** nominal (yaw ~2.4x, pitch ~2.1x; per-trial
  fits 1.96-2.47x). A latest-wins model with that factor and a 75-90 ms
  measurement lag reproduces trajectories to 2.9-8.2 mrad RMS; "zero F6
  does not cancel a running 100-ms timed run" was tested and rejected
  (26.9 mrad). The 2026-09-14 fit used >=4 RPM commands, where gain is ~1,
  so this is a low-speed F6 property; F5 at 1 RPM moved at true speed in
  the bench probes. Cause in firmware unknown.
- Encoder feedback in these trials was ~13 Hz per motor (240 0x31 reads in
  18 s) with ~80 ms effective lag: itself a large loop latency.
- With the measured 1-RPM gain and 13.3 Hz feedback, closed-loop replay
  matches hardware tracking RMS within ~1-2 mrad on most trials and
  reproduces the pitch ranking (hardware Kp4 18.0 < Kp8 21.3 mrad; sim
  19.0 < 20.0). The earlier "pitch model disagreement" was this actuator
  effect, not pitch dynamics. Trajectory residual 5-10 mrad remains.
- Corrected sweep (60 fps camera, `f6_corrected_60fps.json`), P-only
  optimum Kp with 5% band: yaw 30 ms 15 (15-19), 60 ms 11 (10-12),
  120 ms 6.4 (5-8), 200 ms 2.4 (2.2-3.0); pitch 30 ms 19 (15-21),
  60 ms 14 (11-14), 120 ms 7.9 (7-8), 200 ms 2.7 (2.2-3.3). 0 ms still
  hits the grid edge and Ki/Kd refinements remain unreliable until
  measurement noise is modelled.
- Implications: hardware confirmation at low latency needs faster encoder
  feedback than 13 Hz; the 1-RPM gain should be measured directly (and at
  2-3 RPM) before relying on it outside the 0.2 rad/s cap; bearing noise
  from real DeepStream data is still the missing model input.

## 2026-09-27 — F6 speed probe, step-count feedback, first hardware gain sweep

User reports all three motors mechanically uncoupled and unloaded; hardware
runs authorized. Direct bus access, one motor at a time, travel guards and
F6-zero/F7/disable cleanup on every exit. Evidence on Jetson under
`/home/idcs/idcs-devtools/evidence/{f6_speed_probe_*,stepsweep_*}`; code ran
from an exact `git archive` of this branch in `idcs-devtools/claude_stage/`.

- F6 speed (timed, refreshed every 20 ms) is the same on all three motors:
  114/164/228/342 microsteps/s at 1/2/3/5 RPM, i.e. about (n + 1) RPM
  (1 RPM -> 0.224 rad/s, 2.15x nominal). acc 0 and 10 are identical.
  The trials' 0.2 rad/s cap was therefore not enforced on the wire: the
  smallest nonzero F6 already moves 0.224 rad/s. Plain 3-byte F6 did not
  move the motors and its replies disrupted following reads; only the
  timed form is used. The sweep model now uses this measured table.
- 0x33 ("pulses received") is the microstep count (3200/rev, 5.12 encoder
  counts per step) and follows both F6 and F5 motion: usable as the
  step-count feedback. Read cost: 0x33 4.96 ms, 0x31 5.98 ms, 0x39 4.96 ms.
- Bus budget at 38,400 baud per motor per 50 Hz tick: 0x33 read ~5 ms +
  timed F6 write ~2.9 ms = ~7.9 ms. One motor 40% of a 20 ms tick; yaw +
  pitch-A 79%; three motors >100% (would stall). Multi-motor full-rate
  step-count feedback needs a higher baud (to be qualified after the bench
  is built); group F6 frames help only partly; periodic auto-report (5.1.9)
  is avoided on this half-duplex bus. The 13 Hz encoder rate seen in trials
  is the `configs/control.yaml` 100 ms 0x31 schedule, not a link limit.
- `jetson/tools/step_count_gain_sweep.py`: BasicPID on one motor, 0x33
  feedback, emulated 60 fps camera with injected latency, bridge-style
  timed F6. Yaw sweep Kp {4,8,12,16,24} x latency {0,60,120} ms x 2,
  shuffled: all 30 runs at 50.0 Hz, repeats within 0.3%. Mean RMS pointing
  error (mrad), hardware / matched simulation:
  0 ms: 20.7/20.9, 11.5/11.4, 8.5/9.1, 7.7/8.5, 6.6/7.3 (best >=24 both);
  60 ms: 15.1/16.3, 12.9/11.7, 12.2/11.1, 14.2/12.5, 17.2/14.8 (best 12 both);
  120 ms: 18.8/18.4, 20.0/16.8, 23.2/19.4, 25.6/20.7, 27.7/22.6
  (hardware best 4, simulation best 8).
- At 120 ms the hardware penalises high gain 19-23% more than the model. A
  single extra loop delay (best ~10 ms, mean error 7.8%) or a slower
  low-speed motor time constant does not fix it cleanly; one combination
  (tau 25 ms + 10 ms) matches all optima but raises max error. Not adopted.
  Likely cause: the level-0/1 on-off regime under the 0.2 rad/s cap is not
  first-order at 6 ms. Next: identify low-speed actuator dynamics directly
  from the recorded 50 Hz sweep traces, and refine the Kp grid near the
  optimum at the latencies that matter.
- The 0 ms optimum still reaches the grid edge on hardware too: with no
  measurement noise in the synthetic loop, higher gain keeps winning.

## 2026-09-27 — Kp procedure settled; feedforward confirmed on hardware with fast targets

- Decision (user): the Kp procedure (hardware-replay-validated simulation,
  then a synthetic-target, latency-injected, step-count hardware sweep) is
  the accepted way to choose Kp; Kp is considered done. Ki and Kd are
  deferred as low impact.
- F6 speed extended on yaw: 4/6/7/8/10 RPM -> 279/392/454/503/611
  microsteps/s, within ~3% of the (n + 1) RPM extrapolation. The model
  table now holds levels 1-8 and 10 measured.
- Feedforward had looked ineffective because trial targets moved slower than
  one F6 level. `tools/feedforward_sweep.py` uses the real
  `TargetRateKalman` on delayed frames, rate feedforward, and a prediction
  fraction p (PID error = predicted target minus camera angle at
  capture + p * frame age), on smooth targets up to 0.63 rad/s
  (0.2 s cosine velocity blends, <10 rad/s^2) with a 1 rad/s cap and
  10 rad/s^2 slew. At each latency's PID-only Kp, simulation predicted
  ~40% lower RMS for FF 0.5 + prediction 0.5 under both measured-F6 and
  ideal actuators, so F5 is not a prerequisite for feedforward.
- Hardware (yaw, step-count feedback, 50 Hz, 24 runs, repeats within ~2%;
  evidence `ffsweep_yaw_20260927T074444Z`), mean RMS mrad at Kp 19/12/7 for
  30/60/120 ms: PID only 18.0 / 26.7 / 52.9; FF 0.5 11.1 / 16.7 / 42.3;
  FF 0.5 + prediction 0.5 (Kalman accel sigma 8) 8.2 / 14.5 / 32.1
  (-55% / -46% / -39%); same with sigma 2 9.8 / 17.2 / 37.4. Rankings
  match simulation at every latency; costs within ~10-15%.
- Caveats: synthetic loop has no bearing noise and exact capture pose, so
  the responsive sigma 8 is favoured; with real detection jitter the
  estimator noise and prediction fraction must be re-chosen. The slow-target
  simulation remained optimistic versus the earlier hardware result.

## 2026-09-27 — Latency-compensated PID error in the V3 video controller

- `VideoTargetRateEstimator.estimate(..., predict=p)` now also returns the
  predicted target world angle and camera angle at capture + p * (decision -
  capture) and their difference, `predicted_bearing_error_rad`. The target
  uses the existing Kalman state (evaluated at decision time for the
  unchanged capture-age gate, then moved back along the estimated rate).
  The camera angle comes from the same pose history as the capture pose:
  interpolated when bracketed, otherwise the newest sample. "frame" pose
  mode has no fresh pose stream, so it never predicts.
- `ShadowPIDController.decide(..., error_override_rad=...)` uses that error
  instead of the frame bearing; all timing, target, gimbal, and safety gates
  are unchanged. `VideoControllerPolicy` gains `predict` (default 0, the
  previous behaviour) and `feedforward_accel_sigma_rad_s2` (default 0.4);
  an invalid estimate falls back to the frame bearing. Decisions record
  `pid_error_source`. `video_runtime` exposes `--predict {0,0.5,1}` and
  `--feedforward-accel-sigma`, rejects prediction with `--pose-source frame`,
  and logs the predicted error per tick.
- Not changed: the runtime's trial locks (yaw Kp 8, pitch Kp 4/8, 0.2 rad/s
  cap, FF scale 0/0.5). Bench results favour higher Kp per latency, a
  higher cap for fast targets, and FF 0.5 + prediction 0.5; changing live
  runtime limits is a separate, explicit safety decision.

## 2026-09-27 — Feedforward sweep repeated on pitch-A (bare motor)

- Same settings as the yaw sweep on addr 2 (evidence
  `ffsweep_pitchA_20260927T075946Z`, 24 runs, 50 Hz). Mean RMS mrad at
  30/60/120 ms: PID only 18.0 / 26.7 / 52.9; FF 0.5 11.1 / 16.7 / 42.5;
  FF 0.5 + prediction 0.5 (sigma 8) 8.4 / 14.5 / 32.1; sigma 2
  9.8 / 17.2 / 37.5. Every cell is within 0.2 mrad of yaw.
- All bench sweeps so far (yaw and pitch-A) ran uncoupled, unloaded motors,
  so they qualify the controller, actuator, latency handling, and tooling,
  not gimbal mechanics. The yaw-plant model predicts bare pitch-A; the
  loaded-gimbal pitch fit overestimates its error by 25-35%. Pitch-specific
  Kp/feedforward tuning waits for the reassembled gimbal, using the same
  procedure.

## 2026-09-27 — Live V3 video/HIL: feedforward + prediction vs PID only

- New toolkit runners (outside the repo) `run_v3_live_ff_{host,jetson}.sh`,
  fast fixture `v3_live_ff_fast_fixture.yaml` (ellipse ~+/-0.12 rad yaw,
  +/-0.06 rad pitch, ~0.3 rad/s), bridge override
  `hil_live_ff_20260927.yaml` (Codex's with yaw/pitch rate limit 0.8).
  Jetson code from `claude_stage/8318cb2`; host streamer unchanged and
  hash-pinned. Motors uncoupled; yaw and pitch-A recentred into the bridge
  envelope with F4 relative moves first. HIL capture age median ~124 ms.
- Kp by the gain procedure at that latency: PID only 6.0; FF 0.5 +
  prediction 0.5 (Kalman sigma 2) 7.2; rate cap 0.8 rad/s.
- First ABBA set: both PID-only runs overshot the 0.15 rad travel envelope
  on the startup step and the controller's travel hold latched (93% of
  ticks held; it blocked inward motion too). Fixed in `8318cb2` (hold only
  commands that move farther outside). FF + prediction runs braked in time.
- Rerun ABBA (`v3_live_ff2_*`), all analyzer live checks passed, pitch-B
  stationary. RMS pointing error at capture (mrad), yaw / pitch:
  PID only 20.7 / 20.2 and 23.7 / 16.8; FF + prediction 12.7 / 10.3 and
  12.5 / 9.9 -> about -43% yaw, -46% pitch (simulation predicted -37%).
- Caveat (user review, same day): this used the simulator per-frame pose
  path and the renderer-mirroring `_relative_render_pose`, which treat a
  symptom (renderer pose from nominal F6 speeds). It validates the control
  idea live, not a clean V3 architecture. See the audit that follows.

## 2026-09-27 — Cleanup step 1: one measured F6 model, honest caps, no dither

Part of the user-approved V3 cleanup (remove symptom-treating code).

- `common/gimbal/mks_servo42_rs485.py` is the single actuator model:
  `F6_MEASURED_MICROSTEPS_PER_S` (1-8, 10 measured; 9 interpolated; >10
  extrapolated as n+1; 16x subdivision, timed F6 at 50 Hz),
  `f6_level_speed_rad_s`, `f6_level_for_rate` (nearest measured speed,
  never above the cap), `min_f6_speed_rad_s` (0.224 rad/s at 1:1).
  `_encode_speed_payload(..., max_rate_rad_s)` and
  `quantized_speed_rad_s(..., max_rate_rad_s)` use it. Previously the payload
  truncated nominal RPM, so level n ran ~(n+1) RPM and every consumer
  (render predictor, wire tracker, logs, sweeps, simulation) reported speeds
  up to ~2x too low.
- Caps now bound actual speed. The bridge, the step-count sweep tool, and
  the simulation `LoopConfig` refuse a cap below 0.224 rad/s instead of
  silently holding or overrunning it. Codex's toolkit override
  `hil_two_axis_20260926.yaml` (0.2 rad/s) is therefore refused by the new
  bridge; its runs remain historical. Repo configs (10 / 3 rad/s) unaffected.
- `SpeedCommandDither` deleted with its test, the `local_pid_trial`
  `--speed-dither` flag, and its uses in the two trajectory benchmarks.
- Sysid tools encode through the driver; `gimbal_response_sweep` manifests
  now carry `f6_speed_model: measured_2026_09_27` and record measured
  `encoded_rate`. Simulation stopgaps (`f6_one_rpm_gain`,
  `f6_measured_speed`, duplicate table) removed; `--rate-limit` replaces
  them. Earlier sweep reports stay as historical artifacts of the old model.
- Behaviour change for all F6 users (including Pi manual control): the sent
  level is the one whose measured speed is nearest the request, so requests
  below 0.112 rad/s encode as zero (was 0.105 with truncation).
- Hardware recheck on uncoupled yaw (evidence
  `f6_speed_recheck_yaw_*`): levels 1/2/4/8 measured 116.7/165.1/281/503
  steps/s against the table's 114/164/279/503 (within ~2.5%). Full suite 606
  passed.

## 2026-09-27 — Cleanup step 2: step-count control feedback

- `tools/serial_io_service.py` parses/validates 0x33 (4-byte signed
  microstep count). `jetson/gimbal_bridge.py` gains
  `gimbal.position_feedback: steps | encoder` (default steps) and
  `steps_per_rev` (3200). Steps are converted once, at ingestion, to
  encoder-count units (x 16384/3200), so limits, watchdogs, CamState, and the
  F5 planner are unchanged downstream. The bridge refuses to start if the
  serial schedule does not poll the chosen function for every controlled
  motor. In steps mode 0x31 is a cross-check: encoder vs step movement since
  first sample; divergence > 26 counts (~5 microsteps) logs an error.
- `configs/control.yaml` schedule: 0x33 every 40 ms (yaw, pitch-A), 100 ms
  (pitch-B), 0x31 every 1000 ms. Bus budget at 38400 baud with 50 Hz timed
  F6 writes for two motors: ~60%; 50 Hz polling needs a higher baud.
- Read-only hardware check (`stepfeedback_readonly_*`, snapshot 36d26b5,
  no actuation flag): 0x33 at 23.8/23.8/9.8 Hz, median bus 5.45 ms, low
  queue age; 0x31 at 1 Hz; no step/encoder divergence; 955/955 commands
  on the wire, one transient pitch-B timeout retried. The legacy absolute
  pitch A/B divergence warning fires (uncoupled motors parked apart); it is
  replaced by the mirrored relative pair check in step 4.

## 2026-09-27 — Cleanup step 3: no simulator pose paths; live result on clean V3

- Root causes behind the per-frame simulator pose workaround: (1) the bridge
  predicted a render pose from nominal F6 speeds (fixed in step 1), and (2)
  CamState was stamped with its publication time, which the V3 pose history
  used as the measurement time (up to one poll interval off). CamState now
  carries `pan_sample_monotonic_ns` / `tilt_sample_monotonic_ns`; the pose
  history interpolates each axis on its own measurement times and ignores
  republished samples.
- Deleted: bridge render predictor, wire-execution tracker, render fields in
  CamState and `gimbal.render_prediction`; V3 `pose_source` frame/render,
  simulator per-frame pose fields in PerceptionFrameV2, the runtime's
  simulator-pose decoding and `sim_capture_pose_hold`, and the
  renderer-mirroring prediction path. `--sim-camera-fov-y-deg` became the
  generic `--camera-fov-y-deg`. V2 reads measured pan/tilt only.
- Simulator (`pc/streamer.py`): `MeasuredPoseTimeline` stamps each measured
  sample in the host clock as receipt - (published - measured) and, after a
  warm-up, fixes the render delay D just above the p99 sample gap. Each
  frame shows the world at now - D with the interpolated measured pose and
  carries that capture time; frames without a bracketing sample stream but
  carry no truth. `--sim-total-latency-ms` replaces
  `--sim-perception-delay-ms`: truth is published exactly that long after
  capture; the run stops if D exceeds it.
- Analyzer: the all-ticks exact-pose rule became "at most 2% of tracking
  ticks without an aligned capture pose", reported as
  `unaligned_pose_ticks` (late samples fall back to raw-bearing PID).
- Toolkit (outside repo): `hil_live_v3clean_20260927.yaml` (step-count
  schedule; pitch encoders at 5 Hz only for the external pitch guard until
  step 4), `run_v3_live_clean_{host,jetson}.sh` (host streamer from the
  IDCS-v3 worktree, clean tree required; snapshot 8a49bb4 on Jetson).
- First canary with 60 ms total latency stopped itself: D measured 64.6 ms
  (pose 23.3 Hz, p99 gap ~60 ms from low-priority polls behind F6 writes).
  With 100 ms total: controller capture age median 143.5 ms (p5 112, p95
  180). The Kp procedure at that distribution gave PID-only 5.1 and
  FF 0.5 + prediction 0.5 (sigma 2) 5.9, predicting -36%.
- Matched ABBA (`v3_clean_{a1,b1,b2,a2}_*`, all analyzer checks passed,
  pitch-B stationary, 0-7 unaligned ticks): RMS pointing error at capture
  yaw / pitch mrad: PID only 25.2 / 16.7 and 21.7 / 14.7; FF + prediction
  11.3 / 9.6 and 11.5 / 11.2 -> yaw -51%, pitch -34%.

## 2026-09-27 — One mainline: V3 merged, V2 controller removed, versions dropped

- `v3-pid-hardware-verification` fast-forwarded into `main` (117f435..f38ec13).
  "V2" was never a separate branch: it is `main`'s history, and the deployed
  `IDCS-v2-runtime` checkout (db1fdfb) is an ancestor.
- Removed the V2 controller path: `jetson/control_runtime.py`,
  `qualified_controller_profile.py`, `shadow_rate_policy.py`, `los_kalman.py`,
  `sim_control_runtime.py`, `control_replay.py`, the pre-V2 `controller.py`,
  their trace/replay/parity tools, tests and fixtures, the `controller_v2`
  config section, the estimator-study configs and user units,
  `idcs-v2-controller.service`, and `scripts/run_jetson_with_gimbal.sh`.
  Historical artifacts and logs are kept. The PC-only simulated-mount closed
  loop went with `sim_control_runtime`; the simulator's stable mode is now
  video/detection-only.
- Version names dropped: `jetson/control_v3` -> `jetson/control`,
  `controller_v3` -> `controller` (the analyzer still reads `controller_v3`
  and `v3_video_*` from older traces), `idcs-v3-*` -> `idcs-*` units under
  `deploy/systemd/{jetson,rpi,pc}`, runtime checkout `IDCS-runtime`.
  DeepStream and the controller stack now run from the same checkout.

## 2026-09-27 — OpenGL scene and a no-hardware closed loop for the controller

- Shared scene `sim_scene_drone_ellipse_opengl.yaml`: V2's OpenGL renderer and
  mesh drone (daylight sky) on the fast ellipse used for feedforward work
  (~0.3 rad/s peak; the V2 OpenGL paths moved ~0.1 rad/s, too slow for FF to
  matter). ~43 fps at 1080p on the PC including readback. Mode overlays:
  `sim_mode_hil.yaml` (measured pose) and `sim_mode_simulated_mount.yaml`.
- Simulated mount: the streamer accepts `ControlIntent` on `--sim-control-sub`,
  honours the lease on the shared monotonic clock, and quantizes rates with the
  driver's measured F6 model and the gimbal rate caps before integrating; its
  CamState carries exact per-axis sample times. `tools.sim_panel` publishes the
  armed state, loopback only. Units: `idcs-sim.target` on the PC.
- Fixed: controller overlays now set `control.aim_mode: camera_center`. The repo
  default `laser_point` aims ~0.47 rad off axis for the 0.9 m simulated drone;
  the validated HIL runs set camera centre through the toolkit config, and the
  in-repo HIL overlay had missed it.
- Simulated-mount ABBA (40 s each, 100 ms truth latency, exact truth
  detections, per-unique-frame RMS after acquisition), yaw / pitch mrad:
  PID only (Kp 5.1) 28.4 / 19.0 and 28.2 / 19.6; FF 0.5 + predict 0.5
  (Kp 5.9) 12.3 / 10.2 and 12.6 / 10.2 -> yaw -56%, pitch -47%, consistent
  with the live HIL ABBA (-51% / -34%).

- Correction (same day): the stripped 60-degree scene replaced the V2 scene;
  restored. Both loops now render the V2 scene stack unchanged, and the
  controller aims with the shared 135x73 deg model and laser point (the aim
  override is gone). The V2 drone sits ~0.33 rad yaw / 0.26 rad pitch from
  home, so the controller travel cap is now configurable up to 1.0 rad
  (simulated mount 1.0, HIL 0.45 inside the bridge bench envelope).
- HUD: the control-status line, aim/parallax cue, feedforward indicator and
  command freshness were fed by the removed V2 controller. The controller now
  publishes `ControlDiagnostics` (PID/FF terms, final rates, normalized target
  and aim points) on `net.zmq_control_diagnostics`, and the HUD draws those
  cues from it when there is no `ControlCmd`. The UI no longer device-binds
  loopback subscriptions. Units `idcs-ui` (hardware loop) and `idcs-sim-ui`.
- The analyzer's live-trial calibration check now requires the controller's
  aim field of view (`aim_fov_deg`) to equal the simulated camera's
  (`sim_camera_fov_{x,y}_deg`); trial-era traces keep the 60-degree check.
- Simulated mount on the V2 scene (40 s each, same method), yaw / pitch mrad:
  FF + predict 10.5 / 9.6, PID only 17.0 / 13.1 (-38% / -27%); the V2 drone
  moves slower than the ellipse, so feedforward has less to correct.

## 2026-09-28 — Overnight checklist: homing, F6 command pattern, NvDCF, real detections

- **Firmware homing (MKS 91H) is unsafe as configured.** After `92H` set-zero,
  `91H 01` (coordinate homing) ran an endless search (no origin homing had
  been done, default endstop mode, no endstop) and `F7` did **not** stop it;
  only `F3 0` (de-energize) did. 90H homing parameters were written to both
  motors (10 RPM, CW, endstop limit off). Any firmware homing on the assembled
  mount needs de-energize as its stop path and a watchdog; mechanical-limit
  homing (94H mode 1, low current) is the candidate for geared pitch.
  Bench homing is software instead: `jetson/tools/home_axes.py` (F4 relative
  moves to the envelope centre by encoder counts; yaw came back from ~21
  turns to 27 counts). Every tuning hardware stage homes first.
- **F6 speed depends on the command re-send rate** (yaw, step count = encoder):
  level 1/3/5 run 0.106/0.316/0.527 rad/s sent once (exactly n RPM),
  0.115/0.327/0.535 re-sent every 0.2 s, 0.268/0.447/0.632 every 20 ms (the
  bridge's pattern, which the measured table reflects). Sysid refreshed every
  0.2 s, which is what the fitter absorbed as a 0.11 rad/s deadband and why
  the simulator over-predicted error by 27-42%. Sysid now commands like the
  bridge; fitted gains are 0.998-1.001 and the deadband 0.027 rad/s.
- **NvDCF (DeepStream 9.1).** It tracks independently when inference does not
  run (nvinfer interval=2 on NVIDIA's sample: 2,895 tracker-only objects), but
  when inference runs and misses it holds the target in shadow mode and
  reports nothing (every reported object had a detector box, at any
  minTrackerConfidence). With `outputShadowTracks: 1` the pipeline now reads
  NVDS_TRACKER_SHADOW_LIST_META and publishes the tracker's own estimate for
  missed frames, gated by `deepstream.shadow_tracks` (our min confidence and
  consecutive-miss limit; NvDCF's `age` is total target age). Against truth on
  the drone scene: coverage 71% -> 100%, 500/500 YOLO misses bridged,
  tracker-only box error 1.74 px median (YOLO's own: 1.84 px). Snapshots now
  also publish YOLO's pre-tracker detections.
- **Controller on real detections.** Verified RTP frame identity end to end
  (`deepstream.verified_rtp_headers`, streamer `--verified-rtp-headers`);
  `controller.local_clock` decides whether DeepStream's Jetson receipt times
  are kept. `tools/score_pointing_truth.py` scores true pointing error from the
  simulator's truth (validated against a truth-fed trace: 10.47/9.47 vs
  10.5/9.6 mrad). Simulated mount, 40 s each, yaw / pitch mrad:
  truth FF 10.7/9.0, truth PID 17.2/12.8; YOLO+NvDCF FF 9.4/76.1,
  YOLO+NvDCF PID 17.3/77.5. Yaw is unaffected by detection noise; pitch has a
  constant ~77 mrad aim bias from known-size ranging: YOLO's box is ~50% wider
  than the drone (54 vs 36 px), so range reads 1.73 m instead of 2.61 m and
  the laser-parallax aim point lands low (0.4 m mount offset x
  (1/1.73 - 1/2.61) = 78 mrad).
- Detection noise for the tuning sim (plan `detection`): 3.0 mrad per axis,
  28% misses (independent), from the YOLO-vs-truth measurement before shadow
  tracks; with shadow tracks the miss rate the controller sees is ~0.
- The target selector's TensorRT policy engine is now built on the Jetson
  (`prepare_jetson_runtime.sh`) instead of committed; it matches ONNX to ~1e-3.
- **First complete tuning run** (`tune-20260928-final`, all nine stages passed):
  fit gains 0.998-1.001 (theta gate 15 mrad, reasoned in the plan); limits
  0.8 rad/s at accel byte 10; sim under measured detection noise chose FF 0.5
  / predict 0.5 / sigma 2 (not the noise-free sigma 8); agreement median
  sim/hardware cost error 0.10-0.14 (was 0.27-0.42), sim-chosen Kp 0% worse on
  hardware; emitted yaw Kp 6.8, pitch Kp 6.0. Live A/B (HIL, ABBA, truth-scored)
  yaw / pitch mrad: PID 13.5/9.9 and 13.5/11.6, tuned 8.2/9.7 and 9.0/8.9 ->
  qualified_config.yaml.
- Fixed on the way: the bridge now anchors the step count to the encoder frame
  at startup (the step count does not follow F4/homing moves and had drifted
  ~30 rad, so envelope limits blocked every command); the video runtime uses a
  restart-safe intent sequence base (a restarted controller was ignored as
  out-of-order); live A/B homes first and records its own serial evidence.

## 2026-09-28 — Fast target, NvDCF stage cost

- `configs/sim_target_fast.yaml` (simulated-mount streamer): the V2 drone on a
  wider path with vertical legs at 1.0 m/s (V2: 0.25 m/s, level), up to
  ~0.35-0.4 rad/s apparent. Truth-fed controller (FF 0.5/0.5, sigma 2, Kp
  5.9): 23.7 / 13.7 mrad RMS yaw / pitch (slow path 10.5 / 9.6). Detection
  against truth (20 s): YOLO 61% of frames (misses up to 46 frames), NvDCF
  coverage 91% (349/452 misses bridged, tracker-only error 2.3 px median),
  stray tracker boxes in 101 frames; misses beyond the 30-frame coast limit
  stay uncovered. The shadow-track confidence gate (currently 0) should be
  tuned on this scene.
- The OSD/report "infer" stage runs from detector input to the metadata probe,
  i.e. YOLO + tracker. It is now split: live on the simulated-mount stream at
  60 fps, YOLO 12.1 ms mean, NvDCF 6.3 ms mean, combined 16.5 ms p50 /
  20.3 ms p95 (NvSORT on the unpaced sample replay: 1.7 ms vs NvDCF 8.2 ms).
- Motors de-energized (F3 0 acknowledged, enable status 0 on all three);
  Jetson gimbal services stopped; PC on the simulated-mount loop.
