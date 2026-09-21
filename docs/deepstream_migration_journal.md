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
