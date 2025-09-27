# Camera Calibration and Known-Size Ranging Tuning

This guide walks through calibrating the camera intrinsics and tuning the
per-class real-world sizes used by the known-size ranging feature. It assumes you
have already enabled the `camera.intrinsics` and `camera.known_size_ranging`
configuration blocks described in `configs/dev.yaml`.

## 1. Calibrate Camera Intrinsics

Accurate focal lengths (`fx_px`, `fy_px`) are the foundation of distance
estimates. There are two supported workflows:

### Option A: Full Calibration (Recommended)
1. Print or display a checkerboard calibration target large enough to fill most
   of the frame. Keep the pattern flat and well lit.
2. Capture 15–20 images or frames of the checkerboard while varying viewpoint
   and distance. Ensure the entire grid stays in view and avoid motion blur.
3. Feed the collected images into a calibration tool such as OpenCV's
   `calibrateCamera` to solve for the intrinsic matrix. The resulting matrix
   provides pixel focal lengths and principal point offsets.
4. Copy the solved values into `camera.intrinsics` using the `source: direct`
   mode:
   ```yaml
   camera:
     intrinsics:
       source: direct
       fx_px: 912.4
       fy_px: 908.9
       cx_px: 640.1
       cy_px: 358.6
   ```
5. Record the calibration date in your team notes so you can repeat the process
   if the optical stack changes.

### Option B: Derive from Field of View (Quick Start)
1. Confirm the video resolution configured under `video.width`/`video.height`.
2. Measure or obtain the camera's horizontal and vertical field of view (FOV)
   from its datasheet. Avoid marketing values if possible and prefer calibrated
   specs.
3. Populate `camera.intrinsics` with `source: fov` so the shared helper computes
   focal lengths from the FOV and frame size:
   ```yaml
   camera:
     intrinsics:
       source: fov
       fov_deg:
         h: 78.0
         v: 47.5
       # cx_px/cy_px default to frame center when omitted
   ```
4. Expect a scale bias when using catalog FOV values. You can tune the effective
   FOV angles later if you observe consistent distance bias across targets.

## 2. Define Canonical Object Sizes

Known-size ranging requires a lookup table that maps detection classes to their
real-world dimensions:

1. Identify the classes where ranging is valuable (for example, `person`,
   `cone`, `stop_sign`).
2. Measure representative samples with a tape measure or refer to authoritative
   specs (DOT signage tables, manufacturer datasheets, etc.). Capture both height
   and width if you plan to average dimensions.
3. Update `camera.known_size_ranging.class_sizes_m` with the measured sizes in
   meters and ensure the YOLO class-label map references the same names:
   ```yaml
   camera:
     known_size_ranging:
       enabled: true
       dimension: height  # or width / average
       min_pixels: 40
       class_sizes_m:
         person: 1.70
         cone: 0.30
       ema_alpha: 0.4
     yolo:
       class_labels:
         "0": person
         "1": cone
   ```
4. Leave classes unset if their size varies widely or the detector is unreliable
   for that category. Detections whose label is missing from
   `class_sizes_m` will be skipped by the ranging pipeline.

## 3. Field Validation and Bias Adjustment

Use a short validation exercise to verify the ranging accuracy:

1. Place a calibration target whose real size matches one of your configured
   classes at several measured distances (e.g., 2 m, 4 m, 8 m). Align it upright
   so the detector sees the canonical orientation.
2. Run the Jetson server and PC UI so you can see live annotations and log the
   detection stream. Capture at least 10–15 frames per distance.
3. For each distance:
   - Record the instantaneous `box.distance_m` and the smoothed
     `selected_target.distance_ema_m` (if available) from the published
     detection messages.
   - Note the pixel height/width reported in telemetry. Ensure boxes exceed the
     configured `min_pixels` threshold.
4. Compute the mean error (bias) and standard deviation (jitter) for each test
   distance. A spreadsheet or notebook makes this quick.
5. Apply corrections based on the error pattern:
   - **Consistent bias at all ranges:** adjust `fx_px`/`fy_px` (or the FOV values
     if using `source: fov`) by the observed scale factor, or tweak the canonical
     class size.
   - **Bias grows with distance:** revisit the intrinsic calibration—FOV-derived
     estimates may need better measurements, or the chessboard solve could have
     insufficient coverage.
   - **High jitter but low bias:** lower `ema_alpha` to smooth more aggressively
     or raise `min_pixels` to ignore noisy tiny boxes.
6. Repeat the validation steps until bias stays within your acceptable error
   window (e.g., ±10%) and jitter matches control-loop needs.

## 4. Maintenance Checklist

- Re-run the calibration whenever you change optics, resolution, or detector
  training that materially shifts box geometry.
- Version-control your measured class sizes alongside `configs/dev.yaml` so
  deployments stay in sync.
- Document tuned parameters and test results in your internal runbooks so other
  operators can replicate the setup.
