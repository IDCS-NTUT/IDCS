# DeepStream passive-video cutover

The DeepStream service replaces only the legacy CPU/OpenCV video path. Do not
run it together with `jetson.server`: both own the RTP, header, and result
ports. This runbook does not authorize controller, gimbal, serial, or physical
actuation changes.

1. Run `python -m jetson.deepstream.runtime --check` on Jetson. For a
   Jetson-local camera, replace the final runtime profile with
   `configs/deepstream_argus_runtime.yaml`. Its DetectionMsg timing is
   same-host and headerless, so it is not PC-to-Jetson latency evidence.
2. Run a bounded passive canary with `--duration-s` and `--report` while the
   PC streamer uses `--deepstream-shadow`; retain the report and PC logs.
3. Run `pc.metadata_monitor` during the canary, then evaluate both ends with
   `python -m jetson.deepstream.acceptance REPORT.json --receiver-report PC_REPORT.json`.
   Use `--min-steady-fps 60` only after PC hardware encoding is safe; CPU/x264
   is a functional canary, not end-to-end 60-FPS acceptance.
4. Inspect return video and DetectionMsg continuity, then run the same
   configuration for a reconnect/soak interval before enabling the systemd
   service.
5. To cut over, stop the legacy server, enable/start
   `idcs-deepstream-video.service`, and monitor its report/logs.

Rollback is immediate: stop/disable `idcs-deepstream-video.service`, restore
the legacy server, and confirm only one process owns the RTP/header/result
ports. Keep the legacy path until detector, tracker, UI, and hardware-encoder
acceptance gates are all recorded.
