# V2 DeepStream deployment

The V2 runtime owns RTP ingest, frame-header correlation, detector/tracker
metadata, `PerceptionSnapshotV2`, GPU OSD, and the H.264 return stream.

## Passive validation

1. Run `python -m jetson.deepstream.runtime --check`.
2. Start a bounded runtime with `--duration-s` and `--report`.
3. Run `pc.metadata_monitor` and retain its V2 receiver report.
4. Evaluate both ends with:

   ```bash
   python -m jetson.deepstream.acceptance REPORT.json \
     --receiver-report PC_REPORT.json
   ```

5. Confirm frames, GPU OSD, encoded H.264 buffers, V2 publications, monotonic
   frame identity, and expected return/display FPS.

The passive runtime opens no control or serial socket. Argus input is allowed
to publish headerless same-host timing; it is not PC-to-Jetson latency proof.

## Controller deployment

The video controller (`jetson.control.video_runtime`) takes all policy from
the validated `controller` config section; `mode: shadow` never publishes.
Each unit in `deploy/systemd/jetson` (`idcs-serial`, `idcs-bridge`,
`idcs-controller`, grouped by `idcs-hil.target`) runs `--check` before start.
Authority is fail-safe: absent or stale manual state yields zero-rate intents.
Installing or enabling units is a separate operational act; repository tests
do not enable services or touch serial hardware.

## Rollback

Rollback means checking out the prior git revision. Do not run old and new
runtimes together because they use the same RTP/header/metadata ports. There
is no in-tree legacy runtime or dual-publish mode after the V2 cutover.
