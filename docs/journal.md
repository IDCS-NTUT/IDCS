# Development Journal

Chronological evidence, decisions and test results for IDCS, newest entries at
the end. Each entry is dated and says what was run, on which host, with which
commit or release, and where its artifacts are kept. Old paths and results can
be superseded; do not execute historical commands as current procedures.

This journal continues the
[DeepStream and low-latency control migration journal](deepstream_migration_journal.md),
which covers 2026-08 to 2026-10-19 (DeepStream migration, controller V3,
deployments up to deploy-99, camera ROI visual lock, and the 2026-10-19
performance work). That file is closed; link to its entries by date.

State at the hand-over (2026-10-20):

- `deploy-99` (`11cf19e`) is running on PC, Jetson and Pi; source mode Camera;
  physical bridge/serial/controller/HIL units inactive (hardware hold).
- Visual lock: registered CUDA input is the default; live camera processing
  p50 12.9 ms (48 fps) with a static seed. It is not run by any deployed unit.
- Open performance work: the single GStreamer streaming thread runs tracker,
  overlay and conversions serially (~6.8 ms beyond the tracker per frame);
  VPI LK round trips cost ~7.9 ms per frame.
