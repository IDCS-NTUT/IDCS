# Live configuration implementation — 2026-10-17

Branch `codex/live-config`, task baseline `245d103` (`lab/config-cleanup-reviewed`).
No lab hosts, runtime network peers, serial/RS485, motor bridge, HIL or motors were
contacted. No deployment, activation, push, tag, PR or history rewrite was run.
Local tests use fakes and deterministic observations. Generated logs, merged
maps and fold examples are ignored under `.task-tmp/`, not committed.

## Design as built

- Explicit per-section field contract in `common/config_schema.py` and
  `common/config_fields.json`; existing typed validators are retained. Unknown
  paths, bad types, nonfinite values, invalid combinations and bridge-cap
  bypasses fail closed.
- `common/live_config.py` owns the final host overlay, atomic writes, lock/CAS,
  append-only journal, unit stacks parsed from systemd templates, per-unit
  receipts, host-local REQ/REP and update PUB/SUB, with durable-file recovery.
- `tools/idcs_config.py` implements get/set/unset/diff/revert/clear/status/fold,
  local full-stack preflight and agent revalidation. Folding prints a real
  applicable patch; a sample passed `git apply --check` without being applied.
- `jetson.operator_agent` accepts configuration on its existing REP channel and
  local IPC; `pc.mode_agent` adds local IPC while keeping its outbound mode poll.
  Restart planning considers effective changed values and conservative consumer
  sections. Only active software units are try-restarted, with one transaction
  for affected active SIM dependencies. Physical/HIL/current-agent units return
  `cannot_restart`. Startup/schema-only bridge/serial edits add no live loader,
  subscription, socket or device access.
- Controller tick swaps preserve core/PID/estimator state and sequence history;
  idle caps require fresh manual-panel proof and no preceding live motion.
  Safe-point changes happen before assembling the next observation. A startup
  snapshot is retained through initialization, preventing revision races.
- DeepStream swaps shadow/stitch policies in the metadata probe, retaining
  stitch history for parameter changes. Each queued selector request carries
  desired settings and revision so latest-only coalescing cannot drop an
  update. The worker reconfigures selection/timing without reloading the model.
  Old-revision results are discarded. Probe and worker digests are separate.
- Controller diagnostics/health/ready/recorder records, DeepStream health/ready/
  V2 recorder snapshots, agent state, and existing startup outputs expose
  provenance. Pi PanelState carries its startup digest. Constants restore the
  removed authored values for existing typed readers. The swarm model metadata
  pins model and external tensor bytes to normalization; no model was changed.
- `tools/deployment.py` remains a standalone installer. A nonempty host overlay
  blocks stage unless an explicit folded/discard choice is recorded, including
  exact file hash and contents. Verification retains the decision. Stage never
  clears runtime state or claims the patch has been folded.

## Full key-to-tier contract

See [the complete table](live_configuration_keys.md) for every field, type,
domain/default and tier. The table contains **834** entries including
structural maps/list members: **730 restart, 37 fixed, 60 live, 7 idle**. No removed cleanup key is restored.
Optional supported controller endpoint/manual/FOV settings and tracker-path
overrides are explicitly declared even where no committed overlay writes them.

## Keys moved to constants or model metadata

Values below are exactly those authored in baseline base files. Lists count as
one leaf in the merged-map audit. There are nine groups, 29 authored leaves.

| Former YAML path | Fixed value |
|---|---|
| `camera.known_size_ranging.class_aspect_ratio_limits` | `{"drone": {"max": 2.0, "min": 0.4}, "person": {"max": 4.5, "min": 1.2}}` |
| `camera.known_size_ranging.min_pixels` | `10.0` |
| `camera.known_size_ranging.ema_alpha` | `0.9` |
| `camera.known_size_ranging.quality` | `{"edge_margin_px": 2.0, "hold_s": 0.25, "jump_slack_fraction": 0.1, "max_fractional_change_per_s": 1.0}` |
| `laser.render` | `{"beam_length_m": 5.0, "color_bgr": [0, 0, 255], "hit_tolerance_px": 30.0, "thickness_px": 2}` |
| `gimbal.steps_per_rev` | `3200` |
| `gimbal.counts_per_rev` | `16384` |
| `rpi.runtime_control.indicators` | `{"control_stale_s": 0.5, "fire_control_off_delay_s": 1.0, "hardware_stale_s": 0.5, "perception_stale_s": 0.5, "safety_off_delay_s": 1.0}` |
| `swarm_eval.learned_model.normalization` | `{"max_angle_rad": 0.8, "max_breakthrough_time_s": 40.0, "max_closing_speed_m_s": 12.0, "max_damage_weight": 10.0, "max_distance_m": 80.0, "max_time_to_engage_s": 8.0, "max_track_age_s": 5.0, "max_track_observations": 10}` |

The checkpoint's own normalization matches these values (read safely with
`pickletools`, without executing/loading the checkpoint). Source metadata is
`assets/models/swarm/normalization.json`; engine sidecars are created only from
reviewed builds and checked on load. An existing unbound engine will use the
configured planner fallback until its provenance is supplied. This is a
required target preparation step, not evidence that the engine was qualified.

## Merged-config comparison

`python -m tools.audit_live_config` reconstructs the baseline with local
`git show`, validates all 21 systemd stacks, and compares complete resolved maps
with fixed policy materialized. **Zero effective value changes, additions or
removals in every row.** The raw authored maps only remove fixed leaves; bridge
and serial readers use the same named motor-format values. Recorder profiles
contain none of these leaves. Full maps and SHA-256 hashes are in
`.task-tmp/live-config-audit/`; the table below is the requested summary.

| Unit | Fixed authored leaves removed | Effective changes | Digest prefix |
|---|---:|---:|---|
| `jetson/idcs-bridge.service` | 29 | 0 | `058803f5f275` |
| `jetson/idcs-controller-camera.service` | 29 | 0 | `f0e7dad6bad2` |
| `jetson/idcs-controller.service` | 29 | 0 | `2b747c6de8c1` |
| `jetson/idcs-deepstream-camera.service` | 29 | 0 | `717a6ccfcf19` |
| `jetson/idcs-deepstream-video.service` | 29 | 0 | `1c207ad56e56` |
| `jetson/idcs-operator-agent.service` | 29 | 0 | `1c207ad56e56` |
| `jetson/idcs-recorder.service` | 0 | 0 | `0765c08771a0` |
| `jetson/idcs-serial.service` | 29 | 0 | `02b5e8c58262` |
| `pc/idcs-camera-ui.service` | 29 | 0 | `05b461a27a88` |
| `pc/idcs-hil-streamer.service` | 29 | 0 | `36548ae0f171` |
| `pc/idcs-hud-relay.service` | 29 | 0 | `1c207ad56e56` |
| `pc/idcs-pc-mode-agent.service` | 29 | 0 | `1c207ad56e56` |
| `pc/idcs-sim-controller.service` | 29 | 0 | `306b970df456` |
| `pc/idcs-sim-recorder.service` | 0 | 0 | `4ea5ca3f589d` |
| `pc/idcs-sim-streamer.service` | 29 | 0 | `e9d1b5dbdb51` |
| `pc/idcs-sim-swarm-controller.service` | 29 | 0 | `c2ba23c379ac` |
| `pc/idcs-sim-swarm-streamer.service` | 29 | 0 | `df549a52288b` |
| `pc/idcs-sim-ui.service` | 29 | 0 | `ff9b11acb599` |
| `pc/idcs-ui.service` | 29 | 0 | `4efcc9604b40` |
| `rpi/idcs-manual.service` | 29 | 0 | `1c207ad56e56` |
| `rpi/idcs-operator-display.service` | 29 | 0 | `1c207ad56e56` |

## Verification

Task-local dependencies: `python -m pip install --target .task-tmp/deps
Pillow==13.0.0 trimesh==5.1.1`; pip also installed numpy 2.5.4 in that isolated
directory. No global dependencies changed. For runs needing those packages:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) '.task-tmp/deps')
python -m pytest -q --continue-on-collection-errors --basetemp=.task-tmp/pytest-full-verified
```

Final full suite: **1,076 passed, 50 skipped, 1 failed, 4 collection errors,
20 subtests passed** (63.94 s). The failure is unchanged
`test_validate_gimbal_fit.py::test_existing_low_range_fit_is_not_offline_control_qualified`,
because the pre-existing external `artifacts/gimbal_fit/.../fit_report_external_selected.json`
is absent from this clone. Collection errors are the four streamer modules
(`test_streamer_frame_ids`, `test_streamer_gst_video_writer`,
`test_streamer_hil_pose`, `test_streamer_sim_mount`) requiring unavailable `gi`/
GStreamer on Windows. These were reported, not silently skipped or changed.

The final targeted run
`python -m pytest -q tests/test_live_config.py tests/test_control_video_runtime_check.py
 tests/test_deepstream_target_selection.py tests/test_panel_indicators.py
 --basetemp=.task-tmp/pytest-last-contract` passed **113 tests**. There are
**51 dedicated live-config contract cases**. A further
`python -m pytest -q tests/test_live_config.py tests/test_deepstream_pipeline_split.py
 tests/test_deepstream_snapshot_transport.py --basetemp=.task-tmp/pytest-probe-owner`
passed **59 tests**, including an actual callback test with detector/metadata
probes on different fake threads. Local merged-stack audit:
**21 stacks, zero effective changes**. Syntax compilation and `git diff --check`
pass. Full logs remain in `.task-tmp/`.

Coverage includes schema inventory/unknown paths, all stacks, gain application
without sequence reset, pending caps, fail-closed idle proof, invalid update and
callback rejection, affected active-only restarts, SIM dependency grouping,
protected units, HIL/remote-intent conversion refusal, durable restart overlay,
unset/revert/clear, CAS/lock, per-unit pending/apply receipts and journal,
recorded digest serialization, fake REQ/REP/PUB, pure DeepStream probe updates,
model-byte/normalization binding, startup races, fold output, and deployment
refusal/decision records. Existing GPIO/safety tests remain in the full suite.

## Needs Jetson (not run here)

Prerequisites: reviewer-approved code/environment, no competing GPU/video
owner, valid reviewed H.264 clip, model assets and engine metadata. These are
software/perception checks and do not authorize any controller, bridge, serial,
HIL or motor activation. Do not start a production agent merely to run the
isolated replay check.

1. Run the same fake contracts with target dependencies:

   ```sh
   bin/python3 -m pytest -q tests/test_live_config.py tests/test_deepstream_runtime.py \
     tests/test_deepstream_target_selection.py tests/test_operator_agent.py
   bin/python3 -m tools.swarm_model_metadata assets/models/swarm/swarm_policy.engine
   sudo -n -l
   ```

   Pass: tests pass, engine binding verifies, and the existing sudoers policy
   permits `systemctl try-restart` for the allowed passive units. Missing
   permission must produce `restart_failed`; there is no stop/start fallback or
   escalation to another unit. The sudoers policy was not edited here.

2. Passive file replay with isolated paths and no network command endpoint.
   Set `CLIP` to an existing reviewed clip lasting at least 20 s. Use the
   repository's built nvinfer profile/custom parser and approved model assets.

   ```sh
   export IDCS_LIVE_CONFIG=/tmp/idcs-live-canary/live.yaml
   mkdir -p /tmp/idcs-live-canary
   printf '{}\n' > "$IDCS_LIVE_CONFIG"
   args=()
   for f in configs/base/*.yaml; do args+=(--idcs-config "$f"); done
   timeout --signal=INT 25s bin/python3 -m jetson.deepstream.verify_pipeline "$CLIP" \
     --paced --tracker nvdcf --tracker-config configs/deepstream/tracker_nvdcf.yml \
     --nvinfer-config configs/deepstream/nvinfer_yolo26s_736_drone_person_smoke.txt \
     --shadow-min-confidence 0.3 --shadow-max-age 30 --id-stitch-max-gap 45 \
     --target-selection --snapshot-result-bind ipc:///tmp/idcs-live-canary/snapshots.sock \
     --health-file /tmp/idcs-live-canary/health.json \
     --report /tmp/idcs-live-canary/report.json "${args[@]}"
   ```

   In a second local terminal with the same environment, while frames run:

   ```sh
   bin/python3 -c 'from common.live_config import atomic_write,live_path; atomic_write(live_path(), "deepstream:\n  shadow_tracks:\n    min_confidence: 0.6\n  id_stitching:\n    max_gap_frames: 20\n")'
   cat /tmp/idcs-live-canary/health.json
   ```

   Pass: frames continue with one GPU/video owner; health/receipt digest changes
   to the validated desired revision; selector digest converges, pending worker
   acknowledgement clears, shadow/stitch values change in the probe without
   replacing its track history; exit/cleanup completes normally. No command
   transport, controller or motor service is involved. This checks hot delivery,
   not learned detector quality or hardware acceptance. Windows cannot run it.

3. After the reviewer has an authorized passive agent session, qualify real
   restart permissions and process ordering using the local config CLI:

   ```sh
   systemctl show idcs-deepstream-video.service idcs-deepstream-camera.service \
     idcs-controller.service idcs-controller-camera.service idcs-bridge.service \
     idcs-serial.service -p Id -p MainPID -p ActiveState
   bin/python3 -m tools.idcs_config --host jetson set deepstream.return_iframe_interval 15
   bin/python3 -m tools.idcs_config --host jetson status
   bin/python3 -m tools.idcs_config --host jetson unset deepstream.return_iframe_interval
   ```

   Pass: only the already-active affected passive DeepStream unit restarts per
   mutation. Inactive units remain inactive and all controller/bridge/serial
   PIDs/states stay unchanged. CLI reports denied permissions/unavailable units
   explicitly. Do not run this check until that passive software restart is
   separately authorized by the reviewer. This task ran its fake equivalent.

## Deviations and open decisions

- Bridge and serial are deliberately release-only, including startup. Initial
  automatic review rejected a proposed integration because it included them in
  overlay loading. The revised implementation gives them schema/constants only,
  with no overlay/subscription. Physical/HIL/current-agent restarts, including
  the camera controller even when shadow, require manual reviewed handling.
- PC had no inbound REQ/REP channel to extend. Its new endpoint is host-local
  IPC, preserving the existing no-inbound-PC-network architecture.
- Live/idle tiers describe capability by owning consumer. Legacy constructor
  consumers use a conservative affected-unit restart, explicitly reported.
  Learned-model lifecycle/scheduling, intrinsics, defended-asset position and
  enabling/changing operational modes remain restart tier.
- A host-wide overlay cannot change the software/HIL motion source or move a PC
  software controller's intent output off loopback. Those require a separately
  reviewed release, rather than an automatic restart of an active SIM session.
- Native nvinfer/label artifacts are restart-only opaque inputs; their internals
  were not converted into hot YAML keys or a new nvinfer parser (phase 2 remains
  deferred). The native tracker YAML fields are inventoried, but native-profile
  edits still follow release review.
- Worker application is asynchronous: the probe and worker expose separate
  applied digests while acknowledgement is pending. A worker fault drops old
  decisions and reports an error; there is no blocking GPU-probe round trip.
- No Pi config-command agent was requested/added. Pi software loads a manually
  provided local overlay on startup, with fixed safety policy; PC/Jetson agents
  cannot manage Pi units. A future Pi command path needs an explicit design.
- Stage records folded/discard intent; it does not clear the file or assert
  semantic folding. Operator review/clear/revalidation is required before
  activation. The standalone script has the explicit option; the multi-host
  shell wrapper does not silently choose it.
- Journal and overlay are separate durable files. Incomplete write-ahead
  transactions require digest/receipt review and explicit recovery; automatic
  revert includes completed transactions only. Target filesystem power-loss
  behavior and IPC permission/lifecycle handling need Linux qualification.
- Optional provenance fields extend strict V2 snapshots and diagnostics.
  Consumers must receive the reviewed code release together; older strict
  consumers may reject the new fields during a mixed-version rollout.

No physical bench test is needed to qualify the fake contracts. Wiring repair,
every RS485 motor responding, and all existing bench authorization gates remain
prerequisites for any later motor/HIL work.
