# Configuration cleanup — 2026-10-17

Base: `lab/deploy-98-merged`, `0bdff18`. Branch: `codex/config-dead-removal`.
All work was local. No lab-host, serial, RS485, bridge activation, hardware/HIL,
deployment, push, tag or PR operation was performed.

## Result and exceptions

The dead configuration and obsolete code paths listed below were removed.
Every audited unit's merged map has **only removals**, with no changed values,
added keys or changed lists. Reader-level comparisons preserve aiming geometry,
laser policy, image signs, limits, planner configuration, controller clocks,
bridge caps and active streamer scenes. The shared GPIO/ADC definitions are
textually unchanged. This is software/config evidence, not hardware acceptance.

The audit disproved three proposed dead-key classifications. These remain:
`control.sign_convention`, `sim.freeze_frame`, and
`controller.local_clock` / `controller.source_clock`. The required base
`controller.rate_limit_rad_s` also remains duplicated in tuning output; the
strict parser requires the base key for untuned shadow startup. Achieving a
literal single source for that cap needs a separately approved configuration
contract change or a change to tuner output. Neither was made here.

## Item-by-item disposition

| Item | Disposition and evidence |
|---|---|
| 1. MPC/controller/debug | Removed all `Mpc*Config` and debug-overlay types/parsers/exports in `common/control.py`; the unused `MpcAxisDiagnostic`, `ControlCmd.mpc` and `ControlCmd.controller_mode`; PC overlay and `--mpc-overlay`; the legacy Pi overlay; trace-analysis MPC status/limits code; and `tests/test_ui_overlay.py`. No runtime producer sets those schema fields. The HUD compatibility status uses its existing `CONTROL` fallback. Active `ControlIntent`/diagnostic wire models are unchanged. |
| 2. Negotiation | Removed `control.negotiation`; no deployed reader or import-closure consumer. |
| 3. Old control fields | Removed `mode`, `pid`, `target_selector`, `smooth_px_alpha`, `motion_vel_alpha`, `reinit_on_lost`, `deadband_px`, `lost_target_timeout_ms`, `loop_hz`, their typed fields/compatibility gain properties, and V2 scene copies. Production pixel-delta callers always disabled the old deadband. Kept `aim_mode`, laser policy, and the paired rate/acceleration interfaces consumed by `pc.streamer` and `SwarmPlannerSettings`. Caps now come directly from `controller.*`. Offline trajectory benchmarks use local PID defaults (Kp 1, Ki/Kd 0.5 per axis) and a local 50 Hz default. **Kept `sign_convention`:** `pixel_delta` applies its signs in both `common.aiming` and `jetson.swarm_planner`. |
| 4. Pi camera | Removed `camera.libcamera`; no deployed reader. Camera intrinsics/ranging and actual DeepStream Argus settings remain. |
| 5. Serial defaults | Removed `gimbal.backend`, `serial_port`, `baudrate`, `timeout`, `retries`, `use_group_writes`. The deployed serial service uses its existing CLI defaults; its port remains `/dev/ttyCH341USB0`, baud 38400. The old standalone Pi path was their remaining config reader. |
| 6. Pi standalone control | Removed `rpi/manual_control.py`'s old parser/main, serial/motor construction and associated helpers/imports. Kept `ManualSwitchIO`, `map_value_to_rate`, `read_adc`, `resolve_gpio_config` and their dependency closure. Their function/class source is unchanged from the base. Fake GPIO/ADC tests pass. |
| 7. Pi return video | Removed `rpi/return_video.py` and `rpi.return_video` settings. No unit or production import references it; `rpi.operator_display` is the managed replacement. There was no dedicated Pi return-video test module. PC return-video tests remain. |
| 8. Controller clocks | **Kept.** `ControlRuntimeConfig.from_config` directly reads both keys and validates their domain relationship. `video_runtime` uses `source_clock` for observation domains and `local_clock` for upstream receipt handling. Removing them changes SIM/local-camera semantics. Both overlay values and all operator-mode source-clock keys are unchanged in the final tree. |
| 9. SIM frame options | Removed unread `sim.render_fps`. **Kept `sim.freeze_frame`:** `pc.streamer.open_source` reads it and its source uses it to freeze frames. |
| 10. Base scene | Removed base `sim.scene`, including targets/buildings/cubes and planner-evaluation defaults. Every managed streamer loads `sim/v2_scene.yaml`; its lists replace the base lists. `sim/swarm.yaml` supplies all former planner-evaluation keys. Other stacks do not instantiate this scene. Active static/swarm scenes compare identically; the unused planner-evaluation mapping disappears from static-target stacks. |
| 11. Unused files | Removed the five config files below after whole-repository reference searches. Four had no references; `camera_open_loop.yaml` had only its README listing. Kept the documented/manual and smoke/pipeline files below. |
| 12. Config sync | Removed ZMQ server/client/startup handshake, deadlines/retries, endpoint resolver, marker/lock machinery, sync-only metadata serialization/comparison, config-writing helper and exports/imports. `common/config_sync.py` retains local `expand_config_paths`, `load_merged_config`, `merge_config_maps`, `parse_config_text`, `read_snapshot`, snapshot types and video-profile helpers at their existing import paths. `tests/test_config_merge_semantics.py` is unchanged and passes. |

## Duplicate sources

| Setting | Final source / result |
|---|---|
| Aiming FOV/focal length | `camera.intrinsics`; removed `control.fov_deg` and `control.fx_fy_from_fov`. 135/73 degree calibration and focal lengths are unchanged on every unit stack. The explicit simulated vertical-FOV override builds a `CameraIntrinsics` value, preserving aspect-derived focal lengths without rewriting config maps. |
| Bridge yaw/pitch rate caps | `configs/bench/tuned.yaml`, both 0.8 rad/s. Removed stale base caps and redundant uncoupled copies. Every deployed bridge loads tuned; its effective cap is identical. This uses the task's tuner exception rather than editing generated output or changing raw values in passive stacks. Missing raw cap keys on non-bridge units were not their settings. |
| Controller acceleration cap | `configs/bench/tuned.yaml`, 3.5 rad/s²; removed base copy. Untuned parser defaults already supply 3.5, so effective values remain identical. |
| Controller rate cap | **Retained duplicate:** base and generated tuned remain 0.8 rad/s. `ControlRuntimeConfig` requires this key; base-only shadow startup is supported and tested. Removing the base copy would break that path, and changing the required-key check or tuner format was outside this task's no-behaviour/no-safety-change constraints. |

## Removed files and release gate

- `configs/deepstream/nvinfer_small_960_drone_person.txt`
- `configs/deepstream/nvinfer_yolo26n_1088_drone_person_smoke.txt`
- `configs/sim/selection_trial.yaml`
- `configs/sim/visual_lock_fast_trial.yaml`
- `configs/controller/camera_open_loop.yaml`

Also removed: `rpi/return_video.py` and `tests/test_ui_overlay.py`.
No model, recording, trial-data, rollback or historical artifact files were removed.
The managed deployment removal gate is unchanged and fails closed on removed
config files. **The next release requires explicit review of these five removals.**

Retained manual/compatibility files:

- `configs/controller/pid_only.yaml`: documented A/B baseline.
- `configs/sim/visual_lock_trial.yaml`: command in `docs/visual_lock.md`.
- `configs/deepstream/nvinfer_yolo26n_960.txt` and `labels_coco.txt`:
  `scripts/run_deepstream_file_smoke.sh`, pipeline default and label reference.

## Merged-config proof

Reproduce from the repository root with:

```powershell
python docs/config_dead_removal_20261017/audit.py
```

This reconstructs the base configs with local `git show`, extracts each unit's
ordered CLI stack (including matching preflight arguments), and loads the final
stack through the strict bundle loader. It covers all **16 explicit stacks**
and **five implicit `configs/base` stacks**. `pc/idcs-sim-panel.service` launches
`tools.sim_panel`, which loads no config, so it has no merged map. No service is
started by the audit.

Full before/after maps and SHA-256 digests:
[before](config_dead_removal_20261017/merged-before.json),
[after](config_dead_removal_20261017/merged-after.json).
Exact removed paths and original values are in
[differences](config_dead_removal_20261017/differences.json).
[Runtime equivalence](config_dead_removal_20261017/runtime-equivalence.json)
records all compared values. The initial snapshots taken before editing match
the reconstructed original stacks.

The following groups enumerate every removed path. Removed mapping entries are
counted once, with their complete original subtrees in the JSON evidence.

**D (23 common removals):** `camera.libcamera`, `control.controller`, `control.deadband_px`, `control.debug_overlay`, `control.fov_deg`, `control.fx_fy_from_fov`, `control.loop_hz`, `control.lost_target_timeout_ms`, `control.mode`, `control.motion_vel_alpha`, `control.mpc`, `control.negotiation`, `control.pid`, `control.reinit_on_lost`, `control.smooth_px_alpha`, `control.target_selector`, `gimbal.backend`, `gimbal.baudrate`, `gimbal.retries`, `gimbal.serial_port`, `gimbal.timeout`, `gimbal.use_group_writes`, `rpi.return_video`.

**A:** `controller.accel_limit_rad_s2` (redundant configured default).
**G:** `gimbal.yaw_rate_limit_rad_s`, `gimbal.pitch_rate_limit_rad_s`
(unused by these unit entry points; the bridge retains tuned values).
**S:** `sim.scene` (not used by these entry points).
**R:** `sim.render_fps`.
**P:** `sim.scene.planner_eval` (inactive static-target scene).
D includes both dead keys and the explicitly requested duplicate control FOV keys.
There are **zero additions and zero value changes** in every row.

Paths in the stack column are relative to `configs/`; every overlay follows
`base` in the shown order. Unit paths are relative to `deploy/systemd/`.

| Unit | Ordered config stack | Removed paths | Count |
|---|---|---|---:|
| `jetson/idcs-bridge.service` | `base` + `bench/uncoupled.yaml` + `bench/tuned.yaml` | D, S | 24 |
| `jetson/idcs-controller-camera.service` | `base` + `bench/uncoupled.yaml` + `bench/tuned.yaml` + `controller/local_camera.yaml` | D, S | 24 |
| `jetson/idcs-controller.service` | `base` + `bench/uncoupled.yaml` + `controller/hil.yaml` + `bench/tuned.yaml` | D, S | 24 |
| `jetson/idcs-deepstream-camera.service` | `base` + `controller/local_camera.yaml` | A, D, G, S | 27 |
| `jetson/idcs-deepstream-video.service` | `base` (CLI default) | A, D, G, S | 27 |
| `jetson/idcs-operator-agent.service` | `base` (CLI default) | A, D, G, S | 27 |
| `jetson/idcs-recorder.service` | `recorder/jetson.yaml` | none | 0 |
| `jetson/idcs-serial.service` | `base` + `bench/uncoupled.yaml` | A, D, G, S | 27 |
| `pc/idcs-camera-ui.service` | `base` + `pc/operator_local.yaml` | A, D, G, S | 27 |
| `pc/idcs-hil-streamer.service` | `base` + `sim/v2_scene.yaml` + `sim/mode_hil.yaml` | A, D, G, P, R | 28 |
| `pc/idcs-hud-relay.service` | `base` (CLI default) | A, D, G, S | 27 |
| `pc/idcs-pc-mode-agent.service` | `base` | A, D, G, S | 27 |
| `pc/idcs-sim-controller.service` | `base` + `sim/v2_scene.yaml` + `controller/sim_mount.yaml` + `bench/tuned.yaml` | D, P, R | 25 |
| `pc/idcs-sim-recorder.service` | `recorder/sim.yaml` | none | 0 |
| `pc/idcs-sim-streamer.service` | `base` + `sim/v2_scene.yaml` + `sim/target_fast.yaml` + `sim/mode_simulated_mount.yaml` | A, D, G, P, R | 28 |
| `pc/idcs-sim-swarm-controller.service` | `base` + `sim/v2_scene.yaml` + `controller/sim_mount.yaml` + `bench/tuned.yaml` + `controller/detections.yaml` | D, P, R | 25 |
| `pc/idcs-sim-swarm-streamer.service` | `base` + `sim/v2_scene.yaml` + `sim/target_fast.yaml` + `sim/mode_simulated_mount.yaml` + `sim/swarm.yaml` | A, D, G, R | 27 |
| `pc/idcs-sim-ui.service` | `base` + `sim/v2_scene.yaml` + `sim/mode_simulated_mount.yaml` | A, D, G, P, R | 28 |
| `pc/idcs-ui.service` | `base` + `sim/v2_scene.yaml` | A, D, G, P, R | 28 |
| `rpi/idcs-manual.service` | `base` (CLI default) | A, D, G, S | 27 |
| `rpi/idcs-operator-display.service` | `base` (CLI default) | A, D, G, S | 27 |

## Verification

Final per-step full-suite runs used the following command, changing the
`<step>` suffix for separate temporary test directories:

```powershell
python -m pip install --target .task-tmp/deps Pillow==13.0.0 trimesh==5.1.1
$env:PYTHONPATH = (Join-Path (Get-Location) '.task-tmp/deps')
python -m pytest -q --continue-on-collection-errors --basetemp=.task-tmp/pytest-<step>
```

Pillow and trimesh were installed in a task-local dependency directory, without
changing repository dependencies or the machine's global Python installation.
`--continue-on-collection-errors` allows all runnable tests to execute despite
missing GStreamer bindings. The plain initial `python -m pytest -q` stopped at
collection; it was not treated as a pass. No test was changed to hide an
environment failure or skip a hardware gate.

| Step / commit | Passed | Skipped | Failed | Collection errors | Subtests |
|---|---:|---:|---:|---:|---:|
| 1. MPC — `c3f9afc` | 1021 | 50 | 1 | 4 | 20 passed |
| 2. Dead keys — `9370213` | 1021 | 50 | 1 | 4 | 20 passed |
| 3. Legacy Pi — `4eb28c2` | 1021 | 50 | 1 | 4 | 20 passed |
| 4. Config sync — `b552382` | 1021 | 50 | 1 | 4 | 20 passed |
| 5. Unused files — `50de7a8` | configuration-only; no code test rerun required | | | | |
| 6. Duplicates/live-reader correction — `1f9b527` | 1025 | 50 | 1 | 4 | 20 passed |
| 7. Documentation/evidence | 1025 | 50 | 1 | 4 | 20 passed |

The single failure is unchanged
`test_validate_gimbal_fit.py::test_existing_low_range_fit_is_not_offline_control_qualified`:
this fresh clone lacks
`artifacts/gimbal_fit/controller_sysid_step_timed_unloaded_20260908/fit_report_external_selected.json`.
The test and missing-artifact status are unchanged from the base.
The four collection errors are `test_streamer_frame_ids.py`,
`test_streamer_gst_video_writer.py`, `test_streamer_hil_pose.py` and
`test_streamer_sim_mount.py`, all due to absent `gi`/GStreamer on Windows.
Existing skips include unavailable platform/GPU dependencies. These runs do not
qualify GStreamer/GstVideo, rendering or hardware on target Linux hosts.

Additional checks:

- `python -m pytest -q tests/test_config_merge_semantics.py`: **1 passed**, file unchanged.
- `python -m pytest -q tests/test_control_camera_geometry.py`: **3 passed**.
- The fourth new geometry test exercises the video runtime's non-actuating
  config check, proving the explicit vertical-FOV override reaches aiming.
- Local config audit: **21 stacks verified**; only allowlisted removals,
  effective values and active scenes identical, GPIO/ADC source unchanged.
- `python -m compileall -q common jetson pc rpi tools tests` and `git diff --check`: pass.

Final per-step logs are retained in
[`test-results/`](config_dead_removal_20261017/test-results/).

## Other findings and remaining work

No additional dead config key was conclusively established and removed.
The unused private helper `common.control._extract_optional_axis_pair` and
optional `osqp` package declaration in `pyproject.toml` appear orphaned; neither
was removed because they were outside the listed configuration cleanup.
Other matching base/tuned gains and policy values were left outside scope.

The local/source clock keys, image-axis signs and frame-freeze setting are live,
not cleanup candidates. The required controller-rate duplicate remains a review
item. No live-update protocol, per-section key schema, deployment tooling or
hardware acceptance was added. Hardware testing was neither needed nor run;
physical wiring/RS485 readiness and all bench actuation gates remain with the
project lead and bench operator.
