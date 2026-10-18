# Pi panel Settings menu — 2026-10-18

Candidate on `codex/panel-config-menu`, based on `9023faf` (`lab/live-config-followup`).
Local software/fake verification only. No lab host, serial, bridge, HIL, motor,
unit activation or deployment was used.

## Implementation

- `rpi/display/config_menu.py`: a Settings workflow with Speed limits, Lost
  target, Detection persistence, Engagement rules and guarded Calibration.
  Editable maps and excluded-class membership come from the effective config;
  no detector class list is embedded in the menu. The current base exposes 29
  managed leaves (the excluded-class list is one leaf with per-class toggles).
- `rpi/display/menu.py`, `rpi/operator_display.py`: register Settings beside the
  existing top-level pages; delegate its navigation/view without changing the
  existing pages or ROI behavior. No GPIO button is assigned. Joystick manual
  suppression and pitch/menu inversion remain in `PanelNavigator`.
- `rpi/display/inputs.py`: opt-in USB numeric input, including keypad digits,
  decimal point, minus and backspace. Backspace retains Back outside numeric
  editing. Kernel repeats and existing joystick repeats drive edits; after
  eight closely spaced repeats, numeric steps become five times larger.
- `rpi/display/commands.py`: ConfigCommand/ConfigReply use the same REQ socket
  as operator commands, with one request in flight, nonblocking send/receive
  and the unchanged 1.5-second timeout. Config requests are never silently
  evicted by ordinary commands. Guarded requests recheck local panel authority
  immediately before transmission, including after queueing behind status.
- `rpi/display/render.py`: only Settings action pages get a wider panel. Numeric
  editors and per-unit results use the existing information panel. Existing
  safety/fire/E-stop overlays remain above these panels.
- `common/live_config.py`: get/diff additionally return overlay-free release
  values and effective parser defaults. `set_many` accepts dotted-key `values`
  and `unset_keys`, validates the complete candidate against all host stacks,
  and uses the existing single transaction, digest, journal, atomic replace,
  publication and per-unit restart/receipt rules. `calibration_save` appends
  one validated summary record without changing the overlay or restarting.
- `tools/idcs_config.py`: CLI `set_many` with the existing local preflight and
  ConfigCommand envelope. `tests/test_panel_config_menu.py` covers this path
  with fake transport and no host access.

Numbers open an editor with requested effective and release values. Up increases
and Down decreases. Left cancels the proposal. Select opens Cancel/Confirm, with
Cancel first; only Confirm submits one set. USB typed calibration values also
use this confirmation. The same review page has Revert to release, which opens
its own Cancel/Confirm before sending unset. Boolean/list entry proposes a
change directly on the confirmation page. Asterisks mark differences from release.

Calibration Up/Down previews immediately, with actor
`pi-panel:calibration-preview`; requests are suppressed while another is pending.
Leaving the calibration group (or pressing Menu from its editor) asks Keep
editing / Save / Discard, with Keep editing selected first. Save appends one
`calibration_save` record, actor `pi-panel:calibration-save`. Discard sends one
`set_many`, actor `pi-panel:calibration-discard`, restoring original override
values and unsetting keys absent on entry. This restores each stack's original
effective value, including where release stacks differ. Even a timed-out preview
counts as touched because its write might have landed. A failed finish retains
the session for explicit recovery. Preview writes are durable: process/power
loss does not automatically undo them; their normal journal evidence remains.

Revert all removes only managed menu leaves in one confirmed transaction. It
never sends clear. Settings lists remaining overlay keys and has a separate
Remaining overlay keys information page. Unrelated CLI tuning overrides survive.

Values are labelled requested: an overlay reply is never treated as a service
application. Last result shows every affected unit's applied, pending until
idle, restart required, rejected/reason, awaiting service, inactive or other
existing outcome. Later fresh receipts with the matching requested digest can
advance a pending result to applied. Stale-digest rejection refreshes via get;
no mutation is retried automatically. Timeout reports agent offline/outcome
unknown, rebuilds the socket and permits read-only refresh. Enter/review again
before resubmitting any change.

The reference value prefers the managed camera controller for controller keys,
the operator agent for operator keys and DeepStream camera for perception/rules/
calibration. Numeric editors list per-stack requested/release variants when
these differ. All host stacks are validated regardless of the displayed reference.

## Fixed editing steps and bands

`EDIT_BANDS` in `rpi/display/config_menu.py` is the single numeric UI table.
Bands below are inclusive and are intersected with the unchanged schema ranges.
Rate bands are additionally capped by the minimum release bridge rate cap across
host stacks; idle-return rate is capped by the current controller rate. A band
that has no overlap is CLI-only. Cross-field constraints still run in the agent;
for coordinated changes outside the panel policy use CLI set_many.

| Key | Step | Panel band |
|---|---:|---:|
| controller.rate_limit_rad_s | 0.05 rad/s | 0.25–0.8 rad/s, narrowed by bridge caps |
| controller.manual_rate_limit_rad_s | 0.05 rad/s | 0.25–0.8 rad/s, narrowed by bridge caps |
| controller.accel_limit_rad_s2 | 0.5 rad/s² | 0.5–10 rad/s² |
| controller.manual_accel_limit_rad_s2 | 0.5 rad/s² | 0.5–10 rad/s² |
| controller.coast_s | 0.1 s | 0.1–2 s |
| operator.lock_lost_s | 0.1 s | 0.1–5 s |
| controller.idle_return_s | 0.1 s | 0.1–10 s |
| controller.idle_return_rate_rad_s | 0.05 rad/s | 0.05–0.8 rad/s, narrowed by controller rate |
| deepstream.shadow_tracks.max_age_frames | 5 frames | 0–300 frames |
| deepstream.id_stitching.max_gap_frames | 5 frames | 1–300 frames |
| deepstream.shadow_tracks.min_confidence | 0.05 | 0–1 |
| threat_eval.zones.{warning,restricted,critical}.radius_m | 5 m | 0–100 m |
| swarm_eval.max_engage_distance_m | 5 m | 0–200 m |
| swarm_eval.damage_by_class.* | 0.1 | 0–10 |
| laser.dir_cam.{x,y} | 0.0005 component (~0.5 mrad near forward) | −0.05–0.05 |
| laser.dir_cam.z | 0.0005 component | 0.95–1.05 |
| laser.offset_m.{x,z} | 0.001 m | −0.5–0.5 m |
| laser.offset_m.y | 0.001 m | −0.9–0.1 m |
| camera.known_size_ranging.class_sizes_m.* | 0.05 m | 0.05–5 m |

The direction vector is normalized by the existing laser parser; a z-only
component change is a scale change, not a guaranteed angular displacement.
Null disabling, scientific notation, advanced values and controller gains/
feedforward/predict remain CLI-only. Their tiers have not changed.

## Authority and host-scope findings

Pi-local editing authority requires a fresh `SystemStatus.panel` message (the
existing one-second freshness bound), `inputs_valid.fire_control is True` and
`inputs.fire_control is False`. Missing, stale, malformed or unreadable state
refuses Rules and Calibration. This is checked on entry, nudges, confirmation,
reverts, finishing calibration and again at actual transport send. Speed limits
retain the original idle hold; no new authority check is added to them.

The operator agent itself subscribes to perception and PC polling, not the Pi
PanelState stream (which is Pi-local). The Jetson controller does receive
ManualControlState over its existing PUSH/PULL path, but that message carries
`effective_control_on`, not proof of the raw GPIO12 switch state: ROI can suppress
it. Moreover `rpi.runtime_control` passes `control_cmd_valid` to a model that does
not declare that field, so Pydantic drops it before serialization. The recorder
rebroadcast therefore lacks raw switch validity too. Stealing the controller's
PULL ownership would be incorrect. No existing agent message supplies a reliable
raw switch+validity pair, so **no agent-side physical fire-control check was
added**. The display enforces the rule; CLI callers remain governed by their
existing tuning-session procedures. Adding raw authority to a shared protocol
would require separate schema/protocol review, excluded by this task.

`jetson.pc_mode_channel.PcModeChannel` carries `PcModeTask` service actions;
`common.pc_mode.PC_ACTIONS` and task parsing strictly allow-list verb/unit pairs.
It has no ConfigCommand payload or config digest/receipt return path, and
`pc.mode_agent` accepts ConfigCommand only through its local AgentChannel.
Forwarding config would require extending that wire format. No forwarding was
added. Laser and threat-zone results explicitly show **PC not updated** (also
after calibration Save). Use the existing PC-local CLI and that host's release
validation for a separately reviewed PC change. No Jetson-to-PC config copy or
new network endpoint was introduced.

Tier/schema tables, motor vetoes, bridge/serial exclusion, restart allow-list,
deployment gate and all physical bench gates are unchanged.

## Verification

Final local commands and results:

```powershell
python -m pytest -q tests/test_panel_config_menu.py
# 60 passed in 8.00 s

python -m pytest -q tests/test_operator_display.py tests/test_operator_menu.py tests/test_live_config.py
# Initial existing-contract run: 118 passed, 25 skipped in 44.07 s

$env:PYTHONPATH = (Join-Path (Get-Location) '.task-tmp/deps')
python -m pytest -q --continue-on-collection-errors --basetemp=.task-tmp/pytest-panel-final --tb=short
# 1,184 passed, 50 skipped, 1 failed, 4 collection errors,
# 20 subtests passed in 91.75 s

python -m compileall -q common/live_config.py tools/idcs_config.py rpi/display rpi/operator_display.py tests/test_panel_config_menu.py
git diff --check
# Both passed
```

The full-suite failure is the existing
`test_validate_gimbal_fit.py::test_existing_low_range_fit_is_not_offline_control_qualified`:
this fresh clone lacks
`artifacts/gimbal_fit/controller_sysid_step_timed_unloaded_20260908/fit_report_external_selected.json`.
The four collection errors are `test_streamer_frame_ids`,
`test_streamer_gst_video_writer`, `test_streamer_hil_pose` and
`test_streamer_sim_mount`, all requiring unavailable `gi`/GStreamer on Windows.
These are the same failure categories recorded for the base follow-up. The
1,184 passes are the base report's 1,124 plus 60 new cases. Skips include
platform/optional dependency boundaries; they do not qualify Pi/GPU rendering.
Raw local logs are retained under ignored `.task-tmp/` (final full log:
`full-suite-final.txt`). All new tests use fakes,
injected clocks and isolated overlays. They exercise navigation, stepping,
clamping, acceleration, numeric typing/backspace, Cancel/Confirm, Boolean/list
proposals, Revert/Revert all, calibration preview/Save/atomic Discard, fire-control
invalid/stale/send-time refusal, idle receipts, stale digest refresh, offline
timeout, per-stack reference/caps, set_many full-candidate validation, write
failure preservation, journal reversion and CLI transport/preflight.

Local Python: 3.14.4 / Windows. Optional Pillow 13.0.0 and trimesh 5.1.1 were
installed only in `.task-tmp/deps` with `--no-deps`; no global packages changed.
The first full run exposed a missing temporary parent directory; creating
`.task-tmp` resolved those setup errors. An intermediate scope assertion counted
28 rather than 29 leaves; that test expectation was corrected.

## Reviewer qualification — not performed here

All checks below are for the reviewer after code review and a separately
approved installation. Do not start a bridge, serial service, controller, HIL or
motor owner. The unresolved wiring/RS485 bench prohibition remains in force.

1. **Needs Pi — real platform imports and fake contracts.** In the reviewed
   checkout, run `bin/python3 -m pytest -q tests/test_panel_config_menu.py
   tests/test_operator_menu.py tests/test_operator_display.py`. Pass: no new
   failures, and previously unavailable GI/Cairo menu tests execute with the
   Pi's installed libraries. The fixtures do not create device owners or
   contact lab endpoints.
2. **Needs Jetson — fake config/agent validation on target Python.** Run
   `bin/python3 -m pytest -q tests/test_panel_config_menu.py tests/test_live_config.py
   tests/test_operator_agent.py`. Pass: all applicable contracts pass, existing
   platform-specific skips are explained, set_many changes share one transaction
   and invalid candidates leave the isolated overlay unchanged. This command
   uses fake service runners, not actual systemctl mutations.
3. **Needs Pi — USB and joystick usability.** With an already approved,
   motor-free display/agent session and readable fire-control OFF, open Menu →
   Settings → Speed limits → controller.rate_limit_rad_s. Move Down once, then
   Left: no mutation journal record. Re-enter, hold Down: repeat starts after
   0.5 s and larger steps appear after sustained repeats; value stops at 0.25
   (or the narrowed band). Type `0.73`, Backspace, `2`: proposal is 0.72. Select
   then Cancel: no set. Select then Confirm: exactly one set transaction; a
   second press while pending sends no second set. With manual ON, the joystick
   must not navigate; USB keys still navigate. Pass: proposals, labels, clamp,
   Cancel and count match, and video/status tick remains responsive.
4. **Needs Pi — visual and safety display.** At the actual panel resolution,
   inspect all five groups, numeric editor, confirmation and Last result.
   Pass: effective/release/proposed/band values are legible; cursor stays visible;
   long results scroll; every affected unit is inspectable; mandatory E-stop,
   fire and authority indications stay above menu overlays. An idle-tier value
   with a pending receipt shows held until idle and never claims applied before
   a matching fresh receipt. Exercise pending → applied with fake receipts or
   the isolated contract, not a physical controller start.
5. **Needs Pi — fail closed.** With no motor owners, turn fire control ON, then
   attempt Rules, Calibration and a guarded Confirm queued behind a slow status
   reply. Repeat with panel publication unavailable or input validity false.
   Pass: all guarded writes are refused, including at transmission; no mutation
   is journalled. Do not infer switch-off from lamp darkness. Restore readable
   OFF before continuing. Speed editing remains subject to existing idle hold.
6. **Needs Pi + Jetson — calibration durability/reticle.** First run the read-only command
   `systemctl is-active idcs-bridge.service idcs-serial.service idcs-controller.service idcs-controller-camera.service idcs-hil.target`.
   Require every listed unit to report inactive before any interactive change;
   otherwise stop this check without changing any unit state. On an already approved passive
   camera/display pipeline, fire control readable OFF: record entry laser x
   offset and direction, nudge each once, wait for matching replies and observe
   the video reticle. Leave: Keep editing is selected first and makes no write.
   Choose Discard: exactly one transaction restores both keys and their original
   override presence; reticle returns. Repeat previews then Save: one summary
   record with entry/final values, no extra set and no restart. Pass: no motor
   activity, no new device owner, visible reticle follows acknowledged geometry,
   and Last result includes PC not updated. This qualifies display geometry only.
7. **Needs Pi + Jetson — race/offline and selective reset.** With an isolated or
   archived/reviewed test overlay, edit a menu proposal; use CLI on the same
   intended agent to change another key before Confirm. Pass: stale digest
   refuses the proposal, refresh shows new effective values, and no auto retry
   occurs. With a fake responder withholding replies for >1.5 s, pass: offline/
   outcome unknown appears and video/menu ticks continue. Seed an unrelated
   CLI tuning override and a menu override; confirmed Revert all removes only
   the menu override and lists the tuning key as remaining. Restore the reviewed
   test overlay through the normal agent path; preserve the journal evidence.
8. **Needs Jetson — request latency.** On the same passive session measure get
   and mutation reply latency while watching return video and selection. Pass:
   replies arrive within the unchanged 1.5-second client timeout; no display
   tick blocks waiting for transport, no source transition occurs and no
   additional GPU/video owner appears. Agent full-stack validation and its
   existing receipt wait can be expensive on target; this timing is unqualified
   locally. If it exceeds timeout, report it rather than treating a persisted
   overlay as a successful application or automatically retrying the mutation.
