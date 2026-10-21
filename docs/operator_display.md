# Operator display (Pi panel screen)

Read [deployment.md](deployment.md) and the host manifests for current source
paths. Dated trial/release deployment paragraphs below are historical, not
instructions to reinstall old sources. Consolidation preserves panel policies;
real switch/motor acceptance remains deferred.

Resident-mode rework (deployed in deploy-111,2026-10-21):
**Targets -> Tracking mode: YOLO / ROI**, independent of Camera /
Simulator. ROI exposes **Select ROI / Release ROI / Box size**. Closing the
editor, release or loss does not change mode. Only choosing YOLO returns to
detection/planning. Source/encoder/services do not restart. Missing readiness
has a specific refusal, not a hidden source fallback or a generic panel warning.

Selection requires Manual, readable OFF then ON and the exact displayed source
frame. Resident mode retains an explicit lock after Manual OFF and follows only
through the ordinary auto gate; Auto cannot seed/edit. Manual retains steering
when not editing; editing captures the stick without mount motion. E-stop,
unreadable-input, freshness and lease guards remain. This release is not
physical switch/motor acceptance. See newest journal entries.

Shared UI styling, rendering ownership and controlled preview/benchmark checks:
[ui_style.md](ui_style.md). Styling does not change GPIO authority or lamp policy.

Deployed HUD follow-up (introduced in `deploy-79`, retained in `deploy-84`): E-stop/fire and authority/selection/REC
strip are mandatory, even with a menu open; old hide/detail preferences are
ignored and their Display menu page is removed. Steady source/panel-mode/fps
menu repeats are removed, but source-switch progress/errors, actions, freshness,
HUD telemetry and raw Panel diagnostics remain. Decoded return <20fps for more
than one second after a one-second sampling warm-up raises LOW VIDEO FPS; loss
and reconnect have separate/reset states. Fire-arm unreadability shows unknown,
never infers switch-off from lamp darkness. Range in the target picker uses
one decimal place without an approximate marker; full-precision data is preserved.

`rpi.operator_display` puts the Jetson's return video on the control panel's
HDMI screen and draws the operator interface over it. It subscribes to
perception snapshots, controller diagnostics and the local panel state; what
the operator changes from the menu (target lock, target type, mode,
recording) is a request to the Jetson's operator agent, which owns that
authority.

Native-return recovery (introduced in deploy-98): the Pi and PC use
GStreamer's loss notification and complete-access-unit keyframe gating, with
RTCP feedback to the Jetson. After damage, video may briefly pause until a valid
keyframe rather than display a corrupt prediction chain. Existing freshness,
LOW VIDEO FPS and exact-frame ROI checks remain active; recovery does not
authorize motion or reseed a tracker. Default ports are RTP5002, feedback5003
on Jetson, sender reports5004 on receivers. See [visual_lock.md](visual_lock.md)
for configurable ports and all-IDR fallback. Recovery is not a guarantee of
zero loss or a qualification of physical camera control.

### Pi receiver recovery (deployed in deploy-110)

The display separates the UDP/RTCP/depay/decoder/recording receiver bin from
the persistent black source, selector, colour conversion, overlay and screen
sink. Receiver-local errors and EOS withdraw local video/ROI validity, send
release/cancel, clear the exact-frame join and rebuild only that bin. Menus,
GPIO subscriptions, mandatory overlays and the application process stay alive.
Recovery never reseeds a lock; lost release delivery remains bounded by the
existing ROI lease/watchdogs, not a new motion-authority mechanism.

Retries back off from250ms to4s, with at most five automatic rebuilds in a
fault burst. Ten seconds of healthy decoding resets the burst. Persistent
failure keeps the black/UI branch running with VIDEO FAILED and exposes
**Operations -> Reconnect video**. VIDEO RECOVERING identifies an active retry.
Reports include `receiver_recovery` state, attempts, rebuilds, generation and
last error; receiver statistics belong to that generation. Output/compositor
failures remain fatal. A decoder without a confirmed NULL state is never
replaced by a second decoder.

Recording finalization is polled without sleeping on the UI thread, up to the
existing2s bound; the replacement waits for it. Resumed recording starts at a
fresh keyframe. The fallback runs30fps to match the return feed and avoid the
10-to30fps converter renegotiation observed in the trial. Controlled SSRC,
EOS/error and full live-camera display trials passed on the Pi; see the newest
journal entry. Deploy-110 also passed live Camera -> Simulator -> Camera
switching with the same Pi display PID and zero automatic service restarts.

The scene menu, introduced in `deploy-81` (`0ce7a11`), is now under
**Operations -> Source & scene -> Simulator -> Single / Swarm**, with
Cancel-first confirmation. The external
mode stays Simulator; the agent reports the actual scene separately from fresh
PC service state, not a display preference. Scene changes use the full ordered
source switch and readiness/rollback guards and clear source-scoped locks.
Single preserves synthetic-observation SIM control. Swarm preserves the
existing detection-fed software-mount preset, so its picker/locks use Jetson's
visual catalog, not simulator truth IDs. The internal `sim_swarm` mode does not
become another top-level menu item and cannot grant hardware/HIL authority.
Swarm target locks use the detector's session; SIM HUD pose uses the separate
fresh `sim_hud_session` from the PC producer. Missing/stale identity hides geometry
instead of substituting detector identity. The last recorded deploy-84 scene is Swarm; live state is in the host manifests.
Physical panel switching acceptance remains deferred to the user at the bench.

## Resident visual lock (enabled in deploy-109)

With `deepstream.seamless_roi: true` (the default since deploy-109), normal **Camera** and **Simulator**
(Single or Swarm), Targets adds
**Select box**, **Release** and **Box size** when the normal runtime reports
fresh ROI capability. Confirming a box gives ROI selection authority inside
the existing video process: YOLO/NvDCF stop receiving frames until release or
loss, then YOLO resumes. Source mode, process, encoder and return session stay
the same. While ROI owns selection, detector planner/classes/lock and score
actions are suppressed. Idle detector controls remain available alongside box
selection. Missing release-native libraries leave normal YOLO working and hide
the unavailable box action.

Opening Select box starts a session-scoped selecting cache, renewed on the
existing seed socket. Closing it, loss/release, a missing heartbeat (500 ms) or
15 s without confirmation stops caching. Reopen Select box after timeout.
An uncached displayed frame is refused explicitly; no other image is seeded.

All Manual/GPIO12, freshness, E-stop, exact displayed-frame identity, presets,
refusal and new-readable-OFF rules below apply unchanged. Release/loss returns
the editor's box; there is no automatic reseed. In normal modes the explicit
return to YOLO is part of this workflow. Single's software controller receives
the Jetson ROI selection through its existing loopback publisher; Swarm and
camera shadow already use the normal Jetson V2 port. Only a fresh accepted SIM
lease grants simulated auto-follow; physical ROI auto-block is unchanged.

Five bounded live-camera seed/track/release cycles passed on 2026-10-21 with
unchanged DeepStream/Pi PIDs, tracker session and return RTP sender. This is
software protocol acceptance, not physical joystick/switch or moving-target
qualification; see `docs/journal.md` for evidence and the remaining display
restart on a genuine source-mode change in deploy-109. Deploy-110's receiver
recovery and unconditional menu hardening address both issues.

## Managed visual lock fallback (2026-10-20)

The maintenance modes `camera_visual_lock` and `sim_visual_lock` replace the
detector video owner; they are not seamless ROI switches. Keep these modes
available to maintenance tools, not the normal operator menu. Use
**Targets -> Select box / Release** in Camera or Simulator instead. A temporary
loss of ROI capability must not offer a restart-based fallback. Actual Camera /
Simulator scene changes still use the ordered source transaction and rollback.
Nothing starts motors. Deploy-109 hid the camera fallback while seamless
capability was fresh; deploy-110 removes both fallbacks regardless
of capability, including Simulator's nested Visual lock action.

In either mode, Targets contains **Select box**, **Release** and **Box size**
instead of detector planner/classes. Box size is Pi-local in
`~/.config/idcs/operator_display.json`: Small64x48, **Medium96x72** (default),
Large160x120 and X-large240x180. Each new editor opens centred at the chosen
size, clamped to the image. The joystick positions it. Optional keyboard or
`menu_select` toggles the existing fine resize mode; no menu-select GPIO is
required to choose a preset. Close/release an editor to return to the menu.

Select box requires fresh source video/identity and panel state. Select Manual;
readable GPIO12 OFF edits and its OFF-to-ON edge requests the exact displayed
frame. Initial ON, unreadable contacts, E-stop, stale source/video/session or
panel never create a lock. Refusal/loss reasons appear on screen. Release or
loss clears the request, readies a new tracker and requires a fresh readable
OFF before selecting again; there is no reseed or detector fallback. Manual
OFF closes/releases the workflow. Menu/Back cancels/releases before returning
to ordinary navigation. Changing source releases the outgoing receiver.

Simulator / Visual lock runs one slow drone (`idcs-sim-visual-lock.target`) with a
software mount controller fed by Jetson perception, following accepted
V2 geometry without a class-derived range or parallax. Only a fresh accepted
SIM request plus the panel/video gates grants a PC loopback lease. Physical
Pi output keeps the ROI auto block. Camera / Visual lock uses the existing
shadow controller. Isolated `--visual-roi-trial` remains control-free on5597;
its workflow, diagnostics and guard remain available. See the exact build,
plan/validate and three-host acceptance checklist in [visual lock](visual_lock.md).


## Deployed SIM score/error HUD addition (deploy-82)

PC and Pi render one shared, compact metrics strip below the heading tape:

- Swarm: `SWARM HIT 18 BREACH 4 LIVE 3`. HIT is simulator dwell-based elimination,
  BREACH is crossing the defended radius, LIVE is the active scenario population,
  not the number of YOLO detections. The simulator owns these counts; receiver
  HUDs do not score detector identities or inject simulator truth into perception.
- Single: `ERR +1.25 deg / -0.50 deg`, rendered with degree symbols, in yaw/pitch
  order. This is raw measured parallax-adjusted bearing error, not a Kalman
  prediction or a new controller input. Diagnostic age plus the controller's
  capture-age upper bound must be <=250ms; unknown/stale/lost/manual/non-tracking
  error displays `ERR --`. Latest telemetry is not exactly joined to return video.
- **Operations -> Clear Swarm score** exists only while Simulator / Swarm is selected,
  with Cancel-first confirmation. It clears eliminated/breached counters and
  starts a new score epoch; current live targets form its initial spawn population.
  Targets, dwell, pose, RNG, spawn schedule and recordings survive.
  A `score_reset` event is appended to recorder history; nothing is deleted.
  Preserving target locks is also intended, but a delayed acknowledgement can
  currently expire an otherwise valid operator lock (R2 below).

The confirmation freezes the simulator session. The operator agent checks fresh
software Swarm state, no pending action and no motor/HIL activity, then uses the
existing acknowledged PC polling channel. The PC worker sends a bounded request
to the current streamer on loopback only (`operator.sim_score_endpoint`, default
`tcp://127.0.0.1:5580`); it never runs `systemctl clear_score`, restarts a source,
or reaches motion command sockets. The render thread applies the counter change.
Source-session mismatch, duplicate/retired requests and timeouts cannot silently
clear another scenario; success requires a matching epoch acknowledgement.

**Open review finding (2026-10-11):** score reset shares the agent's broad
`busy` state with source transitions. While awaiting a slow PC acknowledgement,
this suppresses the active catalog and SIM activity relay; after `lock_lost_s`
it can clear a still-present lock and withhold the safety indication. The
immediate-success test below did not cover this case. See R2 in the
[review](codebase_review_20261011.md); the code has not been fixed by this doc edit.

New source files: `common/sim_score.py` (typed score, reset identity, shared
formatter), `pc/sim_score.py` (bounded PC loopback client),
`tests/test_sim_score.py` (deterministic reset/wire/freshness/render contracts).
Committed as `753339e`, deployed to PC, Jetson and Pi, and live verified on
2026-10-09. One operator-command-path reset advanced epoch0 to1 with zero
HIT/BREACH and unchanged service owners/session. Single produced402 fresh
numeric error samples in8s; PC/Pi screenshots confirmed the degree glyphs and
exclusive strip, then Swarm was restored. Physical joystick/menu interaction
still requires bench acceptance; the menu/confirmation path has deterministic
native tests. Host manifests and the journal hold current-source evidence.

## Panel controls

2026-10-10 deployed in `deploy-83` (`eedb303`): target IDs are internal, absent
from box labels, picker/previews, lock/engagement notices and target status.
Picker/status rows show planner rank only with multiple targets; unavailable
rank is omitted, never replaced by the monotonically increasing tracker ID.
Presentation joins require fresh perception from the same camera tracker
session as the control catalog; SIM truth catalogs cannot borrow visual ranks.
Internal row keys and source/session-scoped lock commands are unchanged.
The native GPU corner-fill/glyph correction is documented in [hud_design.md](hud_design.md).

| Control | Effect |
|---|---|
| Fire-control switch (GPIO12) | Enables automatic tracking only. Off: SAFE in auto mode; it does not block manual joystick slew. |
| Safety switch (GPIO16) | Enables the fire button and its screen indication. It has no effect on motor movement. |
| Control switch (GPIO5, latching) | Selects manual mode: the joystick slews through the controller (`controller.manual_rate_limit_rad_s`, `manual_accel_limit_rad_s2`, same travel envelope), even with GPIO12 low. Screen: MANUAL. |
| Joystick | GPIO5 manual mode: gimbal rate. Otherwise: target picker and menu navigation. |
| Fire button | With GPIO16 on, shows FIRE BUTTON on the screen. A tracked-target press while ARMED is also recorded as `engage`; other presses are recorded as `engage_refused`. No effector is connected. |
| E-stop | Blocks all motion; EMERGENCY STOP on screen. |

Status bar mode chip: E-STOP > MANUAL (GPIO5 latched) > SAFE (auto mode,
GPIO12 off) > ARMED (auto mode, GPIO12 on).
In automatic mode with the menu closed, left opens the target picker and
right opens the ordinary menu; up/down do nothing. In the picker, up/down
browse with a yellow preview only, right confirms and closes, and left
cancels without sending a command. The picker is a compact one-candidate-at-a-time
overlay, with no planner row or target-ID display. **Targets -> Planner selection**
is the separate on/off toggle. Candidate rows are frozen when opened;
a disappearing/reordered row cannot turn confirmation into another target.
The agent revalidates its identity and availability before accepting it.
In the ordinary menu, up/down navigate, right enters/selects and left goes
back/closes. Joystick events are disabled in GPIO5 manual mode and during
E-stop; entering either cancels an open picker. These actions never slew the
mount, bias aim, press fire or change GPIO permissions. PC keyboard/terminal
input is not required for this console workflow.
Menu/picker vertical navigation is separate from manual pitch: push forward
to move the cursor up, pull backward to move it down. Manual movement retains
forward = pitch down and backward = pitch up; left/right and keyboard arrows
are unchanged. The conversion occurs only in `PanelNavigator`, not the ADC,
motion configuration or `ManualControlState`.
The menu-only correction is included in the consolidated release; old
`IDCS-trials/menu-pitch-*`, safety and target-picker paths are historical
rollbacks, not current installation instructions. See [deployment](deployment.md).
Fire/E-stop state is part of the mandatory console strip, including when the
menu is open. The Panel diagnostics page also shows raw and unreadable inputs.

| Indicator | Meaning |
|---|---|
| Fire-control lamp (GPIO22) | A current detection or observed target track is available. |
| Safety lamp (GPIO25) | Fresh applied live target tracking at either the simulator or gimbal intent consumer, or the manual-mode exception. E-stop forces it off. A visual selection alone or shadow-only controller does not qualify. |
| Yellow (GPIO23) | Fresh physical bridge proof that every configured motor has valid, recent feedback. Not simulator availability or fire-button state. |
| Green (GPIO18) | Panel GPIO runtime running (unchanged). |
| Red (GPIO24) | E-stop active (unchanged). |

These outputs are active-low. Qualifying conditions turn the lamps on
immediately; each has an independent configurable off delay (default one
second of continuous absence). Reacquisition cancels that delay. Fire-control
uses progressing perception, with a 0.5-second stale-frame watchdog. Safety
uses progressing applied-control status, with a separate 0.5-second watchdog;
loss of that status bypasses its off delay, unless local manual mode qualifies.
E-stop overrides safety immediately. A normal fresh inactive/coasting status
starts the safety off delay; coasting, idle-return and limit holds cannot
independently turn it on. Zero-rate tracking of an already centred target
does qualify: tracking is not the same as nonzero motion.

The **manual exception** lights safety when GPIO5 selects manual mode even
without a target, perception, or actuator status. Fresh applied live manual
status from either consumer also qualifies. Shadow manual status does not.
Leaving local manual starts the normal off delay; losing remote manual status
still obeys the stale-status watchdog. This exception does not change the
existing `engage`/`engage_refused` rules or connect any effector.
Physical availability has its own 0.5-second receive watchdog and no off delay.
Missing hardware proof is off, including old bridge and synthetic publishers.
The bridge checks all configured motor replies against its encoder stale limit;
yellow indicates communication availability, not mechanically verified motion.

**Only the fire-control and safety contacts share lamp power**: GPIO22 feeds
GPIO12, and GPIO25 feeds GPIO16. Lamp-off makes that contact unreadable, not
proof the operator opened it. Effective auto/fire permission is false while
unreadable; no old high is latched. On power-up the contact is withheld for one
poll (normally 5 ms), then freshly sampled. This guard is not measured electrical
settling qualification. The user reported the change tested before requesting
deployment; automated checks do not establish electrical or motor qualification.
The Panel page marks unreadable contacts separately from readable/off contacts.
Fire-control depends on perception; safety depends on consumer activity or
manual mode, never on GPIO16/fire-button authority. Neither lamp is gated by
its own contact, so there is no dark-contact startup cycle. No flashing probe
pulses are used. GPIO5 manual mode, the fire button and E-stop are independent
of lamp power. E-stop blocks actuation immediately and extinguishes safety;
fire-control/yellow continue showing their respective conditions. The Panel
page reports the safety reason (sim/gimbal tracking, manual exception,
off-delay, inactive, stale status or emergency), separately from ARMED/SAFE.

The lamp hysteresis does not hold a controller target alive or bypass its timing
watchdogs. Loss beyond the off delay removes GPIO12 auto authority, so automatic
idle-return cannot continue after that cut; manual authority is unchanged.
Receive-age checks use same-host monotonic elapsed time, not subtraction of
PC/Jetson clocks; these are indication watchdogs, not production control-timing
qualification. Applied-control sessions and progressing sequences reject
duplicate/out-of-order/retired-session status without renewing its watchdog
and support consumer restarts. The perception frontier still needs a Pi GPIO
runtime restart if its source clock resets after a host reboot.

### Applied-control indication path

The simulator reports activity only after applying an accepted live intent.
The physical bridge reports activity after forwarding an accepted live intent
to the serial queue, and requires fresh valid feedback from every configured
motor. This proves receiver application/forwarding, **not** a firmware write
acknowledgement, mechanical alignment, or tracking accuracy. Old publishers
without the optional `CamState.control_activity` cannot assert safety tracking.

- Simulator `CamState` -> PC mode agent loopback SUB
  (`operator.sim_activity_sub`, default port 5572) -> existing PC-initiated
  polls to Jetson (5592) -> read-only `net.zmq_sim_activity` PUB (5593) -> Pi.
- Physical bridge `CamState` -> Pi's existing gimbal-state subscription (5558).

The PC agent forwards simulator status only with exactly one matching
streamer/controller pair active, never with a HIL streamer or conflicting pair.
Each relay accumulates local elapsed age; repeated polls cannot renew an old
consumer sample. Mode transitions withhold simulator activity. Network transit
time is not qualified by this indication protocol, which grants no motion or
control-clock authority and requires no fixed PC identity/inbound PC port.

Historical software tests covered camera safety OFF, simulator tracking ON,
relay loss/contact invalidation and recovery. FakeGPIO tests covered manual
exception and E-stop without operating physical switches. These are not motor
or electrical acceptance. The consolidated release includes the bridge activity
code but hardware services remain inactive in the last recorded deployment;
do not activate them until the repaired-wiring/RS485 standing gate is satisfied.
Reports and old unit backups are retained in the journal's dated evidence paths.

## Menu

- **Targets**: **Planner selection** on/off and **Eligible classes**. With
  Planner off, the operator agent chooses the closest visible eligible target
  with a fresh, positive, non-held range estimate. It keeps that identity even if another target
  becomes nearer or the operator explicitly picks another with joystick-left;
  after `operator.lock_lost_s` without seeing it, it chooses the next closest.
  Missing/held range never causes a box-size guess. If no eligible ranged
  target exists, it waits without a lock. Camera and Simulator/Swarm use the
  visual catalog; Simulator/Single uses its controlled synthetic catalog.
  IDs remain internal and source/session scoped. GPIO5 manual motor control is
  unrelated.
- **Operations**: **Source & scene** (`operator.modes`: Camera, Simulator,
  Standby; never HIL and refused while a motor unit runs), recording on/off,
  and conditional Swarm score reset. Destructive
  or source-changing actions retain Cancel-first confirmation.
- **Information**: wide, scrollable, read-only **System**, **Connections**,
  **Control**, **Panel**, and **About** pages. Actions stay in the compact menu;
  ordinary HUD graphics are hidden while an Information page is open, while
  the foreground status strip retains source, authority and recording state
  and the E-stop/fire indicators remain visible above the menu.
  **Display** remains removed because its old controls hid required HUD data.

## Command path

```
display REQ --OperatorCommand--> jetson.operator_agent REP (net.zmq_operator_command)
                                   | PUB OperatorSelection (net.zmq_operator_selection, loopback)
                                   v
                            DeepStream target selector: lock + target classes
```

The agent replies with its state (source mode, selection mode, recording, lock,
target classes and source/session-scoped target catalog) and is polled every
0.2 s; a request unanswered for 1.5 s is reported and the
socket rebuilt. Mode changes coordinate allowlisted Jetson units and the
outbound-connected PC mode agent; recording commands act on the allowlisted
Jetson recorder. Messages: `common/operator_commands.py`.

Camera catalogs carry a UUID per DeepStream tracker process. Simulator catalogs
carry the streamer's applied-control session. Locks require the active source,
exact session and an observed target; busy transitions, stale catalogs and
foreign/restarted source IDs are refused. Duplicate/out-of-order catalogs cannot
renew the 0.5 s freshness window; relays add local elapsed age rather than
subtracting cross-host clocks. Mode remains Operator after agent restart, but
lock IDs do not persist. Configured selectors wait for their first authority
message, and stale Operator authority suppresses selection rather than falling
back. These watchdogs grant no physical motion or control-clock qualification.

For Simulator/Single, the PC mode agent sends synthetic catalogs in its existing outbound polls;
Jetson returns selection authority in replies, which the PC republishes only
on `operator.sim_selection_bind` (loopback 5579). The ground-truth perception
stream (5574), simulator controller and PC SIM HUD then share the applied
choice. Pi uses that same catalog for its picker/lock indication. Swarm uses
the Jetson visual catalog instead. Jetson's rendered
video still contains learned detections/NvDCF tracks; their IDs are never
mistaken for simulator control IDs. No inbound PC LAN control port is required.

### Camera / simulator switching

Open **Operations -> Source & scene -> Camera / Simulator -> Confirm**. Cancel is first.
The operator agent replies immediately with a busy state; the Pi display
continues receiving status and keeps its black fallback/menu visible while
the video source changes. The Information -> System page shows the transition stage, PC
agent connectivity, pending acknowledgement, and completion or failure.

`pc.mode_agent` connects out to `net.zmq_operator_pc` on Jetson (5592), polls
every 200 ms, and runs only allowlisted **user** service operations. Jetson
never needs the PC's SSH key, hostname, or inbound control address. The agent
must be online before a mode change is accepted. Its service must be enabled
once after the tagged release is deployed:

```bash
systemctl --user enable --now idcs-pc-mode-agent.service  # on PC
sudo systemctl restart idcs-operator-agent.service       # on Jetson
systemctl --user restart idcs-operator-display.service   # on Pi
```

Transition order:

1. Stop outgoing PC simulator targets and any outgoing camera UI.
2. Stop both Jetson video pipelines, releasing their shared ports and trackers.
3. Start the chosen Jetson pipeline: camera target (shadow controller only)
   or control-free RTP DeepStream.
4. Start PC camera UI or the software-only simulated-mount target.
5. Require the individual incoming services, three unique fresh source
   snapshots in the expected clock domain, and a fresh health record with
   encoded return frames. Detector recognition is **not** a readiness condition.

The exclusive `idcs-sim.target` and `idcs-sim-swarm.target` share a required
`idcs-sim-session.target`. Panel, simulator UI and simulator recorder are
`PartOf` only this session, not both conflicting mode targets: otherwise a
conflict's stop propagation can cancel their incoming start jobs. The session
stops when unneeded and the menu explicitly stops it on source changes.

Simulator motion is Single synthetic observations or Swarm visual observations
→ software controller → simulated mount. It neither consumes physical encoder
pose nor sends motor commands. The current controller units still inherit bench
gains and the streamer uses F6 quantization; independence from hardware tuning
is an open implementation gap, not a completed property (R6 in the review).
Camera mode retains its shadow controller. Mode actions are
refused while any Jetson HIL/bridge/serial/live-controller owner or PC HIL
streamer is active (including starting/stopping). Unknown unit status also
refuses the switch. These guards do not grant bench authority.

Failed startup/readiness restores the previous source when its PC session
and service states are confirmed. A timed-out PC action stays pending until
acknowledged; no later stop/start may overtake it. A disconnected or restarted
PC session cannot silently replay an old task; the menu reports recovery
required if a safe rollback cannot be confirmed.

The **control connection** is PC-address independent. Simulated-mount sockets
no longer bind to a fixed PC LAN address; their clock endpoint is loopback
because their controller is same-host. The **PC return-video destination**
still uses deployment configuration `net.return_ip`: update it through the
reviewed deployment workflow if the PC address changes. Pi video uses its independent `net.return_mirror_ips` entry.
Hardware/HIL timing endpoints are unchanged. This ZMQ channel, like the
existing operator channel, assumes a trusted LAN; it is not an authenticated
Internet-facing service. The controller now binds its configured addresses
unchanged, including loopback for the software SIM. This was verified in local
tests, not by a live socket/firewall audit. The PC task channel remains a
separate outbound path.

## Return-video recording (2026-10-20)

While the menu's **Operations -> Recording** is on (the Jetson flight
recorder runs), the Pi also saves the return video it receives to
`rpi.operator_display.recording.dir` (`/home/idcs/idcs-recordings/video`).
The encoded stream is tapped before decoding (no re-encode); each recording
starts at a keyframe and is written as 60 s Matroska segments named
`<recording start>-<segment>-return.mkv`; every segment starts at 0:00 and
plays on its own (no gap between segments, at most a few frames repeated). Stopping finalises the last segment;
a segment cut off by power loss remains playable. Oldest files are removed
first beyond `max_total_gb` (15). The video contains the Jetson's OSD (boxes,
labels, lock box), not the Pi's own HUD or menu, which are drawn locally. It
lines up with the flight-recorder data by wall-clock time. The display report
(`~/.local/state/idcs/operator-display.json`) shows `recording` state and the
current file. The Pi records even when the PC is absent.

## Pieces

| Module | Role |
|---|---|
| `rpi/display/video.py` | GStreamer: RTP → hardware H.264 decode (`v4l2h264dec`) → selector → hardware colour conversion/BGRx → `overlaycomposition` → `waylandsink` full screen. A black `videotestsrc` stands in while no stream arrives, so alerts stay visible. |
| `rpi/display/render.py` | cairo drawing of the status bar, alert banner and menu, each as its own small premultiplied-BGRA rectangle; redrawn only when its text changes. |
| `rpi/display/menu.py` | The menu: pages of submenus, settings (`Choice`), actions, or live information lines, driven by six `NavEvent`s. Settings persist in `~/.config/idcs/operator_display.json`. |
| `rpi/display/inputs.py` | Inputs → `NavEvent`: panel joystick (hysteresis, up/down auto-repeat), GPIO menu buttons, USB keyboard read from `/dev/input`. |
| `rpi/display/commands.py` | Operator agent client: one request in flight, timeout and reconnect, status polling. |
| `common/pc_mode.py`, `jetson/pc_mode_channel.py`, `pc/mode_agent.py` | Narrow PC-initiated task/acknowledgement protocol and simulator service owner. |
| `rpi/display/status.py` | Link freshness and rates (video, perception, controller, panel) and derived display state (panel mode, alerts, track list). |
| `rpi/display/hud.py` | Source-aware Cairo operational HUD, bounded histories, cached tiles and stale suppression. |
| `common/hud_geometry.py`, `common/hud_telemetry.py` | Shared smooth tape/FF geometry and bounded display-only simulator telemetry contract. |
| `pc/hud_relay.py` | Read-only simulator loopback subscriptions, outbound PUSH to Pi; no video or controller authority. |
| `rpi/operator_display.py` | The app: config, sockets, 30 Hz tick, menu tree and optional runtime report. |

Only `rpi.runtime_control` touches the ADC and GPIO. It publishes a local
`PanelState` (`rpi.operator_display.panel_state_endpoint`, loopback): joystick
normalized motion rates in [-1, 1] (yaw right, pitch up positive) and the state
of every configured GPIO input role. A new menu button is therefore a config change only: add a pin
for the role `menu`, `menu_select` or `menu_back` under `rpi.gpio.inputs`.

The Pi runtime also mirrors its sampled `ManualControlState` on
`rpi.runtime_control.sim_panel_pub_endpoint` for the PC software-SIM panel
bridge. When the Pi transport connects, this physical GPIO/joystick state
replaces the synthetic auto-armed SIM state. A missing or stale Pi stream
disables SIM control; it does not silently restore synthetic authority during
that bridge process lifetime. The installed service expects the Pi and holds
SIM control disabled even if Pi service never connects. Explicit panel-less
SIM runs omit the physical-panel endpoint and retain synthetic auto enable
after a short startup guard.
The bridge output remains PC loopback-only and cannot arm the Jetson gimbal.
GPIO12 gates SIM automatic tracking just as it gates gimbal automatic tracking;
GPIO5 selects manual joystick control; GPIO16 only gates the fire-button
indication. The Pi mirror is not a path for menu navigation or an exact-frame
ROI seed; those remain separate display/trial protocols.

## Operational HUD (implemented 2026-10-07)

The mirrored RTP feed contains Jetson's detector/tracker OSD, not the final
PC window. Previously the PC added its operational HUD locally and Pi showed
only panel/menu overlays. This implementation gap was not an intentional
information restriction. Pi now draws the centre reticle, smooth fractional
heading/elevation tapes, target histories, selection/range/threat/rank cues,
parallax aim cue, PID/feedforward rates and states, and freshness locally.
Cues depend on fresh available source data; absent range/threat/rank or physical
pose is not invented. MPC terms remain excluded. Menu, alerts and fire chip
are composited above the HUD.

Video remains Jetson RTP directly to Pi, with no PC re-encode hop. Camera HUD
uses existing Jetson metadata, perception, pose and diagnostics. SIM HUD uses
`pc.hud_relay`: loopback pose5572, perception5574 and diagnostics15565 ->
PC-initiated 30 Hz PUSH -> Pi PULL5594. Configure `operator.hud` on PC and
`rpi.operator_display.hud_sim_bind` on Pi. PC identity need not be fixed.
This is display-only, bounded JSON (64 KiB), not an actuator/control-clock
channel; it assumes a trusted LAN, not an authenticated Internet endpoint.

Source mode, progressing samples, simulator catalog/session and receive
freshness must agree. Sender/source restarts clear the cache; duplicate samples
cannot refresh component age. Dynamic geometry disappears after 250 ms without
sample progress. Camera pose requires fresh actual measurements and hardware
availability; unavailable hardware does not imply a zero pose. HIL does not
fall back to SIM telemetry. Feedforward explicitly distinguishes ACTIVE,
LIMITED, INHIBITED, STALE and UNAVAILABLE.

**Timing limitation:** decoded video has no frame-ID join on Pi. The screen
states `HUD latest telemetry; video frame join unavailable`; SIM age is
`sample+rx age; transit unknown`. These are latest-data visual aids, not
qualified capture-to-display registration or cross-host control timing.

The initial panel-HUD trial and deploy-75 acceptance are retained in the journal.
Current installed-source authority is the managed host manifest, not those trial
folders. The software SIM session requires its viewer: restarting SIM UI also
cycles the session/target and software source/controller. Recover the intended
software target and verify fresh session/HUD afterward. This is not a qualified
hardware-position handover. Managed visual lock is the candidate source mode described above; isolated trials remain opt-in.

## Cost

The older small-overlay CPU estimate is not applicable to this full HUD.
Initial composition on YUV reduced live return to ~19.8 fps. Moving composition
after hardware colour conversion to BGRx restored ~30 fps. The actual chain is
RTP -> hardware H.264 decode -> selector -> hardware colour conversion ->
BGRx -> tiled overlaycomposition -> waylandsink.

Pi deterministic 300-frame Cairo build p95 was 3.23 ms. Final live 720p30
return on the 1080p monitor held ~30.06 fps, overlay-build p95 10.03 ms; gates
were >=28 fps and <=12 ms p95. Build timing includes tile preparation and
overlay submission, not downstream blending, transit or display latency.
Relay-loss testing kept ~30 fps while removing stale dynamic cues, then
recovered on relay restart. See journal for retained failing and passing runs.

## Extending

- A new page: add a `Page(title, lines=...)` or `Page(title, items=...)` to
  `build_menu`. `lines` is called on every redraw, so it can show live data
  from `SystemStatus`.
- A new display setting: add a `SettingSpec` to `SETTINGS` and read it in
  `OperatorDisplay._draw`.
- Anything that changes the system goes through an `Action(..., confirm=...)`
  and a command the operator agent owns (`common/operator_commands.py`); the
  display itself changes nothing directly.

## Testing without a screen

### Opt-in display-only ROI trial

This code is packaged for software validation but is not enabled by the
managed Pi display unit. A guarded trial must start the display with
`--visual-roi-trial` and an exclusive exact-frame Jetson ROI return pipeline;
the ordinary camera/SIM return does not supply the required frame identity.
The candidate isolated receiver can use verified SIM or native Argus camera
frames. The trial action requires its fresh control-disabled source status and
exact displayed identity under **Operations -> Visual ROI tracking (isolated)**.
It does not switch the operator agent's mode. Enter from Manual ON using a configured menu button or
keyboard, or enter with Manual OFF and GPIO12 auto authority OFF, then turn
Manual ON.

In the trial, readable GPIO12 ON leaves the joystick in the manual-control
path (if a separately authorized software mount owner is active). Readable
GPIO12 OFF captures the joystick for a centred selection
box and removes controller-facing manual authority. A fresh OFF-to-ON edge
requests visual tracking of the exact displayed frame. The joystick returns
to manual camera control only after it is centred. The accepted visual box is
drawn in the return video, but its ROI-SIM auto-follow lease stays disabled.
Unreadable GPIO12, E-stop, Manual OFF, stale panel/video/source or leaving the
editor releases the request. A lost display lease cannot resume a deflected
joystick; ordinary auto control requires a fresh readable GPIO12 OFF after
the trial ends. GPIO16/fire-button behavior is unchanged. This is not live
panel or motor qualification.

For the current bounded ROI session, open Operations -> Visual ROI tracking
(isolated), then select Manual. GPIO12 OFF moves the selection box; OFF-to-ON locks
the current exact image, and OFF releases it. After Manual has been selected,
Manual OFF closes the editor and restores joystick menu navigation. Tracking
geometry/reticle/status are burned into the Jetson return; Pi draws only its
panel, menu, E-stop and fire information over that ROI view. The session is
VPI-only, has no automatic camera follow and restores normal services when its
bound ends. Latest software acceptance and physical-input scope are recorded
in the journal.

While OFF/selecting, Select (a configured menu-select button or USB Enter)
toggles movement and resizing. ROI SIZE marks resize mode: right grows width,
up grows height. Texture refusals show the original `seed:`/`catch-up:` reason
and suggest resizing after OFF. No new GPIO assignment is made. A larger box
may include background and must be checked against target localization.
Camera validation and remaining limits are in [camera ROI validation](camera_roi_validation.md).

Menu navigation/confirmation tests require no screen, GPIO, or service changes:

```bash
python -m unittest discover -s tests -p test_operator_menu.py -v
python -m unittest discover -s tests -p test_panel_hud.py -v
python -m tools.verify_panel_hud --frames 300 --output /absolute/artifact/path
python -m pytest -q tests/test_pc_mode.py tests/test_operator_agent.py
```

The latter tests inject fake systemd runners, guaranteed empty-track source
snapshots, startup failures and dropped acknowledgements. They do not qualify
live switching; verify Camera → Simulator → Camera from the Pi after deployment,
checking return display continuity and that every motor/HIL owner stays off.

### Historical live switching evidence

The 2026-10-06 uncommitted menu trial is superseded by managed releases. Its
backups/reports remain evidence; do not reinstall `IDCS-trials/menu-20261006-02`
or use the old trial installer over managed units. Bounded menu-driven software
switch tests passed, with source readiness roughly 9–16 seconds plus a
three-second stable-frame check in that trial. These are cold source changes,
not seamless cuts. Pi display remains alive with black fallback/menu.

`tools/verify_menu_switch.py` uses the real command/menu construction but does
not operate the physical joystick. Camera/Simulator switching still requires
separate physical-input acceptance. See the journal and current deployment guide.

Replace the sink with a JPEG writer and feed a test stream:

```bash
python -m rpi.operator_display --duration-s 30 \
    --sink "videorate ! video/x-raw,framerate=1/1 ! videoconvert ! jpegenc ! multifilesink location=/tmp/shot-%03d.jpg"
```
