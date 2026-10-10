# Operator display (Pi panel screen)

Read [deployment.md](deployment.md) and the host manifests for current source
paths. Dated trial/release deployment paragraphs below are historical, not
instructions to reinstall old sources. Consolidation preserves panel policies;
real switch/motor acceptance remains deferred.

Shared UI styling, rendering ownership and controlled preview/benchmark checks:
[ui_style.md](ui_style.md). Styling does not change GPIO authority or lamp policy.

Deployed HUD follow-up (introduced in `deploy-79`, retained in `deploy-82`): E-stop/fire and authority/selection/REC
strip are mandatory, even with a menu open; old hide/detail preferences are
ignored and their Display menu page is removed. Steady source/panel-mode/fps
menu repeats are removed, but source-switch progress/errors, actions, freshness,
HUD telemetry and raw Panel diagnostics remain. Decoded return <20fps for more
than one second after a one-second sampling warm-up raises LOW VIDEO FPS; loss
and reconnect have separate/reset states. Fire-arm unreadability shows unknown,
never infers switch-off from lamp darkness. Range in target lists/picker uses
whole metres without an approximate marker; full-precision data is preserved.

`rpi.operator_display` puts the Jetson's return video on the control panel's
HDMI screen and draws the operator interface over it. It subscribes to
perception snapshots, controller diagnostics and the local panel state; what
the operator changes from the menu (target lock, target type, mode,
recording) is a request to the Jetson's operator agent, which owns that
authority.

Deployed scene menu (`deploy-81`, `0ce7a11`): **System -> Mode ->
Simulator -> Single / Swarm**, with Cancel-first confirmation. The external
mode stays Simulator; the agent reports the actual scene separately from fresh
PC service state, not a display preference. Scene changes use the full ordered
source switch and readiness/rollback guards and clear source-scoped locks.
Single preserves synthetic-observation SIM control. Swarm preserves the
existing detection-fed software-mount preset, so its picker/locks use Jetson's
visual catalog, not simulator truth IDs. The internal `sim_swarm` mode does not
become another top-level menu item and cannot grant hardware/HIL authority.
Swarm target locks use the detector's session; SIM HUD pose uses the separate
fresh `sim_hud_session` from the PC producer. Missing/stale identity hides geometry
instead of substituting detector identity. The running scene is Swarm; physical
panel switching acceptance remains deferred to the user at the bench.

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
- **System -> Clear Swarm score** exists only while Simulator / Swarm is selected,
  with Cancel-first confirmation. It clears eliminated/breached counters and
  starts a new score epoch; current live targets form its initial spawn population.
  Targets, dwell, pose, RNG, spawn schedule, target locks and recordings survive.
  A `score_reset` event is appended to recorder history; nothing is deleted.

The confirmation freezes the simulator session. The operator agent checks fresh
software Swarm state, no pending action and no motor/HIL activity, then uses the
existing acknowledged PC polling channel. The PC worker sends a bounded request
to the current streamer on loopback only (`operator.sim_score_endpoint`, default
`tcp://127.0.0.1:5580`); it never runs `systemctl clear_score`, restarts a source,
or reaches motion command sockets. The render thread applies the counter change.
Source-session mismatch, duplicate/retired requests and timeouts cannot silently
clear another scenario; success requires a matching epoch acknowledgement.

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

2026-10-10 tested candidate (not deployed): target IDs are internal, absent
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
cancels without sending a command. The fixed first row **Use planner**
explicitly returns to Planner selection. Target rows are frozen when opened;
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
This menu-only correction is active on the Pi from snapshot
`IDCS-trials/menu-pitch-20261007-02` (2026-10-07), now retained in
`IDCS-trials/panel-hud-20261007-01`. Only the display unit was
changed/restarted; GPIO/manual, video producers and mode authority were not.
The fire state is shown in a separate bottom-right chip as `fire off`,
`fire ready`, or `FIRE BUTTON`. The Panel page shows both switch states and
the raw button.
**Deployment distinction:** the policy below is active from a pre-commit snapshot
`IDCS-trials/safety-20261006-01` on Pi GPIO/display, Jetson operator agent,
PC mode agent and simulator streamer units. The tagged runtime checkouts are
unchanged (Pi/Jetson deploy-74, PC deploy-62); deploy-74's old safety policy was
visual selection. The target-picker trial now replaces active software service
paths with `IDCS-trials/target-20261006-01`; historical safety/menu snapshots
remain for rollback. Jetson's physical bridge remains inactive and its new activity
source is not activated. Camera mode was restored after SSH verification.

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

PC simulator/PC mode agent, Jetson operator agent and Pi GPIO/display consumers
were activated together and live-tested. Reports and scoped unit backups are
in the active snapshot's `.safety-deployment/`. Camera-mode safety OFF, simulator
tracking ON, status-relay loss OFF/contact invalidation and recovery ON passed
with real PanelState and read-only pin samples. Manual exception and E-stop
passed deterministic FakeGPIO tests on Pi; no physical switch was operated.
Gimbal activity remains contract-tested only: the bridge's updated source must
be activated in a future permitted rollout and must not be started until the
repaired-wiring/RS485 standing rule is satisfied. Keep the rollback snapshots;
an isolated test-only snapshot must not be confused with an active unit source.

## Menu

- **Targets**: the active control source's track list, plus **Use planner**.
  Camera uses Jetson tracker IDs; software Simulator uses simulator ground-truth
  IDs. The same numeric ID in these sources is not the same target. A confirmed
  lock sets **Operator** selection and is red (LOCK #id) only after it is applied.
  Loss clears the lock after `operator.lock_lost_s` but keeps Operator mode:
  no automatic replacement or planner fallback. Select another target or
  explicitly choose **Use planner**. GPIO5 manual motor control is unrelated.
- **System**: mode (`operator.modes`: Camera, Simulator, Standby; never HIL,
  and refused while a motor unit runs), recording on/off, target type (which
  detector classes the planner may select; persisted on the Jetson). Each
  asks for confirmation (Cancel first).
- **Status**, **HUD telemetry**, **Panel** (every input's state), **About**.
  **Display** was removed in deploy-79 because its only controls hid required
  HUD information.

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
socket rebuilt. Mode and recording changes run `sudo -n systemctl` on the
allowlisted Jetson units only. Messages: `common/operator_commands.py`.

Camera catalogs carry a UUID per DeepStream tracker process. Simulator catalogs
carry the streamer's applied-control session. Locks require the active source,
exact session and an observed target; busy transitions, stale catalogs and
foreign/restarted source IDs are refused. Duplicate/out-of-order catalogs cannot
renew the 0.5 s freshness window; relays add local elapsed age rather than
subtracting cross-host clocks. Mode remains Operator after agent restart, but
lock IDs do not persist. Configured selectors wait for their first authority
message, and stale Operator authority suppresses selection rather than falling
back. These watchdogs grant no physical motion or control-clock qualification.

The PC mode agent sends simulator catalogs in its existing outbound polls;
Jetson returns selection authority in replies, which the PC republishes only
on `operator.sim_selection_bind` (loopback 5579). The ground-truth perception
stream (5574), simulator controller and PC SIM HUD then share the applied
choice. Pi uses that same catalog for its picker/LOCK overlay. Jetson's rendered
video still contains learned detections/NvDCF tracks; their IDs are never
mistaken for simulator control IDs. No inbound PC LAN control port is required.

### Camera / simulator switching

Open **System → Mode → Camera / Simulator → Confirm**. Cancel is first.
The operator agent replies immediately with a busy state; the Pi display
continues receiving status and keeps its black fallback/menu visible while
the video source changes. The System page shows the transition stage, PC
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

Simulator motion is local synthetic input → simulator-only controller →
ideal simulated plant. It neither consumes physical encoder pose nor sends
motor commands. Camera mode retains its shadow controller. Mode actions are
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
still uses deployment configuration `net.return_ip`: update it if the PC
address changes. Pi video uses its independent `net.return_mirror_ips` entry.
Hardware/HIL timing endpoints are unchanged. This ZMQ channel, like the
existing operator channel, assumes a trusted LAN; it is not an authenticated
Internet-facing service.

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

Initial scoped snapshot: `~/Desktop/project/IDCS-trials/panel-hud-20261007-01`
for Pi display and the new PC user `idcs-hud-relay.service` only. Pi GPIO/manual
and PC/Jetson source/controller units retain their previous snapshots. Current
live SIM was not switched; camera rendering passed deterministic fixtures, not
a new live camera trial. Unit backups, installer, reports and screenshots:
`~/Desktop/project/deployments/panel-hud-20261007-01/` on PC and Pi.
Rollback restores the saved Pi display unit, daemon-reloads and restarts only
display; stop/disable PC relay. The installer does not automatically roll back
a failed activation. No GPIO, physical controller, bridge or serial activation.

Superseding committed deployment: `24c824f` / `deploy-75` at
`~/Desktop/project/IDCS-releases/deploy-75` on all hosts. PC viewers/relay and
Pi display now reference this clean worktree. ROI tools/libraries are installed
but not started as a console/video mode. Reports/backups/native builds are in
`deployments/release-75-20261007/`. GPIO/manual and Jetson production owners
remain unchanged. The software SIM session **Requires its viewer**: restarting
SIM UI also cycles the session/target and software source/controller; explicitly
recover the same software target and verify fresh source/session/HUD afterward.
Do not treat this as a hardware-safe source transition or promise unchanged SIM
worker PIDs. Final live return30.06fps/build p958.44ms; timing limits above remain.

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

### Uncommitted live trial (2026-10-06)

User requested deployment/testing before committing. Active units therefore
use the separate `~/Desktop/project/IDCS-trials/menu-20261006-02` snapshot,
not the clean tagged runtime checkout. The PC mode agent is enabled and the
system was left in Camera after two successful switch cycles. Source package
hashes, live reports and physical-input limitations are recorded in the journal.

`tools/install_menu_trial.py --host pc|jetson|pi` installs a snapshot's affected
units with rollback backups; `--rollback` restores unit files only, with no
service restarts. These backups are `.deployment/` artifacts outside Git.
`tools/verify_menu_switch.py --report /absolute/artifact/path.json` runs a
bounded live trial from the Pi using the real menu construction and command
client. It navigates a menu instance, not the physical joystick or running
display's input device; actual video is checked from metadata plus Pi display
logs/screenshots. Its commands are restricted to Camera/Simulator.

Current switches are cold source changes, not seamless camera cuts: successful
video readiness took roughly 9–16 seconds, followed by a 3-second stable-frame
check. The Pi display process remains alive with its black fallback and menu.

Replace the sink with a JPEG writer and feed a test stream:

```bash
python -m rpi.operator_display --duration-s 30 \
    --sink "videorate ! video/x-raw,framerate=1/1 ! videoconvert ! jpegenc ! multifilesink location=/tmp/shot-%03d.jpg"
```
