# Operator display (Pi panel screen)

`rpi.operator_display` puts the Jetson's return video on the control panel's
HDMI screen and draws the operator interface over it. It subscribes to
perception snapshots, controller diagnostics and the local panel state; what
the operator changes from the menu (target lock, target type, mode,
recording) is a request to the Jetson's operator agent, which owns that
authority.

## Panel controls

| Control | Effect |
|---|---|
| Fire-control switch (GPIO12) | Enables automatic tracking only. Off: SAFE in auto mode; it does not block manual joystick slew. |
| Safety switch (GPIO16) | Enables the fire button and its screen indication. It has no effect on motor movement. |
| Control switch (GPIO5, latching) | Selects manual mode: the joystick slews through the controller (`controller.manual_rate_limit_rad_s`, `manual_accel_limit_rad_s2`, same travel envelope), even with GPIO12 low. Screen: MANUAL. |
| Joystick | Manual mode: gimbal rate. Otherwise: menu navigation. |
| Fire button | With GPIO16 on, shows FIRE BUTTON on the screen. A tracked-target press while ARMED is also recorded as `engage`; other presses are recorded as `engage_refused`. No effector is connected. |
| E-stop | Blocks all motion; EMERGENCY STOP on screen. |

Status bar mode chip: E-STOP > MANUAL (GPIO5 latched) > SAFE (auto mode,
GPIO12 off) > ARMED (auto mode, GPIO12 on).
The fire state is shown in a separate bottom-right chip as `fire off`,
`fire ready`, or `FIRE BUTTON`. The Panel page shows both switch states and
the raw button.
The indicator policy below is deployed to the Pi as **deploy-74**. Jetson
runtime source matches; its physical motor bridge remains inactive. PC/Jetson
menu/video processes remain on the already-tested menu trial 02.

| Indicator | Meaning |
|---|---|
| Fire-control lamp (GPIO22) | A current detection or observed target track is available. |
| Safety lamp (GPIO25) | The selected visual track is currently observed, not coasting. This does not claim the physical gimbal is moving. |
| Yellow (GPIO23) | Fresh physical bridge proof that every configured motor has valid, recent feedback. Not simulator availability or fire-button state. |
| Green (GPIO18) | Panel GPIO runtime running (unchanged). |
| Red (GPIO24) | E-stop active (unchanged). |

These outputs are active-low. Target lamps turn on immediately; each has a
configurable off delay (default one second of continuous target absence).
Reacquisition cancels that delay. Stale perception overrides the delay and
turns both off after 0.5 seconds without a progressing accepted frame.
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
Lamps depend on perception, never on these contacts or controller authority,
so an initially dark contact cannot prevent target reacquisition from lighting
its lamp. No flashing probe pulses are used. GPIO5 manual mode, the fire button,
and E-stop are independent of lamp power. E-stop still blocks actuation
immediately; status lamps continue showing their respective conditions.

The lamp hysteresis does not hold a controller target alive or bypass its timing
watchdogs. Loss beyond the off delay removes GPIO12 auto authority, so automatic
idle-return cannot continue after that cut; manual authority is unchanged.
Receive-age checks use Pi monotonic time, not subtraction of PC/Jetson clocks;
these are indication watchdogs, not production control-timing qualification.
Replay/out-of-order source times cannot refresh them. After a remote monotonic
clock resets (host reboot), restart the Pi GPIO runtime to reset stream frontiers.

## Menu

- **Targets**: live track list. The cursor outlines the track on the video
  (yellow); right locks it. A locked track (red, LOCK #id) is the selection
  for the whole system (policy `operator_lock`) until released or lost for
  `operator.lock_lost_s`.
- **System**: mode (`operator.modes`: Camera, Simulator, Standby; never HIL,
  and refused while a motor unit runs), recording on/off, target type (which
  detector classes the planner may select; persisted on the Jetson). Each
  asks for confirmation (Cancel first).
- **Status**, **Panel** (every input's state), **Display**, **About**.

## Command path

```
display REQ --OperatorCommand--> jetson.operator_agent REP (net.zmq_operator_command)
                                   | PUB OperatorSelection (net.zmq_operator_selection, loopback)
                                   v
                            DeepStream target selector: lock + target classes
```

The agent replies with its state (mode, recording, lock, target classes) and
is polled every 2 s; a request unanswered for 1.5 s is reported and the
socket rebuilt. Mode and recording changes run `sudo -n systemctl` on the
allowlisted Jetson units only. Messages: `common/operator_commands.py`.

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
| `rpi/display/video.py` | GStreamer: RTP → hardware H.264 decode (`v4l2h264dec`) → `overlaycomposition` → `waylandsink` full screen. A black `videotestsrc` behind an `input-selector` stands in while no stream arrives, so alerts stay visible. |
| `rpi/display/render.py` | cairo drawing of the status bar, alert banner and menu, each as its own small premultiplied-BGRA rectangle; redrawn only when its text changes. |
| `rpi/display/menu.py` | The menu: pages of submenus, settings (`Choice`), actions, or live information lines, driven by six `NavEvent`s. Settings persist in `~/.config/idcs/operator_display.json`. |
| `rpi/display/inputs.py` | Inputs → `NavEvent`: panel joystick (hysteresis, up/down auto-repeat), GPIO menu buttons, USB keyboard read from `/dev/input`. |
| `rpi/display/commands.py` | Operator agent client: one request in flight, timeout and reconnect, status polling. |
| `common/pc_mode.py`, `jetson/pc_mode_channel.py`, `pc/mode_agent.py` | Narrow PC-initiated task/acknowledgement protocol and simulator service owner. |
| `rpi/display/status.py` | Link freshness and rates (video, perception, controller, panel) and derived display state (panel mode, alerts, track list). |
| `rpi/operator_display.py` | The app: config, sockets, 20 Hz tick, menu tree. |

Only `rpi.runtime_control` touches the ADC and GPIO. It publishes a local
`PanelState` (`rpi.operator_display.panel_state_endpoint`, loopback): joystick
deflection in [-1, 1] (x right, y up) and the state of every configured GPIO
input role. A new menu button is therefore a config change only: add a pin
for the role `menu`, `menu_select` or `menu_back` under `rpi.gpio.inputs`.

## Cost

Measured on the Pi 4 at 720p30: about 5% of one core for receive, decode and
display; the overlay blend adds under 1%, because only its rectangles are
blended and they are re-rendered only on change.

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
