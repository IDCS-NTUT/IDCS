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
| Safety switch | Master arm. No motion, auto or manual, without it. Screen: SAFE while off. |
| Fire-control switch | Auto control enable (with the master arm, no manual, no E-stop): ARMED. |
| Control switch (latching) | Manual: the joystick slews the gimbal through the controller (`controller.manual_rate_limit_rad_s`, `manual_accel_limit_rad_s2`, same travel envelope). Screen: MANUAL. |
| Joystick | Manual mode: gimbal rate. Otherwise: menu navigation. |
| Fire button | Engages the controller's current target while ARMED and tracking: recorded (`engage` in the flight record and controller log; refused presses as `engage_refused` with the reason) and shown as ENGAGE #id. No effector yet. |
| E-stop | Blocks all motion; EMERGENCY STOP on screen. |

Status bar mode chip: E-STOP > SAFE (master arm off) > MANUAL > ARMED >
STANDBY (armed, auto off).

## Menu

- **Targets**: live track list. The cursor outlines the track on the video
  (yellow); right locks it. A locked track (red, LOCK #id) is the selection
  for the whole system (policy `operator_lock`) until released or lost for
  `operator.lock_lost_s`.
- **System**: mode (`operator.modes`: Camera, PC video, Standby; never HIL,
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
configured units only. Messages: `common/operator_commands.py`.

## Pieces

| Module | Role |
|---|---|
| `rpi/display/video.py` | GStreamer: RTP → hardware H.264 decode (`v4l2h264dec`) → `overlaycomposition` → `waylandsink` full screen. A black `videotestsrc` behind an `input-selector` stands in while no stream arrives, so alerts stay visible. |
| `rpi/display/render.py` | cairo drawing of the status bar, alert banner and menu, each as its own small premultiplied-BGRA rectangle; redrawn only when its text changes. |
| `rpi/display/menu.py` | The menu: pages of submenus, settings (`Choice`), actions, or live information lines, driven by six `NavEvent`s. Settings persist in `~/.config/idcs/operator_display.json`. |
| `rpi/display/inputs.py` | Inputs → `NavEvent`: panel joystick (hysteresis, up/down auto-repeat), GPIO menu buttons, USB keyboard read from `/dev/input`. |
| `rpi/display/commands.py` | Operator agent client: one request in flight, timeout and reconnect, status polling. |
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

Replace the sink with a JPEG writer and feed a test stream:

```bash
python -m rpi.operator_display --duration-s 30 \
    --sink "videorate ! video/x-raw,framerate=1/1 ! videoconvert ! jpegenc ! multifilesink location=/tmp/shot-%03d.jpg"
```
