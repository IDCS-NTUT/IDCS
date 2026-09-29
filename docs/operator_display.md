# Operator display (Pi panel screen)

`rpi.operator_display` puts the Jetson's return video on the control panel's
HDMI screen and draws the operator interface over it. It is read-only: it
subscribes to perception snapshots, controller diagnostics and the local
panel state, and commands nothing.

## Pieces

| Module | Role |
|---|---|
| `rpi/display/video.py` | GStreamer: RTP → hardware H.264 decode (`v4l2h264dec`) → `overlaycomposition` → `waylandsink` full screen. A black `videotestsrc` behind an `input-selector` stands in while no stream arrives, so alerts stay visible. |
| `rpi/display/render.py` | cairo drawing of the status bar, alert banner and menu, each as its own small premultiplied-BGRA rectangle; redrawn only when its text changes. |
| `rpi/display/menu.py` | The menu: pages of submenus, settings (`Choice`), actions, or live information lines, driven by six `NavEvent`s. Settings persist in `~/.config/idcs/operator_display.json`. |
| `rpi/display/inputs.py` | Inputs → `NavEvent`: panel joystick (hysteresis, up/down auto-repeat), GPIO menu buttons, USB keyboard read from `/dev/input`. |
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
- Anything that changes the system (mode switching, service control) must go
  through an explicit, confirmed action and a service that owns that
  authority; the display itself stays read-only.

## Testing without a screen

Replace the sink with a JPEG writer and feed a test stream:

```bash
python -m rpi.operator_display --duration-s 30 \
    --sink "videorate ! video/x-raw,framerate=1/1 ! videoconvert ! jpegenc ! multifilesink location=/tmp/shot-%03d.jpg"
```
