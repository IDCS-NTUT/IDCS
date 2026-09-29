import json
import struct

import pytest

from rpi.display.inputs import JoystickNavigator, PanelNavigator, key_events
from rpi.display.menu import Action, Choice, Menu, NavEvent, Page, Settings, SettingSpec
from rpi.display.status import SystemStatus

SPECS = {"status_bar": SettingSpec("Status bar", ("on", "off"), "on")}


def _menu(tmp_path, ran=None):
    ran = [] if ran is None else ran
    root = Page("Menu", items=(
        Page("Status", lines=lambda: ["line"]),
        Page("Display", items=(Choice("Status bar", "status_bar"),)),
        Action("Ping", lambda: ran.append(1)),
    ))
    return Menu(root, Settings(SPECS, tmp_path / "s.json")), ran


def test_menu_opens_navigates_and_closes(tmp_path):
    menu, ran = _menu(tmp_path)
    assert menu.view() is None
    assert menu.handle(NavEvent.UP) is False  # closed: only open events act
    menu.handle(NavEvent.RIGHT)
    assert menu.view().rows[0].label == "Status" and menu.view().cursor == 0
    menu.handle(NavEvent.UP)  # wraps
    assert menu.view().cursor == 2
    menu.handle(NavEvent.SELECT)
    assert ran == [1]
    menu.handle(NavEvent.DOWN)
    menu.handle(NavEvent.RIGHT)  # into Status: an information page
    view = menu.view()
    assert view.title == "Menu / Status" and view.lines == ("line",) and view.cursor is None
    menu.handle(NavEvent.LEFT)
    menu.handle(NavEvent.LEFT)
    assert menu.view() is None
    menu.handle(NavEvent.MENU)
    assert menu.is_open
    menu.handle(NavEvent.MENU)
    assert not menu.is_open


def test_choice_steps_and_persists(tmp_path):
    menu, _ = _menu(tmp_path)
    menu.handle(NavEvent.MENU)
    menu.handle(NavEvent.DOWN)
    menu.handle(NavEvent.RIGHT)
    assert menu.view().rows[0].value == "on"
    menu.handle(NavEvent.RIGHT)
    assert menu.view().rows[0].value == "off"
    assert json.loads((tmp_path / "s.json").read_text()) == {"status_bar": "off"}
    assert Settings(SPECS, tmp_path / "s.json")["status_bar"] == "off"


def test_settings_ignore_invalid_stored_values(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"status_bar": "sideways", "unknown": 1}))
    assert Settings(SPECS, path)["status_bar"] == "on"
    path.write_text("{not json")
    assert Settings(SPECS, path)["status_bar"] == "on"


def test_joystick_hysteresis_and_repeat():
    nav = JoystickNavigator()
    assert nav.update(0.5, 0.0, 0.0) == []  # below engage
    assert nav.update(0.0, 0.9, 0.0) == [NavEvent.UP]
    assert nav.update(0.0, 0.4, 0.3) == []  # held above release, before repeat delay
    assert nav.update(0.0, 0.9, 0.6) == [NavEvent.UP]  # repeat
    assert nav.update(0.0, 0.9, 0.7) == []
    assert nav.update(0.0, 0.9, 0.81) == [NavEvent.UP]
    assert nav.update(0.0, 0.1, 0.9) == []  # released
    assert nav.update(-0.9, 0.2, 1.0) == [NavEvent.LEFT]
    assert nav.update(-0.9, 0.2, 5.0) == []  # left/right never repeat
    assert nav.update(0.0, 0.0, 5.1) == []
    assert nav.update(0.0, -0.8, 5.2) == [NavEvent.DOWN]


def test_panel_navigator_ignores_joystick_in_manual_and_edges_buttons():
    nav = PanelNavigator()
    manual = {"manual_active": True, "joystick": [0.0, 1.0], "inputs": {}}
    assert nav.update(manual, 0.0) == []
    idle = {"manual_active": False, "joystick": [0.0, 1.0], "inputs": {"menu_select": True}}
    assert nav.update(idle, 0.1) == [NavEvent.SELECT, NavEvent.UP]
    assert nav.update(idle, 0.2) == []  # button held, stick before repeat
    released = {"manual_active": False, "joystick": [0.0, 0.0], "inputs": {"menu_select": False}}
    assert nav.update(released, 0.3) == []
    estop = {"manual_active": False, "emergency": True, "joystick": [1.0, 0.0], "inputs": {}}
    assert nav.update(estop, 0.4) == []


def test_keyboard_event_decoding():
    fmt = struct.Struct("llHHi")
    data = b"".join([
        fmt.pack(0, 0, 1, 103, 1),  # KEY_UP press
        fmt.pack(0, 0, 1, 103, 0),  # release: ignored
        fmt.pack(0, 0, 0, 0, 0),  # SYN
        fmt.pack(0, 0, 1, 28, 2),  # ENTER auto-repeat
        fmt.pack(0, 0, 1, 30, 1),  # KEY_A: unmapped
    ])
    assert key_events(data) == [NavEvent.UP, NavEvent.SELECT]


def test_status_modes_alerts_and_links():
    status = SystemStatus()
    assert status.panel_mode(10.0) == "NO PANEL"
    assert status.alerts(10.0) == ["NO VIDEO"]
    status.on_panel({"manual_active": False, "control_cmd_enabled": True}, 10.0)
    assert status.panel_mode(10.2) == "ARMED"
    status.on_panel({"emergency": True, "manual_active": True}, 10.3)
    assert status.panel_mode(10.4) == "E-STOP"
    for i in range(31):
        status.on_video_frame(10.0 + i / 30)
    assert status.video.rate_hz(11.0) == pytest.approx(30.0, rel=0.05)
    assert status.alerts(11.0) == ["EMERGENCY STOP"]
    assert status.alerts(13.0) == ["VIDEO LOST"]  # panel stale too: no E-STOP claim
    assert status.panel_mode(13.0) == "NO PANEL"
    assert status.link_lines(13.0)[2].startswith("Controller  never received")


def test_status_tracks_from_snapshot():
    from common.perception import PerceptionSnapshotV2

    frame = {"frame_id": 1, "source_time_ns": 1, "observed_time_ns": 2, "source_clock_domain": "pc",
             "observation_clock_domain": "jetson", "width": 1280, "height": 720}
    box = {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}
    snapshot = PerceptionSnapshotV2.model_validate({
        "sequence": 1, "frame": frame,
        "tracks": [
            {"track_id": 3, "box": box, "class_id": "drone", "confidence": 0.5, "missed_frames": 2},
            {"track_id": 7, "box": box, "class_id": "drone", "confidence": 0.9, "missed_frames": 0},
        ],
        "assessments": [{"track_id": 7, "distance_m": 42.0}],
        "selection": {"track_id": 7, "source_frame_id": 1, "applied_frame_id": 1, "selected_time_ns": 2,
                      "selection_clock_domain": "jetson", "policy": "p"},
    })
    status = SystemStatus()
    status.on_snapshot(snapshot, 5.0)
    assert status.targets_text(5.1) == "2 trk  sel #7"
    lines = status.track_lines(5.1)
    assert lines[0].startswith("> #7") and "42.0 m" in lines[0]
    assert "coast" in lines[1]
    assert status.targets_text(7.0) == "no perception"


def test_render_places_every_element_inside_the_frame(tmp_path):
    pytest.importorskip("cairo")
    from rpi.display.render import StatusBar, render

    menu, _ = _menu(tmp_path)
    menu.handle(NavEvent.MENU)
    images = render(1280, 720, bar=StatusBar("ARMED", ("30 fps", "2 trk", "ctrl tracking")),
                    alerts=["NO VIDEO"], menu=menu.view())
    assert len(images) == 3
    for image in images:
        assert 0 <= image.x and image.x + image.width <= 1280
        assert 0 <= image.y and image.y + image.height <= 720
        assert len(image.data) == image.stride * image.height
        assert any(image.data[3::4])  # something drawn (alpha)
    assert render(1280, 720, bar=None, alerts=[], menu=None) == []


def test_panel_state_message_scales_joystick():
    pytest.importorskip("smbus")
    from common.schemas import ManualControlState
    from rpi.runtime_control import panel_state_message

    state = ManualControlState(src_ts_ms=1, source="t", active=False, emergency=False,
                               joystick_raw=(255, 0), joystick_rate_cmd=(0.5, -1.2))
    message = panel_state_message(state, {"menu_select": True}, max_rate_rad_s=1.0)
    assert message["type"] == "PanelState"
    assert message["joystick"] == [0.5, -1.0]
    assert message["inputs"] == {"menu_select": True} and message["manual_active"] is False


def test_return_mirror_hosts_reach_the_pipeline(tmp_path):
    from jetson.deepstream.runtime import _return_mirror_hosts, build_pipeline_argv

    assert _return_mirror_hosts({}, "pc") == ()
    assert _return_mirror_hosts({"return_mirror_ips": ["192.168.0.3"]}, "pc") == ("192.168.0.3",)
    for bad in ("192.168.0.3", ["pc"], ["a", "a"], ["bad host"]):
        with pytest.raises(ValueError):
            _return_mirror_hosts({"return_mirror_ips": bad}, "pc")
    from dataclasses import replace

    from jetson.deepstream.runtime import RuntimeSettings

    settings = RuntimeSettings("rtp", 5000, tmp_path / "m.txt", "tcp://0.0.0.0:5555", "tcp://0.0.0.0:5564",
                               "pc", 5002, 1280, 720, 30, 20000, False)
    argv = build_pipeline_argv(replace(settings, return_mirror_hosts=("192.168.0.3",)), [])
    assert argv[argv.index("--return-mirror-host") + 1] == "192.168.0.3"
    assert "--return-mirror-host" not in build_pipeline_argv(settings, [])


def test_display_pipeline_description():
    gi = pytest.importorskip("gi")
    try:
        gi.require_version("GstVideo", "1.0")
    except ValueError:
        pytest.skip("GstVideo typelib not installed")
    from rpi.display.video import pipeline_description

    text = pipeline_description(port=5002, jitter_ms=20, sink="waylandsink fullscreen=true", width=1280, height=720)
    assert "udpsrc port=5002" in text and "v4l2h264dec" in text and "overlaycomposition" in text
    assert "waylandsink fullscreen=true" in text
