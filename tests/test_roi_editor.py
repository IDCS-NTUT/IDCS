"""Software-only panel ROI editor contracts."""

from rpi.display.menu import NavEvent
from rpi.display.inputs import PanelNavigator
from rpi.display.roi_editor import RoiEditor


def test_four_way_editor_confirms_bounded_rectangle() -> None:
    editor = RoiEditor()
    editor.open(1280, 720)
    assert editor.box() == (592 / 1280, 324 / 720, 96 / 1280, 72 / 720)
    editor.handle(NavEvent.LEFT)
    assert editor.x == 584
    editor.handle(NavEvent.DOWN)
    editor.handle(NavEvent.RIGHT)
    assert editor.y == 332
    for _ in range(3):
        editor.handle(NavEvent.DOWN)
    assert editor.view().rows[editor.cursor].label == "Confirm ROI"
    assert editor.handle(NavEvent.RIGHT) == "confirm"
    assert not editor.active


def test_editor_cancel_and_bounds() -> None:
    editor = RoiEditor()
    editor.open(32, 32)
    for _ in range(100):
        editor.handle(NavEvent.RIGHT)
    x, y, w, h = editor.box()
    assert 0 <= x <= 1 and 0 <= y <= 1
    assert 12 / 32 <= w <= 1 and 12 / 32 <= h <= 1
    assert x + w <= 1 and y + h <= 1
    assert editor.handle(NavEvent.BACK) == "cancel"
    assert editor.view() is None


def test_editor_never_uses_fire_or_manual_events() -> None:
    editor = RoiEditor()
    editor.open(1280, 720)
    assert editor.handle(NavEvent.SELECT) is None
    assert editor.active
    assert editor.handle(NavEvent.MENU) == "cancel"


def test_horizontal_repeat_is_enabled_only_during_roi_editing() -> None:
    nav = PanelNavigator()
    panel = {"joystick": [1.0, 0.0], "inputs": {}}
    assert nav.update(panel, 0.0) == [NavEvent.RIGHT]
    assert nav.update(panel, 1.0) == []
    assert nav.update(panel, 1.2, roi_editing=True) == [NavEvent.RIGHT]
    assert nav.update(panel, 1.41, roi_editing=True) == [NavEvent.RIGHT]
    assert nav.update({"joystick": [0.0, 0.0]}, 1.5, roi_editing=True) == []
