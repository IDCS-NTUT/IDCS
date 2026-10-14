"""Software-only ROI switch authority and direct box steering."""

import pytest

from rpi.display.roi_editor import RoiEditor
from rpi.display.roi_panel import RoiPanelWorkflow


def panel(*, on=False, valid=True, manual=True, emergency=False):
    return dict(control_cmd_enabled=False, inputs={"fire_control": on},
                inputs_valid={"fire_control": valid},
                manual_active=manual, emergency=emergency)


def test_manual_off_selects_and_readable_rising_on_locks_without_camera_follow():
    workflow = RoiPanelWorkflow()
    workflow.open()
    assert workflow.state == "camera"
    assert workflow.update(panel(on=True), available=True) is None
    assert workflow.update(panel(on=True), available=True) is None
    assert workflow.update(panel(), available=True) is None
    assert workflow.state == "selecting"
    assert workflow.update(panel(on=True), available=True) == "lock"
    workflow.submitted("seed-a")
    assert workflow.result("seed-a", True)
    assert workflow.state == "tracking"
    assert workflow.sim_lease() == dict(type="roi_sim_lease_v1", enabled=False,
                                        request_id=None)
    assert workflow.update(panel(on=True), available=True) is None
    assert workflow.update(panel(), available=True) == "release"
    assert workflow.state == "selecting"
    assert workflow.sim_lease()["enabled"] is False
    assert workflow.update(panel(on=True), available=True) == "lock"


def test_unreadable_auto_emergency_or_loss_release_without_false_off_edge():
    for unsafe in (panel(on=False, valid=False), panel(on=True, manual=False),
                   panel(on=True, emergency=True)):
        workflow = RoiPanelWorkflow()
        workflow.open()
        workflow.update(panel(), available=True)
        assert workflow.update(panel(on=True), available=True) == "lock"
        workflow.submitted("seed-b")
        assert workflow.update(unsafe, available=True) == "release"
        assert workflow.state == "camera"
        assert workflow.update(panel(on=True), available=True) is None
        assert workflow.update(panel(), available=True) is None
        assert workflow.update(panel(on=True), available=True) == "lock"
    workflow = RoiPanelWorkflow()
    workflow.open()
    workflow.update(panel(), available=True)
    workflow.update(panel(on=True), available=True)
    workflow.submitted("seed-c")
    assert workflow.update(panel(on=True), available=False) == "release"
    assert workflow.update(panel(on=True), available=True) is None


def test_delayed_seed_ack_cannot_rearm_after_off():
    workflow = RoiPanelWorkflow()
    workflow.open()
    workflow.update(panel(), available=True)
    workflow.update(panel(on=True), available=True)
    workflow.submitted("seed-d")
    assert workflow.update(panel(), available=True) == "release"
    assert not workflow.result("seed-d", True)
    assert workflow.state == "selecting"


def test_unreadable_off_and_initial_on_never_lock():
    workflow = RoiPanelWorkflow()
    workflow.open()
    assert workflow.update(panel(on=True), available=True) is None
    assert workflow.update(panel(on=False, valid=False), available=True) is None
    assert workflow.update(panel(on=True), available=True) is None
    assert workflow.update(panel(on=False), available=True) is None
    assert workflow.update(panel(on=True), available=True) == "lock"


def test_direct_joystick_moves_box_in_two_axes_without_resizing():
    editor = RoiEditor()
    editor.open(1280, 720, tracking_trial=True, direct_select=True)
    original = (editor.x, editor.y, editor.w, editor.h)
    editor.move_joystick(1.0, 1.0, 1.0)
    editor.move_joystick(1.0, 1.0, 1.1)
    assert editor.x > original[0] and editor.y < original[1]
    assert (editor.w, editor.h) == original[2:]
    assert editor.view() is None
    editor.move_joystick(1.0, -1.0, 100.0)
    assert editor.x - original[0] < 100  # a stalled panel cannot cause a large jump
