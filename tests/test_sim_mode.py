from __future__ import annotations

import pytest

from common.sim_mode import resolve_simulation_motion_mode


def test_existing_false_selector_uses_stable_substitute() -> None:
    mode = resolve_simulation_motion_mode({"use_jetson_cam_state": False})

    assert mode.name == "stable_substitute"
    assert mode.use_jetson_cam_state is False
    assert mode.moves_physical_mount is False


def test_existing_true_selector_is_hardware_in_loop() -> None:
    mode = resolve_simulation_motion_mode({"use_jetson_cam_state": True})

    assert mode.name == "hardware_in_loop"
    assert mode.use_jetson_cam_state is True
    assert mode.moves_physical_mount is True


def test_selector_rejects_truthy_non_boolean_values() -> None:
    with pytest.raises(ValueError, match="must be boolean"):
        resolve_simulation_motion_mode({"use_jetson_cam_state": "false"})
