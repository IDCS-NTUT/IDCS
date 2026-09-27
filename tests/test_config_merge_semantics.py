from __future__ import annotations

from common.config import merge_config_layers
from common.config_sync import merge_config_maps


def test_overlays_merge_recursively_in_every_loader() -> None:
    base = {"gimbal": {"yaw_addr": 1, "pitch_motor_a_addr": 2, "limits": {"a": 1, "b": 2}},
            "serial_io": {"schedule": [{"name": "x"}, {"name": "y"}]}}
    overlay = {"gimbal": {"pitch_motor_b_enabled": False, "limits": {"b": 3}},
               "serial_io": {"schedule": [{"name": "z"}]}}
    merged = merge_config_maps(base, overlay)
    assert merged == merge_config_layers(base, overlay)
    assert merged["gimbal"] == {"yaw_addr": 1, "pitch_motor_a_addr": 2,
                                "pitch_motor_b_enabled": False, "limits": {"a": 1, "b": 3}}
    assert merged["serial_io"]["schedule"] == [{"name": "z"}]  # lists replace
    assert base["gimbal"]["limits"] == {"a": 1, "b": 2}  # inputs untouched
