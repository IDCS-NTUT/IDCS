"""Simulated mount driven by ControlIntent (no hardware)."""

from __future__ import annotations

import time

import pytest

from common.gimbal.mks_servo42_rs485 import MksServo42Axis
from common.schemas import CamState
from common.sim_mode import require_simulation_loopback_endpoint
from pc.streamer import open_source


def _cap():
    cfg = {
        "sim": {"renderer": "cpu", "use_jetson_cam_state": False, "plant_model": {"mode": "ideal"},
                "scene": {"mode": "static_targets", "targets": [], "buildings": [], "cubes": []}},
        "gimbal": {"yaw_rate_limit_rad_s": 0.8, "pitch_rate_limit_rad_s": 0.8},
    }
    return open_source("sim", 64, 36, 60, cfg=cfg, sim_control_enabled=True)


def _intent(yaw: float, pitch: float, *, lease_ns: int = 50_000_000, mode: str = "live") -> dict:
    now = time.monotonic_ns()
    return {"type": "ControlIntent", "sequence": 1, "observation_sequence": 1,
            "issued_monotonic_ns": now, "valid_until_monotonic_ns": now + lease_ns,
            "mode": mode, "yaw_rate_rad_s": yaw, "pitch_rate_rad_s": pitch, "reason": "tracking"}


def test_live_intent_moves_the_mount_at_the_measured_f6_speed() -> None:
    cap = _cap()
    cap.handle_control_cmd(_intent(0.3, -0.12))
    assert cap._resolve_command(time.monotonic()) == pytest.approx((
        MksServo42Axis.quantized_speed_rad_s(0.3, 1.0, 0.8),
        MksServo42Axis.quantized_speed_rad_s(-0.12, 1.0, 0.8),
    ))
    assert cap._resolve_command(time.monotonic())[0] != 0.3  # quantized, not ideal


def test_expired_or_shadow_intent_does_not_move_the_mount() -> None:
    cap = _cap()
    cap.handle_control_cmd(_intent(0.3, 0.3, lease_ns=0))
    time.sleep(0.001)
    assert cap._resolve_command(time.monotonic()) == (0.0, 0.0)
    cap.handle_control_cmd(_intent(0.3, 0.3, mode="shadow"))
    assert cap._resolve_command(time.monotonic()) == (0.0, 0.0)


def test_simulated_camstate_carries_exact_sample_times() -> None:
    cap = _cap()
    ok, _ = cap.read()
    assert ok
    state = CamState(**cap.build_cam_state(1, 0))
    assert state.pan_sample_monotonic_ns == state.tilt_sample_monotonic_ns == cap.last_frame_source_ns
    assert state.state_monotonic_ns >= state.pan_sample_monotonic_ns


def test_sim_panel_endpoint_must_be_loopback() -> None:
    assert require_simulation_loopback_endpoint("tcp://127.0.0.1:15559", "x")
    with pytest.raises(ValueError):
        require_simulation_loopback_endpoint("tcp://192.168.0.5:5559", "x")
