from __future__ import annotations

import pytest

from common.control import AxisPair, ControlConfig, LaserAimingControlConfig
from common.schemas import (
    ControlGimbalObservation, ControlObservation, ControlSafetyObservation,
    ControlTargetObservation, ControlTransportObservation, ManualControlState,
)
from jetson.control.engagement import EngagementMonitor
from jetson.control.pid import AxisPIDConfig, BasicPID
from jetson.control.video_controller import VideoControllerCore, VideoControllerPolicy
from jetson.control_observation import ControlObservationAssembler


def _core(**policy) -> VideoControllerCore:
    axis = AxisPIDConfig(6.0, 0.0, 0.0, 0.0, 0.8, 3.5)
    return VideoControllerCore(BasicPID(axis, axis), VideoControllerPolicy(
        live_authorized=True, max_travel_rad=0.3, **policy))


def _obs(tick: int, *, manual=True, arm=True, emergency=False, joystick=(0.4, -0.2), yaw=0.0,
         auto=False, fire=False, target_valid=False) -> ControlObservation:
    return ControlObservation(
        sequence=tick + 1, created_monotonic_ns=1_000_000_000 + tick * 20_000_000,
        target=(ControlTargetObservation(
            valid=True, track_id=7, target_center_px=(640.0, 360.0), aim_reference_px=(640.0, 360.0),
            pixel_error=(0.0, 0.0), bearing_error_rad=(0.0, 0.0), on_target=True,
        ) if target_valid else ControlTargetObservation(valid=False)),
        gimbal=ControlGimbalObservation(valid=True, yaw_rad=yaw, pitch_rad=0.0,
                                        yaw_rate_rad_s=0.0, pitch_rate_rad_s=0.0, sample_age_ms=5.0),
        transport=ControlTransportObservation(),
        safety=ControlSafetyObservation(valid=True, auto_allowed=auto, manual_active=manual,
                                        emergency_active=emergency, sample_age_ms=10.0, master_arm=arm,
                                        fire=fire, manual_rate_rad_s=joystick),
    )


def test_manual_joystick_drives_live_intent_slew_limited_and_clamped() -> None:
    core = _core(manual_rate_limit_rad_s=0.3, manual_accel_limit_rad_s2=2.0)
    rates = []
    for tick in range(30):
        intent = core.decide(_obs(tick, joystick=(1.0, -0.2)), None).intent
        assert intent.mode == "live" and intent.reason == "manual"
        rates.append((intent.yaw_rate_rad_s, intent.pitch_rate_rad_s))
    # 2 rad/s^2 over 20 ms ticks: 0.04 rad/s per tick, up to the 0.3 clamp.
    assert rates[0] == (0.0, 0.0)
    assert rates[1] == pytest.approx((0.04, -0.04))
    assert rates[-1] == pytest.approx((0.3, -0.2))


@pytest.mark.parametrize("kwargs", [{"arm": False}, {"emergency": True}, {"manual": False}])
def test_manual_needs_master_arm_and_no_estop(kwargs) -> None:
    core = _core()
    for tick in range(5):
        intent = core.decide(_obs(tick, **kwargs), None).intent
    assert intent.reason != "manual"
    assert (intent.yaw_rate_rad_s, intent.pitch_rate_rad_s) == (0.0, 0.0)


def test_manual_stops_at_travel_envelope_but_may_return() -> None:
    core = _core(manual_accel_limit_rad_s2=20.0)
    core.decide(_obs(0, yaw=0.0), None)  # origin at yaw 0
    out = [core.decide(_obs(tick, yaw=0.3, joystick=(0.5, 0.0)), None) for tick in range(1, 5)][-1]
    assert out.travel_held[0] and out.intent.yaw_rate_rad_s == 0.0
    back = [core.decide(_obs(tick, yaw=0.3, joystick=(-0.5, 0.0)), None) for tick in range(5, 15)][-1]
    assert back.intent.yaw_rate_rad_s < 0.0


def test_auto_requires_master_arm() -> None:
    def state(arm: bool) -> ManualControlState:
        return ManualControlState(src_ts_ms=0, source="test", active=False, emergency=False,
                                  control_cmd_enabled=True, master_arm=arm,
                                  joystick_raw=(128, 128), joystick_rate_cmd=(0.0, 0.0))

    assembler = ControlObservationAssembler(ControlConfig(
        mode="rate", loop_hz=50.0, fx_px=1000.0, fy_px=1000.0, cx_px=640.0, cy_px=360.0,
        aim_mode="camera_center", kp=AxisPair(1, 1), kd=AxisPair(0, 0), ki=AxisPair(0, 0),
        rate_limits=AxisPair(1, 1), accel_limits=AxisPair(1, 1), deadband_px=0, smooth_px_alpha=0,
        lost_target_timeout_ms=100, reinit_on_lost=True, target_selector="preselected", yaw_sign=1,
        pitch_sign=-1, frame_size=(1280, 720), fov_deg=None,
        laser=LaserAimingControlConfig(1, "infinite", 10)))
    assembler.update_manual_state(state(False), received_at=10.0)
    assert not assembler.build(now=10.01).safety.auto_allowed
    assembler.update_manual_state(state(True), received_at=10.0)
    assert assembler.build(now=10.01).safety.auto_allowed


def test_fire_engages_tracked_target_once_per_press() -> None:
    monitor = EngagementMonitor()
    core = _core()
    obs = _obs(0, manual=False, auto=True, fire=True, target_valid=True)
    intent = core.decide(obs, None).intent.model_copy(update={"reason": "tracking"})
    record = monitor.update(obs, intent)
    assert record is not None and record["type"] == "engage" and record["track_id"] == 7
    assert monitor.last is not None and monitor.last.track_id == 7
    assert monitor.update(_obs(1, manual=False, auto=True, fire=True, target_valid=True), intent) is None


@pytest.mark.parametrize("kwargs,why", [
    ({"arm": False}, "master_arm_off"),
    ({"manual": True, "auto": False}, "manual"),
    ({"emergency": True}, "emergency"),
    ({"auto": False, "manual": False}, "auto_not_armed"),
])
def test_fire_refused_outside_armed_tracking(kwargs, why) -> None:
    monitor = EngagementMonitor()
    params = {"manual": False, "auto": True, "fire": True, "target_valid": True, **kwargs}
    obs = _obs(0, **params)
    intent = _core().decide(obs, None).intent.model_copy(update={"reason": "tracking"})
    record = monitor.update(obs, intent)
    assert record is not None and record["type"] == "engage_refused" and record["why"] == why
    assert monitor.last is None
