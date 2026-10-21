import json

import pytest

from common.perception import PerceptionSnapshotV2
from common.schemas import CamState
from rpi.panel_indicators import (IndicatorState, PanelIndicators, RoiLampLease,
                                  RoiManualCapture, RoiAutoBlock)
from common.control_activity import ActivityEnvelope, AppliedControlActivity, SimActivityRelay


def test_resident_roi_preserves_normal_auto_gate_without_clearing_legacy_latch():
    lease=RoiLampLease()
    assert lease.observe(json.dumps(dict(type='roi_lamp_v1',available=True,roi_active=True,
                                        resident_control=True)),1.)
    assert lease.resident_control_live(1.1) and not lease.resident_control_live(1.3)
    block=RoiAutoBlock()
    assert not block.suppress(roi_active=True,resident_control=True,fire_control_valid=True,fire_control_on=True)
    assert block.suppress(roi_active=True,resident_control=False,fire_control_valid=True,fire_control_on=True)
    assert block.suppress(roi_active=True,resident_control=True,fire_control_valid=True,fire_control_on=True)
    assert not block.suppress(roi_active=True,resident_control=True,fire_control_valid=True,fire_control_on=False)


@pytest.fixture
def fake_smbus(monkeypatch):
    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "smbus", SimpleNamespace(SMBus=object))


def sim_activity(seq=1, *, reason="tracking", applied=True, mode="live", age_ms=0.0):
    state = AppliedControlActivity(backend="sim", session="sim-session", sequence=seq,
        created_monotonic_ns=seq, intent_sequence=seq, mode=mode, reason=reason, applied=applied)
    return SimActivityRelay(activity=ActivityEnvelope(activity=state, age_ms=age_ms)).model_dump_json()


def snapshot(stamp=1, *, seen=True, selected=True, missed=0, detection=False, domain="jetson_monotonic"):
    box = dict(x=0.4, y=0.4, w=0.1, h=0.1)
    payload = dict(sequence=stamp, frame=dict(frame_id=stamp, source_time_ns=stamp,
        observed_time_ns=stamp, source_clock_domain=domain, observation_clock_domain="jetson_monotonic",
        width=1280, height=720), tracks=[], detections=[])
    if seen:
        payload["tracks"] = [dict(track_id=7, class_id="0", confidence=0.8, missed_frames=missed, box=box)]
    if selected and seen:
        payload["selection"] = dict(track_id=7, source_frame_id=stamp, applied_frame_id=stamp,
            selected_time_ns=stamp, selection_clock_domain="jetson_monotonic", policy="test")
    if detection:
        payload["detections"] = [dict(detection_id=2, class_id="0", confidence=0.9, box=box)]
    return PerceptionSnapshotV2.model_validate(payload).model_dump_json(exclude_none=True)


def hardware(stamp=1, available=True):
    return CamState(frame_id=stamp, src_ts_ms=stamp, state_monotonic_ns=stamp,
                    pan=0, tilt=0, hardware_available=available).model_dump_json(exclude_none=True)


def test_visual_selection_alone_never_means_control_is_tracking():
    lamps = PanelIndicators()
    assert lamps.state(0) == IndicatorState()
    lamps.on_perception(snapshot(seen=False, selected=False, detection=True), 0)
    assert lamps.state(0) == IndicatorState(True, False, False)
    lamps.on_perception(snapshot(2, selected=False), .1)
    assert lamps.state(.1) == IndicatorState(True, False, False)
    lamps.on_perception(snapshot(3), .2)
    assert lamps.state(.2) == IndicatorState(True, False, False)
    lamps.on_sim_activity(sim_activity(), .2)
    assert lamps.state(.2) == IndicatorState(True, True, False)


def test_roi_selection_powers_only_fire_control_contact_with_short_lease():
    lamps = PanelIndicators()
    lease = RoiLampLease(stale_s=.25)
    assert lease.observe('{"type":"roi_lamp_v1","available":true}', 1.0)
    assert lamps.state(1.0, roi_available=lease.live(1.0)) == IndicatorState(True, False, False)
    assert lamps.state(1.26, roi_available=lease.live(1.26)) == IndicatorState()
    assert not lease.observe('{"type":"roi_lamp_v1","available":1}', 1.27)
    assert lamps.state(1.27, roi_available=lease.live(1.27)) == IndicatorState()


def test_roi_manual_selection_capture_requires_on_and_neutral_to_resume():
    lease = RoiLampLease(stale_s=.25)
    capture = RoiManualCapture()
    assert lease.observe('{"type":"roi_lamp_v1","available":true,"roi_active":true}', 1.0)
    assert lease.roi_active_live(1.0)
    def held(now, *, on, valid=True, rates=(0.4, -0.2)):
        return capture.suppress(roi_active=lease.roi_active_live(now), manual_active=True,
                                fire_control_valid=valid, fire_control_on=on,
                                joystick_rates=rates)
    assert not held(1.0, on=True)
    assert held(1.01, on=False)
    assert held(1.02, on=True)  # Lock edge cannot resume a deflected stick.
    assert held(1.26, on=False)  # Lost display lease cannot resume movement.
    assert held(1.27, on=True)
    assert not held(1.28, on=True, rates=(0.0, 0.0))
    assert held(1.29, on=False, valid=False) is False  # Stale lease, no new capture.


def test_roi_manual_capture_rejects_unreadable_switch_and_bad_lease_type():
    lease = RoiLampLease()
    capture = RoiManualCapture()
    assert not lease.observe('{"type":"roi_lamp_v1","available":true,"roi_active":1}', 1.0)
    assert lease.observe('{"type":"roi_lamp_v1","available":false,"roi_active":true}', 1.1)
    assert capture.suppress(roi_active=lease.roi_active_live(1.1), manual_active=True,
                            fire_control_valid=False, fire_control_on=False,
                            joystick_rates=(0.2, 0.0))
    assert capture.suppress(roi_active=False, manual_active=True,
                            fire_control_valid=False, fire_control_on=True,
                            joystick_rates=(0.0, 0.0))
    assert not capture.suppress(roi_active=False, manual_active=True,
                                fire_control_valid=True, fire_control_on=True,
                                joystick_rates=(0.0, 0.0))


def test_roi_blocks_auto_until_new_readable_off_after_display_loss():
    block = RoiAutoBlock()
    assert not block.suppress(roi_active=False, fire_control_valid=True,
                              fire_control_on=True)
    assert block.suppress(roi_active=True, fire_control_valid=True,
                          fire_control_on=True)
    assert block.suppress(roi_active=False, fire_control_valid=False,
                          fire_control_on=False)
    assert block.suppress(roi_active=False, fire_control_valid=True,
                          fire_control_on=True)
    assert not block.suppress(roi_active=False, fire_control_valid=True,
                              fire_control_on=False)


def test_roi_capture_zero_controller_demand_keeps_local_box_joystick(fake_smbus):
    from common.schemas import ManualControlState
    from rpi.runtime_control import panel_state_message

    state = ManualControlState(src_ts_ms=1, source="rpi.runtime_control", active=False,
                               emergency=False, control_cmd_enabled=False,
                               joystick_raw=(128, 128), joystick_rate_cmd=(0.0, 0.0))
    message = panel_state_message(state, {"fire_control": False},
                                  max_rate_rad_s=1.0, joystick_rates=(0.4, -0.2),
                                  physical_manual_active=True)
    assert state.joystick_rate_cmd == (0.0, 0.0)
    assert message["joystick"] == [0.4, -0.2]
    assert message["manual_active"] is True


def test_independent_off_delays_reacquisition_and_continuous_loss():
    lamps = PanelIndicators(safety_off_delay_s=.2, fire_control_off_delay_s=.4)
    lamps.on_perception(snapshot(), 0)
    lamps.on_sim_activity(sim_activity(), 0)
    lamps.state(0)
    lamps.on_perception(snapshot(2, selected=False), .1)
    lamps.on_sim_activity(sim_activity(2, reason="safety_hold"), .1)
    lamps.state(.1)
    assert lamps.state(.29).safety_lamp
    assert not lamps.state(.31).safety_lamp
    lamps.on_perception(snapshot(3), .32)
    lamps.on_sim_activity(sim_activity(3), .32)
    assert lamps.state(.32).safety_lamp
    lamps.on_perception(snapshot(4, seen=False), .4)
    lamps.on_sim_activity(sim_activity(4, reason="coasting"), .4)
    lamps.state(.4)
    assert lamps.state(.59).safety_lamp
    assert not lamps.state(.61).safety_lamp
    assert lamps.state(.79).fire_control_lamp
    assert not lamps.state(.81).fire_control_lamp


def test_tracker_coasting_does_not_count_as_current_tracking():
    lamps = PanelIndicators(safety_off_delay_s=0, fire_control_off_delay_s=0)
    lamps.on_perception(snapshot(), 0)
    lamps.on_perception(snapshot(2, missed=1), .1)
    assert lamps.state(.1) == IndicatorState()


def test_link_watchdog_overrides_hysteresis_and_duplicate_packets():
    lamps = PanelIndicators()
    lamps.on_perception(snapshot(), 0)
    assert not lamps.on_perception(snapshot(), .49)
    assert not lamps.on_perception(snapshot(0), .49)
    assert lamps.state(.5) == IndicatorState()
    lamps.on_perception(snapshot(2, seen=False), .51)
    assert lamps.state(.51) == IndicatorState()


def test_source_switch_does_not_carry_old_tracking_delay():
    lamps = PanelIndicators()
    lamps.on_perception(snapshot(), 0)
    lamps.on_perception(snapshot(seen=False, domain="pc_monotonic"), .1)
    assert lamps.state(.1) == IndicatorState()


def test_invalid_perception_never_refreshes_watchdog():
    lamps = PanelIndicators()
    lamps.on_perception(snapshot(), 0)
    with pytest.raises(ValueError):
        lamps.on_perception('{"type":"PerceptionSnapshot","version":1}', .49)
    assert lamps.state(.5) == IndicatorState()


def test_hardware_requires_explicit_fresh_proof_and_progress():
    lamps = PanelIndicators()
    lamps.on_hardware(hardware(1, None), 0)
    assert not lamps.state(0).yellow_light
    lamps.on_hardware(hardware(2), .1)
    assert lamps.state(.1).yellow_light
    assert not lamps.on_hardware(hardware(2), .59)
    assert not lamps.state(.61).yellow_light
    lamps.on_hardware(hardware(3), .7)
    assert lamps.state(.7).yellow_light
    lamps.on_hardware(hardware(4, False), .71)
    assert not lamps.state(.71).yellow_light
    with pytest.raises(ValueError):
        lamps.on_hardware(json.dumps(dict(type="CamState", frame_id=5, src_ts_ms=5,
             state_monotonic_ns=5, pan=0, tilt=0, hardware_available="true")), .8)


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_invalid_configuration(bad):
    with pytest.raises(ValueError):
        PanelIndicators(safety_off_delay_s=bad)


def test_zero_watchdog_is_rejected():
    with pytest.raises(ValueError):
        PanelIndicators(hardware_stale_s=0)


def test_panel_message_and_ui_distinguish_unpowered_contact_from_off_switch(fake_smbus):
    from common.schemas import ManualControlState
    from rpi.runtime_control import panel_state_message
    from rpi.display.status import SystemStatus

    payload = panel_state_message(ManualControlState(src_ts_ms=1, source="test", active=False,
        emergency=False, joystick_raw=(128, 128), joystick_rate_cmd=(0, 0)),
        {"fire_control": False, "safety": False, "fire": False}, max_rate_rad_s=1,
        input_validity={"fire_control": False, "safety": True, "fire": True},
        indicators={"fire_control_lamp": False, "safety_lamp": True, "yellow_light": False})
    status = SystemStatus()
    status.on_panel(payload, 0)
    lines = status.panel_lines(0)
    assert any("fire_control" in line and "unreadable" in line for line in lines)
    assert any("safety " in line and "off" in line for line in lines)
    assert status.panel_mode(0) == "SAFE"


@pytest.mark.parametrize("pitch,event_name", [(-0.8, "UP"), (0.8, "DOWN")])
def test_panel_navigation_conversion_leaves_manual_pitch_unchanged(fake_smbus, pitch, event_name):
    from common.schemas import ManualControlState
    from rpi.runtime_control import panel_state_message
    from rpi.display.inputs import PanelNavigator
    from rpi.display.menu import NavEvent

    state = ManualControlState(src_ts_ms=1, source="test", active=False,
        emergency=False, joystick_raw=(128, 128), joystick_rate_cmd=(0, pitch))
    panel = panel_state_message(state, {}, max_rate_rad_s=1)
    assert PanelNavigator().update(panel, 0) == [NavEvent[event_name]]
    assert state.joystick_rate_cmd == (0, pitch)
    assert panel["joystick"] == [0, pitch]
    manual = state.model_copy(update={"active": True})
    panel = panel_state_message(manual, {}, max_rate_rad_s=1)
    assert PanelNavigator().update(panel, 0) == []
    assert manual.joystick_rate_cmd == (0, pitch)


def test_runtime_receive_boundary_ignores_empty_and_invalid_records(fake_smbus):
    import logging
    import zmq
    from rpi.runtime_control import _receive_indicator

    class Subscriber:
        def __init__(self, raw):
            self.raw = raw

        def recv(self, *, flags):
            assert flags == zmq.NOBLOCK
            if self.raw is None:
                raise zmq.Again()
            return self.raw

    lamps = PanelIndicators()
    log = logging.getLogger("test")
    _receive_indicator(Subscriber(None), lamps.on_perception, 0, log)
    _receive_indicator(Subscriber(b"broken"), lamps.on_perception, 0, log)
    assert lamps.state(0) == IndicatorState()
    _receive_indicator(Subscriber(snapshot().encode()), lamps.on_perception, 0, log)
    assert lamps.state(0) == IndicatorState(True, False)
    _receive_indicator(Subscriber(snapshot().encode()), lamps.on_perception, .49, log)
    assert lamps.state(.5) == IndicatorState()


def test_production_config_powers_only_the_two_shared_contacts(fake_smbus):
    from pathlib import Path
    from common.config import load_config_bundle
    from rpi.manual_control import resolve_gpio_config

    cfg = load_config_bundle([Path(__file__).resolve().parents[1] / "configs/base/panel.yaml"]).data
    gpio = resolve_gpio_config(cfg["rpi"]["gpio"])
    assert gpio["shared_lamp_inputs"] == {"fire_control": "fire_control_lamp", "safety": "safety_lamp"}
    assert gpio["inputs"]["fire_control"] == 12 and gpio["outputs"]["fire_control_lamp"] == 22
    assert gpio["inputs"]["safety"] == 16 and gpio["outputs"]["safety_lamp"] == 25
    assert gpio["input_modes"] == {"control_switch": "latch"}
    from common.config_constants import RPI_RUNTIME_CONTROL_INDICATORS
    assert "indicators" not in cfg["rpi"]["runtime_control"]
    lamps = PanelIndicators(**RPI_RUNTIME_CONTROL_INDICATORS)
    assert lamps.fire.delay_s == lamps.safety.delay_s == 1


@pytest.mark.parametrize("reason", ["idle_return", "coasting", "gimbal_invalid", "safety_hold", "position_limit_hold"])
def test_nontracking_receiver_status_does_not_light_safety(reason):
    lamps = PanelIndicators()
    lamps.on_sim_activity(sim_activity(reason=reason), 0)
    assert not lamps.state(0).safety_lamp


@pytest.mark.parametrize("mode,applied", [("shadow", True), ("live", False), ("shadow", False)])
def test_shadow_or_unapplied_tracking_is_not_active(mode, applied):
    lamps = PanelIndicators()
    lamps.on_sim_activity(sim_activity(mode=mode, applied=applied), 0)
    assert not lamps.state(0).safety_lamp


def test_manual_exception_without_target_or_backend_and_emergency_override():
    lamps = PanelIndicators()
    assert lamps.state(0, manual_active=True) == IndicatorState(False, True, False)
    assert lamps.safety_reason == "manual exception"
    assert not lamps.state(.1, manual_active=True, emergency=True).safety_lamp
    assert lamps.safety_reason == "emergency"
    assert lamps.state(.2, manual_active=True).safety_lamp
    assert lamps.state(.3, manual_active=False).safety_lamp
    assert lamps.state(1.29).safety_lamp
    assert not lamps.state(1.31).safety_lamp


@pytest.mark.parametrize("backend", ["sim", "gimbal"])
def test_live_manual_receiver_also_qualifies_for_the_manual_exception(backend):
    lamps = PanelIndicators()
    if backend == "sim":
        lamps.on_sim_activity(sim_activity(reason="manual"), 0)
    else:
        activity = AppliedControlActivity(backend="gimbal", session="bridge", sequence=1,
            created_monotonic_ns=1, mode="live", reason="manual", applied=True)
        lamps.on_hardware(CamState(frame_id=1, src_ts_ms=1, state_monotonic_ns=1,
            pan=0, tilt=0, hardware_available=True, control_activity=activity).model_dump_json(), 0)
    assert lamps.state(0).safety_lamp
    assert lamps.safety_reason == "manual exception"
    assert not lamps.state(.1, emergency=True).safety_lamp


def test_shadow_manual_receiver_does_not_qualify_as_manual_mode():
    lamps = PanelIndicators()
    lamps.on_sim_activity(sim_activity(reason="manual", mode="shadow"), 0)
    assert not lamps.state(0).safety_lamp


def test_remote_manual_loss_cannot_turn_the_watchdog_into_an_off_delay():
    lamps = PanelIndicators()
    lamps.on_sim_activity(sim_activity(reason="manual"), 0)
    assert lamps.state(0).safety_lamp
    assert not lamps.state(.5).safety_lamp
    assert lamps.state(.6, manual_active=True).safety_lamp  # explicit local exception still works


def test_receiver_loss_duplicate_relay_and_explicit_invalidations_fail_closed():
    lamps = PanelIndicators()
    payload = sim_activity()
    lamps.on_sim_activity(payload, 0)
    assert lamps.state(0).safety_lamp
    assert not lamps.on_sim_activity(payload, .49)
    assert not lamps.state(.5).safety_lamp  # watchdog overrides the 1s off-delay
    lamps.on_sim_activity(sim_activity(2), .6)
    assert lamps.state(.6).safety_lamp
    lamps.on_sim_activity(SimActivityRelay().model_dump_json(), .61)
    assert not lamps.state(.61).safety_lamp
    assert not lamps.on_sim_activity(sim_activity(2), .62)
    assert not lamps.state(.62).safety_lamp
    lamps.on_sim_activity(sim_activity(3), .63)
    assert lamps.state(.63).safety_lamp


def test_forwarded_age_counts_toward_watchdog():
    lamps = PanelIndicators()
    lamps.on_sim_activity(sim_activity(age_ms=450.0), 10)
    assert lamps.state(10).safety_lamp
    assert not lamps.state(10.06).safety_lamp


def test_physical_tracking_requires_fresh_hardware_and_applied_intent():
    lamps = PanelIndicators()
    activity = AppliedControlActivity(backend="gimbal", session="bridge", sequence=1,
        created_monotonic_ns=1, mode="live", reason="tracking", applied=True)
    state = CamState(frame_id=1, src_ts_ms=1, state_monotonic_ns=1, pan=0, tilt=0,
        hardware_available=True, control_activity=activity)
    lamps.on_hardware(state.model_dump_json(), 0)
    assert lamps.state(0) == IndicatorState(False, True, True)
    assert lamps.safety_reason == "gimbal tracking"
    lamps.on_hardware(state.model_copy(update={"state_monotonic_ns": 2,
        "hardware_available": False}).model_dump_json(), .1)
    assert not lamps.state(.1).safety_lamp
    assert not lamps.state(.1).yellow_light
    restarted = state.model_copy(update={"state_monotonic_ns": 0,
        "control_activity": activity.model_copy(update={"session": "restarted-bridge"})})
    assert lamps.on_hardware(restarted.model_dump_json(), .2)
    assert lamps.state(.2).safety_lamp
    assert not lamps.on_hardware(state.model_copy(update={"state_monotonic_ns": 3}).model_dump_json(), .3)
