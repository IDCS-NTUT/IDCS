import json

import pytest

from common.perception import PerceptionSnapshotV2
from common.schemas import CamState
from rpi.panel_indicators import IndicatorState, PanelIndicators


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


def test_startup_is_off_then_available_and_selected_are_distinct():
    lamps = PanelIndicators()
    assert lamps.state(0) == IndicatorState()
    lamps.on_perception(snapshot(seen=False, selected=False, detection=True), 0)
    assert lamps.state(0) == IndicatorState(True, False, False)
    lamps.on_perception(snapshot(2, selected=False), .1)
    assert lamps.state(.1) == IndicatorState(True, False, False)
    lamps.on_perception(snapshot(3), .2)
    assert lamps.state(.2) == IndicatorState(True, True, False)


def test_independent_off_delays_reacquisition_and_continuous_loss():
    lamps = PanelIndicators(safety_off_delay_s=.2, fire_control_off_delay_s=.4)
    lamps.on_perception(snapshot(), 0)
    lamps.on_perception(snapshot(2, selected=False), .1)
    assert lamps.state(.29).safety_lamp
    assert not lamps.state(.31).safety_lamp
    lamps.on_perception(snapshot(3), .32)
    assert lamps.state(.32).safety_lamp
    lamps.on_perception(snapshot(4, seen=False), .4)
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


def test_panel_message_and_ui_distinguish_unpowered_contact_from_off_switch():
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


def test_runtime_receive_boundary_ignores_empty_and_invalid_records():
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
    assert lamps.state(0) == IndicatorState(True, True)
    _receive_indicator(Subscriber(snapshot().encode()), lamps.on_perception, .49, log)
    assert lamps.state(.5) == IndicatorState()


def test_production_config_powers_only_the_two_shared_contacts():
    from pathlib import Path
    from common.config import load_config_bundle
    from rpi.manual_control import resolve_gpio_config

    cfg = load_config_bundle([Path(__file__).resolve().parents[1] / "configs/base/panel.yaml"]).data
    gpio = resolve_gpio_config(cfg["rpi"]["gpio"])
    assert gpio["shared_lamp_inputs"] == {"fire_control": "fire_control_lamp", "safety": "safety_lamp"}
    assert gpio["inputs"]["fire_control"] == 12 and gpio["outputs"]["fire_control_lamp"] == 22
    assert gpio["inputs"]["safety"] == 16 and gpio["outputs"]["safety_lamp"] == 25
    assert gpio["input_modes"] == {"control_switch": "latch"}
    lamps = PanelIndicators(**cfg["rpi"]["runtime_control"]["indicators"])
    assert lamps.fire.delay_s == lamps.safety.delay_s == 1
