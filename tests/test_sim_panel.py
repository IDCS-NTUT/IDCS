from common.schemas import ManualControlState
from tools.sim_panel import PanelAuthority, armed_state


def test_sim_panel_uses_current_schema_and_enables_only_auto():
    state = armed_state()
    assert ManualControlState.model_validate_json(state.model_dump_json()) == state
    assert state.control_cmd_enabled and not state.active and not state.emergency
    assert not state.fire_arm


def test_shared_sim_services_have_one_nonconflicting_lifetime():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "deploy/systemd/pc"
    for name in ("idcs-sim-panel.service", "idcs-sim-ui.service", "idcs-sim-recorder.service"):
        parts = [line for line in (root / name).read_text().splitlines() if line.startswith("PartOf=")]
        assert parts == ["PartOf=idcs-sim-session.target"]
    for name in ("idcs-sim.target", "idcs-sim-swarm.target"):
        assert "Requires=idcs-sim-session.target" in (root / name).read_text()


def test_panel_less_sim_falls_back_only_after_discovery_grace():
    authority = PanelAuthority(started_s=10.0, allow_panel_less=True)
    assert not authority.state(11.99).control_cmd_enabled
    assert authority.state(12.0).control_cmd_enabled


def test_configured_physical_panel_does_not_auto_arm_when_pi_service_is_down():
    authority = PanelAuthority(started_s=10.0)
    assert not authority.state(100.0).control_cmd_enabled


def test_physical_panel_switches_take_precedence_and_stale_fails_closed():
    authority = PanelAuthority(started_s=10.0)
    authority.mark_connected()
    assert not authority.state(12.0).control_cmd_enabled
    panel = armed_state().model_copy(update={
        "source": "rpi.runtime_control", "active": True, "emergency": False,
        "control_cmd_enabled": False, "fire_arm": True, "fire": True,
        "joystick_rate_cmd": (0.4, -0.3),
    })
    assert authority.observe(panel.model_dump_json(), now_s=12.01)
    assert authority.state(12.02) == panel
    estop = panel.model_copy(update={
        "active": False, "emergency": True, "control_cmd_enabled": True,
        "fire_arm": False, "fire": False,
    })
    assert authority.observe(estop.model_dump_json(), now_s=12.03)
    assert authority.state(12.04) == estop
    assert not authority.state(12.39).control_cmd_enabled
    assert not authority.state(100.0).active
    assert not authority.state(100.0).fire
    assert not authority.state(100.0).fire_arm
    assert authority.physical_seen  # loss never restores synthetic auto authority


def test_invalid_or_spoofed_payload_never_refreshes_physical_state():
    authority = PanelAuthority(started_s=0.0)
    authority.mark_connected()
    assert not authority.observe("bad JSON", now_s=1.0)
    assert not authority.observe(armed_state().model_dump_json(), now_s=1.0)
    assert not authority.state(10.0).control_cmd_enabled


def test_sim_service_subscribes_to_pi_manual_state_mirror():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    service = (root / "deploy/systemd/pc/idcs-sim-panel.service").read_text()
    config = (root / "configs/base/panel.yaml").read_text()
    assert "--physical-panel-endpoint tcp://192.168.0.3:5580" in service
    assert 'sim_panel_pub_endpoint: "tcp://192.168.0.3:5580"' in config


def test_live_loopback_bridge_physical_precedence_and_loss(monkeypatch):
    """Exercise the ZMQ sockets; no camera, panel GPIO or motor is involved."""
    import socket
    import sys
    import threading
    import time

    import zmq
    import tools.sim_panel as sim_panel

    def free_port():
        with socket.socket() as candidate:
            candidate.bind(("127.0.0.1", 0))
            return candidate.getsockname()[1]

    output_endpoint = f"tcp://127.0.0.1:{free_port()}"
    physical_endpoint = f"tcp://127.0.0.1:{free_port()}"
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind(output_endpoint)
    publisher = context.socket(zmq.PUB)
    publisher.bind(physical_endpoint)
    stop = threading.Event()
    monkeypatch.setattr(sim_panel, "install_signal_handlers", lambda: stop)
    monkeypatch.setattr(sys, "argv", ["sim_panel", "--manual-state-endpoint", output_endpoint,
        "--physical-panel-endpoint", physical_endpoint, "--discovery-grace-s", "0.2",
        "--physical-stale-s", "0.25", "--publish-hz", "50"])
    worker = threading.Thread(target=sim_panel.main, daemon=True)
    try:
        worker.start()
        poller = zmq.Poller()
        poller.register(receiver, zmq.POLLIN)

        def next_state(timeout_ms=500):
            assert dict(poller.poll(timeout_ms)).get(receiver) == zmq.POLLIN
            return ManualControlState.model_validate_json(receiver.recv())

        assert not next_state().control_cmd_enabled
        physical = armed_state().model_copy(update={
            "source": "rpi.runtime_control", "active": True,
            "control_cmd_enabled": False, "joystick_rate_cmd": (0.2, -0.1),
        })
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            publisher.send_string(physical.model_dump_json())
            state = next_state()
            if state.source == "rpi.runtime_control":
                assert state.active and state.joystick_rate_cmd == (0.2, -0.1)
                break
        else:
            raise AssertionError("physical Pi state never replaced synthetic SIM state")
        publisher.close(0)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            state = next_state()
            if state.source == "sim_panel_guard" and "stale" in state.note:
                assert not state.active and not state.control_cmd_enabled
                break
        else:
            raise AssertionError("lost Pi stream did not fail closed")
    finally:
        stop.set()
        worker.join(timeout=2.0)
        if not publisher.closed:
            publisher.close(0)
        receiver.close(0)
        context.destroy(linger=0)
        assert not worker.is_alive()
