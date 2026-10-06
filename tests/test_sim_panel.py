from common.schemas import ManualControlState
from tools.sim_panel import armed_state


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
