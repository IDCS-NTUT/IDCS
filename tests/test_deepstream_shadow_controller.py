import json
from pathlib import Path

import pytest

from common.perception import TargetSelectionV2
from common.synthetic_perception import load_synthetic_scenario, snapshot_at
from jetson.deepstream import shadow_controller


CONFIG_PATHS = [
    Path("configs/network.yaml"),
    Path("configs/perception.yaml"),
    Path("configs/control.yaml"),
    Path("configs/system.yaml"),
]


def test_synthetic_preselection_reaches_v2_simulation_boundary():
    scenario = load_synthetic_scenario(
        Path("tests/fixtures/synthetic_tracking_v1.json")
    )
    source = snapshot_at(scenario, 2)
    selected = source.model_copy(update={
        "selection": TargetSelectionV2(
            track_id=41,
            source_frame_id=2,
            applied_frame_id=2,
            selected_time_ns=source.frame.observed_time_ns,
            selection_clock_domain="synthetic",
            policy="deterministic_test",
        )
    })

    assert selected.selection is not None
    assert selected.selection.track_id == 41
    assert shadow_controller._has_valid_preselection(selected)


def test_check_mode_validates_without_constructing_zmq_context(monkeypatch, capsys):
    def fail_context():
        raise AssertionError("check mode must not construct a ZMQ context")

    monkeypatch.setattr(shadow_controller.zmq, "Context", fail_context)
    args = []
    for path in CONFIG_PATHS:
        args.extend(["--idcs-config", str(path)])
    args.extend([
        "--snapshot-sub", "tcp://127.0.0.1:6550",
        "--sim-control-bind", "tcp://127.0.0.1:6551",
        "--check",
    ])

    assert shadow_controller.run(args) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["physical_control_disabled"] is True
    assert result["controller"] == "pid"
    assert result["target_selector"] == "preselected"
    assert result["loop_hz"] == 50.0
    assert len(result["config_digest"]) == 64
    assert len(result["config_sources"]) == len(CONFIG_PATHS)


def test_check_mode_rejects_production_control_port_before_zmq():
    args = []
    for path in CONFIG_PATHS:
        args.extend(["--idcs-config", str(path)])
    args.extend([
        "--snapshot-sub", "tcp://127.0.0.1:6550",
        "--sim-control-bind", "tcp://127.0.0.1:5557",
        "--check",
    ])

    with pytest.raises(SystemExit):
        shadow_controller.run(args)
