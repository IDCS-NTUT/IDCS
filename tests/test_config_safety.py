"""Agent-side raw panel guard and restart/mode exclusion; no hardware."""
import json
from pathlib import Path
import threading
import time
from unittest.mock import Mock

import pytest
from common.live_config import ConfigAgent, digest, live_path, runtime_bundle
from common.config import resolve_config_paths
from jetson.operator_agent import AgentConfig, OperatorAgent
from pc.mode_agent import PcModeWorker
from common.pc_mode import PcModeTask

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("IDCS_LIVE_CONFIG", str(tmp_path / "live.yaml"))
    return ConfigAgent(ROOT, "jetson", run=Mock(return_value=(0, "fake")), result_wait_s=0)


@pytest.mark.parametrize("enabled,valid,age,health_age,allowed", [
    (False, True, .01, .01, True), (True, True, .01, .01, False),
    (False, False, .01, .01, False), (False, True, .5, .01, False),
    (False, True, .01, 4, False), (False, True, .3, .1, False),
    (False, True, None, .01, False), (False, True, .01, -.5, False),
])
def test_agent_fire_control_raw_freshness(config, monkeypatch, enabled, valid, age, health_age, allowed):
    config.active = lambda: {"idcs-controller-camera.service"}
    original = Path.read_text
    def health(path, *a, **kw):
        if path.name == "controller-health.json":
            return json.dumps(dict(monotonic_s=time.monotonic()-health_age, control_cmd_enabled=enabled,
                                   control_cmd_valid=valid, control_cmd_age_s=age))
        return original(path, *a, **kw)
    monkeypatch.setattr(Path, "read_text", health)
    result = config.handle(dict(actor="pi-panel", action="set", key="laser.offset_m.x", value=.1))
    assert result["ok"] == allowed, result
    assert live_path().exists() == allowed
    if not allowed:
        assert "fresh, readable and off" in result["error"]


def test_no_controller_allows_with_notice(config):
    result = config.handle(dict(actor="pi-panel", action="set", key="laser.offset_m.x", value=.1))
    assert result["ok"] and "no controller active" in result["notice"]


def operator(tmp_path):
    cfg = runtime_bundle(resolve_config_paths(ROOT / "configs/base"), overlay={}).mutable_copy()
    cfg["operator"]["state_file"] = str(tmp_path / "operator.json")
    return OperatorAgent(AgentConfig.from_config(cfg), run=Mock(return_value=(0, "fake")))


def test_mode_in_progress_refuses_restart_but_keeps_overlay(config, tmp_path):
    owner = operator(tmp_path)
    owner.busy = "mode transition"
    config.restart_guard = owner.config_restart_guard
    config.active = lambda: {"idcs-deepstream-camera.service"}
    result = config.handle(dict(actor="pi-panel", action="set", key="video.return_jitter_ms", value=21))
    assert result["ok"], result
    unit = result["units"]["jetson/idcs-deepstream-camera.service"]
    assert unit["state"] == "cannot_restart" and unit["reason"] == "mode change in progress"
    assert live_path().exists()
    config.run.assert_not_called()


def test_config_restart_prevents_operator_mode_path(config, tmp_path):
    owner = operator(tmp_path)
    with owner.config_restart_guard() as refusal:
        assert refusal is None
        assert not owner._start_mode_job(owner.config.mode("standby"))[0]
        assert owner.busy == "config restart"
    assert owner.busy is None


def test_pc_mode_and_config_restart_are_exclusive():
    worker = PcModeWorker(run=Mock())
    task = PcModeTask(1, "start", "idcs-sim.target")
    with worker.config_restart_guard() as refusal:
        assert refusal is None
        with pytest.raises(RuntimeError, match="config restart"):
            worker.accept(task)
    worker._mode_running = True
    with worker.config_restart_guard() as refusal:
        assert refusal == "mode change in progress"
    worker.run.assert_not_called()
