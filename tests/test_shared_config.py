"""Two agents, fake transports and deterministic safety input; no lab access."""
from copy import deepcopy
import json
from pathlib import Path
import statistics
import threading
import time
from unittest.mock import Mock

import pytest

from common.config import ConfigError
import common.live_config as live
from common.shared_config import ConfigLink, SharedConfigAgent, shared_keys

ROOT = Path(__file__).resolve().parents[1]
KEY = "laser.offset_m.x"


class FakeLink:
    def __init__(self, target):
        self.target, self.connected = target, True
        self.requests = []
    def online(self):
        return self.connected
    def call(self, request, *, timeout):
        self.requests.append(deepcopy(request))
        if not self.connected:
            raise TimeoutError("fake PC offline")
        return self.target.receive(request)


@pytest.fixture
def agents(tmp_path, monkeypatch):
    monkeypatch.setenv("IDCS_LIVE_CONFIG", str(tmp_path / "unused.yaml"))
    result = []
    for host in ("jetson", "pc"):
        local = live.ConfigAgent(ROOT, host, run=Mock(return_value=(0, "fake")),
                                 result_wait_s=0, path=tmp_path / host / "live.yaml")
        agent = SharedConfigAgent(local)
        result.append(agent)
    jetson, pc = result
    jetson.peer, pc.peer = FakeLink(pc), FakeLink(jetson)
    assert pc.reconcile()["state"] == "ready"
    return jetson, pc


def request(agent, action="set", **kw):
    return agent.handle(dict(type="ConfigCommand", actor="pi-panel", action=action, **kw))


def overlay(agent):
    return live.flatten(live.read_overlay(agent.local.path))


def records(agent):
    return [json.loads(line) for line in agent.local.path.with_name("live-journal.jsonl").read_text().splitlines()]


def test_ownership_comes_from_stack_table():
    keys = shared_keys(ROOT)
    assert KEY in keys and "threat_eval.zones.warning.radius_m" in keys
    assert "deepstream.shadow_tracks.max_age_frames" not in keys
    assert "rpi.joystick.deadzone" not in keys


def test_shared_change_commits_both_and_one_transaction(agents):
    jetson, pc = agents
    result = request(jetson, key=KEY, value=.1)
    assert result["ok"], result
    assert overlay(jetson)[KEY] == overlay(pc)[KEY] == .1
    assert result["hosts"]["pc"]["state"] == "ready"
    for agent in agents:
        assert any(r.get("transaction") == result["transaction"] for r in records(agent))
    assert result["pc_round_trip_ms"] >= 0


@pytest.mark.parametrize("host", [0, 1])
def test_validation_rejection_writes_neither(agents, host):
    jetson, pc = agents
    agents[host].local.bundles = Mock(side_effect=ConfigError("fake stack rejection"))
    result = request(jetson, key=KEY, value=.1)
    assert not result["ok"] and "rejection" in result["error"]
    assert overlay(jetson) == overlay(pc) == {}


def test_offline_commit_and_reconcile(agents):
    jetson, pc = agents
    jetson.peer.connected = False
    result = request(jetson, key=KEY, value=.1)
    assert result["ok"] and result["hosts"]["pc"]["state"] == "pending"
    assert overlay(pc) == {}
    jetson.peer.connected = True
    assert pc.reconcile()["state"] == "ready"
    assert overlay(pc)[KEY] == .1
    assert any(r.get("transaction") == result["transaction"] for r in records(pc))
    assert request(jetson, "status")["shared"]["hosts"]["pc"]["state"] == "ready"


def test_reconcile_rejection_retains_previous_and_reports(agents):
    jetson, pc = agents
    jetson.peer.connected = False
    assert request(jetson, key=KEY, value=.1)["ok"]
    pc.local.bundles = Mock(side_effect=ConfigError("PC validation rejected"))
    jetson.peer.connected = True
    result = pc.reconcile()
    assert result["state"] == "rejected" and overlay(pc) == {}
    status = request(jetson, "status")
    assert status["shared"]["hosts"]["pc"]["state"] == "rejected"
    assert "PC validation rejected" in status["shared"]["hosts"]["pc"]["error"]
    assert not live.shared_receipt_status(pc.local.path)["shared_config_ready"]


def test_pc_cli_forwarding_and_local_key(agents):
    jetson, pc = agents
    result = request(pc, key=KEY, value=.1, expected_digest=live.digest({}))
    assert result["ok"], result
    assert overlay(jetson)[KEY] == overlay(pc)[KEY] == .1
    assert any(r.get("phase") == "forward" for r in pc.peer.requests)
    before = list(jetson.peer.requests)
    result = request(jetson, key="deepstream.shadow_tracks.max_age_frames", value=20)
    assert result["ok"] and jetson.peer.requests == before
    assert "deepstream.shadow_tracks.max_age_frames" not in overlay(pc)


@pytest.mark.parametrize("host", [0, 1])
def test_stale_digest_on_either_host(agents, host):
    jetson, pc = agents
    if host == 0:
        kw = dict(expected_digest="stale")
    else:
        jetson.state["hosts"]["pc"]["digest"] = "stale"
        kw = {}
    result = request(jetson, key=KEY, value=.1, **kw)
    assert not result["ok"] and "overlay changed" in result["error"]
    assert overlay(jetson) == overlay(pc) == {}


def test_existing_pc_overlay_requires_explicit_adopt_or_drop(agents):
    jetson, pc = agents
    live.atomic_write(pc.local.path, "laser:\n  offset_m:\n    x: 0.1\n")
    assert request(pc, "status")["shared"]["conflicts"] == {KEY: .1}
    assert pc.reconcile()["state"] == "rejected"
    assert overlay(pc)[KEY] == .1
    result = request(pc, "adopt_shared")
    assert result["ok"], result
    assert overlay(jetson)[KEY] == overlay(pc)[KEY] == .1
    live.atomic_write(pc.local.path, "laser:\n  offset_m:\n    x: 0.2\n")
    result = request(pc, "drop_shared")
    assert result["ok"], result
    assert overlay(pc) == {}
    assert pc.reconcile()["state"] == "ready"
    assert overlay(pc)[KEY] == .1


def test_prepare_timeout_writes_neither_and_keeps_callable(agents):
    jetson, pc = agents
    jetson.peer.call = Mock(side_effect=TimeoutError("bounded timeout"))
    result = request(jetson, key=KEY, value=.1)
    assert not result["ok"] and "timeout" in result["error"]
    assert overlay(jetson) == overlay(pc) == {}
    assert request(jetson, "status")["ok"]


def test_lost_commit_reply_recovers_idempotently(agents):
    jetson, pc = agents
    call = jetson.peer.call
    def lose_reply(request, **kw):
        result = call(request, **kw)
        if request.get("phase") == "commit":
            raise TimeoutError("reply lost")
        return result
    jetson.peer.call = lose_reply
    result = request(jetson, key=KEY, value=.1)
    assert result["ok"] and result["hosts"]["pc"]["state"] == "commit_pending"
    transaction = result["transaction"]
    assert pc.reconcile()["state"] == "ready"
    assert not jetson.state["decision"]
    assert len([r for r in records(pc) if r.get("type") == "result" and r.get("transaction") == transaction]) == 1


def test_internal_phase_not_available_from_cli(agents):
    for agent in agents:
        assert not request(agent, phase="commit", transaction="bad")["ok"]
        assert not request(agent, action="install", candidate={})["ok"]


def test_real_inprocess_link_forwarding_and_timeouts(agents):
    import zmq
    jetson, pc = agents
    context = zmq.Context()
    endpoint = "inproc://config-link-contract"
    server = ConfigLink(context, endpoint, jetson, server=True)
    client = ConfigLink(context, endpoint, pc)
    try:
        deadline = time.monotonic() + 5
        while not (client.online() and server.online()):
            assert time.monotonic() < deadline
            time.sleep(.005)
        assert pc.reconcile()["state"] == "ready"
        result = request(pc, key=KEY, value=.1)
        assert result["ok"], result
        assert overlay(jetson)[KEY] == overlay(pc)[KEY] == .1
        # A missing response is bounded and neither socket owner is blocked.
        original = pc.receive
        release, entered = threading.Event(), threading.Event()
        def stall(request):
            entered.set()
            release.wait(1)
            return original(request)
        pc.receive = stall
        with pytest.raises(TimeoutError):
            server.call(dict(phase="abort", transaction="none"), timeout=.05)
        assert entered.is_set()
        assert client.thread.is_alive() and server.thread.is_alive()
        assert server.online()  # heartbeat I/O continues during handler work
        release.set()
    finally:
        client.close()
        server.close()
        context.term()


def test_warm_shared_panel_timings(agents):
    jetson, pc = agents
    assert request(jetson, "get")["ok"]
    elapsed, pc_ms = [], []
    for value in (.1, .11, .12):
        start = time.perf_counter()
        result = request(jetson, key=KEY, value=value)
        elapsed.append(time.perf_counter() - start)
        assert result["ok"], result
        pc_ms.append(result["pc_round_trip_ms"])
    print(f"shared panel mutation n=3 median={statistics.median(elapsed)*1000:.2f} ms "
          f"max={max(elapsed)*1000:.2f} ms; PC phase median={statistics.median(pc_ms):.2f} ms")
    assert max(elapsed) < .8

@pytest.mark.parametrize("change", ["conflict", "pending", "stale", "pc_rejected", "jetson_pending"])
def test_deployment_gate_refuses_unreconciled_either_host(agents, monkeypatch, change):
    from tools.deployment import shared_config_gate
    jetson, pc = agents
    assert request(jetson, key=KEY, value=.1)["ok"]
    assert pc.reconcile()["state"] == "ready"
    monkeypatch.setenv("IDCS_LIVE_CONFIG", str(pc.local.path))
    state = json.loads(pc.state_path.read_text())
    if change == "conflict":
        state["conflicts"] = {KEY: .2}
    elif change == "pending":
        state["prepared"] = {"transaction": "pending"}
    elif change == "stale":
        state["updated_ns"] = 0
    elif change == "pc_rejected":
        state["hosts"]["pc"]["state"] = "rejected"
    else:
        state["hosts"]["jetson"]["state"] = "pending"
    live.atomic_write(pc.state_path, json.dumps(state))
    with pytest.raises(RuntimeError, match="unreconciled"):
        shared_config_gate(live.read_overlay(pc.local.path))


def test_deployment_gate_accepts_reconciled_evidence_and_checks_live_overlay(agents, monkeypatch):
    from tools.deployment import shared_config_gate
    jetson, pc = agents
    assert request(jetson, key=KEY, value=.1)["ok"]
    assert pc.reconcile()["state"] == "ready"
    for agent in agents:
        monkeypatch.setenv("IDCS_LIVE_CONFIG", str(agent.local.path))
        assert shared_config_gate(live.read_overlay(agent.local.path))["ready"]
    live.atomic_write(pc.local.path, "laser:\n  offset_m:\n    x: .2\n")
    assert live.shared_receipt_status(pc.local.path)["shared_config_mismatch"]
    with pytest.raises(RuntimeError, match="unreconciled"):
        shared_config_gate(live.read_overlay(pc.local.path))


def test_both_agent_channel_owners_remain_responsive_during_remote_timeout(agents, monkeypatch):
    import zmq
    jetson, pc = agents
    context = zmq.Context()
    channels = []
    for host, agent in zip(("jetson", "pc"), agents):
        monkeypatch.setattr(live, "command_endpoint", lambda host=host: "inproc://" + host + "-cli")
        monkeypatch.setattr(live, "update_endpoint", lambda host=host: "inproc://" + host + "-updates")
        channels.append(live.AgentChannel(context, agent))
    entered, release = threading.Event(), threading.Event()
    replies, ticks = [], [0, 0]
    def timeout(request, **kw):
        entered.set()
        assert release.wait(2)
        raise TimeoutError("bounded fake timeout")
    jetson.peer.call = timeout
    channels[0].submit(dict(actor="pi-panel", action="set", key=KEY, value=.1), replies.append)
    try:
        assert entered.wait(2)
        deadline = time.monotonic() + .2
        while time.monotonic() < deadline:
            for i, channel in enumerate(channels):
                start = time.perf_counter()
                channel.poll()
                assert time.perf_counter() - start < .05
                ticks[i] += 1
            time.sleep(.001)
        assert min(ticks) > 20 and not replies
        release.set()
        deadline = time.monotonic() + 2
        while not replies:
            assert time.monotonic() < deadline
            for channel in channels:
                channel.poll()
            time.sleep(.002)
        assert not replies[0]["ok"] and overlay(jetson) == overlay(pc) == {}
    finally:
        release.set()
        for channel in channels:
            channel.close()
        context.term()


def test_pc_mixed_batch_and_clear_are_one_coordinated_transaction(agents):
    jetson, pc = agents
    result = request(pc, "set_many", values={KEY: .1, "sim.freeze_frame": True})
    assert result["ok"], result
    assert overlay(jetson)[KEY] == overlay(pc)[KEY] == .1
    assert overlay(pc)["sim.freeze_frame"] is True
    assert "sim.freeze_frame" not in overlay(jetson)
    result = request(pc, "clear")
    assert result["ok"], result
    assert overlay(jetson) == overlay(pc) == {}


def test_pc_local_only_change_does_not_contact_coordinator(agents):
    jetson, pc = agents
    pc.peer.call = Mock(side_effect=AssertionError("local key contacted peer"))
    result = request(pc, key="sim.freeze_frame", value=True)
    assert result["ok"], result
    assert overlay(pc)["sim.freeze_frame"] is True and overlay(jetson) == {}


def test_late_prepare_after_timeout_is_aborted_before_reconcile(agents):
    jetson, pc = agents
    call = jetson.peer.call
    def lose_prepare(request, **kw):
        response = call(request, **kw)
        if request.get("phase") == "prepare":
            raise TimeoutError("prepared response lost")
        return response
    jetson.peer.call = lose_prepare
    assert not request(jetson, key=KEY, value=.1)["ok"]
    assert pc.state["prepared"]
    assert jetson.state["hosts"]["pc"]["state"] == "prepare_unknown"
    assert overlay(jetson) == overlay(pc) == {}
    assert pc.reconcile()["state"] == "ready"
    assert not pc.state["prepared"] and overlay(jetson) == overlay(pc) == {}


def test_agent_restart_recovers_commit_after_pc_overlay_write(agents):
    jetson, pc = agents
    save = pc._save
    def crash_after_write():
        if pc.state.get("last_result") and overlay(pc).get(KEY) == .1:
            raise OSError("fake crash before saving commit receipt")
        return save()
    pc._save = crash_after_write
    result = request(jetson, key=KEY, value=.1)
    assert result["hosts"]["pc"]["state"] == "commit_pending"
    assert overlay(jetson)[KEY] == overlay(pc)[KEY] == .1
    # New wrappers read only durable state, as in a process restart.
    restarted_jetson = SharedConfigAgent(jetson.local)
    restarted_pc = SharedConfigAgent(pc.local)
    restarted_jetson.peer, restarted_pc.peer = FakeLink(restarted_pc), FakeLink(restarted_jetson)
    restarted_pc.local.apply_plan = Mock(side_effect=AssertionError("committed overlay reapplied"))
    assert restarted_pc.reconcile()["state"] == "ready"
    assert not restarted_jetson.state["decision"] and not restarted_pc.state["prepared"]
    assert overlay(restarted_jetson)[KEY] == overlay(restarted_pc)[KEY] == .1


def test_pi_staging_requires_both_host_reconciliation_evidence(agents, tmp_path, monkeypatch):
    from tools.deployment import Deployment
    jetson, pc = agents
    assert pc.reconcile()["state"] == "ready"
    monkeypatch.setenv("IDCS_LIVE_CONFIG", str(tmp_path / "pi" / "live.yaml"))
    manager = Deployment(tmp_path / "project", "pi", "candidate", templates=tmp_path)
    manager.source.mkdir(parents=True)
    manager.safety = Mock()
    manager.run = Mock(side_effect=["a" * 40, ""])
    manager.plan = lambda: manager.load("stage.json")
    bundle = tmp_path / "fake.bundle"
    bundle.write_bytes(b"fake")
    with pytest.raises(RuntimeError, match="status unavailable"):
        manager.stage(bundle, "a" * 40)
    result = manager.stage(bundle, "a" * 40, shared_status=pc.state_path)
    assert result["live_overlay"]["shared"]["ready"]


def test_pc_forward_requires_the_jetson_digest_from_reconciliation(agents):
    jetson, pc = agents
    # An unrelated Jetson-local change makes the PC's Jetson view stale.
    assert request(jetson, key="deepstream.shadow_tracks.max_age_frames", value=20)["ok"]
    result = request(pc, key=KEY, value=.1)
    assert not result["ok"] and "overlay changed" in result["error"]
    assert KEY not in overlay(jetson) and KEY not in overlay(pc)
    assert pc.reconcile()["state"] == "ready"
    assert request(pc, key=KEY, value=.1)["ok"]
    # A second command can use the forwarded reply's new Jetson digest immediately.
    assert request(pc, key=KEY, value=.2)["ok"]


def test_fire_control_is_rechecked_after_remote_prepare_before_any_write(agents):
    jetson, pc = agents
    jetson.local.fire_control_notice = Mock(side_effect=[None, ConfigError("fire control became unreadable")])
    result = request(jetson, key=KEY, value=.1)
    assert not result["ok"] and "unreadable" in result["error"]
    assert overlay(jetson) == overlay(pc) == {}
    assert not pc.state["prepared"]
