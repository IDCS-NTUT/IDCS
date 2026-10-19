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


def test_pi_staging_checks_reconciliation_evidence_when_given(agents, tmp_path, monkeypatch):
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
    # The Pi has no config agent; without evidence the PC/Jetson stages gate.
    assert manager.stage(bundle, "a" * 40)["live_overlay"]["shared"]["not_applicable"]
    manager.run = Mock(side_effect=["a" * 40, ""])
    with pytest.raises(RuntimeError, match="status unavailable"):
        manager.stage(bundle, "a" * 40, shared_status=tmp_path / "absent.json")
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


def stuck_commit(jetson, pc, *, delivered=False):
    call = jetson.peer.call
    def lose_commit(command, **kw):
        if command.get("phase") == "commit":
            if delivered:
                call(command, **kw)
            raise TimeoutError("fake lost commit")
        return call(command, **kw)
    jetson.peer.call = lose_commit
    result = request(jetson, key=KEY, value=.1)
    jetson.peer.call = call
    assert result["hosts"]["pc"]["state"] == "commit_pending"
    return result["transaction"]


def recover(agent, action, *, yes=False):
    return agent.handle(dict(actor="recovery-reviewer", action="recover_shared",
                             recovery_action=action, yes=yes))


@pytest.mark.parametrize("action", ["complete", "abandon"])
@pytest.mark.parametrize("state", ["commit_lost", "reply_lost", "jetson_drift", "pc_drift", "both_drift"])
def test_jetson_recovery_stuck_decisions(agents, action, state):
    jetson, pc = agents
    transaction = stuck_commit(jetson, pc, delivered=state == "reply_lost")
    if state in ("jetson_drift", "both_drift"):
        live.atomic_write(jetson.local.path, "laser:\n  offset_m:\n    x: 0.2\n")
        # Startup must preserve evidence and keep the local recovery IPC alive.
        jetson = SharedConfigAgent(jetson.local)
        jetson.peer, pc.peer = FakeLink(pc), FakeLink(jetson)
    if state in ("pc_drift", "both_drift"):
        live.atomic_write(pc.local.path, "laser:\n  offset_m:\n    x: 0.3\nsim:\n  freeze_frame: true\n")
        assert pc.reconcile()["state"] == ("rejected" if state == "pc_drift" else "pending")
    before = [live.read_overlay(a.local.path) for a in (jetson, pc)]
    persisted = [a.state_path.read_bytes() for a in (jetson, pc)]
    counts = [len(records(a)) for a in (jetson, pc)]
    status = request(jetson, "status")
    assert status["shared"]["recovery_required"]
    assert status["shared"]["recovery"][action]["overlay_before"] == before[0]
    preview = recover(jetson, action)
    assert preview["ok"] and preview["dry_run"], preview
    assert [a.state_path.read_bytes() for a in (jetson, pc)] == persisted
    assert [len(records(a)) for a in (jetson, pc)] == counts
    assert [live.read_overlay(a.local.path) for a in (jetson, pc)] == before
    result = recover(jetson, action, yes=True)
    assert result["ok"], result
    assert live.read_overlay(jetson.local.path) == preview["recovery"][action]["overlay_after"]
    assert pc.reconcile()["state"] == "ready"
    expected = .2 if action == "abandon" and state in ("jetson_drift", "both_drift") else .1
    assert overlay(pc)[KEY] == overlay(jetson)[KEY] == expected
    if state in ("pc_drift", "both_drift"):
        assert overlay(pc).get("sim.freeze_frame", False) == (action == "abandon")
    assert not jetson.state["decision"] and not pc.state["prepared"]
    assert not request(jetson, "status")["shared"]["recovery_required"]
    for agent in (jetson, pc):
        audit = [r for r in records(agent) if r.get("type") == "shared_recovery"]
        assert audit and any(r["transaction"] == transaction for r in audit)
        assert all(r["actor"] == "recovery-reviewer" and r["action"] == action for r in audit)
        assert all("overlay_before" in r and "overlay_after" in r for r in audit)
    from tools.deployment import shared_config_gate
    for agent in (jetson, pc):
        assert shared_config_gate(live.read_overlay(agent.local.path), agent.state_path)["ready"]


@pytest.mark.parametrize("action", ["complete", "abandon"])
@pytest.mark.parametrize("drift", [False, True])
def test_pc_recovery_prepared_plan(agents, action, drift):
    jetson, pc = agents
    stuck_commit(jetson, pc)
    if drift:
        live.atomic_write(pc.local.path, "laser:\n  offset_m:\n    x: 0.3\n")
    preview = recover(pc, action)
    assert preview["ok"] and preview["dry_run"]
    before = live.read_overlay(pc.local.path)
    assert recover(pc, action, yes=True)["ok"]
    assert live.read_overlay(pc.local.path) == preview["recovery"][action]["overlay_after"]
    assert pc.reconcile()["state"] == "ready"
    assert overlay(pc)[KEY] == overlay(jetson)[KEY] == .1
    assert not pc.state["prepared"] and not jetson.state["decision"]
    assert pc.state["ready"] and jetson.state["ready"]
    if action == "abandon":
        assert any(r.get("action") == action for r in records(jetson))
        assert before == preview["recovery"][action]["overlay_after"]


@pytest.mark.parametrize("host", [0, 1])
@pytest.mark.parametrize("action", ["complete", "abandon"])
def test_undecided_prepare_recovery(agents, host, action):
    jetson, pc = agents
    call = jetson.peer.call
    def lose_prepare(command, **kw):
        result = call(command, **kw)
        if command.get("phase") == "prepare":
            raise TimeoutError("fake lost prepare")
        return result
    jetson.peer.call = lose_prepare
    assert not request(jetson, key=KEY, value=.1)["ok"]
    jetson.peer.call = call
    result = recover(agents[host], action, yes=True)
    if host == 0 and action == "complete":
        assert not result["ok"] and "no durable decision" in result["error"]
        assert recover(jetson, "abandon", yes=True)["ok"]
    else:
        assert result["ok"], result
    assert pc.reconcile()["state"] == "ready"
    assert overlay(jetson) == overlay(pc) == {}
    assert not pc.state["prepared"] and not jetson.state["decision"]


@pytest.mark.parametrize("host", [0, 1])
@pytest.mark.parametrize("failure", ["validation", "fire_control"])
def test_complete_recovery_revalidates_both_hosts_before_writing(agents, host, failure):
    jetson, pc = agents
    stuck_commit(jetson, pc)
    # Both hosts need an actual value change to exercise the fire-control gate.
    live.atomic_write(jetson.local.path, "laser:\n  offset_m:\n    x: 0.2\n")
    before = [live.read_overlay(a.local.path) for a in agents]
    saved = [a.state_path.read_bytes() for a in agents]
    if failure == "validation":
        agents[host].local.bundles = Mock(side_effect=ConfigError("current stack rejected"))
    else:
        agents[host].local.fire_control_notice = Mock(side_effect=ConfigError("fire control unavailable"))
    for yes in (False, True):
        result = recover(jetson, "complete", yes=yes)
        assert not result["ok"], result
        assert [live.read_overlay(a.local.path) for a in agents] == before
        assert [a.state_path.read_bytes() for a in agents] == saved


def test_recovery_requires_action_digest_and_cli_actor(agents):
    jetson, pc = agents
    stuck_commit(jetson, pc)
    assert not jetson.handle(dict(actor="reviewer", action="recover_shared", yes=True))["ok"]
    assert not request(jetson, "recover_shared", recovery_action="abandon", yes=True)["ok"]
    result = jetson.handle(dict(actor="reviewer", action="recover_shared", recovery_action="abandon",
                               yes=True, expected_digest="stale"))
    assert not result["ok"] and "overlay changed" in result["error"]


@pytest.mark.parametrize("action", ["complete", "abandon"])
def test_recovery_replay_does_not_republish_or_restart(agents, action):
    jetson, pc = agents
    stuck_commit(jetson, pc)
    assert recover(jetson, action, yes=True)["ok"]
    assert pc.reconcile()["state"] == "ready"
    for agent in agents:
        agent.local.apply_plan = Mock(side_effect=AssertionError("completed recovery republished"))
    result = recover(jetson, action, yes=True)
    assert result["ok"] and result["idempotent"], result
    assert pc.reconcile()["state"] == "ready"


@pytest.mark.parametrize("host", [0, 1])
@pytest.mark.parametrize("action", ["complete", "abandon"])
def test_cli_recovery_uses_local_agent_and_is_dry_run_by_default(agents, monkeypatch, capsys, host, action):
    from contextlib import nullcontext
    import zmq
    from tools import idcs_config
    jetson, pc = agents
    stuck_commit(jetson, pc)
    agent = agents[host]
    monkeypatch.setenv("IDCS_LIVE_CONFIG", str(agent.local.path))
    commands, endpoints = [], []
    class Socket:
        def setsockopt(self, *_): pass
        def connect(self, endpoint): endpoints.append(endpoint)
        def send_json(self, command): commands.append(command)
        def recv_json(self): return agent.handle(commands[-1])
    context = Mock()
    context.socket.side_effect = lambda *_: nullcontext(Socket())
    monkeypatch.setattr(zmq, "Context", lambda: nullcontext(context))
    argv = ["--host", agent.local.host, "--actor", "recovery-reviewer", "recover_shared", "--action", action]
    before = live.read_overlay(agent.local.path)
    assert idcs_config.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"]
    assert live.read_overlay(agent.local.path) == before
    assert commands[-1]["yes"] is False
    assert endpoints == [live.command_endpoint()]
    assert idcs_config.main(argv + ["--yes"]) == 0
    assert not json.loads(capsys.readouterr().out)["dry_run"]
    assert commands[-1]["yes"] is True
    assert pc.reconcile()["state"] == "ready"


def test_complete_startup_revalidation_failure_keeps_recovery_available(agents):
    jetson, pc = agents
    stuck_commit(jetson, pc)
    live.atomic_write(jetson.local.path, "{}\n")  # decision durable, overlay not yet replaced
    jetson.local.bundles = Mock(side_effect=ConfigError("new release rejects decision"))
    restarted = SharedConfigAgent(jetson.local)
    assert request(restarted, "status")["shared"]["recovery_required"]
    assert not recover(restarted, "complete", yes=True)["ok"]
    assert overlay(restarted) == {}
    assert recover(restarted, "abandon", yes=True)["ok"]


def test_abandon_reconciliation_retains_overlay_when_validation_fails(agents):
    jetson, pc = agents
    stuck_commit(jetson, pc)
    assert recover(jetson, "abandon", yes=True)["ok"]
    pc.local.bundles = Mock(side_effect=ConfigError("reconciliation rejects current stack"))
    assert pc.reconcile()["state"] == "rejected"
    assert overlay(pc) == {} and overlay(jetson)[KEY] == .1
    assert not pc.state["ready"]


def test_pc_abandon_is_durable_while_offline(agents):
    jetson, pc = agents
    stuck_commit(jetson, pc)
    pc.peer.connected = False
    assert recover(pc, "abandon", yes=True)["ok"]
    assert pc.reconcile()["state"] == "pending"
    pc = SharedConfigAgent(pc.local)
    pc.peer, jetson.peer = FakeLink(jetson), FakeLink(pc)
    assert pc.reconcile()["state"] == "ready"
    assert not jetson.state["decision"] and not pc.state["prepared"]
    assert overlay(pc)[KEY] == .1


def test_jetson_complete_offline_refuses_and_abandon_reconciles_next_connect(agents):
    jetson, pc = agents
    stuck_commit(jetson, pc)
    jetson.peer.connected = False
    before = jetson.state_path.read_bytes()
    assert not recover(jetson, "complete", yes=True)["ok"]
    assert jetson.state_path.read_bytes() == before
    assert recover(jetson, "abandon", yes=True)["ok"]
    jetson.peer.connected = True
    assert pc.reconcile()["state"] == "ready"


def test_complete_retries_after_lost_peer_recovery_reply(agents):
    jetson, pc = agents
    stuck_commit(jetson, pc)
    call = jetson.peer.call
    def lose_recovery(command, **kw):
        result = call(command, **kw)
        if command.get("phase") == "recover" and command.get("yes"):
            raise TimeoutError("recovery response lost")
        return result
    jetson.peer.call = lose_recovery
    assert not recover(jetson, "complete", yes=True)["ok"]
    assert jetson.state["decision"] and not pc.state["prepared"]
    jetson.peer.call = call
    pc.local.apply_plan = Mock(side_effect=AssertionError("peer recovery repeated overlay write"))
    assert recover(jetson, "complete", yes=True)["ok"]
    assert pc.reconcile()["state"] == "ready"


def test_pc_abandon_refuses_to_cancel_a_newer_coordinator_decision(agents):
    jetson, pc = agents
    stuck_commit(jetson, pc)
    assert recover(pc, "abandon", yes=True)["ok"]
    jetson.state["decision"]["transaction"] = "newer-decision"
    before = deepcopy(jetson.state)
    result = pc.reconcile()
    assert result["state"] == "pending" and "newer decision" in result["error"]
    assert jetson.state == before and pc.state["reconciliation_requested"]


def test_stale_report_cannot_clear_abandon_reconciliation(agents):
    jetson, pc = agents
    transaction = stuck_commit(jetson, pc)
    assert recover(jetson, "abandon", yes=True)["ok"]
    reply = jetson.receive(dict(phase="report", transaction=transaction,
        result=dict(state="ready", shared_digest=live.digest(jetson.state["shared"]))))
    assert reply["result"]["state"] == "mismatch"
    assert pc.reconcile()["state"] == "ready"


def test_pc_abandon_journals_original_prepare_when_coordinator_evidence_was_lost(agents):
    jetson, pc = agents
    transaction = "prepare-without-coordinator-decision"
    assert pc.receive(dict(phase="prepare", actor="reviewer", transaction=transaction,
                           shared={KEY: .1}, expected_digest=live.digest({})))["ok"]
    assert recover(pc, "abandon", yes=True)["ok"]
    assert pc.reconcile()["state"] == "ready"
    for agent in agents:
        audit = [r for r in records(agent) if r.get("type") == "shared_recovery"]
        assert audit and all(r["transaction"] == transaction for r in audit)
