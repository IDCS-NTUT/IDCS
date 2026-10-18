"""Bounded local config/owner-loop contracts; no hosts, systemd or hardware."""
import json
import os
from pathlib import Path
import queue
import statistics
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
import zmq

from common.config import resolve_config_paths
import common.live_config as live
from jetson.operator_agent import AgentConfig, OperatorAgent, SELECTION_PERIOD_S, serve_loop


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("IDCS_LIVE_CONFIG", str(tmp_path / "live.yaml"))


def request(service, action, **kwargs):
    return service.handle(dict(type="ConfigCommand", actor="pi-panel", action=action, **kwargs))


def service(**kwargs):
    return live.ConfigAgent(ROOT, "jetson", run=Mock(return_value=(0, "fake")), **kwargs)


def test_cached_get_keeps_parser_defaults_without_runtime_or_parser_rebuild(monkeypatch):
    agent = service()
    first = request(agent, "get")
    assert first["ok"], first
    unit = "jetson/idcs-controller-camera.service"
    assert first["effective"][unit]["controller.manual_accel_limit_rad_s2"] == 2.
    build = Mock(side_effect=AssertionError("cached get rebuilt stack"))
    monkeypatch.setattr(live, "runtime_bundle", build)
    from jetson.control.runtime_config import ControlRuntimeConfig
    monkeypatch.setattr(ControlRuntimeConfig, "from_config", build)
    timings = []
    for _ in range(10):
        start = time.perf_counter()
        second = request(agent, "get")
        timings.append(time.perf_counter() - start)
        assert second == first
    assert request(agent, "status")["ok"]
    build.assert_not_called()
    # Callers cannot corrupt the cached lists/maps.
    second["effective"][unit].clear()
    assert request(agent, "get") == first
    print(f"cached Jetson-stack get (n=10): median={statistics.median(timings) * 1000:.2f} ms, "
          f"max={max(timings) * 1000:.2f} ms")


def test_mutation_validates_only_candidate_and_reuses_it_for_next_get(monkeypatch):
    agent = service()
    assert request(agent, "get")["ok"]
    build = Mock(wraps=live.runtime_bundle)
    monkeypatch.setattr(live, "runtime_bundle", build)
    start = time.perf_counter()
    result = request(agent, "set", key="operator.lock_lost_s", value=.9)
    elapsed = time.perf_counter() - start
    assert result["ok"], result
    assert build.call_count == len(agent.stacks)
    assert request(agent, "get")["ok"]
    assert build.call_count == len(agent.stacks)
    print(f"warm Jetson-stack panel mutation (fake units): {elapsed * 1000:.2f} ms")


def test_stack_file_and_external_overlay_changes_invalidate(tmp_path, monkeypatch):
    paths = list(resolve_config_paths(ROOT / "configs/base"))
    operator = tmp_path / "operator.yaml"
    source = next(p for p in paths if "operator" in yaml.safe_load(p.read_text(encoding="utf-8")))
    operator.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    paths[paths.index(source)] = operator
    unit = "jetson/idcs-operator-agent.service"
    agent = service(stacks={unit: dict(module="jetson.operator_agent", paths=paths)})
    first = request(agent, "get")
    assert first["ok"], first
    build = Mock(wraps=live.runtime_bundle)
    monkeypatch.setattr(live, "runtime_bundle", build)
    data = yaml.safe_load(operator.read_text(encoding="utf-8"))
    data["operator"]["lock_lost_s"] = .8
    operator.write_text(yaml.safe_dump(data), encoding="utf-8")
    second = request(agent, "get")
    assert second["release"][unit]["operator.lock_lost_s"] == .8
    assert build.call_count == 1  # empty overlay shares release validation
    # A same-size timestamp-only change must invalidate too.
    stat = operator.stat()
    os.utime(operator, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert request(agent, "get")["ok"] and build.call_count == 2
    live.atomic_write(live.live_path(), "operator:\n  lock_lost_s: 1.2\n")
    third = request(agent, "get")
    assert third["digest"] != first["digest"]
    assert third["effective"][unit]["operator.lock_lost_s"] == 1.2
    assert third["release"][unit]["operator.lock_lost_s"] == .8
    assert build.call_count == 3
    live.atomic_write(live.live_path(), "operator:\n  lock_lost_s: wrong\n")
    assert not request(agent, "get")["ok"]
    operator.unlink()
    assert not request(agent, "get")["ok"]


@pytest.mark.parametrize("actor", ["pi-panel", "pi-panel:calibration-preview",
                                   "pi-panel:calibration-discard", "pi-panel:calibration-save"])
def test_panel_never_collects_receipts_in_mutation_reply(actor, monkeypatch):
    unit = "jetson/idcs-operator-agent.service"
    spec = live.unit_stacks(ROOT, "jetson")[unit]
    agent = service(stacks={unit: spec}, active=lambda: {unit.split("/")[1]}, result_wait_s=1.)
    agent.status = Mock(side_effect=AssertionError("panel waited for a receipt"))
    result = agent.handle(dict(actor=actor, action="set", key="operator.lock_lost_s", value=.8))
    assert result["ok"], result
    assert result["units"][unit]["state"] == "awaiting_service"
    records = [json.loads(line) for line in live.live_path().with_name("live-journal.jsonl").read_text().splitlines()]
    assert records[-1]["type"] == "result"
    assert records[-1]["units"][unit]["state"] == "awaiting_service"


def test_worker_serializes_writers_and_owner_publishes_before_reply(monkeypatch):
    monkeypatch.setattr(live, "command_endpoint", lambda: "inproc://config")
    monkeypatch.setattr(live, "update_endpoint", lambda: "inproc://updates")
    agent = service()
    owner = threading.get_ident()
    handlers, trace, replies = [], [], []
    handle = agent.handle
    def worker_handle(request):
        handlers.append(threading.get_ident())
        return handle(request)
    agent.handle = worker_handle
    ctx = zmq.Context()
    channel = live.AgentChannel(ctx, agent)
    channel.on_update = lambda event: trace.append(("update", threading.get_ident()))
    def reply(value):
        trace.append(("reply", threading.get_ident()))
        replies.append(value)
    try:
        for value in (.8, 1.2):
            channel.submit(dict(actor="pi-panel", action="set", key="operator.lock_lost_s",
                                value=value, expected_digest=live.digest({})), reply)
        deadline = time.monotonic() + 5
        while len(replies) < 2:
            assert time.monotonic() < deadline
            channel.poll()
            time.sleep(.002)
        assert replies[0]["ok"] and not replies[1]["ok"]
        assert "overlay changed" in replies[1]["error"]
        assert len(set(handlers)) == 1 and handlers[0] != owner
        assert trace == [("update", owner), ("reply", owner), ("reply", owner)]
        records = [json.loads(line) for line in live.live_path().with_name("live-journal.jsonl").read_text().splitlines()]
        assert len(records) == 2 and records[-1]["type"] == "result"
        assert records[0]["transaction"] == records[1]["transaction"]
    finally:
        channel.close()
        ctx.term()


def test_completed_request_wakes_the_idle_owner_poller(monkeypatch):
    monkeypatch.setattr(live, "command_endpoint", lambda: "inproc://config")
    monkeypatch.setattr(live, "update_endpoint", lambda: "inproc://updates")
    started, release = threading.Event(), threading.Event()
    def handle(request):
        started.set()
        assert release.wait(2)
        return dict(type="ConfigReply", ok=True)
    ctx = zmq.Context()
    channel = live.AgentChannel(ctx, SimpleNamespace(handle=handle))
    replies = []
    try:
        poller = zmq.Poller()
        channel.register(poller)
        channel.submit(dict(action="get"), replies.append)
        assert started.wait(2)
        release.set()
        assert channel.ready in dict(poller.poll(1000))
        channel.poll()
        assert replies == [dict(type="ConfigReply", ok=True)]
    finally:
        release.set()
        channel.close()
        ctx.term()


@pytest.mark.parametrize("via", ["panel", "cli"])
def test_operator_loop_cadence_during_slow_config_get_and_set(tmp_path, monkeypatch, via):
    """Exercise the production loop and both REQ entry points with 1s builds."""
    monkeypatch.setattr(live, "command_endpoint", lambda: "inproc://config")
    monkeypatch.setattr(live, "update_endpoint", lambda: "inproc://updates")
    cfg = live.runtime_bundle(resolve_config_paths(ROOT / "configs/base")).mutable_copy()
    cfg["operator"]["state_file"] = str(tmp_path / "operator-state.json")
    runner = Mock(side_effect=lambda args, **_: (0, " ".join("inactive" for _ in args[1:])))
    operator = OperatorAgent(AgentConfig.from_config(cfg), run=runner)
    operator.refresh_units()
    operator.pc.poll = lambda _: "{}"
    expiry = Mock(wraps=operator.expire_lock)
    operator.expire_lock = expiry
    unit = "jetson/idcs-operator-agent.service"
    agent = service(stacks={unit: live.unit_stacks(ROOT, "jetson")[unit]},
                    active=lambda: {unit.split("/")[1]})
    original = agent.bundles
    slow = threading.Event()
    def bundles(overlay):
        slow.set()
        time.sleep(1.)
        return original(overlay)
    agent.bundles = bundles
    ctx = zmq.Context()
    channel = live.AgentChannel(ctx, agent)
    rep = ctx.socket(zmq.ROUTER)
    rep.bind("inproc://operator")
    pc_rep = ctx.socket(zmq.REP)
    pc_rep.bind("inproc://pc")
    poller = zmq.Poller()
    channel.register(poller)
    poller.register(rep, zmq.POLLIN)
    poller.register(pc_rep, zmq.POLLIN)
    stamps, control_latencies, pc_latencies, replies = [], [], [], []
    failures = queue.SimpleQueue()
    stop = threading.Event()
    def publisher(*args, **kwargs):
        stamps.append(time.monotonic())
    def client():
        socket = ctx.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 8000)
        socket.setsockopt(zmq.REQ_CORRELATE, 1)
        socket.connect("inproc://operator" if via == "panel" else live.command_endpoint())
        try:
            for action in (dict(action="get"), dict(action="set", key="operator.lock_lost_s", value=.8)):
                socket.send_json(dict(type="ConfigCommand", actor="pi-panel" if via == "panel" else "cli-test", **action))
                result = socket.recv_json()
                assert result["ok"], result
                replies.append(result)
        except BaseException as exc:
            failures.put(exc)
            stop.set()
        finally:
            socket.close(0)
    def controls():
        sockets = []
        try:
            assert slow.wait(3)
            for endpoint in ("inproc://operator", "inproc://pc"):
                socket = ctx.socket(zmq.REQ)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVTIMEO, 1000)
                socket.connect(endpoint)
                sockets.append(socket)
            while not stop.is_set() and len(replies) < 2:
                start = time.monotonic()
                sockets[0].send_json(dict(type="OperatorCommand", command="status" if control_latencies else "release"))
                assert sockets[0].recv_json()["ok"]
                control_latencies.append(time.monotonic() - start)
                start = time.monotonic()
                sockets[1].send_json({})
                assert "selection" in sockets[1].recv_json()
                pc_latencies.append(time.monotonic() - start)
                stop.wait(.05)
            stop.set()
        except BaseException as exc:
            failures.put(exc)
            stop.set()
        finally:
            for socket in sockets:
                socket.close(0)
    threads = [threading.Thread(target=client), threading.Thread(target=controls)]
    timer = threading.Timer(12, stop.set)
    try:
        timer.start()
        for thread in threads:
            thread.start()
        serve_loop(operator, channel, SimpleNamespace(tick=lambda **_: None), poller,
                   rep, pc_rep, SimpleNamespace(send_string=publisher),
                   SimpleNamespace(send_string=lambda *a, **kw: None), None, stop)
    finally:
        stop.set()
        timer.cancel()
        for thread in threads:
            thread.join(9)
        channel.close()
        rep.close(0)
        pc_rep.close(0)
        ctx.term()
    assert all(not t.is_alive() for t in threads)
    if not failures.empty():
        raise failures.get()
    assert len(replies) == 2
    assert len(stamps) > 10 and expiry.call_count > 20
    gap = max(b - a for a, b in zip(stamps, stamps[1:]))
    assert gap <= 2 * SELECTION_PERIOD_S
    assert len(control_latencies) > 10 and max(control_latencies) <= 2 * SELECTION_PERIOD_S
    assert len(pc_latencies) > 10 and max(pc_latencies) <= 2 * SELECTION_PERIOD_S
    print(f"{via} slow-config selection max gap: {gap:.3f}s; "
          f"operator reply: {max(control_latencies):.3f}s; PC reply: {max(pc_latencies):.3f}s")
