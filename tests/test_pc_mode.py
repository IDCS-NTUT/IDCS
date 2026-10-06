import json
import threading

import pytest

from common.pc_mode import PC_STATUS_UNITS, PcModePoll, PcModeTask, parse_poll, parse_task
from jetson.pc_mode_channel import PcModeChannel
from pc.mode_agent import PcModeWorker, active_units


class Systemd:
    def __init__(self):
        self.active = set()
        self.actions = []
        self.states = {}

    def __call__(self, args, **_):
        verb, *units = args
        if verb == "is-active":
            return 0, "\n".join(self.states.get(u, "active" if u in self.active else "inactive") for u in units)
        self.actions.append(args)
        if verb == "start":
            self.active.update(units)
        else:
            self.active.difference_update(units)
        return 0, ""


def test_protocol_roundtrip_and_allowlist():
    task = PcModeTask(1, "start", "idcs-sim.target")
    assert parse_task(task.to_json()) == task
    poll = PcModePoll(frozenset({"idcs-sim.target"}), 1, True, "done", "session")
    assert parse_poll(poll.to_json()) == poll
    assert parse_task('{"type":"PcModeTask","id":null}') is None
    for unit in ("idcs-hil.target", "idcs-hil-streamer.service", "sshd.service"):
        with pytest.raises(ValueError):
            PcModeTask(1, "start", unit)
    for value in (True, 0, "1"):
        with pytest.raises(ValueError):
            parse_task(json.dumps({"type": "PcModeTask", "id": value, "verb": "start", "unit": "idcs-sim.target"}))
    with pytest.raises(ValueError):
        parse_poll('{"type":"PcModePoll","active_units":["sshd.service"],"session":"s"}')


def test_worker_duplicate_delivery_never_repeats_action():
    systemd = Systemd()
    worker = PcModeWorker(run=systemd)
    task = PcModeTask(42, "start", "idcs-sim.target")
    worker.accept(task)
    worker.wait_idle()
    worker.accept(task)
    worker.wait_idle()
    assert systemd.actions == [["start", "idcs-sim.target"]]
    assert worker.poll().completed_id == 42 and worker.poll().ok


@pytest.mark.parametrize("state", ["active", "activating", "reloading", "deactivating"])
def test_hil_owner_blocks_all_pc_actions(state):
    systemd = Systemd()
    systemd.states["idcs-hil-streamer.service"] = state
    worker = PcModeWorker(run=systemd)
    worker.accept(PcModeTask(1, "stop", "idcs-sim.target"))
    worker.wait_idle()
    assert not worker.poll().ok and not systemd.actions


def test_unknown_unit_state_fails_closed():
    with pytest.raises(RuntimeError):
        active_units(lambda *_a, **_k: (1, "unavailable"))
    systemd = Systemd()
    systemd.states[PC_STATUS_UNITS[0]] = "garbage"
    with pytest.raises(RuntimeError):
        active_units(systemd)


def test_channel_lost_ack_retains_action_and_rejects_overtaking():
    channel = PcModeChannel()
    channel.poll(PcModePoll(frozenset(), session="s").to_json())
    result = channel.run(["start", "idcs-sim.target"], timeout_s=0)
    assert result[0] and channel.pending()
    task = parse_task(channel.poll(PcModePoll(frozenset(), session="s").to_json()))
    assert channel.run(["stop", "idcs-sim.target"], timeout_s=0)[0]
    assert parse_task(channel.poll(PcModePoll(frozenset(), session="s").to_json())) == task
    channel.poll(PcModePoll(frozenset({task.unit}), task.task_id, True, session="s").to_json())
    assert not channel.pending()


def test_channel_peer_restart_does_not_replay_old_task():
    now = [100.0]
    channel = PcModeChannel(clock=lambda: now[0])
    channel.poll(PcModePoll(frozenset(), session="old").to_json())
    channel.run(["start", "idcs-sim.target"], timeout_s=0)
    with pytest.raises(ValueError, match="already connected"):
        channel.poll(PcModePoll(frozenset(), session="new").to_json())
    now[0] += 4
    assert parse_task(channel.poll(PcModePoll(frozenset(), session="new").to_json())) is None
    assert not channel.pending()


def test_offline_channel_has_no_pending_action():
    channel = PcModeChannel()
    assert channel.run(["start", "idcs-sim.target"], timeout_s=0)[0]
    assert not channel.pending()


def test_channel_worker_contract_integration():
    systemd = Systemd()
    worker = PcModeWorker(run=systemd)
    channel = PcModeChannel()
    channel.poll(worker.poll().to_json())
    done = []
    thread = threading.Thread(target=lambda: done.append(channel.run(["start", "idcs-sim.target"], timeout_s=2)))
    thread.start()
    try:
        for _ in range(100):
            worker.accept(parse_task(channel.poll(worker.poll().to_json())))
            worker.wait_idle(0.02)
            if not thread.is_alive():
                break
            thread.join(0.01)
    finally:
        thread.join(3)
    assert done == [(0, "")]
    assert systemd.actions == [["start", "idcs-sim.target"]]
