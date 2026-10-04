from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.config import load_config_bundle, resolve_config_paths
from common.operator_commands import CommandError, OperatorCommand, parse_command, parse_selection
from common.perception import PerceptionSnapshotV2
from jetson.deepstream.async_target_selection import OPERATOR_LOCK_POLICY, _apply_completed_snapshot
from jetson.operator_agent import AgentConfig, OperatorAgent, reply


class FakeSystemd:
    def __init__(self, active=()):
        self.active = set(active)
        self.calls = []

    def __call__(self, args, **_):
        self.calls.append(list(args))
        verb, units = args[0], args[1:]
        if verb == "is-active":
            return 0, "\n".join("active" if u in self.active else "inactive" for u in units)
        if verb == "start":
            self.active.update(units)
            if "idcs-deepstream-video.service" in units:
                self.active.discard("idcs-camera.target")
            if "idcs-camera.target" in units:
                self.active.discard("idcs-deepstream-video.service")
        elif verb == "stop":
            self.active.difference_update(units)
        return 0, ""


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


@pytest.fixture
def config(tmp_path) -> AgentConfig:
    bundle = load_config_bundle(resolve_config_paths("configs/base"))
    data = bundle.mutable_copy()
    data["operator"]["state_file"] = str(tmp_path / "state.json")
    return AgentConfig.from_config(data)


def _agent(config, active=("idcs-deepstream-video.service",)):
    systemd, clock = FakeSystemd(active), Clock()
    agent = OperatorAgent(config, run=systemd, clock=clock)
    agent.refresh_units()
    return agent, systemd, clock


def test_config_defaults_target_classes_from_swarm_exclusion(config) -> None:
    assert config.class_names == ("drone", "person")
    assert config.default_target_classes == ("drone",)
    assert config.command_bind == "tcp://*:5590"
    assert config.selection_bind == "tcp://127.0.0.1:5591"
    assert [m.name for m in config.modes] == ["camera", "pc_video", "standby"]


def test_lock_needs_a_present_track_and_expires_when_lost(config) -> None:
    agent, _, clock = _agent(config)
    assert agent.handle(OperatorCommand("lock", track_id=3)) == (False, "no perception: nothing to lock")
    agent.on_snapshot({"tracks": [{"track_id": 3}, {"track_id": 5}]})
    assert not agent.handle(OperatorCommand("lock", track_id=9))[0]
    ok, _ = agent.handle(OperatorCommand("lock", track_id=5))
    assert ok and agent.selection().lock_track_id == 5
    clock.t += 0.5
    agent.on_snapshot({"tracks": [{"track_id": 3}]})
    agent.expire_lock()
    assert agent.lock_track_id == 5
    clock.t += 0.6
    agent.on_snapshot({"tracks": [{"track_id": 3}]})
    agent.expire_lock()
    assert agent.lock_track_id is None and "lost" in agent.last_message


def test_target_classes_validated_and_persisted(config) -> None:
    agent, _, _ = _agent(config)
    assert not agent.handle(OperatorCommand("target_classes", classes=("bird",)))[0]
    assert agent.handle(OperatorCommand("target_classes", classes=("drone", "person")))[0]
    assert json.loads(config.state_file.read_text()) == {"target_classes": ["drone", "person"]}
    again, _, _ = _agent(config)
    assert again.target_classes == ("drone", "person")


def test_mode_switch_runs_units_and_reports_mode(config) -> None:
    agent, systemd, _ = _agent(config)
    assert agent.current_mode() == "pc_video"
    ok, message = agent.handle(OperatorCommand("set_mode", mode="camera"))
    assert ok, message
    agent.wait_idle()
    assert ["start", "idcs-camera.target"] in systemd.calls
    assert agent.current_mode() == "camera" and agent.busy is None
    assert agent.handle(OperatorCommand("set_mode", mode="standby"))[0]
    agent.wait_idle()
    assert agent.current_mode() == "standby"


def test_mode_switch_refused_while_motors_live(config) -> None:
    agent, systemd, _ = _agent(config, active=("idcs-hil.target", "idcs-bridge.service"))
    ok, message = agent.handle(OperatorCommand("set_mode", mode="camera"))
    assert not ok and "motor" in message
    assert not any(call[0] in ("start", "stop") for call in systemd.calls)


def test_recording_toggles_recorder_unit(config) -> None:
    agent, systemd, _ = _agent(config)
    assert agent.handle(OperatorCommand("recording", on=True))[0]
    agent.wait_idle()
    assert agent.state()["recording"] is True
    assert ["start", "idcs-recorder.service"] in systemd.calls


def test_reply_wraps_errors_and_state(config) -> None:
    agent, _, _ = _agent(config)
    out = json.loads(reply(agent, b'{"type": "OperatorCommand", "command": "explode"}'))
    assert out["ok"] is False and "unknown command" in out["message"]
    assert out["state"]["mode"] == "pc_video"


def test_command_round_trip_and_validation() -> None:
    for command in (OperatorCommand("lock", track_id=4), OperatorCommand("release"),
                    OperatorCommand("target_classes", classes=("drone",)),
                    OperatorCommand("set_mode", mode="camera"), OperatorCommand("recording", on=False)):
        assert parse_command(command.to_json()) == command
    for bad in ('{"type": "OperatorCommand", "command": "lock", "track_id": -1}',
                '{"type": "OperatorCommand", "command": "lock", "track_id": true}',
                '{"type": "OperatorCommand", "command": "target_classes", "classes": []}',
                '{"type": "OperatorCommand", "command": "recording", "on": "yes"}', "nope"):
        with pytest.raises(CommandError):
            parse_command(bad)
    assert parse_selection('{"type": "OperatorSelection", "sequence": 1, "lock_track_id": "x", '
                           '"target_classes": []}') is None


def _snapshot(tracks, selection=None) -> PerceptionSnapshotV2:
    payload = {
        "sequence": 1,
        "frame": {"frame_id": 10, "width": 1280, "height": 720, "source_time_ns": 1,
                  "source_clock_domain": "pc_monotonic", "observed_time_ns": 1,
                  "observation_clock_domain": "pc_monotonic"},
        "tracks": [{"track_id": tid, "class_id": cls, "confidence": 0.8, "missed_frames": 0,
                    "box": {"x": 0.1 * i, "y": 0.1, "w": 0.05, "h": 0.05}}
                   for i, (tid, cls) in enumerate(tracks)],
    }
    if selection is not None:
        payload["selection"] = {"track_id": selection, "source_frame_id": 10, "applied_frame_id": 10,
                                "selected_time_ns": 1, "selection_clock_domain": "pc_monotonic",
                                "policy": "swarm_planner"}
    return PerceptionSnapshotV2.model_validate(payload)


def test_operator_lock_overrides_planner_while_present() -> None:
    current = _snapshot([(1, "drone"), (2, "drone")])
    completed = _snapshot([(1, "drone"), (2, "drone")], selection=1)
    applied, selected = _apply_completed_snapshot(current, completed, lock_track_id=2)
    assert selected and applied.selection.track_id == 2
    assert applied.selection.policy == OPERATOR_LOCK_POLICY
    gone = _snapshot([(1, "drone")])
    applied, _ = _apply_completed_snapshot(gone, completed, lock_track_id=2)
    assert applied.selection.track_id == 1 and applied.selection.policy == "swarm_planner"


def test_planner_choice_of_untargeted_class_is_dropped() -> None:
    current = _snapshot([(1, "person"), (2, "drone")])
    completed = _snapshot([(1, "person"), (2, "drone")], selection=1)
    applied, selected = _apply_completed_snapshot(current, completed, target_classes=("drone",))
    assert not selected and applied.selection is None
