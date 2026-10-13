from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from common.config import load_config_bundle, resolve_config_paths
from common.operator_commands import CommandError, OperatorCommand, parse_command, parse_selection
from common.perception import PerceptionSnapshotV2
from jetson.deepstream.async_target_selection import OPERATOR_LOCK_POLICY, _apply_completed_snapshot
from jetson.operator_agent import AgentConfig, OperatorAgent, reply


class FakeSystemd:
    def __init__(self, active=(), trace=None):
        self.active = set(active)
        self.calls = []
        self.trace = [] if trace is None else trace

    def __call__(self, args, **_):
        self.calls.append(list(args))
        verb, units = args[0], args[1:]
        if verb == "is-active":
            return 0, "\n".join("active" if u in self.active else "inactive" for u in units)
        if verb == "start":
            self.trace.append(("Jetson", verb, *units))
            self.active.update(units)
            if "idcs-deepstream-video.service" in units:
                self.active.discard("idcs-camera.target")
            if "idcs-camera.target" in units:
                self.active.discard("idcs-deepstream-video.service")
                self.active.update(("idcs-deepstream-camera.service", "idcs-controller-camera.service"))
        elif verb == "stop":
            self.trace.append(("Jetson", verb, *units))
            self.active.difference_update(units)
        return 0, ""


class FakePC:
    def __init__(self, trace=None):
        self.connected = True
        self.unacked = False
        self.active_units = frozenset({"idcs-sim.target", "idcs-sim-session.target", "idcs-sim-streamer.service", "idcs-sim-ui.service",
                                       "idcs-sim-controller.service", "idcs-sim-panel.service"})
        self.trace = [] if trace is None else trace

    def online(self):
        return self.connected

    def pending(self):
        return self.unacked

    def run(self, args, **_):
        self.trace.append(("PC", *args))
        verb, unit = args
        units = set(self.active_units)
        if verb == "stop":
            units.discard(unit)
            if unit in ("idcs-sim.target", "idcs-sim-swarm.target", "idcs-sim-session.target"):
                units.difference_update(u for u in units.copy() if u.startswith("idcs-sim-"))
        else:
            units.add(unit)
            if unit == "idcs-sim.target":
                units.update(("idcs-sim-session.target", "idcs-sim-streamer.service", "idcs-sim-ui.service",
                              "idcs-sim-controller.service", "idcs-sim-panel.service"))
            if unit == "idcs-sim-swarm.target":
                units.update(("idcs-sim-session.target", "idcs-sim-swarm-streamer.service", "idcs-sim-ui.service",
                              "idcs-sim-swarm-controller.service", "idcs-sim-panel.service"))
        self.active_units = frozenset(units)
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
    agent = OperatorAgent(config, run=systemd, clock=clock, pc=FakePC(systemd.trace),
                          ready_probe=lambda *_: True)
    agent.refresh_units()
    return agent, systemd, clock


def test_score_clear_only_routes_to_fresh_software_swarm_without_lock_changes(config):
    from types import SimpleNamespace as NS
    from common.target_authority import TargetCatalog, CatalogEnvelope
    agent, systemd, _ = _agent(config)
    assert agent.handle(OperatorCommand('set_mode', mode='sim', scene='swarm'))[0]
    agent.wait_idle()
    catalog = CatalogEnvelope(catalog=TargetCatalog(source='sim', session='sim-A', sequence=1,
        created_monotonic_ns=1), age_ms=0.)
    agent.pc.targets = NS(forward=lambda now: catalog)
    agent.lock_track_id, agent.selection_mode = 42, 'operator'
    units, calls = agent.pc.active_units, []
    agent.pc.run = lambda args, **kw: calls.append((args, kw)) or (0, 'Swarm score cleared (epoch 1)')
    command = OperatorCommand('clear_swarm_score', source_session='sim-A')
    assert agent.handle(command)[0]
    agent.wait_idle()
    assert calls[0][0] == ['clear_score', 'idcs-sim-swarm-streamer.service']
    assert calls[0][1]['source_session'] == 'sim-A'
    assert agent.last_message == 'Swarm score cleared (epoch 1)'
    assert (agent.lock_track_id, agent.selection_mode, agent.pc.active_units) == (42, 'operator', units)
    assert not agent.handle(OperatorCommand('clear_swarm_score', source_session='old'))[0]
    agent.pc.targets = NS(forward=lambda now: None)
    assert not agent.handle(command)[0] and len(calls) == 1
    agent.pc.targets = NS(forward=lambda now: catalog)
    systemd.active.add('idcs-serial.service')
    assert not agent.handle(command)[0] and len(calls) == 1


def test_score_clear_rejects_single_or_pending_and_reports_ack_failure(config):
    from types import SimpleNamespace as NS
    from common.target_authority import TargetCatalog, CatalogEnvelope
    agent, _, _ = _agent(config)
    command = OperatorCommand('clear_swarm_score', source_session='sim-A')
    assert not agent.handle(command)[0]
    assert agent.handle(OperatorCommand('set_mode', mode='sim', scene='swarm'))[0]
    agent.wait_idle()
    agent.pc.unacked = True
    assert not agent.handle(command)[0]
    agent.pc.unacked = False
    catalog = CatalogEnvelope(catalog=TargetCatalog(source='sim', session='sim-A', sequence=1,
        created_monotonic_ns=1), age_ms=0.)
    agent.pc.targets = NS(forward=lambda now: catalog)
    agent.pc.run = lambda *a, **kw: (1, 'outcome unconfirmed')
    assert agent.handle(command)[0]
    agent.wait_idle()
    assert 'failed' in agent.last_message and 'unconfirmed' in agent.last_message


def test_sim_scene_switch_uses_ordered_transition_and_reports_nested_mode(config):
    agent, systemd, _ = _agent(config)
    assert agent.handle(OperatorCommand('set_mode',mode='sim',scene='swarm'))[0]
    agent.wait_idle()
    assert agent.current_mode() == 'sim_swarm'
    assert agent.state()['mode'] == 'sim' and agent.state()['sim_scene'] == 'swarm'
    assert [m['name'] for m in agent.state()['modes']] == ['camera','sim','standby']
    assert ('PC','stop','idcs-sim.target') in systemd.trace
    assert systemd.trace[-1] == ('PC','start','idcs-sim-swarm.target')
    assert 'idcs-sim-streamer.service' not in agent.pc.active_units
    assert agent.handle(OperatorCommand('set_mode',mode='sim',scene='single'))[0]
    agent.wait_idle()
    assert agent.current_mode() == 'sim' and agent.state()['sim_scene'] == 'single'
    assert 'idcs-sim-swarm-streamer.service' not in agent.pc.active_units


def test_swarm_uses_visual_catalog_not_sim_truth_and_restores_exact_scene(config):
    agent, systemd, _ = _agent(config)
    assert agent.handle(OperatorCommand('set_mode',mode='sim',scene='swarm'))[0]
    agent.wait_idle()
    agent.on_snapshot(_snapshot([(900,'drone')]).model_dump(mode='json'))
    catalog = agent.active_catalog()
    assert catalog.catalog.source == 'camera' and catalog.catalog.tracks[0].track_id == 900
    assert agent.handle(OperatorCommand('lock',track_id=900,source='camera',
                                       source_session=catalog.catalog.session))[0]
    assert agent.selection().source == 'camera'
    from types import SimpleNamespace
    from common.target_authority import CatalogEnvelope,catalog_from_snapshot
    sim_catalog=catalog_from_snapshot(_snapshot([(1,'drone')]),'sim','sim-pose-session',1)
    agent.pc.targets=SimpleNamespace(forward=lambda now:CatalogEnvelope(catalog=sim_catalog))
    assert agent.state()['sim_hud_session']=='sim-pose-session'
    assert agent.state()['target_catalog']['catalog']['session']!=agent.state()['sim_hud_session']
    def fail_camera(args,**kw):
        return (1,'injected') if args==['start','idcs-camera.target'] else systemd(args,**kw)
    agent.run = fail_camera
    assert agent.handle(OperatorCommand('set_mode',mode='camera'))[0]
    agent.wait_idle()
    assert agent.current_mode() == 'sim_swarm' and agent.lock_track_id is None
    assert 'restored Simulator / Swarm' in agent.last_message


@pytest.mark.parametrize('mode,scene',[('camera','swarm'),('sim','unknown'),('sim',True)])
def test_invalid_scene_command_is_rejected_without_actions(config,mode,scene):
    with pytest.raises(CommandError):
        parse_command(OperatorCommand('set_mode',mode=mode,scene=scene).to_json())
    agent, systemd, _ = _agent(config)
    assert not agent.handle(OperatorCommand('set_mode',mode=mode,scene=scene))[0]
    assert not systemd.trace


@pytest.mark.parametrize('unit',['idcs-controller.service','idcs-hil.target'])
def test_scene_choice_cannot_bypass_physical_gate(config,unit):
    agent, systemd, _ = _agent(config,active=(unit,))
    assert not agent.handle(OperatorCommand('set_mode',mode='sim',scene='swarm'))[0]
    assert not systemd.trace


def test_sim_scene_command_roundtrip():
    for scene in ('single','swarm'):
        command=OperatorCommand('set_mode',mode='sim',scene=scene)
        assert parse_command(command.to_json()) == command


def test_config_defaults_target_classes_from_swarm_exclusion(config) -> None:
    assert config.class_names == ("drone", "person")
    assert config.default_target_classes == ("drone",)
    assert config.command_bind == "tcp://*:5590"
    assert config.selection_bind == "tcp://127.0.0.1:5591"
    assert [m.name for m in config.modes] == ["camera", "sim", "sim_swarm", "standby"]
    assert config.pc_bind == "tcp://*:5592"


def test_lock_needs_a_present_track_and_expires_when_lost(config) -> None:
    agent, _, clock = _agent(config, active=("idcs-camera.target", "idcs-deepstream-camera.service", "idcs-controller-camera.service"))
    agent.pc.active_units = frozenset({"idcs-camera-ui.service"})
    assert agent.handle(OperatorCommand("lock", track_id=3)) == (False, "no perception: nothing to lock")
    agent.on_snapshot(_snapshot([(3, "drone"), (5, "drone")]).model_dump(mode="json"))
    assert not agent.handle(OperatorCommand("lock", track_id=9))[0]
    ok, _ = agent.handle(OperatorCommand("lock", track_id=5))
    assert ok and agent.selection().lock_track_id == 5
    clock.t += 0.5
    agent.on_snapshot(_snapshot([(3, "drone")]).model_dump(mode="json"))
    agent.expire_lock()
    assert agent.lock_track_id == 5
    clock.t += 0.6
    agent.on_snapshot({"tracks": [{"track_id": 3}]})
    agent.expire_lock()
    assert agent.lock_track_id is None and "lost" in agent.last_message


def test_operator_mode_holds_nearest_then_reacquires_after_loss(config) -> None:
    from types import SimpleNamespace
    from common.target_authority import CatalogEnvelope, catalog_from_snapshot

    agent, _, clock = _agent(config)
    def envelope(ids, ranges, seq):
        raw = _snapshot([(track_id, "drone") for track_id in ids]).model_dump(mode="json")
        raw["sequence"] = seq
        raw["assessments"] = [dict(track_id=track_id, distance_m=distance)
                              for track_id, distance in ranges.items()]
        return CatalogEnvelope(catalog=catalog_from_snapshot(
            PerceptionSnapshotV2.model_validate(raw), "sim", "sim-A", int(clock.t * 1e9)))

    current = [envelope((1, 2), {1: 20., 2: 10.}, 1)]
    agent.pc.targets = SimpleNamespace(forward=lambda _now: current[0])
    assert agent.handle(OperatorCommand("selection_mode", mode="operator"))[0]
    assert agent.lock_track_id == 2 and agent.selection().source_session == "sim-A"
    current[0] = envelope((1, 2), {1: 5., 2: 30.}, 2)
    agent.expire_lock()
    assert agent.lock_track_id == 2  # Do not chase a newly closer target.
    assert agent.handle(OperatorCommand("lock", track_id=1, source="sim", source_session="sim-A"))[0]
    assert agent.lock_track_id == 1
    current[0] = envelope((2,), {2: 30.}, 3)
    clock.t += config.lock_lost_s / 2
    agent.expire_lock()
    assert agent.lock_track_id == 1  # One dropout cannot cause a switch.
    clock.t += config.lock_lost_s / 2 + .01
    agent.expire_lock()
    assert agent.lock_track_id == 2 and agent.selection_mode == "operator"
    assert agent.selection().source_session == "sim-A"


def test_operator_mode_waits_without_qualified_range(config) -> None:
    from types import SimpleNamespace
    from common.target_authority import CatalogEnvelope, catalog_from_snapshot

    agent, _, _ = _agent(config)
    catalog = catalog_from_snapshot(_snapshot([(1, "drone")]), "sim", "sim-A", 1)
    agent.pc.targets = SimpleNamespace(forward=lambda _now: CatalogEnvelope(catalog=catalog))
    assert agent.handle(OperatorCommand("selection_mode", mode="operator"))[0]
    assert agent.selection_mode == "operator" and agent.lock_track_id is None


def test_target_classes_validated_and_persisted(config) -> None:
    agent, _, _ = _agent(config)
    assert not agent.handle(OperatorCommand("target_classes", classes=("bird",)))[0]
    assert agent.handle(OperatorCommand("target_classes", classes=("drone", "person")))[0]
    assert json.loads(config.state_file.read_text()) == {"target_classes": ["drone", "person"], "selection_mode": "planner"}
    again, _, _ = _agent(config)
    assert again.target_classes == ("drone", "person")


def test_mode_switch_runs_units_and_reports_mode(config) -> None:
    agent, systemd, _ = _agent(config)
    assert agent.current_mode() == "sim"
    ok, message = agent.handle(OperatorCommand("set_mode", mode="camera"))
    assert ok, message
    agent.wait_idle()
    assert ["start", "idcs-camera.target"] in systemd.calls
    assert agent.current_mode() == "camera" and agent.busy is None
    assert systemd.trace == [
        ("PC", "stop", "idcs-sim.target"), ("PC", "stop", "idcs-sim-swarm.target"),
        ("PC", "stop", "idcs-sim-session.target"),
        ("Jetson", "stop", "idcs-deepstream-video.service"),
        ("Jetson", "stop", "idcs-camera.target"),
        ("Jetson", "stop", "idcs-deepstream-camera.service"),
        ("Jetson", "stop", "idcs-controller-camera.service"),
        ("Jetson", "start", "idcs-camera.target"), ("PC", "start", "idcs-camera-ui.service"),
    ]
    assert agent.last_message == "mode Camera ready"
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
    assert out["state"]["mode"] == "sim"


@pytest.mark.parametrize("guard", ["offline", "unacked", "hil", "unknown_units"])
def test_mode_preflight_refuses_without_writes(config, guard):
    agent, systemd, _ = _agent(config)
    if guard == "offline":
        agent.pc.connected = False
    elif guard == "unacked":
        agent.pc.unacked = True
    elif guard == "hil":
        agent.pc.active_units |= {"idcs-hil-streamer.service"}
    else:
        agent.run = lambda *_a, **_k: (1, "unavailable")
    assert not agent.handle(OperatorCommand("set_mode", mode="camera"))[0]
    assert not systemd.trace


@pytest.mark.parametrize("unit", ["idcs-controller.service", "idcs-serial.service", "idcs-bridge.service"])
def test_motor_owner_blocks_mode_even_without_hil_target(config, unit):
    agent, systemd, _ = _agent(config, active=(unit,))
    assert not agent.handle(OperatorCommand("set_mode", mode="sim"))[0]
    assert not systemd.trace


def test_camera_to_sim_order_and_lock_reset(config):
    active = ("idcs-camera.target", "idcs-deepstream-camera.service", "idcs-controller-camera.service")
    agent, systemd, clock = _agent(config, active=active)
    agent.pc.active_units = frozenset({"idcs-camera-ui.service"})
    agent.on_snapshot(_snapshot([(7, "drone")]).model_dump(mode="json"))
    assert agent.handle(OperatorCommand("lock", track_id=7))[0]
    assert agent.handle(OperatorCommand("set_mode", mode="sim"))[0]
    agent.wait_idle()
    assert systemd.trace == [
        ("PC", "stop", "idcs-sim.target"), ("PC", "stop", "idcs-sim-swarm.target"),
        ("PC", "stop", "idcs-sim-session.target"),
        ("PC", "stop", "idcs-camera-ui.service"), ("Jetson", "stop", "idcs-camera.target"),
        ("Jetson", "stop", "idcs-deepstream-camera.service"),
        ("Jetson", "stop", "idcs-controller-camera.service"),
        ("Jetson", "stop", "idcs-deepstream-video.service"),
        ("Jetson", "start", "idcs-deepstream-video.service"), ("PC", "start", "idcs-sim.target"),
    ]
    assert agent.current_mode() == "sim" and agent.lock_track_id is None
    assert not agent.handle(OperatorCommand("lock", track_id=7))[0]


def test_failed_start_restores_previous_source(config):
    agent, systemd, _ = _agent(config)
    def fail_camera(args, **kw):
        if args == ["start", "idcs-camera.target"]:
            return 1, "injected start failure"
        return systemd(args, **kw)
    agent.run = fail_camera
    assert agent.handle(OperatorCommand("set_mode", mode="camera"))[0]
    agent.wait_idle()
    assert agent.current_mode() == "sim"
    assert "failed" in agent.last_message and "restored Simulator" in agent.last_message
    assert agent.busy is None


def test_unacknowledged_action_does_not_rollback_or_overtake(config):
    agent, systemd, _ = _agent(config)
    def lost_ack(args, **kw):
        agent.pc.unacked = True
        return 1, "acknowledgement timed out"
    agent.pc.run = lost_ack
    assert agent.handle(OperatorCommand("set_mode", mode="camera"))[0]
    agent.wait_idle()
    assert "recovery required" in agent.last_message
    assert not systemd.trace
    assert not agent.handle(OperatorCommand("set_mode", mode="sim"))[0]


def test_busy_refuses_second_mode_and_lock(config):
    agent, _, _ = _agent(config)
    entered, release = threading.Event(), threading.Event()
    def ready(*_):
        entered.set()
        release.wait(2)
        return True
    agent.ready_probe = ready
    assert agent.handle(OperatorCommand("set_mode", mode="camera"))[0]
    try:
        assert entered.wait(2)
        assert not agent.handle(OperatorCommand("set_mode", mode="sim"))[0]
        assert not agent.handle(OperatorCommand("lock", track_id=1))[0]
    finally:
        release.set()
        agent.wait_idle()


def test_readiness_timeout_restores_source_without_detector(config):
    agent, _, _ = _agent(replace(config, readiness_timeout_s=0.1))
    agent.ready_probe = lambda mode, _began: mode.name == "sim"
    assert agent.handle(OperatorCommand("set_mode", mode="camera"))[0]
    agent.wait_idle()
    assert "did not become ready" in agent.last_message and "restored Simulator" in agent.last_message


def test_target_active_does_not_mask_failed_child(config):
    agent, _, _ = _agent(config, active=("idcs-camera.target",))
    agent.pc.active_units = frozenset({"idcs-camera-ui.service"})
    assert agent.current_mode() == "mixed"


def test_wants_child_failure_cannot_report_ready(config):
    agent, systemd, _ = _agent(replace(config, readiness_timeout_s=0.1))
    def failed_child(args, **kw):
        result = systemd(args, **kw)
        systemd.active.discard("idcs-deepstream-camera.service")
        return result
    agent.run = failed_child
    assert agent.handle(OperatorCommand("set_mode", mode="camera"))[0]
    agent.wait_idle()
    assert "did not become ready" in agent.last_message and "restored Simulator" in agent.last_message


def test_stale_observations_do_not_count_for_readiness(config):
    agent, _, clock = _agent(config)
    agent._transition_s = clock.t
    for i in range(3):
        agent.on_snapshot({"frame": {"frame_id": i, "source_time_ns": i,
                                      "source_clock_domain": "jetson_monotonic",
                                      "observation_clock_domain": "jetson_monotonic",
                                      "observed_time_ns": 99_000_000_000}, "tracks": []})
    assert agent._fresh_frames == 0


def test_video_readiness_needs_unique_fresh_frames_domain_and_return(config, tmp_path):
    config = replace(config, health_file=tmp_path / "health.json")
    agent, _, clock = _agent(config)
    agent._transition_s = clock.t
    mode = config.mode("camera")
    config.health_file.write_text(json.dumps({"last_frame_monotonic_s": clock.t, "frames": 5, "return_frames": 1}))
    def frame(i, domain="jetson_monotonic", observed=100_000_000_000):
        return {"frame": {"frame_id": i, "source_time_ns": i,
                          "source_clock_domain": domain, "observed_time_ns": observed,
                          "observation_clock_domain": "jetson_monotonic"}, "tracks": []}
    for _ in range(3):
        agent.on_snapshot(frame(1))
    assert not agent._video_ready(mode, clock.t)
    agent.on_snapshot(frame(2))
    agent.on_snapshot(frame(3))
    assert agent._video_ready(mode, clock.t)  # no tracks or detections necessary
    agent.on_snapshot(frame(4, "pc_monotonic"))
    assert not agent._video_ready(mode, clock.t)
    config.health_file.write_text(json.dumps({"last_frame_monotonic_s": clock.t, "frames": 5, "return_frames": 0}))
    agent.on_snapshot(frame(5))
    assert not agent._video_ready(mode, clock.t)
    clock.t += 3
    assert not agent._video_ready(mode, 100.0)


def test_config_rejects_motor_action_plan(config):
    cfg = load_config_bundle(resolve_config_paths("configs/base")).mutable_copy()
    cfg["operator"]["modes"]["camera"]["start"] = ["idcs-hil.target"]
    from common.config import ConfigError
    with pytest.raises(ConfigError, match="non-video"):
        AgentConfig.from_config(cfg)


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
