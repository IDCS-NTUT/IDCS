import json
import queue
from pathlib import Path
import pytest

from common.perception import PerceptionSnapshotV2
from common.target_authority import (CatalogCache, CatalogEnvelope, SelectionReceiver,
                                     catalog_from_snapshot, closest_eligible_track)
from common.operator_commands import OperatorCommand, OperatorSelection, parse_command, CommandError
from common.pc_mode import PcModePoll, PC_STATUS_UNITS, parse_poll
from common.config import load_config_bundle, resolve_config_paths
from jetson.operator_agent import AgentConfig, OperatorAgent
from jetson.pc_mode_channel import PcModeChannel
from pc.mode_agent import PcModeWorker, SimActivitySampler
from common.schemas import CamState
from rpi.display.menu import Menu, Page, Action, Settings, NavEvent
from rpi.display.inputs import PanelNavigator
from jetson.deepstream.async_target_selection import AsyncDeepStreamTargetSelector
from jetson.deepstream.range_guard import KnownSizeRangeGuard


def snapshot(ids=(1, 2), seq=1, session="sim-A"):
    return PerceptionSnapshotV2.model_validate(dict(sequence=seq,
        frame=dict(frame_id=seq, source_time_ns=10_000_000_000 + seq, observed_time_ns=10_000_000_000 + seq,
            source_clock_domain="pc_monotonic", observation_clock_domain="pc_monotonic",
            tracker_session=session, source_identity_verified=True, width=1280, height=720),
        tracks=[dict(track_id=tid, class_id="drone", confidence=1., missed_frames=0,
            box=dict(x=.2 if tid == 1 else .7, y=.4, w=.05, h=.1)) for tid in ids],
        selection=None if not ids else dict(track_id=ids[0], source_frame_id=seq, applied_frame_id=seq,
            selected_time_ns=10_000_000_000 + seq, selection_clock_domain="pc_monotonic", policy="sim_ground_truth")))


def selection(seq=1, target=2, mode="operator", source="sim", session="sim-A", authority="agent-A"):
    return OperatorSelection(seq, target, ("drone",), mode, source, session, authority)


def test_operator_selection_no_fallback_on_loss_silence_or_wrong_namespace():
    receiver = SelectionReceiver()
    assert receiver.apply(snapshot(), "sim", "sim-A", 0).selection is None
    receiver.receive(selection().to_json(), 10.)
    assert receiver.apply(snapshot(), "sim", "sim-A", 10.).selection.track_id == 2
    for s, source, session, now in ((snapshot((1,)), "sim", "sim-A", 10.1),
            (snapshot(), "sim", "sim-B", 10.1), (snapshot(), "camera", "sim-A", 10.1),
            (snapshot(), "sim", "sim-A", 12.01)):
        assert receiver.apply(s, source, session, now).selection is None
    receiver.receive(selection(seq=2, target=None, mode="planner").to_json(), 13.)
    assert receiver.apply(snapshot(), "sim", "sim-A", 13.).selection.track_id == 1


def test_receiver_rejects_old_command_and_retired_agent():
    r = SelectionReceiver()
    assert r.receive(selection(seq=3).to_json(), 10.)
    assert not r.receive(selection(seq=2, mode="planner").to_json(), 10.1)
    assert r.receive(selection(authority="agent-B", target=None).to_json(), 10.2)
    assert not r.receive(selection(seq=100, authority="agent-A", mode="planner").to_json(), 10.3)
    assert r.apply(snapshot(), "sim", "sim-A", 10.3).selection is None


def test_camera_selector_waits_for_first_configured_authority():
    # No worker, network or GPU: exercise the actual probe application path.
    selector = object.__new__(AsyncDeepStreamTargetSelector)
    selector._config = load_config_bundle(resolve_config_paths("configs/base")).mutable_copy()
    selector._config_digest = selector._worker_config_digest = "a" * 64
    selector._labels = {0: "drone"}
    selector._range_guard = KnownSizeRangeGuard(
        load_config_bundle(resolve_config_paths("configs/base")).mutable_copy(), selector._labels)
    selector._poll_operator = lambda: (None, None)
    selector._requests, selector._results = queue.Queue(), queue.Queue()
    selector._latest = None
    selector._operator_sub, selector._operator = object(), None
    selector.submitted = 0
    assert selector.submit_and_apply_snapshot(snapshot()).selection is None


def test_catalog_duplicates_do_not_renew_and_restart_retires_previous_source():
    cache = CatalogCache()
    envelope = CatalogEnvelope(catalog=catalog_from_snapshot(snapshot(), "sim", "A", 1), age_ms=100.)
    assert cache.accept(envelope, 1.)
    assert not cache.accept(envelope, 1.2)
    assert cache.forward(1.39) is not None
    assert cache.forward(1.41) is None
    assert cache.accept(envelope.model_copy(update={"catalog": envelope.catalog.model_copy(update={"session": "B"})}), 2.)
    assert not cache.accept(envelope.model_copy(update={"catalog": envelope.catalog.model_copy(update={"sequence": 999})}), 2.1)


def test_nearest_requires_visible_allowed_measured_range_not_box_size():
    data = snapshot((1, 2, 3)).model_dump(mode="json")
    data["tracks"][1]["class_id"] = "person"
    data["tracks"][2]["box"].update(x=.4, w=.5)  # Area is not qualified range.
    data["assessments"] = [dict(track_id=1, distance_m=30., distance_quality="measured"),
                           dict(track_id=2, distance_m=10., distance_quality="measured"),
                           dict(track_id=3, distance_m=5., distance_quality="held")]
    catalog = catalog_from_snapshot(PerceptionSnapshotV2.model_validate(data), "sim", "A", 1)
    assert closest_eligible_track(catalog, ("drone", "person")) == 2
    assert closest_eligible_track(catalog, ("drone",)) == 1
    hidden = catalog.model_copy(update={"tracks": tuple(
        track.model_copy(update={"missed_frames": 1}) if track.track_id == 1 else track
        for track in catalog.tracks)})
    assert closest_eligible_track(hidden, ("drone",)) is None


def agent(tmp_path):
    cfg = load_config_bundle(resolve_config_paths("configs/base")).mutable_copy()
    cfg["operator"]["state_file"] = str(tmp_path / "agent.json")
    clock = [10.]
    pc = PcModeChannel(lambda: clock[0])
    pc.active_units = frozenset({"idcs-sim.target", "idcs-sim-session.target", "idcs-sim-streamer.service",
        "idcs-sim-controller.service", "idcs-sim-panel.service", "idcs-sim-ui.service"})
    pc.last_seen = 10.
    a = OperatorAgent(AgentConfig.from_config(cfg), pc=pc, clock=lambda: clock[0])
    states = {"idcs-deepstream-video.service": "active"}
    a._units = (True, states, frozenset(states))
    pc.targets.accept(CatalogEnvelope(catalog=catalog_from_snapshot(snapshot(), "sim", "sim-A", 1)), 10.)
    return a, clock


def test_agent_requires_active_source_and_preserves_operator_after_loss_restart(tmp_path):
    a, clock = agent(tmp_path)
    assert a.current_mode() == "sim"
    assert not a.handle(OperatorCommand("lock", track_id=2, source="camera", source_session="sim-A"))[0]
    assert not a.handle(OperatorCommand("lock", track_id=2))[0]
    assert a.handle(OperatorCommand("lock", track_id=2, source="sim", source_session="sim-A"))[0]
    assert a.selection().source_session == "sim-A" and a.selection_mode == "operator"
    a.pc.targets.accept(CatalogEnvelope(catalog=catalog_from_snapshot(snapshot((1,), seq=2), "sim", "sim-A", 2)), 10.2)
    clock[0] = 11.1
    a.expire_lock()
    assert a.lock_track_id is None and a.selection_mode == "operator"
    restarted = OperatorAgent(a.config, pc=a.pc, clock=a.clock)
    assert restarted.selection_mode == "operator" and restarted.lock_track_id is None
    assert a.handle(OperatorCommand("selection_mode", mode="planner"))[0]
    assert a.selection_mode == "planner"


def test_pc_initiated_catalog_roundtrip_same_host_age_and_owner_guard(tmp_path):
    a, clock = agent(tmp_path)
    sampler = SimActivitySampler()
    cat = catalog_from_snapshot(snapshot(), "sim", "sim-A", 10_000_000_000)
    state = CamState(frame_id=1, src_ts_ms=10000, pan=0., tilt=0., target_catalog=cat)
    sampler.receive(state.model_dump_json().encode(), 10_100_000_000)
    units = a.pc.active_units
    def run(*_, **__):
        return 0, "\n".join("active" if u in units else "inactive" for u in PC_STATUS_UNITS)
    poll = PcModeWorker(run=run).poll(targets=lambda: sampler.targets.forward(10.2))
    assert parse_poll(poll.to_json()).targets.age_ms == pytest.approx(200.)
    channel = PcModeChannel(lambda: 1000.)
    channel.poll(poll.to_json())
    assert channel.targets.forward(1000.2).age_ms == pytest.approx(400.)
    units = frozenset({"idcs-hil-streamer.service"})
    assert PcModeWorker(run=run).poll(targets=lambda: sampler.targets.forward(10.2)).targets is None


def test_joystick_picker_browse_cancel_confirm_and_return_to_planner():
    calls = []
    picker = Page("Select target", items=(Action("Use planner", lambda: calls.append("planner")),
        Action("one", lambda: calls.append(1)), Action("two", lambda: calls.append(2))))
    menu = Menu(Page("Menu"), Settings({}, None), picker_factory=lambda: picker)
    nav = PanelNavigator()
    t = 0.
    def push(x, y):
        nonlocal t
        for value in ((0., 0.), (x, y)):
            t += .1
            for event in nav.update({"joystick": value, "manual_active": False}, t):
                menu.handle(event)
    push(-1, 0)
    assert menu.picker_active
    push(0, 1); push(0, 1)  # backward/pitch-up browses down
    assert menu.view().cursor == 2 and calls == []
    push(-1, 0)
    assert not menu.is_open and calls == []
    push(-1, 0); push(0, 1); push(0, -1)  # forward/pitch-down browses up
    assert menu.view().cursor == 0 and calls == []
    push(0, 1); push(0, 1); push(1, 0)
    assert calls == [2] and not menu.is_open
    push(-1, 0); push(1, 0)
    assert calls == [2, "planner"]
    push(1, 0)
    assert menu.is_open and not menu.picker_active
    for panel in ({"manual_active": True}, {"emergency": True}):
        assert nav.update({**panel, "joystick": (1., 0.)}, t + 1) == []


@pytest.mark.parametrize("payload", [dict(command="selection_mode", mode="manual"),
    dict(command="lock", track_id=2, source="sim"), dict(command="lock", track_id=2, source="other", source_session="A")])
def test_bad_selection_command_rejected(payload):
    with pytest.raises(CommandError):
        parse_command(json.dumps({"type": "OperatorCommand", **payload}))
