"""Settings workflow tests: deterministic fakes, no hosts or device owners."""
from copy import deepcopy
import json
import struct
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from common.live_config import ConfigAgent, digest, live_path, read_overlay
from rpi.display.config_menu import ConfigMenu, Entry, EDIT_BANDS, fire_control_enabled
from rpi.display.inputs import key_events
from rpi.display.menu import Action, Menu, NavEvent as N, Page, Settings
from rpi.display.status import SystemStatus
from rpi.display.commands import CommandClient


VALUES = {
    "controller.rate_limit_rad_s": .8, "controller.accel_limit_rad_s2": 3.5,
    "controller.manual_rate_limit_rad_s": .5, "controller.manual_accel_limit_rad_s2": 2.,
    "controller.coast_s": .5, "operator.lock_lost_s": .5,
    "controller.idle_return_s": 1., "controller.idle_return_rate_rad_s": .3,
    "deepstream.shadow_tracks.enabled": True, "deepstream.shadow_tracks.max_age_frames": 30,
    "deepstream.shadow_tracks.min_confidence": .5, "deepstream.id_stitching.enabled": True,
    "deepstream.id_stitching.max_gap_frames": 30,
    "threat_eval.zones.warning.radius_m": 10., "threat_eval.zones.restricted.radius_m": 5.,
    "threat_eval.zones.critical.radius_m": .5, "swarm_eval.enabled": True,
    "swarm_eval.max_engage_distance_m": 25., "swarm_eval.excluded_target_classes": ["person"],
    "swarm_eval.damage_by_class.drone": 2., "swarm_eval.damage_by_class.person": 0.,
    "laser.dir_cam.x": 0., "laser.dir_cam.y": 0., "laser.dir_cam.z": 1.,
    "laser.offset_m.x": 0., "laser.offset_m.y": -.4, "laser.offset_m.z": 0.,
    "camera.known_size_ranging.class_sizes_m.drone": .52,
    "camera.known_size_ranging.class_sizes_m.person": .45,
    "perception.class_labels.0": "drone", "perception.class_labels.1": "person",
    "gimbal.yaw_rate_limit_rad_s": .8, "gimbal.pitch_rate_limit_rad_s": .8,
}


@pytest.fixture
def ui():
    sent = []
    menu = ConfigMenu(lambda request: sent.append(request) or True, lambda: False, clock=lambda: 1.)
    menu.open()
    menu.on_reply(dict(ok=True, digest="d0", overlay={}, effective={"camera": deepcopy(VALUES)},
                       release={"camera": deepcopy(VALUES)}, units={}))
    sent.clear()
    return menu, sent


def confirm(menu):
    assert menu.screen == "confirm" and menu.cursor == 0
    menu.handle(N.DOWN)
    menu.handle(N.SELECT)


def ack(menu, **kw):
    menu.on_reply(dict(ok=True, digest="d1", changed_keys=[menu.entry.key] if menu.entry else [],
                       units={"jetson/controller": dict(state="applied")}, **kw))


def test_root_integrates_without_changing_existing_menu(ui):
    config, sent = ui
    menu = Menu(Page("Menu", (Action("Existing", lambda: None), Action("Settings", config.open))), Settings({}, None))
    menu.config_menu = config
    config.active = False
    menu.handle(N.MENU)
    assert menu.view().rows[0].label == "Existing"
    menu.handle(N.DOWN)
    menu.handle(N.RIGHT)
    assert menu.view().title == "Settings"
    config.handle(N.LEFT)
    assert menu.view().title == "Menu"


def test_navigation_edit_cancel_confirmation_and_exactly_one_set(ui):
    menu, sent = ui
    menu.handle(N.SELECT)
    assert menu.screen == "group" and menu.group == 0
    menu._edit(Entry("controller.rate_limit_rad_s"))
    menu.handle(N.DOWN)
    assert menu.proposed == .75 and not sent
    assert any("Release: 0.8" == line for line in menu.view().lines)
    menu.handle(N.LEFT)
    assert not sent
    menu._edit(Entry("controller.rate_limit_rad_s"))
    menu.handle(N.DOWN)
    menu.handle(N.SELECT)
    menu.handle(N.SELECT)  # Cancel first
    assert menu.screen == "edit" and not sent
    menu.handle(N.SELECT)
    confirm(menu)
    assert len(sent) == 1
    assert sent[0] == dict(type="ConfigCommand", action="set", actor="pi-panel",
                          expected_digest="d0", key="controller.rate_limit_rad_s", value=.75)
    menu.handle(N.SELECT)
    assert len(sent) == 1
    assert menu.effective["controller.rate_limit_rad_s"] == .8  # never optimistic


@pytest.mark.parametrize("key,band", EDIT_BANDS.items())
def test_bands_are_finite_and_steps_positive(key, band):
    step, low, high = band
    assert step > 0 and low < high


def test_step_clamp_acceleration_and_bridge_cap(ui):
    menu, sent = ui
    menu._edit(Entry("controller.rate_limit_rad_s"))
    menu.release["gimbal.yaw_rate_limit_rad_s"] = .7
    for _ in range(12):
        menu.handle(N.DOWN)
    assert menu.proposed == .25
    assert menu.repeats >= 8
    for _ in range(12):
        menu.handle(N.UP)
    assert menu.proposed == .7
    assert not sent


def test_keyboard_exact_entry_backspace_integer_and_band_validation(ui):
    menu, sent = ui
    menu._edit(Entry("controller.rate_limit_rad_s"))
    for c in ".769\b":
        menu.handle(c)
    menu.handle(N.SELECT)
    assert menu.confirm_request["value"] == .76
    menu.handle(N.SELECT)
    menu.typed = "-1"
    menu.handle(N.SELECT)
    assert menu.screen == "edit" and "Invalid" in menu.message and not sent
    menu._edit(Entry("deepstream.shadow_tracks.max_age_frames"))
    menu.typed = "12.5"
    menu.handle(N.SELECT)
    assert menu.screen == "edit" and not sent
    menu.typed = "15"
    menu.handle(N.SELECT)
    confirm(menu)
    assert sent[-1]["value"] == 15 and type(sent[-1]["value"]) is int


def test_usb_raw_text_mode_is_opt_in():
    raw = b"".join(struct.pack("llHHi", 0, 0, 1, c, 1) for c in (2, 11, 52, 12, 14, 28))
    assert key_events(raw, text_entry=True) == ["1", "0", ".", "-", "\b", N.SELECT]
    assert key_events(raw) == [N.BACK, N.SELECT]


def test_boolean_and_dynamic_list_membership_need_confirmation(ui):
    menu, sent = ui
    menu._edit(Entry("deepstream.shadow_tracks.enabled"))
    assert menu.screen == "confirm" and not sent
    menu.handle(N.SELECT)
    assert not sent
    entries = menu.entries(3)
    assert {e.member for e in entries if e.member} == {"drone", "person"}
    menu._edit(Entry("swarm_eval.excluded_target_classes", "drone"))
    assert not sent
    confirm(menu)
    assert sent[-1]["value"] == ["person", "drone"]


def test_revert_and_revert_all_preserve_unmanaged_overlay(ui):
    menu, sent = ui
    menu.overlay = {"laser.dir_cam.x": .01, "controller.yaw_kp": 6., "operator.lock_lost_s": .8}
    menu.effective["operator.lock_lost_s"] = .8
    menu.group = 1
    assert any(row.label.startswith("* ") for row in menu.view().rows) is False  # root
    menu._enter_group(1)
    assert any(row.label.startswith("* ") for row in menu.view().rows)
    menu._edit(Entry("operator.lock_lost_s"))
    menu.handle(N.SELECT)
    menu.handle(N.DOWN)
    menu.handle(N.DOWN)
    menu.handle(N.SELECT)  # Revert opens another Cancel/Confirm
    assert menu.confirm_request == dict(action="unset", key="operator.lock_lost_s")
    assert not sent
    confirm(menu)
    assert sent[-1]["action"] == "unset"
    menu.pending = None
    sent.clear()
    menu._revert_all()
    assert not sent
    confirm(menu)
    assert len(sent) == 1 and sent[0]["action"] == "set_many"
    assert sent[0]["unset_keys"] == ["laser.dir_cam.x", "operator.lock_lost_s"]
    menu._remaining()
    assert "controller.yaw_kp" in menu.view().lines


@pytest.mark.parametrize("save", [False, True])
def test_calibration_preview_leave_save_discard_one_transaction(ui, save):
    menu, sent = ui
    menu.overlay = {"laser.offset_m.y": -.41}
    menu.effective["laser.offset_m.y"] = -.41
    menu._enter_group(4)
    menu._edit(Entry("laser.dir_cam.x"))
    menu.handle(N.UP)
    assert len(sent) == 1 and sent[0]["actor"] == "pi-panel:calibration-preview"
    ack(menu)
    assert sent[-1]["action"] == "get"
    menu.on_reply(dict(ok=True, digest="d1", overlay={"laser": {"dir_cam": {"x": .0005}, "offset_m": {"y": -.41}}},
                       effective={"camera": dict(VALUES, **{"laser.dir_cam.x": .0005})},
                       release={"camera": VALUES}, units={}))
    menu._edit(Entry("laser.offset_m.y"))
    menu.handle(N.UP)
    ack(menu)
    menu.on_reply(dict(ok=True, digest="d2", overlay={}, effective={"camera": VALUES}, release={"camera": VALUES}, units={}))
    menu.handle(N.MENU)
    assert [r.label for r in menu.view().rows] == ["Keep editing", "Save", "Discard"]
    menu.handle(N.SELECT)
    assert menu.screen == "group" and menu.session is not None
    menu.handle(N.LEFT)
    sent.clear()
    menu.handle(N.DOWN)
    if not save:
        menu.handle(N.DOWN)
    menu.handle(N.SELECT)
    assert len(sent) == 1
    if save:
        assert sent[0]["action"] == "calibration_save"
        assert sent[0]["entry_values"] == {"laser.dir_cam.x": 0., "laser.offset_m.y": -.41}
    else:
        assert sent[0]["action"] == "set_many"
        assert sent[0]["values"] == {"laser.offset_m.y": -.41}
        assert sent[0]["unset_keys"] == ["laser.dir_cam.x"]
    assert menu.session is not None  # keep until reply
    ack(menu)
    assert menu.session is None


@pytest.mark.parametrize("state", [{}, {"inputs": {"fire_control": False}},
    {"inputs": {"fire_control": False}, "inputs_valid": {"fire_control": False}},
    {"inputs": {"fire_control": True}, "inputs_valid": {"fire_control": True}},
    {"inputs": None, "inputs_valid": None}])
def test_fire_control_invalid_or_enabled_fails_closed(state):
    status = SystemStatus(panel_state=state)
    status.panel.mark(1.)
    assert fire_control_enabled(status, 1.)


def test_fire_control_freshness_and_guard_at_send_time(ui):
    status = SystemStatus(panel_state=dict(inputs=dict(fire_control=False), inputs_valid=dict(fire_control=True)))
    status.panel.mark(1.)
    assert fire_control_enabled(status, 1.) is False
    assert fire_control_enabled(status, 3.) is True
    menu, sent = ui
    menu._edit(Entry("swarm_eval.enabled"))
    menu.fire_control_enabled = lambda: True
    confirm(menu)
    assert not sent and "Refused" in menu.message
    menu._enter_group(4)
    assert menu.session is None
    menu._edit(Entry("controller.rate_limit_rad_s"))
    menu.handle(N.DOWN)
    menu.handle(N.SELECT)
    confirm(menu)
    assert sent[-1]["action"] == "set"


def test_pending_receipt_stale_digest_refresh_and_offline(ui):
    menu, sent = ui
    menu._edit(Entry("controller.rate_limit_rad_s"))
    menu.units = {"unit": dict(stale=False, pending_config_keys=[menu.entry.key])}
    assert "held until idle" in menu.view().lines
    menu.handle(N.DOWN)
    menu.handle(N.SELECT)
    confirm(menu)
    menu.on_reply(dict(ok=False, error="overlay changed; refresh and retry"))
    assert len(sent) == 2 and sent[-1]["action"] == "get"
    assert menu.digest is None
    menu.on_reply(dict(ok=True, digest="new", effective={"unit": VALUES}, release={"unit": VALUES}, overlay={}, units={}))
    assert menu.digest == "new" and "overlay changed" in menu.message
    menu._edit(Entry("controller.rate_limit_rad_s"))
    menu.handle(N.SELECT)
    confirm(menu)
    assert sent[-1]["expected_digest"] == "new"
    menu.on_reply(dict(ok=True, digest="next", units={"unit": dict(state="pending")}))
    assert "pending until idle" in menu.view().lines[0]
    menu.on_reply(dict(ok=False, error="agent offline; outcome unknown"))
    assert menu.digest is None and "offline" in menu.message


def test_pc_gap_is_explicit(ui):
    menu, sent = ui
    menu._edit(Entry("threat_eval.zones.warning.radius_m"))
    menu.handle(N.SELECT)
    confirm(menu)
    ack(menu)
    assert any("PC not updated" in line for line in menu.view().lines)


class Socket:
    def __init__(self):
        self.sent = []
    def setsockopt(self, *args): pass
    def connect(self, *args): pass
    def close(self, *args): pass
    def send_string(self, message, **kw): self.sent.append(json.loads(message))
    def recv(self, *args):
        import zmq
        raise zmq.Again()


def test_command_client_one_in_flight_nonblocking_and_timeout():
    sockets = []
    ctx = SimpleNamespace(socket=lambda _: sockets.append(Socket()) or sockets[-1])
    client = CommandClient(ctx, "fake", timeout_s=1.5)
    assert client.submit_config(dict(type="ConfigCommand", action="get", actor="pi-panel"))
    assert not client.submit_config(dict(action="get"))
    client.poll(0.)
    assert len(sockets[0].sent) == 1
    assert client.poll(.2) == []
    replies = client.poll(2.)
    assert len(replies) == 1 and "offline" in replies[0].config["error"]
    assert len(sockets) == 2


@pytest.fixture
def agent(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setenv("IDCS_LIVE_CONFIG", str(tmp_path / "live.yaml"))
    service = ConfigAgent(Path(__file__).resolve().parents[1], "jetson", run=Mock(return_value=(0, "fake")),
                          active=lambda: (), result_wait_s=0)
    return service


def request(agent, action, **kw):
    return agent.handle(dict(type="ConfigCommand", action=action, actor="test-panel", **kw))


def test_agent_set_many_validates_combination_and_journals_one_transaction(agent):
    reply = request(agent, "set_many", values={"controller.rate_limit_rad_s": .6,
                    "controller.manual_rate_limit_rad_s": .4}, expected_digest=digest({}))
    assert reply["ok"], reply
    records = [json.loads(line) for line in live_path().with_name("live-journal.jsonl").read_text().splitlines()]
    assert len(records) == 3 and len({r["transaction"] for r in records}) == 1
    assert len([r for r in records if r.get("type") == "result"]) == 1
    assert request(agent, "revert")["ok"]
    assert read_overlay() == {}
    assert not request(agent, "set_many", values={"controller.rate_limit_rad_s": .2,
                                                  "laser.dir_cam.x": .01})["ok"]
    assert read_overlay() == {}


@pytest.mark.parametrize("payload", [dict(values={"controller": {}}), dict(values={"gimbal.sync_flag": 1}),
    dict(values={"laser.dir_cam.x": .01}, unset_keys=["laser.dir_cam.x"]), dict(values=[]),
    dict(values={"laser.dir_cam.x": float("nan")}), dict(values={}), dict(unset_keys=["laser.dir_cam.x"]*2)])
def test_agent_rejects_entire_invalid_transaction(agent, payload):
    reply = request(agent, "set_many", **payload)
    assert not reply["ok"], reply
    assert read_overlay() == {} and not live_path().exists()


def test_agent_calibration_save_single_record_and_discard_restores_presence(agent):
    assert request(agent, "set_many", values={"laser.dir_cam.x": .001, "laser.offset_m.x": .01})["ok"]
    before = live_path().with_name("live-journal.jsonl").read_text().splitlines()
    assert request(agent, "calibration_save", entry_values={"laser.dir_cam.x": 0., "laser.offset_m.x": 0.})["ok"]
    after = live_path().with_name("live-journal.jsonl").read_text().splitlines()
    assert len(after) == len(before) + 1 and json.loads(after[-1])["type"] == "calibration_save"
    assert request(agent, "set_many", values={}, unset_keys=["laser.dir_cam.x", "laser.offset_m.x"])["ok"]
    assert read_overlay() == {}
    assert not request(agent, "calibration_save", entry_values={"controller.yaw_kp": 1})["ok"]


def test_agent_set_many_persistence_failure_preserves_old_overlay(agent, monkeypatch):
    import common.live_config as live
    assert request(agent, "set", key="laser.dir_cam.x", value=.001)["ok"]
    before = read_overlay()
    monkeypatch.setattr(live, "atomic_write", Mock(side_effect=OSError("fake disk error")))
    assert not request(agent, "set_many", values={"laser.dir_cam.x": .002, "laser.offset_m.x": .01})["ok"]
    assert read_overlay() == before
    records = [json.loads(line) for line in live_path().with_name("live-journal.jsonl").read_text().splitlines()]
    assert len([r for r in records if r.get("type") == "result"]) == 1


def test_agent_get_includes_release_values_and_parser_defaults(agent):
    result = request(agent, "get")
    assert result["ok"]
    unit = "jetson/idcs-controller-camera.service"
    assert result["effective"][unit]["controller.manual_accel_limit_rad_s2"] == 2.
    assert result["release"][unit]["controller.manual_rate_limit_rad_s"] == .5


def test_config_guard_rechecked_when_queued_request_is_actually_sent():
    from common.operator_commands import OperatorCommand
    socket = Socket()
    client = CommandClient(SimpleNamespace(socket=lambda _: socket), "fake")
    client.submit(OperatorCommand("status"))
    client.poll(0.)
    allowed = [True]
    assert client.submit_config(dict(action="set", key="swarm_eval.enabled", value=False), allowed=lambda: allowed[0])
    allowed[0] = False
    replies = client.poll(2.)  # outstanding status times out; queued mutation must not send
    assert len(socket.sent) == 1
    assert len(replies) == 1 and "before send" in replies[0].config["error"]


def test_canonical_uses_owning_stack_and_caps_take_all_release_stacks(ui):
    menu, sent = ui
    values = dict(VALUES, **{"controller.accel_limit_rad_s2": 4.5})
    stacks = {"jetson/idcs-bridge.service": VALUES, "jetson/idcs-controller-camera.service": values}
    assert menu._canonical(stacks)["controller.accel_limit_rad_s2"] == 4.5
    menu.release_stacks = {"jetson/unit": {"gimbal.yaw_rate_limit_rad_s": .6}}
    assert menu.bounds("controller.rate_limit_rad_s")[2] == .6


def test_calibration_guard_rechecks_nudges_and_discard(ui):
    menu, sent = ui
    menu._enter_group(4)
    menu._edit(Entry("laser.offset_m.x"))
    menu.fire_control_enabled = lambda: None
    menu.handle(N.UP)
    assert not sent
    menu.touched = {"laser.offset_m.x"}
    menu._finish(False)
    assert not sent and menu.session is not None


def test_set_many_stale_digest_and_cross_field_failure_are_atomic(agent):
    assert request(agent, "set", key="laser.offset_m.x", value=.01)["ok"]
    before = read_overlay()
    rejected = request(agent, "set_many", values={"laser.offset_m.x": .02}, expected_digest=digest({}))
    assert not rejected["ok"] and "refresh" in rejected["error"] and read_overlay() == before
    rejected = request(agent, "set_many", values={"controller.rate_limit_rad_s": .3,
                       "controller.idle_return_rate_rad_s": .5, "laser.offset_m.x": .02})
    assert not rejected["ok"] and read_overlay() == before


def test_calibration_summary_rejects_invalid_entry_values(agent):
    assert not request(agent, "calibration_save", entry_values={"laser.dir_cam.x": float("nan")})["ok"]


def test_pending_result_updates_only_from_matching_fresh_receipt(ui):
    menu, sent = ui
    menu._edit(Entry("controller.rate_limit_rad_s"))
    menu.handle(N.SELECT)
    confirm(menu)
    menu.on_reply(dict(ok=True, digest="d1", units={"unit": dict(state="pending")}))
    assert "pending until idle" in menu.view().lines[0]
    menu.on_reply(dict(ok=True, digest="d1", effective={"unit": VALUES}, release={"unit": VALUES}, overlay={},
                       units={"unit": dict(requested_digest="other", stale=False)}))
    assert "pending until idle" in menu.view().lines[0]
    menu.refresh()
    menu.on_reply(dict(ok=True, digest="d1", effective={"unit": VALUES}, release={"unit": VALUES}, overlay={},
                       units={"unit": dict(requested_digest="d1", stale=False, pending_config_keys=[])}))
    assert menu.view().lines == ("unit: applied",)


def test_cli_set_many_uses_existing_config_envelope_and_preflight(agent, monkeypatch):
    from tools.idcs_config import main
    import zmq
    sent = []
    class FakeSocket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def setsockopt(self, *args): pass
        def connect(self, *args): pass
        def send_json(self, request): sent.append(request)
        def recv_json(self): return dict(ok=True)
    class FakeContext:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def socket(self, *args): return FakeSocket()
    monkeypatch.setattr(zmq, "Context", FakeContext)
    assert main(["--host", "jetson", "--actor", "test-cli", "set_many",
                 "{laser.dir_cam.x: 0.001, laser.offset_m.x: 0.02}", "--unset", "laser.dir_cam.y"]) == 0
    assert len(sent) == 1 and sent[0]["type"] == "ConfigCommand" and sent[0]["action"] == "set_many"
    assert sent[0]["unset_keys"] == ["laser.dir_cam.y"]
    assert sent[0]["expected_digest"] == digest({})
    sent.clear()
    assert main(["--host", "jetson", "set_many", "{laser.dir_cam.x: 0.1, laser.dir_cam.x: 0.2}"]) == 1
    assert not sent


def test_calibration_summary_malformed_keys_are_a_reply_not_an_agent_crash(agent):
    assert not request(agent, "calibration_save", entry_values={1: 0.})["ok"]


def test_all_menu_keys_are_exactly_the_requested_scope(ui):
    menu, sent = ui
    assert len(menu.keys()) == 29
    assert not any(k in menu.keys() for k in ("controller.yaw_kp", "controller.feedforward_scale", "controller.predict"))
    assert {entry.key for group in range(5) for entry in menu.entries(group)} == set(menu.keys())


def test_calibration_typed_entry_still_requires_cancel_first_confirmation(ui):
    menu, sent = ui
    menu._enter_group(4)
    menu._edit(Entry("laser.offset_m.x"))
    for c in ".123":
        menu.text(c)
    menu.handle(N.SELECT)
    assert menu.screen == "confirm" and not sent
    confirm(menu)
    assert len(sent) == 1 and sent[0]["value"] == .123
    assert sent[0]["actor"] == "pi-panel:calibration-preview"
