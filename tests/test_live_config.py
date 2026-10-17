"""Local contract tests: no hosts, systemctl, GPU, serial, or motor access."""
import json
from pathlib import Path
import queue
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from common.config import ConfigError, load_config_bundle, resolve_config_paths
from common.config_constants import FIXED_VALUES, with_constants
from common.config_schema import FIELDS, tier, validate_fields
from common.live_config import (ConfigAgent, LiveSettings, atomic_write, changed, digest, flatten,
                                live_path, read_overlay, runtime_bundle, unit_stacks)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("IDCS_LIVE_CONFIG", str(tmp_path / "host" / "live.yaml"))


def write_overlay(value):
    atomic_write(live_path(), yaml.safe_dump(value))


def agent(*, host="pc", active=(), stacks=None, publish=lambda _: None):
    runner = Mock(return_value=(0, "fake restart"))
    service = ConfigAgent(ROOT, host, run=runner, active=lambda: active,
                          stacks=stacks, publish=publish, result_wait_s=0)
    return service, runner


def command(service, action, **kw):
    return service.handle(dict(action=action, actor="test-reviewer", **kw))


def controller_settings(callback=lambda _: None):
    paths = resolve_config_paths(ROOT / "configs/base", [ROOT / "configs/controller/sim_mount.yaml"])
    return LiveSettings(paths, unit="local/test-controller", apply=callback, hot_prefixes=("controller", "laser"))


def test_explicit_schema_covers_all_committed_yaml_and_native_profile():
    def check(value, path):
        assert path in FIELDS, path
        if isinstance(value, dict):
            for key, item in value.items():
                check(item, f"{path}.{key}")
        if isinstance(value, list):
            for item in value:
                check(item, path + "[]")
    for path in (ROOT / "configs").rglob("*"):
        if path.suffix in (".yaml", ".yml"):
            raw = yaml.safe_load(path.read_text(encoding="utf-8").replace("%YAML:1.0", ""))
            for key, value in raw.items():
                check(value, str(key))


@pytest.mark.parametrize("config,path", [
    ({"controller": {"yaw_kpp": 7}}, "controller.yaw_kpp"),
    ({"camera": {"unknown": 3}}, "camera.unknown"),
    ({"mystery": {}}, "mystery"),
    ({"serial_io": {"schedule": [{"unknown": 1}]}}, "serial_io.schedule[].unknown"),
])
def test_unknown_paths_fail_at_startup_and_update(config, path, tmp_path):
    with pytest.raises(ConfigError, match=path.replace("[]", r"\[\]")):
        validate_fields(config)
    file = tmp_path / "bad.yaml"
    file.write_text(yaml.safe_dump(config))
    with pytest.raises(ConfigError, match="unknown configuration key"):
        runtime_bundle([file], overlay={})


@pytest.mark.parametrize("key", FIXED_VALUES)
def test_fixed_keys_cannot_be_reintroduced(key):
    from common.live_config import put
    overlay = {}
    put(overlay, key, FIXED_VALUES[key])
    with pytest.raises(ConfigError, match="fixed"):
        validate_fields(overlay)


def test_all_21_systemd_stacks_validate_and_keep_base_effective_values():
    stacks = unit_stacks(ROOT)
    assert len(stacks) == 21
    # The task baseline is a local commit, never fetched from a host/service.
    for unit, spec in stacks.items():
        before = {}
        from common.config import merge_config_layers
        for path in spec["paths"]:
            name = path.relative_to(ROOT).as_posix()
            text = subprocess.check_output(["git", "show", "245d103:" + name], cwd=ROOT).decode("utf-8")
            before = merge_config_layers(before, yaml.safe_load(text) or {})
        after = runtime_bundle(spec["paths"], overlay={}).mutable_copy()
        assert before == after, (unit, changed(before, after))


def test_held_idle_key_does_not_rebuild_every_engaged_tick(monkeypatch):
    import common.live_config as live
    settings = controller_settings(lambda values: None)
    write_overlay({"controller": {"rate_limit_rad_s": 0.7}})
    settings.tick(engaged=True, force=True)
    assert settings.pending == ["controller.rate_limit_rad_s"]
    builds = []
    real = live.runtime_bundle
    monkeypatch.setattr(live, "runtime_bundle", lambda *a, **k: builds.append(1) or real(*a, **k))
    for _ in range(50):  # one second of 50 Hz engaged ticks, overlay unchanged
        settings.tick(engaged=True)
    assert builds == []
    assert settings.tick(engaged=False)
    assert settings.pending == [] and len(builds) == 1


def test_live_applies_next_tick_idle_waits_until_disengaged():
    applied = []
    settings = controller_settings(applied.append)
    original = settings.bundle.digest
    write_overlay({"controller": {"yaw_kp": 7.0, "rate_limit_rad_s": 0.7}})
    assert settings.tick(engaged=True, force=True)
    assert applied[-1]["controller"]["yaw_kp"] == 7
    assert applied[-1]["controller"]["rate_limit_rad_s"] == 0.8
    assert settings.pending == ["controller.rate_limit_rad_s"]
    assert settings.bundle.digest != original
    assert settings.tick(engaged=False, force=True)
    assert applied[-1]["controller"]["rate_limit_rad_s"] == 0.7
    assert settings.pending == []
    assert settings.status()["config_digest"] == runtime_bundle(settings.paths).digest


def test_invalid_update_and_callback_failure_keep_prior_settings():
    callback = Mock(side_effect=ValueError("fake service refuses swap"))
    settings = controller_settings(callback)
    original = settings.bundle
    write_overlay({"controller": {"yaw_kp": 0.5}})
    assert not settings.tick(engaged=False, force=True)
    callback.assert_not_called()
    assert settings.bundle is original
    write_overlay({"controller": {"yaw_kp": 8.0}})
    assert not settings.tick(engaged=False, force=True)
    assert settings.bundle is original
    assert settings.error == "fake service refuses swap"


@pytest.mark.parametrize("value", [True, "8", float("nan"), 0.1, 21.0])
def test_bad_gain_types_or_ranges_are_rejected(value):
    service, runner = agent()
    result = command(service, "set", key="controller.yaw_kp", value=value)
    assert not result["ok"]
    assert read_overlay() == {}
    runner.assert_not_called()


def test_caps_cannot_exceed_real_bridge_ceiling_or_fit_outside_travel():
    service, runner = agent(host="jetson")
    result = command(service, "set", key="controller.rate_limit_rad_s", value=0.9)
    assert not result["ok"] and "gimbal" in result["error"]
    result = command(service, "set", key="controller.max_travel_rad", value=1.0)
    assert not result["ok"] and "gimbal" in result["error"]
    runner.assert_not_called()


def test_restart_only_affected_active_unit_and_never_inactive_or_physical_owner():
    all_stacks = unit_stacks(ROOT, "jetson")
    selected = {u: s for u, s in all_stacks.items() if "deepstream" in u or "bridge" in u}
    service, runner = agent(host="jetson", active=["idcs-deepstream-video.service", "idcs-bridge.service"], stacks=selected)
    result = command(service, "set", key="deepstream.return_iframe_interval", value=15)
    assert result["ok"]
    runner.assert_called_once_with(["try-restart", "idcs-deepstream-video.service"])
    assert result["units"]["jetson/idcs-deepstream-camera.service"]["state"] == "inactive"
    runner.reset_mock()
    result = command(service, "set", key="gimbal.yaw_rate_limit_rad_s", value=0.9)
    assert result["ok"]
    assert result["units"]["jetson/idcs-bridge.service"]["state"] == "cannot_restart"
    runner.assert_not_called()


def test_revert_clear_unset_journal_digest_and_restart_persistence():
    updates = []
    service, _ = agent(publish=updates.append)
    first = command(service, "set", key="controller.yaw_kp", value=7.0)
    assert first["ok"]
    assert updates[-1]["type"] == "config_update"
    assert updates[-1]["changed_keys"] == ["controller.yaw_kp"]
    assert updates[-1]["digest"] == digest(read_overlay())
    # A new process starts with the durable overlay.
    assert controller_settings().bundle.data["controller"]["yaw_kp"] == 7
    assert command(service, "set", key="controller.yaw_kp", value=8.0)["ok"]
    assert command(service, "revert", n=1)["ok"]
    assert read_overlay()["controller"]["yaw_kp"] == 7
    assert command(service, "unset", key="controller.yaw_kp")["ok"]
    assert read_overlay() == {}
    assert command(service, "set", key="operator.lock_lost_s", value=2.0)["ok"]
    assert command(service, "clear")["ok"]
    assert read_overlay() == {}
    records = [json.loads(line) for line in live_path().with_name("live-journal.jsonl").read_text().splitlines()]
    entries = [r for r in records if "key" in r]
    assert entries[0]["old_present"] is False
    assert entries[0]["new"] == 7
    assert entries[0]["tier"] == "live"
    assert entries[0]["actor"] == "test-reviewer"
    assert entries[0]["time_ns"] > 0
    assert all(len(r["resulting_digest"]) == 64 for r in records)


def test_compare_and_swap_and_writer_lock_fail_closed():
    service, runner = agent()
    result = command(service, "set", key="operator.lock_lost_s", value=2, expected_digest="outdated")
    assert not result["ok"]
    live_path().parent.mkdir(parents=True, exist_ok=True)
    live_path().with_suffix(".lock").touch()
    assert not command(service, "clear")["ok"]
    runner.assert_not_called()


def test_deployment_refuses_nonempty_overlay_before_stage_or_systemctl(tmp_path):
    from tools.deployment import Deployment
    write_overlay({"controller": {"yaw_kp": 7.0}})
    manager = Deployment(tmp_path, "pc", "candidate", templates=tmp_path)
    manager.safety = Mock(side_effect=AssertionError("must not query systemctl"))
    manager.run = Mock(side_effect=AssertionError("must not stage"))
    with pytest.raises(RuntimeError, match="live-overlay"):
        manager.stage(tmp_path / "no-bundle", "a" * 40)
    manager.safety.assert_not_called()
    manager.run.assert_not_called()


def test_fold_prints_reviewable_patch_and_writes_no_repo_files(tmp_path):
    from tools.idcs_config import fold_patch
    root = tmp_path / "release"
    base = root / "configs/base/controller.yaml"
    base.parent.mkdir(parents=True)
    original = "controller:\n  yaw_kp: 5.9\n"
    base.write_text(original)
    patch = fold_patch(root, {"controller": {"yaw_kp": 7.0}})
    assert "a/configs/base/controller.yaml" in patch
    assert "-  yaw_kp: 5.9" in patch and "+  yaw_kp: 7.0" in patch
    assert base.read_text() == original


def test_deepstream_probe_update_preserves_stitch_history_without_gpu():
    from jetson.deepstream.live_settings import apply_pipeline_settings
    from jetson.deepstream.metadata_adapter import TrackIdStitcher
    stitcher = TrackIdStitcher(max_gap_frames=45)
    stitcher._frame = 123
    stats = SimpleNamespace(stitcher=stitcher, shadow_policy=None)
    selector = Mock()
    values = runtime_bundle(resolve_config_paths(ROOT / "configs/base"), overlay={}).mutable_copy()
    values["deepstream"]["id_stitching"]["max_gap_frames"] = 20
    values["deepstream"]["shadow_tracks"]["min_confidence"] = 0.6
    apply_pipeline_settings(stats, selector, values, shadow_factory=lambda *args: args,
                            stitch_factory=TrackIdStitcher)
    assert stats.stitcher is stitcher and stitcher._frame == 123
    assert stitcher._max_gap == 20 and stats.shadow_policy == (0.6, 30)
    selector.reconfigure.assert_called_once_with(values, digest(values))


def test_controller_gain_swap_keeps_sequence_history_and_caps_wait():
    from tests.test_control_video_controller import _controller, _steps
    from jetson.control.runtime_config import ControlRuntimeConfig
    controller = _controller(live=True, scale=0.0)
    observations, clocks = _steps()
    settings = controller_settings(lambda values: controller.reconfigure(ControlRuntimeConfig.from_config(values)))
    controller.reconfigure(ControlRuntimeConfig.from_config(settings.bundle.data))
    first = controller.decide(observations[0], clocks[0])
    previous_pid = controller.pid._pid
    write_overlay({"controller": {"yaw_kp": 8.0, "rate_limit_rad_s": 0.7}})
    settings.tick(engaged=True, force=True)
    second = controller.decide(observations[1], clocks[1])
    assert second.intent.sequence > first.intent.sequence
    assert controller.pid._pid is previous_pid
    assert controller.pid._last_sequence == observations[1].sequence
    assert previous_pid._config[0].kp == 8
    assert previous_pid._config[0].rate_limit_rad_s == 0.8
    assert second.pid.pid.yaw.proportional_rad_s == pytest.approx(
        observations[1].target.bearing_error_rad[0] * 8)
    settings.tick(engaged=False, force=True)
    assert previous_pid._config[0].rate_limit_rad_s == 0.7


def test_service_receipt_and_recorder_record_have_applied_digest():
    settings = controller_settings()
    write_overlay({"controller": {"yaw_kp": 7.0}})
    settings.tick(engaged=False, force=True)
    from common.live_config import receipt_path
    receipt = json.loads(receipt_path(settings.unit).read_text())
    assert receipt["config_digest"] == settings.bundle.digest
    assert receipt["pending_config_keys"] == []
    from jetson.control.video_runtime import _publish_record
    from tools.flight_recorder import decode
    publisher = Mock()
    _publish_record(publisher, {"type": "tick", "sequence": 123}, config_status=settings.status())
    _, recorded = decode(publisher.send_string.call_args.args[0].encode())
    assert recorded["config_digest"] == settings.bundle.digest
    assert recorded["sequence"] == 123
    from common.perception import PerceptionSnapshotV2
    assert "config_digest" in PerceptionSnapshotV2.model_fields


@pytest.mark.parametrize("valid,manual,automatic,age,active,expected", [
    (True, True, False, 10, False, False),
    (False, True, False, 10, False, True),
    (True, True, False, 751, False, True),
    (True, True, False, None, False, True),
    (True, False, False, 10, False, True),
    (True, False, True, 10, False, True),
    (True, True, False, 10, True, True),
])
def test_idle_proof_fails_closed_for_panel_authority(valid, manual, automatic, age, active, expected):
    from jetson.control.live_settings import configuration_engaged
    safety = SimpleNamespace(valid=valid, manual_active=manual, auto_allowed=automatic, sample_age_ms=age)
    assert configuration_engaged(safety, active) is expected


def test_agent_reports_pending_then_service_journals_later_application():
    stack = unit_stacks(ROOT, "pc")["pc/idcs-sim-controller.service"]
    settings = LiveSettings(stack["paths"], unit="pc/idcs-sim-controller.service",
                            apply=lambda _: None, hot_prefixes=("controller",))
    service, runner = agent(active=["idcs-sim-controller.service"],
        stacks={settings.unit: stack}, publish=lambda _: settings.tick(engaged=True, force=True))
    result = command(service, "set", key="controller.rate_limit_rad_s", value=0.7)
    assert result["units"][settings.unit]["state"] == "pending"
    assert result["units"][settings.unit]["config_digest"] != runtime_bundle(stack["paths"]).digest
    settings.tick(engaged=False, force=True)
    records = [json.loads(line) for line in live_path().with_name("live-journal.jsonl").read_text().splitlines()]
    assert records[-1]["type"] == "application"
    assert records[-1]["units"] == {settings.unit: "applied"}
    assert records[-1]["key"] == "controller.rate_limit_rad_s"
    runner.assert_not_called()


def test_overlay_gimbal_change_cannot_raise_actual_release_bridge_ceiling():
    service, runner = agent(host="jetson")
    assert command(service, "set", key="gimbal.yaw_rate_limit_rad_s", value=0.9)["ok"]
    assert command(service, "set", key="gimbal.pitch_rate_limit_rad_s", value=0.9)["ok"]
    result = command(service, "set", key="controller.rate_limit_rad_s", value=0.9)
    assert not result["ok"] and "release bridge" in result["error"]
    runner.assert_not_called()


@pytest.mark.parametrize("decision", ["folded", "discard"])
def test_deployment_overlay_decision_is_recorded_with_fakes(tmp_path, decision):
    from tools.deployment import Deployment
    write_overlay({"controller": {"yaw_kp": 7.0}})
    manager = Deployment(tmp_path / "project", "pc", "candidate", templates=tmp_path)
    manager.source.mkdir(parents=True)
    manager.safety = Mock()
    manager.run = Mock(side_effect=["a" * 40, ""])
    manager.plan = lambda: manager.load("stage.json")
    bundle = tmp_path / "fake.bundle"
    bundle.write_bytes(b"fake bundle, never executed")
    result = manager.stage(bundle, "a" * 40, decision)
    assert result["live_overlay"]["decision"] == decision
    assert result["live_overlay"]["overlay"] == read_overlay()
    assert len(result["live_overlay"]["sha256"]) == 64


def test_source_and_derived_model_metadata_refuse_drift(tmp_path):
    from common.swarm_metadata import load_normalization, bind_derived_model
    source = ROOT / "assets/models/swarm/swarm_policy.onnx"
    expected = load_normalization(source)
    assert expected == FIXED_VALUES["swarm_eval.learned_model.normalization"]
    engine = tmp_path / "fake.engine"
    engine.write_bytes(b"fake engine, never loaded")
    bind_derived_model(engine, source)
    assert load_normalization(engine) == expected
    engine.write_bytes(b"different fake engine")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_normalization(engine)


def test_derived_model_without_metadata_is_not_guessed(tmp_path):
    from common.swarm_metadata import load_normalization
    engine = tmp_path / "fake.engine"
    engine.write_bytes(b"never loaded")
    with pytest.raises(FileNotFoundError):
        load_normalization(engine)


def test_sim_viewer_dependency_restart_includes_only_active_software_units():
    service, runner = agent(active=["idcs-sim-streamer.service", "idcs-sim-controller.service",
        "idcs-sim-panel.service", "idcs-sim-ui.service"])
    result = command(service, "set", key="video.active_profile", value="720p30")
    assert result["ok"]
    assert runner.call_count == 1
    call = runner.call_args.args[0]
    assert call == ["try-restart", "idcs-sim-controller.service", "idcs-sim-panel.service",
                    "idcs-sim-streamer.service", "idcs-sim-ui.service"]


def test_local_agent_channel_handles_req_rep_and_publishes_with_fake_sockets():
    from common.live_config import AgentChannel
    class Socket:
        def __init__(self): self.incoming = None; self.sent = []; self.endpoint = None
        def setsockopt(self, *_): pass
        def bind(self, endpoint): self.endpoint = endpoint
        def poll(self, _): return self.incoming is not None
        def recv_json(self): value, self.incoming = self.incoming, None; return value
        def send_json(self, value): self.sent.append(value)
        def close(self, _): pass
    class Context:
        def socket(self, _): return Socket()
    service, runner = agent()
    channel = AgentChannel(Context(), service)
    channel.rep.incoming = dict(type="ConfigCommand", action="set", actor="fake-client",
                                key="operator.lock_lost_s", value=2.0)
    channel.poll()
    assert channel.rep.sent[-1]["ok"]
    assert channel.pub.sent[-1]["type"] == "config_update"
    assert channel.pub.sent[-1]["digest"] == digest(read_overlay())
    channel.close()
    runner.assert_not_called()


def test_startup_race_preserves_actual_snapshot_until_safe_point():
    paths = resolve_config_paths(ROOT / "configs/base")
    original = runtime_bundle(paths)
    write_overlay({"controller": {"yaw_kp": 8.0}})
    applied = []
    settings = LiveSettings(paths, unit="local/startup-race", apply=applied.append,
                            hot_prefixes=("controller",), initial_bundle=original)
    assert settings.bundle is original and applied == []
    settings.tick(engaged=True, force=True)
    assert applied[-1]["controller"]["yaw_kp"] == 8
    assert settings.bundle.sources == original.sources  # in-memory candidate has its own digest


def test_invalid_revision_arriving_during_startup_does_not_abort_valid_snapshot():
    paths = resolve_config_paths(ROOT / "configs/base")
    original = runtime_bundle(paths)
    write_overlay({"controller": {"yaw_kp": 0}})
    settings = LiveSettings(paths, unit="local/startup-race", apply=Mock(),
                            hot_prefixes=("controller",), initial_bundle=original)
    assert not settings.tick(engaged=False, force=True)
    assert settings.bundle is original and settings.error


def test_overlay_cannot_convert_software_session_into_hil_or_remote_intent_output():
    service, runner = agent()
    result = command(service, "set", key="sim.use_jetson_cam_state", value=True)
    assert not result["ok"] and "bench-reviewed" in result["error"]
    result = command(service, "set", key="controller.endpoints.intent_bind", value="tcp://192.168.0.5:5557")
    assert not result["ok"] and "loopback" in result["error"]
    assert read_overlay() == {}
    runner.assert_not_called()


def test_disabling_swarm_selection_does_not_start_a_model_or_kill_the_worker():
    from jetson.deepstream.target_selection import DeepStreamTargetSelector
    from tests.test_deepstream_target_selection import _synthetic_snapshot
    values = runtime_bundle(resolve_config_paths(ROOT / "configs/base"), overlay={}).mutable_copy()
    values["swarm_eval"]["enabled"] = False
    factory = Mock(side_effect=AssertionError("must not construct a model"))
    selector = DeepStreamTargetSelector(values, planner_factory=factory)
    assert selector.select_snapshot(_synthetic_snapshot(1)).selection is None
    factory.assert_not_called()


def test_only_metadata_probe_thread_subscribes_and_applies_settings(monkeypatch):
    import sys
    import threading
    from jetson.deepstream import pipeline
    frame = SimpleNamespace(source_frame_width=1280, source_frame_height=720,
                            frame_num=1, buf_pts=1, obj_meta_list=None)
    batch = SimpleNamespace(frame_meta_list=SimpleNamespace(data=frame, next=None))
    cast = SimpleNamespace(cast=lambda value: value)
    monkeypatch.setitem(sys.modules, "pyds", SimpleNamespace(
        gst_buffer_get_nvds_batch_meta=lambda _: batch, NvDsFrameMeta=cast, NvDsObjectMeta=cast))
    calls = []
    settings = SimpleNamespace(_sub=None,
        subscribe=lambda: calls.append(("subscribe", threading.get_ident())),
        tick=lambda **_: calls.append(("tick", threading.get_ident())))
    stats = pipeline.VerificationStats(live_settings=settings)
    gst = SimpleNamespace(PadProbeReturn=SimpleNamespace(OK=1))
    info = SimpleNamespace(get_buffer=lambda: object())
    detector = threading.Thread(target=pipeline._detector_probe, args=(None, info, (gst, stats)))
    detector.start()
    detector.join(timeout=2)
    assert not detector.is_alive()
    assert calls == []
    pipeline._metadata_probe(None, info, (gst, stats, pipeline.StageClock(), None, False, None, {}))
    assert calls == [("subscribe", threading.get_ident()), ("tick", threading.get_ident())]
