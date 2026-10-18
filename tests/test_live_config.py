"""Local contract tests: no hosts, systemctl, GPU, serial, or motor access."""
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
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
    instances = []
    original = LiveSettings.__init__
    def create(self, *args, **kwargs):
        original(self, *args, **kwargs)
        instances.append(self)
    monkeypatch.setattr(LiveSettings, "__init__", create)
    yield
    for settings in instances:
        settings.close()


def wait_until(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "live-settings worker did not settle"
        time.sleep(0.002)


def settle(settings, *, engaged, force=True):
    """Wait off the owner path, then consume one worker-prepared revision."""
    settings.tick(engaged=engaged, force=force)
    wait_until(lambda: settings._prepared is not None and settings._prepared.epoch == settings._epoch)
    applied = settings.tick(engaged=engaged)
    from common.live_config import receipt_path
    def acknowledged():
        try:
            receipt = json.loads(receipt_path(settings.unit).read_text())
            return all(receipt[k] == v for k, v in json.loads(json.dumps(settings.status())).items())
        except (OSError, ValueError):
            return False
    wait_until(acknowledged)
    return applied


def write_overlay(value):
    atomic_write(live_path(), yaml.safe_dump(value))


def agent(*, host="pc", active=(), stacks=None, publish=lambda _: None, **kwargs):
    runner = Mock(return_value=(0, "fake restart"))
    service = ConfigAgent(ROOT, host, run=runner, active=lambda: active,
                          stacks=stacks, publish=publish, result_wait_s=0, **kwargs)
    return service, runner


def command(service, action, **kw):
    return service.handle(dict(action=action, actor="test-reviewer", **kw))


def controller_settings(callback=lambda _: None, **kwargs):
    paths = resolve_config_paths(ROOT / "configs/base", [ROOT / "configs/controller/sim_mount.yaml"])
    return LiveSettings(paths, unit="local/test-controller", apply=callback,
                        hot_prefixes=("controller", "laser"), **kwargs)


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
    try:
        baseline = subprocess.run(["git", "cat-file", "-e", "245d103^{commit}"], cwd=ROOT,
                                  capture_output=True)
    except FileNotFoundError:
        pytest.skip("baseline 245d103 unavailable: git is absent (release tarball)")
    if baseline.returncode:
        pytest.skip("baseline 245d103 unavailable in this checkout/release tarball")
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
        # This release adds exactly the dedicated config link, a restart key.
        if "net" in after:
            assert after["net"].pop("zmq_config_link") == "tcp://192.168.0.5:5598"
        assert before == after, (unit, changed(before, after))


def test_held_idle_key_does_not_rebuild_every_engaged_tick(monkeypatch):
    import common.live_config as live
    settings = controller_settings(lambda values: None)
    write_overlay({"controller": {"rate_limit_rad_s": 0.7}})
    settle(settings, engaged=True)
    assert settings.pending == ["controller.rate_limit_rad_s"]
    builds = []
    real = live.runtime_bundle
    monkeypatch.setattr(live, "runtime_bundle", lambda *a, **k: builds.append(1) or real(*a, **k))
    for _ in range(50):  # one second of 50 Hz engaged ticks, overlay unchanged
        settings.tick(engaged=True)
    assert builds == []
    assert settle(settings, engaged=False, force=False)
    assert settings.pending == [] and len(builds) == 1


def test_live_applies_next_tick_idle_waits_until_disengaged():
    applied = []
    settings = controller_settings(applied.append)
    original = settings.bundle.digest
    write_overlay({"controller": {"yaw_kp": 7.0, "rate_limit_rad_s": 0.7}})
    assert settle(settings, engaged=True)
    assert applied[-1]["controller"]["yaw_kp"] == 7
    assert applied[-1]["controller"]["rate_limit_rad_s"] == 0.8
    assert settings.pending == ["controller.rate_limit_rad_s"]
    assert settings.bundle.digest != original
    assert settle(settings, engaged=False)
    assert applied[-1]["controller"]["rate_limit_rad_s"] == 0.7
    assert settings.pending == []
    assert settings.status()["config_digest"] == runtime_bundle(settings.paths).digest


def test_invalid_update_and_callback_failure_keep_prior_settings():
    callback = Mock(side_effect=ValueError("fake service refuses swap"))
    settings = controller_settings(callback)
    original = settings.bundle
    write_overlay({"controller": {"yaw_kp": 0.5}})
    assert not settle(settings, engaged=False)
    callback.assert_not_called()
    assert settings.bundle is original
    write_overlay({"controller": {"yaw_kp": 8.0}})
    assert not settle(settings, engaged=False)
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
    active = ["idcs-deepstream-video.service"]
    service, runner = agent(host="jetson", active=active, stacks=selected)
    result = command(service, "set", key="deepstream.return_iframe_interval", value=15)
    assert result["ok"]
    runner.assert_called_once_with(["try-restart", "idcs-deepstream-video.service"])
    assert result["units"]["jetson/idcs-deepstream-camera.service"]["state"] == "inactive"
    runner.reset_mock()
    active.append("idcs-bridge.service")
    result = command(service, "set", key="gimbal.yaw_rate_limit_rad_s", value=0.9)
    assert result["ok"]
    assert result["units"]["jetson/idcs-bridge.service"]["state"] == "cannot_restart"
    runner.assert_not_called()


@pytest.mark.parametrize("unit", [
    "jetson/idcs-deepstream-camera.service", "jetson/idcs-deepstream-video.service",
    "jetson/idcs-recorder.service", "pc/idcs-camera-ui.service", "pc/idcs-ui.service", "pc/idcs-hud-relay.service",
    "pc/idcs-sim-streamer.service", "pc/idcs-sim-swarm-streamer.service", "pc/idcs-sim-ui.service",
    "pc/idcs-sim-controller.service", "pc/idcs-sim-swarm-controller.service",
    "pc/idcs-sim-panel.service", "pc/idcs-sim-recorder.service",
])
def test_explicit_restart_allow_list(unit):
    from common.live_config import RESTART_ALLOWED_UNITS
    assert unit in RESTART_ALLOWED_UNITS
    # Deterministic consumer fixture: this tests authorization, not dependency
    # parsing (the real stack/consumer mapping is covered separately).
    spec = unit_stacks(ROOT, "jetson")["jetson/idcs-deepstream-video.service"]
    host, name = unit.split("/")
    service, runner = agent(host=host, active=[name], stacks={unit: spec})
    result = command(service, "set", key="deepstream.return_iframe_interval", value=15)
    assert result["units"][unit]["state"] == "restarted"
    runner.assert_called_once_with(["try-restart", name])


@pytest.mark.parametrize("unit", [
    "jetson/idcs-future-perception.service", "jetson/idcs-bridge.service", "jetson/idcs-serial.service",
    "jetson/idcs-controller.service", "jetson/idcs-controller-camera.service",
    "jetson/idcs-operator-agent.service", "pc/idcs-hil-streamer.service", "pc/idcs-pc-mode-agent.service",
    "jetson/idcs-sim-controller.service",  # allow-list is host scoped
])
def test_unlisted_units_cannot_restart_even_with_unrestricted_runner(unit):
    spec = unit_stacks(ROOT, "jetson")["jetson/idcs-deepstream-video.service"]
    host, name = unit.split("/")
    service, runner = agent(host=host, active=[name], stacks={unit: spec})
    result = command(service, "set", key="deepstream.return_iframe_interval", value=15)
    assert result["units"][unit]["state"] == "cannot_restart"
    assert "allow-list" in result["units"][unit]["reason"]
    runner.assert_not_called()


@pytest.mark.parametrize("name", ["idcs-deepstream-camera.service", "idcs-deepstream-video.service"])
@pytest.mark.parametrize("motors,engagement", [(True, False), (False, True), (True, True)])
def test_perception_restart_refused_while_motors_live_or_controller_engaged(name, motors, engagement):
    unit = "jetson/" + name
    stack = unit_stacks(ROOT, "jetson")[unit]
    service, runner = agent(host="jetson", active=[name], stacks={unit: stack},
        motors_live=lambda: motors, controller_engaged=lambda: engagement)
    result = command(service, "set", key="deepstream.return_iframe_interval", value=15)
    assert result["units"][unit] == dict(state="cannot_restart",
        keys=["deepstream.return_iframe_interval"], reason="motors live")
    assert read_overlay()["deepstream"]["return_iframe_interval"] == 15
    runner.assert_not_called()


def test_perception_restart_rechecks_motor_gate_at_execution():
    unit = "jetson/idcs-deepstream-video.service"
    gate = Mock(side_effect=[False, True])
    service, runner = agent(host="jetson", active=[unit.split("/")[1]],
        stacks={unit: unit_stacks(ROOT, "jetson")[unit]}, motors_live=gate,
        controller_engaged=lambda: False)
    result = command(service, "set", key="deepstream.return_iframe_interval", value=15)
    assert result["units"][unit]["state"] == "cannot_restart"
    assert result["units"][unit]["reason"] == "motors live"
    assert gate.call_count == 2
    runner.assert_not_called()


@pytest.mark.parametrize("report,expected", [
    ({"monotonic_s": 100, "last_reason": "tracking"}, True),
    ({"monotonic_s": 100, "last_reason": "coasting"}, True),
    ({"monotonic_s": 100, "last_reason": "manual"}, False),
    ({"monotonic_s": 90, "last_reason": "manual"}, True),
    ({}, True),
])
def test_controller_health_engagement_gate_is_fail_closed(monkeypatch, report, expected):
    import common.live_config as live
    service, _ = agent(host="jetson", active=["idcs-controller-camera.service"])
    monkeypatch.setattr(live.time, "monotonic", lambda: 100)
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: json.dumps(report))
    assert service.controller_engaged() is expected


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
    from jetson.deepstream.live_settings import prepare_pipeline_settings, apply_pipeline_settings
    from jetson.deepstream.metadata_adapter import TrackIdStitcher
    stitcher = TrackIdStitcher(max_gap_frames=45)
    stitcher._frame = 123
    stats = SimpleNamespace(stitcher=stitcher, shadow_policy=None)
    selector = Mock()
    values = runtime_bundle(resolve_config_paths(ROOT / "configs/base"), overlay={}).mutable_copy()
    values["deepstream"]["id_stitching"]["max_gap_frames"] = 20
    values["deepstream"]["shadow_tracks"]["min_confidence"] = 0.6
    prepared = prepare_pipeline_settings(selector, values, shadow_factory=lambda *args: args,
                                         stitch_factory=TrackIdStitcher)
    apply_pipeline_settings(stats, selector, prepared)
    assert stats.stitcher is stitcher and stitcher._frame == 123
    assert stitcher._max_gap == 20 and stats.shadow_policy == (0.6, 30)
    selector.prepare_settings.assert_called_once_with(values, digest(values))
    selector.apply_settings.assert_called_once_with(prepared[-1])


def test_controller_gain_swap_keeps_sequence_history_and_caps_wait():
    from tests.test_control_video_controller import _controller, _steps
    from jetson.control.runtime_config import ControlRuntimeConfig
    from jetson.control.live_settings import prepare_controller_settings, apply_controller_settings
    controller = _controller(live=True, scale=0.0)
    observations, clocks = _steps()
    assembler = SimpleNamespace()
    current = [None]
    def apply(prepared):
        current[0] = apply_controller_settings(controller, assembler, current[0], prepared)
    settings = controller_settings(apply, prepare=lambda values: prepare_controller_settings(
        values, (1280, 720), policy=controller.policy))
    current[0] = ControlRuntimeConfig.from_config(settings.bundle.data)
    controller.reconfigure(ControlRuntimeConfig.from_config(settings.bundle.data))
    first = controller.decide(observations[0], clocks[0])
    previous_pid = controller.pid._pid
    write_overlay({"controller": {"yaw_kp": 8.0, "rate_limit_rad_s": 0.7}})
    settle(settings, engaged=True)
    second = controller.decide(observations[1], clocks[1])
    assert second.intent.sequence > first.intent.sequence
    assert controller.pid._pid is previous_pid
    assert controller.pid._last_sequence == observations[1].sequence
    assert previous_pid._config[0].kp == 8
    assert previous_pid._config[0].rate_limit_rad_s == 0.8
    assert second.pid.pid.yaw.proportional_rad_s == pytest.approx(
        observations[1].target.bearing_error_rad[0] * 8)
    settle(settings, engaged=False)
    assert previous_pid._config[0].rate_limit_rad_s == 0.7


def test_tick_has_no_io_or_validation_and_ready_controller_swap_is_under_one_ms(monkeypatch):
    import builtins
    import io
    import os
    import statistics
    import common.live_config as live
    from common.control import ControlConfig, LaserMountConfig
    from jetson.control.runtime_config import ControlRuntimeConfig
    from jetson.control.video_controller import VideoControllerPolicy
    from jetson.control.pid import AxisPIDConfig
    from jetson.control.live_settings import prepare_controller_settings, apply_controller_settings
    from tests.test_control_video_controller import _controller
    core, assembler = _controller(live=True, scale=0), SimpleNamespace()
    current = [None]
    def apply(prepared):
        current[0] = apply_controller_settings(core, assembler, current[0], prepared)
    settings = controller_settings(apply, prepare=lambda values: prepare_controller_settings(
        values, (1280, 720), policy=core.policy))
    current[0] = ControlRuntimeConfig.from_config(settings.bundle.data)
    owner, in_tick = threading.get_ident(), [False]
    def guard(function):
        def checked(*args, **kwargs):
            assert not (threading.get_ident() == owner and in_tick[0]), function
            return function(*args, **kwargs)
        return checked
    for obj, name in ((builtins, "open"), (io, "open"), (os, "open"), (os, "stat"),
                      (os, "write"), (os, "replace"), (os, "fsync"),
                      (live, "read_overlay"), (live, "runtime_bundle"), (live, "validate_config"),
                      (live, "atomic_write"), (live, "append_journal"),
                      (ControlRuntimeConfig, "from_config"), (ControlConfig, "from_raw_config"),
                      (VideoControllerPolicy, "__post_init__"), (AxisPIDConfig, "__post_init__"),
                      (LaserMountConfig, "from_raw_config")):
        monkeypatch.setattr(obj, name, guard(getattr(obj, name)))
    def tick(**kwargs):
        in_tick[0] = True
        start = time.perf_counter_ns()
        try:
            return settings.tick(**kwargs), time.perf_counter_ns() - start
        finally:
            in_tick[0] = False
    samples = []
    for i in range(20):
        write_overlay({"controller": {"yaw_kp": 7.0 + i % 2}})
        assert not tick(engaged=True, force=True)[0]
        wait_until(lambda: settings._prepared is not None and settings._prepared.epoch == settings._epoch)
        applied, elapsed = tick(engaged=True)
        assert applied
        samples.append(elapsed / 1e6)
    # Also cover an unchanged poll interval and a disengagement/held-cap retry.
    write_overlay({"controller": {"yaw_kp": 8.0, "rate_limit_rad_s": 0.7}})
    tick(engaged=True, force=True)
    wait_until(lambda: settings._prepared is not None)
    tick(engaged=True)
    tick(engaged=False)
    wait_until(lambda: settings._prepared is not None and settings._prepared.epoch == settings._epoch)
    assert tick(engaged=False)[0]
    assert max(samples) < 0.5, samples  # comfortably below the 1 ms requirement
    print(f"ready controller tick ms: n={len(samples)}, median={statistics.median(samples):.6f}, max={max(samples):.6f}")


def test_worker_discards_overlay_changed_during_preparation_and_journals_only_swap():
    entered, release = threading.Event(), threading.Event()
    calls = []
    def prepare(values):
        if values["controller"]["yaw_kp"] == 7:
            entered.set()
            assert release.wait(3)
        return values
    settings = controller_settings(calls.append, prepare=prepare)
    write_overlay({"controller": {"yaw_kp": 7.0}})
    settings.tick(engaged=True)
    assert entered.wait(3)
    try:
        write_overlay({"controller": {"yaw_kp": 8.0}})
        assert not settings.tick(engaged=True)
    finally:
        release.set()
    desired = digest(read_overlay())
    wait_until(lambda: settings._prepared is not None and settings._prepared.requested_digest == desired)
    journal = live_path().with_name("live-journal.jsonl")
    assert not journal.exists()
    assert settings.tick(engaged=True)
    settings.close()  # flush confirmed applications even on immediate shutdown
    assert [value["controller"]["yaw_kp"] for value in calls] == [8]
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert len(records) == 1 and records[0]["new"] == 8
    assert records[0]["resulting_digest"] == desired


def test_prepared_idle_caps_are_discarded_when_owner_reengages():
    calls = []
    settings = controller_settings(calls.append)
    write_overlay({"controller": {"rate_limit_rad_s": 0.7}})
    settle(settings, engaged=True)
    settings.tick(engaged=False)
    wait_until(lambda: settings._prepared is not None and settings._prepared.epoch == settings._epoch)
    assert not settings.tick(engaged=True)
    settle(settings, engaged=True)
    assert calls == []
    assert settle(settings, engaged=False)
    assert calls[-1]["controller"]["rate_limit_rad_s"] == 0.7


def test_pure_service_parser_rejection_keeps_running_settings():
    callback = Mock()
    settings = controller_settings(callback, prepare=Mock(side_effect=ValueError("typed parser refused")))
    original = settings.bundle
    write_overlay({"controller": {"yaw_kp": 8.0}})
    assert not settle(settings, engaged=True)
    assert settings.bundle is original and settings.error == "typed parser refused"
    callback.assert_not_called()


def test_slow_poll_prepares_without_owner_ticks_and_invalidates_an_older_ready_object():
    calls = []
    settings = controller_settings(calls.append)
    settle(settings, engaged=True)
    write_overlay({"controller": {"yaw_kp": 7.0}})
    first = digest(read_overlay())
    wait_until(lambda: settings._prepared is not None and settings._prepared.requested_digest == first)
    write_overlay({"controller": {"yaw_kp": 8.0}})
    latest = digest(read_overlay())
    wait_until(lambda: settings._prepared is not None and settings._prepared.requested_digest == latest)
    assert calls == []
    assert settings.tick(engaged=True)
    assert [values["controller"]["yaw_kp"] for values in calls] == [8]


def test_invalid_revision_can_be_reverted_to_last_valid_overlay():
    settings = controller_settings()
    settle(settings, engaged=True)
    write_overlay({"controller": {"yaw_kp": 0}})
    assert not settle(settings, engaged=True)
    assert settings.error
    write_overlay({})
    assert not settle(settings, engaged=True)
    assert settings.error is None


def test_tick_never_waits_for_worker_lock():
    settings = controller_settings()
    with settings._lock:
        start = time.perf_counter()
        assert not settings.tick(engaged=True)
        assert time.perf_counter() - start < 0.001


@pytest.mark.skipif(os.name != "nt", reason="Windows reader sharing contract")
def test_atomic_overlay_write_retries_windows_reader_sharing_violation(monkeypatch):
    replace = os.replace
    attempts = []
    def sharing_violation(source, destination):
        attempts.append(1)
        if len(attempts) < 3:
            error = PermissionError("reader still has the previous overlay open")
            error.winerror = 32
            raise error
        replace(source, destination)
    monkeypatch.setattr(os, "replace", sharing_violation)
    write_overlay({"controller": {"yaw_kp": 8.0}})
    assert len(attempts) == 3
    assert read_overlay()["controller"]["yaw_kp"] == 8


def test_worker_owns_subscription_receive_and_teardown(monkeypatch):
    import zmq
    calls = []
    owner = threading.get_ident()
    def record(name):
        calls.append((name, threading.get_ident()))
    class Socket:
        def setsockopt(self, *args): record("option")
        def setsockopt_string(self, *args): record("subscription")
        def connect(self, endpoint): record("connect")
        def recv(self, *args):
            record("receive")
            raise zmq.Again()
        def close(self, *args): record("close")
    class Context:
        def __init__(self): record("context")
        def socket(self, *args): record("socket"); return Socket()
        def term(self): record("term")
    monkeypatch.setattr(zmq, "Context", Context)
    settings = controller_settings()
    settings.subscribe()
    settle(settings, engaged=True)
    settings.close()
    assert {name for name, _ in calls} >= {"context", "socket", "connect", "receive", "close", "term"}
    assert {thread for _, thread in calls} == {settings._worker.ident}
    assert settings._worker.ident != owner


def test_worker_retries_journal_and_receipt_io_without_rejecting_applied_config(monkeypatch):
    import common.live_config as live
    settings = controller_settings()
    settle(settings, engaged=True)
    journal, receipt = live.append_journal, live.atomic_write
    attempts = {"journal": 0, "receipt": 0}
    def append(record):
        attempts["journal"] += 1
        if attempts["journal"] == 1:
            raise OSError("transient journal failure")
        journal(record)
    def write(path, text):
        attempts["receipt"] += 1
        if attempts["receipt"] == 1:
            raise OSError("transient receipt failure")
        receipt(path, text)
    write_overlay({"controller": {"yaw_kp": 8.0}})
    monkeypatch.setattr(live, "append_journal", append)
    monkeypatch.setattr(live, "atomic_write", write)
    assert settle(settings, engaged=True)
    wait_until(lambda: attempts["journal"] >= 2 and attempts["receipt"] >= 2)
    assert settings.error is None
    records = [json.loads(line) for line in live_path().with_name("live-journal.jsonl").read_text().splitlines()]
    assert len(records) == 1 and records[0]["new"] == 8


def test_history_comparison_skips_clearly_when_baseline_is_unavailable(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    with pytest.raises(pytest.skip.Exception, match="baseline 245d103 unavailable"):
        test_all_21_systemd_stacks_validate_and_keep_base_effective_values()


def test_service_receipt_and_recorder_record_have_applied_digest():
    settings = controller_settings()
    write_overlay({"controller": {"yaw_kp": 7.0}})
    settle(settings, engaged=False)
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
        stacks={settings.unit: stack}, publish=lambda _: settle(settings, engaged=True))
    result = command(service, "set", key="controller.rate_limit_rad_s", value=0.7)
    assert result["units"][settings.unit]["state"] == "pending", json.dumps(result)
    assert result["units"][settings.unit]["config_digest"] != runtime_bundle(stack["paths"]).digest
    settle(settings, engaged=False)
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
    atomic_write(live_path().with_name("config-shared.json"), json.dumps(dict(
        ready=True, shared={}, keys=[], hosts={"pc": dict(state="ready"), "jetson": dict(state="ready")},
        updated_ns=time.time_ns())))
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


@pytest.mark.parametrize("name", ["idcs-future-session.service", "idcs-bridge.service"])
def test_sim_dependency_expansion_cannot_bypass_the_restart_allow_list(monkeypatch, name):
    import common.live_config as live
    monkeypatch.setattr(live, "SIM_SESSION_UNITS", live.SIM_SESSION_UNITS | {name})
    service, runner = agent(active=["idcs-sim-streamer.service", name])
    result = command(service, "set", key="video.active_profile", value="720p30")
    assert result["units"]["pc/" + name]["state"] == "cannot_restart"
    assert result["units"]["pc/" + name]["dependency_restart"] is True
    runner.assert_called_once_with(["try-restart", "idcs-sim-streamer.service"])


def test_local_agent_channel_handles_req_rep_and_publishes_with_fake_sockets():
    from common.live_config import AgentChannel
    class Socket:
        def __init__(self): self.incoming = None; self.sent = []; self.endpoint = None
        def setsockopt(self, *_): pass
        def bind(self, endpoint): self.endpoint = endpoint
        def connect(self, endpoint): self.endpoint = endpoint
        def send(self, *args, **kwargs): pass
        def recv(self, *_):
            import zmq
            raise zmq.Again
        def poll(self, _): return self.incoming is not None
        def recv_multipart(self):
            value, self.incoming = self.incoming, None
            return [b"client", b"", json.dumps(value).encode()]
        def send_multipart(self, frames, **_):
            assert frames[:-1] == [b"client", b""]
            self.sent.append(json.loads(frames[-1]))
        def send_json(self, value, **_): self.sent.append(value)
        def close(self, _): pass
    class Context:
        def socket(self, _): return Socket()
    service, runner = agent()
    channel = AgentChannel(Context(), service)
    channel.rep.incoming = dict(type="ConfigCommand", action="set", actor="fake-client",
                                key="operator.lock_lost_s", value=2.0)
    try:
        channel.poll()
        wait_until(lambda: (channel.poll() or channel.rep.sent))
        assert channel.rep.sent[-1]["ok"]
        assert channel.pub.sent[-1]["type"] == "config_update"
        assert channel.pub.sent[-1]["digest"] == digest(read_overlay())
    finally:
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
    settle(settings, engaged=True)
    assert applied[-1]["controller"]["yaw_kp"] == 8
    assert settings.bundle.sources == original.sources  # in-memory candidate has its own digest


def test_invalid_revision_arriving_during_startup_does_not_abort_valid_snapshot():
    paths = resolve_config_paths(ROOT / "configs/base")
    original = runtime_bundle(paths)
    write_overlay({"controller": {"yaw_kp": 0}})
    settings = LiveSettings(paths, unit="local/startup-race", apply=Mock(),
                            hot_prefixes=("controller",), initial_bundle=original)
    assert not settle(settings, engaged=False)
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


def test_only_metadata_probe_thread_ticks_settings(monkeypatch):
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
    assert calls == [("tick", threading.get_ident())]
