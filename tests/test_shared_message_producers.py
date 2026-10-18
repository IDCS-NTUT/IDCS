"""Execute the actual producer constructor expressions without importing hardware."""
import ast
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Optional

import pytest
from pydantic import ValidationError
from common.schemas import CamState, ControlCmd, ManualControlState, control_cmd_from_json, manual_control_state_from_json

ROOT = Path(__file__).resolve().parents[1]
MODELS = {m.__name__: m for m in (CamState, ControlCmd, ManualControlState)}


def producer_calls():
    for directory in ("common", "jetson", "pc", "rpi", "tools"):
        for path in (ROOT / directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id in MODELS and all(k.arg is not None for k in node.keywords)):
                    yield path, node


CALLS = list(producer_calls())


@pytest.mark.parametrize("path,node", CALLS, ids=[f"{p.relative_to(ROOT)}:{n.lineno}" for p, n in CALLS])
def test_each_explicit_producer_constructs_a_strict_message(path, node):
    env = dict(MODELS, time=time, _ns=lambda seconds: int(seconds * 1e9),
        frame_id=1, src_ts_ms=1, pan=.1, tilt=.1, pan_rate=.1, tilt_rate=.1,
        home_pan=0., home_tilt=0., encoder_pan_counts=1, encoder_tilt_counts=1,
        hardware_available=True, control_activity=None, frame=1, began=1.,
        sequence=1, sample_ns=1, rate=.1, seq=1, reason="test", note="test",
        sample=SimpleNamespace(timestamp=1., pan_timestamp=1., tilt_timestamp=1.,
                               pan_rad=.1, tilt_rad=.1, pan_rate_rad_s=.1, tilt_rate_rad_s=.1),
        effective_manual_active=False, effective_manual_changed=False, effective_control_on=False,
        effective_control_changed=False, switch_state={}, joy_x=128, joy_y=128, yaw_rate=0., pitch_rate=0.,
        switch_io=SimpleNamespace(input_validity={"fire_control": True}, role_states={}))
    message = eval(compile(ast.Expression(node), str(path), "eval"), env)
    assert message.model_config["extra"] == "forbid"
    payload = message.model_dump(mode="json")
    assert type(message).model_validate(payload) == message
    with pytest.raises(ValidationError, match="extra_forbidden"):
        type(message).model_validate(dict(payload, accidental_extra=True))
    if node.func.id == "ManualControlState" and path.name == "runtime_control.py":
        assert message.control_cmd_valid is True


def test_streamer_dictionary_producer_is_a_valid_camstate():
    path = ROOT / "pc/streamer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "build_cam_state")
    scope = dict(Optional=Optional, time=time)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    activity = SimpleNamespace(model_dump=lambda **_: dict(backend="sim", session="fixture", sequence=1,
                                created_monotonic_ns=1, reason="tracking"))
    producer = SimpleNamespace(_last_pose={"pan": .1, "tilt": .2},
        gen=SimpleNamespace(get_home_pose=lambda: {}, planner_eval_enabled=lambda: False),
        _pan_rate=0., _tilt_rate=0., _encoder_pose_enabled=False, last_frame_source_ns=1,
        _sim_control_enabled=True, _control_activity=SimpleNamespace(sample=lambda **_: activity))
    state = CamState.model_validate(scope["build_cam_state"](producer, 1, 1))
    assert state.control_activity.backend == "sim"
    producer.gen.planner_eval_enabled = lambda: True
    producer.gen.get_planner_eval_stats = lambda: dict(epoch=0, active=1, spawned=1, eliminated=0, breached=0,
                                                     breach_radius_m=10)
    assert CamState.model_validate(scope["build_cam_state"](producer, 1, 1)).sim_scene == "swarm"


def test_dynamic_decoders_reject_unknown_fields():
    manual = dict(src_ts_ms=1, source="fixture", active=False, emergency=False,
                  joystick_raw=[0, 0], joystick_rate_cmd=[0., 0.], control_cmd_valid=True)
    assert manual_control_state_from_json(manual).control_cmd_valid
    control = dict(frame_id=1, src_ts_ms=1, cmd_ts_ms=1, target_ok=False, target_uv=[0., 0.],
                   err_uv=[0., 0.], err_rad=[0., 0.], pan_rate_cmd=0., tilt_rate_cmd=0.)
    assert control_cmd_from_json(control)
    for decoder, payload in ((manual_control_state_from_json, manual), (control_cmd_from_json, control)):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            decoder(dict(payload, accidental_extra=True))


def test_all_common_pydantic_message_models_forbid_extra():
    import importlib
    from pydantic import BaseModel
    for name in ("schemas", "perception", "control_activity", "target_authority", "sim_score", "synthetic_perception"):
        module = importlib.import_module("common." + name)
        for model in vars(module).values():
            if isinstance(model, type) and issubclass(model, BaseModel) and model is not BaseModel:
                assert model.model_config.get("extra") == "forbid", model.__name__
