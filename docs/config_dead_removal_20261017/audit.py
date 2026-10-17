"""Reproduce local config equivalence evidence; never opens runtime interfaces.

Run from the repository root:
    python docs/config_dead_removal_20261017/audit.py
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import re
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.config import load_config_bundle, resolve_active_video_profile, resolve_config_paths
from common.config_sync import merge_config_maps, parse_config_text
from common.control import ControlConfig
from jetson.control.runtime_config import ControlRuntimeConfig

BASE = '0bdff18'
OUT = Path(__file__).resolve().parent
DEFAULT_CONFIG_MODULES = {
    'jetson.deepstream.runtime', 'jetson.operator_agent', 'pc.hud_relay',
    'rpi.runtime_control', 'rpi.operator_display',
}


@lru_cache(maxsize=None)
def original(path: str) -> str:
    return subprocess.check_output(['git', 'show', f'{BASE}:{path}'], cwd=ROOT).decode('utf-8')


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True))


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def differences(before, after, prefix='') -> list[dict]:
    result = []
    for key in sorted(before.keys() | after.keys()):
        path = f'{prefix}.{key}' if prefix else key
        if key not in after:
            result.append({'path': path, 'kind': 'removed', 'before': before[key]})
        elif key not in before:
            result.append({'path': path, 'kind': 'added', 'after': after[key]})
        elif isinstance(before[key], dict) and isinstance(after[key], dict):
            result.extend(differences(before[key], after[key], path))
        elif before[key] != after[key]:
            result.append({'path': path, 'kind': 'changed', 'before': before[key], 'after': after[key]})
    return result


def main() -> None:
    tracked = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE], cwd=ROOT).decode().splitlines()
    original_paths = set(tracked)
    before_units, after_units, diffs, runtime = {}, {}, {}, {}
    legacy = types.ModuleType('idcs_config_audit_baseline_control')
    sys.modules[legacy.__name__] = legacy
    exec(compile(original('common/control.py'), 'baseline_common_control.py', 'exec'), legacy.__dict__)
    no_config = []
    geometry_keys = ('fx_px', 'fy_px', 'cx_px', 'cy_px', 'aim_mode', 'frame_size', 'fov_deg',
                     'laser', 'yaw_sign', 'pitch_sign', 'swarm_eval', 'threat_eval')
    for unit in sorted((ROOT / 'deploy/systemd').rglob('*.service')):
        lines = original(unit.relative_to(ROOT).as_posix()).splitlines()
        commands = [line for line in lines if line.startswith(('ExecStart=', 'ExecStartPre='))]
        main_command = next((line for line in commands if line.startswith('ExecStart=')), '')
        module_match = re.search(r' -m ([\w.]+)', main_command)
        if not module_match:
            continue
        module = module_match[1]
        explicit = re.search(r'--config(?:=|\s+)(\S+)', main_command)
        extra = re.search(r'--config-extra(?:=|\s+)(\S+)', main_command)
        if explicit is None and module not in DEFAULT_CONFIG_MODULES:
            no_config.append({'unit': unit.relative_to(ROOT).as_posix(), 'module': module})
            continue
        stack = [explicit[1] if explicit else 'configs/base', extra[1] if extra else '']
        for command in commands:
            pre_config = re.search(r'--config(?:=|\s+)(\S+)', command)
            pre_extra = re.search(r'--config-extra(?:=|\s+)(\S+)', command)
            if pre_config:
                assert [pre_config[1], pre_extra[1] if pre_extra else ''] == stack
        old_paths = []
        for path in [stack[0], *filter(None, stack[1].split(','))]:
            if path in original_paths:
                old_paths.append(path)
            else:
                old_paths.extend(sorted(p for p in tracked if p.startswith(path + '/') and p.endswith('.yaml') and '/' not in p[len(path)+1:]))
        before = merge_config_maps(*(parse_config_text(original(p), p) for p in old_paths))
        after = load_config_bundle(resolve_config_paths(*stack)).mutable_copy()
        name = unit.relative_to(ROOT).as_posix()
        context = {'module': module, 'stack': stack, 'implicit_default': explicit is None}
        before_units[name] = dict(context, sha256=digest(before), config=before)
        after_units[name] = dict(context, sha256=digest(after), config=after)
        changes = differences(before, after)
        assert all(change['kind'] == 'removed' for change in changes), (name, changes)
        allowed_removed = {
            'camera.libcamera', 'control.controller', 'control.deadband_px',
            'control.debug_overlay', 'control.fov_deg', 'control.fx_fy_from_fov',
            'control.loop_hz', 'control.lost_target_timeout_ms', 'control.mode',
            'control.motion_vel_alpha', 'control.mpc', 'control.negotiation',
            'control.pid', 'control.reinit_on_lost', 'control.smooth_px_alpha',
            'control.target_selector', 'controller.accel_limit_rad_s2',
            'gimbal.backend', 'gimbal.baudrate', 'gimbal.pitch_rate_limit_rad_s',
            'gimbal.retries', 'gimbal.serial_port', 'gimbal.timeout',
            'gimbal.use_group_writes', 'gimbal.yaw_rate_limit_rad_s',
            'rpi.return_video', 'sim.render_fps', 'sim.scene', 'sim.scene.planner_eval',
        }
        assert all(change['path'] in allowed_removed for change in changes), name
        diffs[name] = changes
        checks = {}
        if 'control' in before:
            video, _ = resolve_active_video_profile(before)
            frame_size = (int(video['width']), int(video['height']))
            old_aim = legacy.ControlConfig.from_raw_config(before, frame_size)
            new_aim = ControlConfig.from_raw_config(after, frame_size)
            def aiming_values(parsed):
                values = {key: dataclasses.asdict(parsed)[key] for key in geometry_keys}
                values.update(rate_limits=dataclasses.asdict(parsed.rate_limits),
                              accel_limits=dataclasses.asdict(parsed.accel_limits))
                return normalized(values)
            old_values, new_values = aiming_values(old_aim), aiming_values(new_aim)
            assert old_values == new_values, name
            checks['aiming_laser_signs_limits_planner'] = new_values
            old_controller = normalized(dataclasses.asdict(ControlRuntimeConfig.from_config(before)))
            new_controller = normalized(dataclasses.asdict(ControlRuntimeConfig.from_config(after)))
            assert old_controller == new_controller, name
            checks['controller_including_clocks'] = new_controller
        if module == 'jetson.gimbal_bridge':
            keys = ('yaw_rate_limit_rad_s', 'pitch_rate_limit_rad_s')
            checks['bridge_caps'] = {key: after['gimbal'][key] for key in keys}
            assert checks['bridge_caps'] == {key: before['gimbal'][key] for key in keys}
        if module == 'pc.streamer':
            old_scene, new_scene = dict(before['sim']['scene']), dict(after['sim']['scene'])
            if old_scene['mode'] == 'static_targets':
                old_scene.pop('planner_eval', None)
                new_scene.pop('planner_eval', None)
            assert old_scene == new_scene, name
            checks['active_scene'] = 'identical'
        runtime[name] = checks
    # Compare saved initial snapshots too, when running in the original task checkout.
    saved = ROOT / '.task-tmp/before.json'
    if saved.exists():
        for name, value in json.loads(saved.read_text(encoding='utf-8')).items():
            assert before_units[name]['config'] == value['config']
    old_panel = original('rpi/manual_control.py')
    new_panel = (ROOT / 'rpi/manual_control.py').read_text(encoding='utf-8')
    for name in ('ManualSwitchIO', 'map_value_to_rate', 'read_adc', 'resolve_gpio_config'):
        def segment(source):
            node = next(n for n in ast.parse(source).body if getattr(n, 'name', None) == name)
            return ast.get_source_segment(source, node)
        assert segment(old_panel) == segment(new_panel), name
    assert original('tests/test_config_merge_semantics.py') == (ROOT / 'tests/test_config_merge_semantics.py').read_text(encoding='utf-8')
    dump('merged-before.json', before_units)
    dump('merged-after.json', after_units)
    dump('differences.json', diffs)
    dump('runtime-equivalence.json', {'base': BASE, 'units': runtime, 'without_config': no_config,
                                     'gpio_adc_definitions': 'unchanged', 'merge_semantics_test': 'unchanged'})
    print(f'{len(before_units)} config-loading units; {sum(not v["implicit_default"] for v in before_units.values())} explicit stacks; {len(no_config)} unit without config')
    print('Only removals in merged maps; all compared runtime values, clocks, streamer scenes and GPIO/ADC definitions unchanged.')
    for name, changes in diffs.items():
        print(name, ':', ', '.join(change['path'] for change in changes) or '(no differences)')


if __name__ == '__main__':
    main()
