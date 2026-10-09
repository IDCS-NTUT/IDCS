"""Bounded, read-only SIM display wire; never a control/clock qualification.

Sender sample age + receiver residence are reported separately from unknown
network transit. Do not subtract unrelated monotonic clocks.
"""
import json
import math
import copy
from dataclasses import dataclass
from typing import Mapping
from common.sim_score import SwarmScore

MAX_BYTES = 65536
STALE_S = .25


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_data(name, data):
    if name == 'pose':
        keys = {'pan', 'tilt', 'home_pan', 'home_tilt'}
        if not keys <= set(data) or set(data) - keys - {'sim_scene', 'swarm_score'} or not all(finite(data.get(k)) for k in ('pan', 'tilt')):
            raise ValueError('invalid HUD pose')
        if any(data.get(k) is not None and not finite(data[k]) for k in ('home_pan', 'home_tilt')):
            raise ValueError('invalid HUD home')
        if 'sim_scene' in data and data['sim_scene'] not in ('single', 'swarm'):
            raise ValueError('invalid SIM scene')
        if data.get('swarm_score') is not None:
            if data.get('sim_scene') != 'swarm':
                raise ValueError('score outside Swarm scene')
            SwarmScore.model_validate(data['swarm_score'])
    elif name == 'diagnostics':
        if data.get('type') != 'ControlDiagnostics' or data.get('version') != 1 or not isinstance(data.get('reason'), str):
            raise ValueError('invalid HUD diagnostics')
        for axis in ('yaw', 'pitch'):
            values = data.get(axis)
            if not isinstance(values, dict) or type(values.get('estimator_enabled')) is not bool or not finite(values.get('final_rate_rad_s')):
                raise ValueError('invalid HUD axis')
            for key in ('feedforward_term_rad_s', 'desired_rate_pre_limit_rad_s', 'desired_rate_post_limit_rad_s'):
                if values.get(key) is not None and not finite(values[key]):
                    raise ValueError('invalid HUD term')
        for key in ('aim_reference_norm', 'target_center_norm'):
            point = data.get(key)
            if point is not None and (not isinstance(point, (tuple, list)) or len(point) != 2 or not all(finite(v) and 0 <= v <= 1 for v in point)):
                raise ValueError('invalid HUD aim geometry')
    elif name == 'perception':
        tracks, assessments = data.get('tracks'), data.get('assessments')
        if not isinstance(tracks, list) or not isinstance(assessments, list) or len(tracks) > 32 or len(assessments) > 32:
            raise ValueError('invalid HUD track lists')
        ids = []
        for track in tracks:
            if not isinstance(track, dict) or type(track.get('track_id')) is not int or track['track_id'] < 0:
                raise ValueError('invalid HUD track ID')
            ids.append(track['track_id'])
            box = track.get('box')
            if not isinstance(box, dict) or not all(finite(box.get(k)) for k in ('x', 'y', 'w', 'h')):
                raise ValueError('invalid HUD box')
            if min(box['x'], box['y']) < 0 or min(box['w'], box['h']) <= 0 or box['x'] + box['w'] > 1.000000000001 or box['y'] + box['h'] > 1.000000000001:
                raise ValueError('HUD box outside frame')
            if type(track.get('missed_frames', 0)) is not int or track.get('missed_frames', 0) < 0:
                raise ValueError('invalid HUD track lifecycle')
        if len(ids) != len(set(ids)) or data.get('selected_id') is not None and data['selected_id'] not in ids:
            raise ValueError('invalid HUD selection identity')
        for assessment in assessments:
            if not isinstance(assessment, dict) or assessment.get('track_id') not in ids:
                raise ValueError('invalid HUD assessment identity')
            if assessment.get('distance_m') is not None and (not finite(assessment['distance_m']) or assessment['distance_m'] < 0):
                raise ValueError('invalid HUD range')
    return data


def validate_message(raw):
    if isinstance(raw, (str, bytes)):
        if len(raw) > MAX_BYTES:
            raise ValueError('oversized HUD telemetry')
        raw = json.loads(raw)
    if not isinstance(raw, dict) or set(raw) != {'type', 'version', 'session', 'source_session', 'sequence', 'pose', 'diagnostics', 'perception'}:
        raise ValueError('invalid HUD wire keys')
    if raw['type'] != 'OperatorHudTelemetry' or type(raw['version']) is not int or raw['version'] != 1:
        raise ValueError('invalid HUD type/version')
    for name in ('session', 'source_session'):
        if not isinstance(raw[name], str) or not 1 <= len(raw[name]) <= 80:
            raise ValueError('invalid HUD session')
    if type(raw['sequence']) is not int or raw['sequence'] < 0:
        raise ValueError('invalid HUD sequence')
    for name in ('pose', 'diagnostics', 'perception'):
        value = raw[name]
        if value is not None:
            if not isinstance(value, dict) or set(value) != {'sequence', 'age_ms', 'data'} or type(value['sequence']) is not int or value['sequence'] < 0:
                raise ValueError('invalid HUD sample')
            if not finite(value['age_ms']) or value['age_ms'] < 0 or not isinstance(value['data'], dict):
                raise ValueError('invalid HUD sample age/data')
            validate_data(name, value['data'])
    def inspect(value, depth=0):
        if depth > 12:
            raise ValueError('HUD nesting limit')
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('non-finite HUD field')
        if isinstance(value, dict):
            if len(value) > 64:
                raise ValueError('HUD mapping limit')
            for key, item in value.items():
                if key in {'control_activity', 'intent', 'limits', 'fire', 'manual_active'}:
                    raise ValueError('authority not accepted by HUD')
                inspect(item, depth + 1)
        elif isinstance(value, (list, tuple)):
            if len(value) > 32:
                raise ValueError('HUD list limit')
            for item in value:
                inspect(item, depth + 1)
        elif isinstance(value, str) and len(value) > 160:
            raise ValueError('HUD string limit')
    inspect(raw)
    return raw


@dataclass(frozen=True)
class HudView:
    source: str
    session: str
    pose: Mapping | None = None
    diagnostics: Mapping | None = None
    perception: Mapping | None = None
    pose_age_s: float | None = None
    diagnostics_age_s: float | None = None
    perception_age_s: float | None = None
    # SIM wire age excludes unmeasured transit, not exact capture-to-display.
    age_basis: str = 'rx age'


class HudCache:
    def __init__(self):
        self.value = None
        self.received_s = None
        self.retired = set()
        self.samples = {}

    def accept(self, raw, now_s):
        value = validate_message(copy.deepcopy(raw) if isinstance(raw, dict) else raw)
        identity = (value['session'], value['source_session'])
        if identity in self.retired:
            return False
        if self.value is not None:
            old = (self.value['session'], self.value['source_session'])
            if identity == old and value['sequence'] <= self.value['sequence']:
                return False
            if old != identity:
                self.retired.add(old)
                if len(self.retired) > 64:
                    raise ValueError('HUD session retirement limit; restart display')
                self.samples.clear()
        for name in ('pose', 'diagnostics', 'perception'):
            sample = value[name]
            previous = self.samples.get(name)
            if sample is None:
                self.samples.pop(name, None)
            elif previous is None or sample['sequence'] > previous[0]['sequence']:
                self.samples[name] = sample, now_s
        self.value, self.received_s = value, now_s
        return True

    def view(self, now_s, *, source_session):
        if self.value is None or self.value['source_session'] != source_session or now_s < self.received_s:
            return HudView('SIM-ONLY', source_session or '', age_basis='sample+rx age; transit unknown')
        values, ages = {}, {}
        for name in ('pose', 'diagnostics', 'perception'):
            entry = self.samples.get(name)
            age = None if entry is None else entry[0]['age_ms'] / 1000 + now_s - entry[1]
            ages[name + '_age_s'] = age
            values[name] = entry[0]['data'] if age is not None and 0 <= age <= STALE_S else None
        return HudView('SIM-ONLY', source_session, **values, **ages, age_basis='sample+rx age; transit unknown')
