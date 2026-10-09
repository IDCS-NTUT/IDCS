"""Read-only loopback SIM telemetry -> Pi; no inbound PC address or video encode."""
import argparse
import json
import signal
import threading
import time
import uuid

import zmq

from common.config import load_config_bundle, resolve_config_paths
from common.hud_telemetry import validate_message, validate_data
from common.perception import PerceptionSnapshotV2
from common.schemas import CamState, ControlDiagnostics
from common.sim_mode import require_simulation_loopback_endpoint


class HudSampler:
    def __init__(self, session=None):
        self.session = session or uuid.uuid4().hex
        self.sequence = 0
        self.source_session = None
        self.source_started_ns = 0
        self.retired = set()
        self.samples = {}

    def receive(self, name, raw, now_ns):
        if name == 'pose':
            state = CamState.model_validate_json(raw)
            cat = state.target_catalog
            if cat is None or cat.source != 'sim' or state.state_monotonic_ns is None:
                self.source_session = None
                self.samples.clear()
                return
            if cat.session != self.source_session:
                if cat.session in self.retired:
                    return
                if self.source_session:
                    self.retired.add(self.source_session)
                self.samples.clear()
                self.source_started_ns = state.state_monotonic_ns
            self.source_session = cat.session
            data = {key: getattr(state, key) for key in ('pan', 'tilt', 'home_pan', 'home_tilt')}
            if state.sim_scene is not None:
                data['sim_scene'] = state.sim_scene
                data['swarm_score'] = None if state.swarm_score is None else state.swarm_score.model_dump(mode='json')
            samples = (state.pan_sample_monotonic_ns, state.tilt_sample_monotonic_ns)
            if not all(type(s) is int and 0 <= s <= state.state_monotonic_ns for s in samples):
                self.samples.pop('pose', None)
                return
            seq, stamp = state.frame_id, min(samples)
        elif name == 'diagnostics':
            state = ControlDiagnostics.model_validate_json(raw)
            if state.timing.source_clock_domain not in (None, 'pc_monotonic'):
                return
            data = state.model_dump(mode='json', exclude_none=True)
            # Controller intent counters restart independently of the simulator.
            seq = stamp = state.created_monotonic_ns
            if stamp < self.source_started_ns:
                return
        elif name == 'perception':
            state = PerceptionSnapshotV2.model_validate_json(raw)
            if state.frame.source_clock_domain != 'pc_monotonic' or state.frame.observation_clock_domain != 'pc_monotonic':
                return
            if state.frame.tracker_session != self.source_session:
                return
            tracks = list(state.tracks[:31])
            if state.selection is not None:
                selected = next(t for t in state.tracks if t.track_id == state.selection.track_id)
                if selected not in tracks:
                    tracks.append(selected)
            ids = {t.track_id for t in tracks}
            data = dict(sequence=state.sequence, frame_id=state.frame.frame_id,
                tracks=[t.model_dump(mode='json') for t in tracks],
                assessments=[a.model_dump(mode='json', exclude_none=True) for a in state.assessments if a.track_id in ids],
                selected_id=None if state.selection is None else state.selection.track_id)
            seq, stamp = state.sequence, state.frame.observed_time_ns
        else:
            raise ValueError('unknown HUD source')
        if stamp > now_ns or stamp < 0:
            self.samples.pop(name, None)
            return
        validate_data(name, data)
        old = self.samples.get(name)
        if old is None or seq > old[0]:
            self.samples[name] = seq, stamp, data

    def message(self, now_ns):
        if self.source_session is None:
            return None
        self.sequence += 1
        values = {}
        for name in ('pose', 'diagnostics', 'perception'):
            sample = self.samples.get(name)
            age = None if sample is None else (now_ns - sample[1]) / 1e6
            values[name] = None if age is None or not 0 <= age <= 250 else dict(sequence=sample[0], age_ms=age, data=sample[2])
        message = dict(type='OperatorHudTelemetry', version=1, session=self.session,
            source_session=self.source_session, sequence=self.sequence, **values)
        return validate_message(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/base')
    parser.add_argument('--config-extra')
    parser.add_argument('--duration-s', type=float)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    cfg = load_config_bundle(resolve_config_paths(args.config, args.config_extra)).data
    wire = (cfg.get('operator') or {}).get('hud') or {}
    dest = str(wire['push_endpoint'])
    inputs = {name: require_simulation_loopback_endpoint(str(wire[name + '_sub']), name + '_sub')
        for name in ('pose', 'diagnostics', 'perception')}
    if not dest.startswith('tcp://') or dest.endswith(':5557'):
        raise ValueError('HUD destination must be a dedicated read-only endpoint')
    if args.check:
        print(json.dumps(dict(destination=dest, inputs=inputs)))
        return 0
    if args.duration_s is not None and args.duration_s <= 0:
        parser.error('duration must be positive')
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    ctx = zmq.Context()
    sockets = {}
    for name, endpoint in inputs.items():
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.setsockopt(zmq.LINGER, 0)
        sub.setsockopt_string(zmq.SUBSCRIBE, '')
        sub.connect(endpoint)
        sockets[name] = sub
    push = ctx.socket(zmq.PUSH)
    push.setsockopt(zmq.CONFLATE, 1)
    push.setsockopt(zmq.IMMEDIATE, 1)
    push.setsockopt(zmq.LINGER, 0)
    push.connect(dest)
    sampler = HudSampler()
    deadline = None if args.duration_s is None else time.monotonic() + args.duration_s
    sent = 0
    try:
        while not stop.is_set() and (deadline is None or time.monotonic() < deadline):
            for name, sub in sockets.items():
                try:
                    sampler.receive(name, sub.recv(zmq.NOBLOCK), time.monotonic_ns())
                except (zmq.Again, ValueError):
                    pass
            message = sampler.message(time.monotonic_ns())
            if message is not None:
                try:
                    push.send_string(json.dumps(message, separators=(',', ':')), flags=zmq.NOBLOCK)
                    sent += 1
                except zmq.Again:
                    pass
            stop.wait(1 / 30)
    finally:
        for sock in (*sockets.values(), push):
            sock.close(0)
        ctx.term()
    print(json.dumps(dict(hud_messages_sent=sent)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
