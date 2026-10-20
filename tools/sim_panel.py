"""SIM-only panel bridge: physical Pi state when present, synthetic fallback otherwise.

The real panel is the Pi (``rpi.runtime_control``). In pure simulation there
is no panel, this publishes an armed auto state. Once the Pi panel connects,
its full GPIO/joystick state wins. A stale/lost Pi stream holds SIM control
disabled instead of silently falling back to synthetic authority. Output is
loopback-only, so it cannot arm the real gimbal controller on the Jetson.
"""

from __future__ import annotations

import argparse
import json
import time

import zmq
from zmq.utils.monitor import recv_monitor_message

from common.schemas import ManualControlState, manual_control_state_from_json
from common.return_frame_identity import RoiSourceLease
from common.shutdown import install_signal_handlers
from common.sim_mode import require_simulation_loopback_endpoint


def armed_state() -> ManualControlState:
    return ManualControlState(
        src_ts_ms=int(time.time() * 1000), source="sim_panel", active=False,
        emergency=False, control_cmd_enabled=True, control_cmd_valid=True, fire_arm=False, joystick_raw=(0, 0),
        joystick_rate_cmd=(0.0, 0.0), note="simulated mount only",
    )


def disabled_state(reason: str) -> ManualControlState:
    return ManualControlState(
        src_ts_ms=int(time.time() * 1000), source="sim_panel_guard", active=False,
        emergency=False, control_cmd_enabled=False, fire_arm=False, fire=False,
        joystick_raw=(0, 0), joystick_rate_cmd=(0.0, 0.0), note=reason,
    )


class PanelAuthority:
    """Use local receive time; never compare unsynchronized PC/Pi clocks."""

    def __init__(self, *, started_s: float, startup_grace_s: float = 2.0,
                 stale_s: float = 0.35, allow_panel_less: bool = False):
        self.started_s = started_s
        self.startup_grace_s = startup_grace_s
        self.stale_s = stale_s
        self.allow_panel_less = allow_panel_less
        self.physical_seen = False
        self.physical_state: ManualControlState | None = None
        self.received_s: float | None = None

    def mark_connected(self) -> None:
        self.physical_seen = True

    def observe(self, payload: str | bytes, *, now_s: float) -> bool:
        try:
            state = manual_control_state_from_json(payload)
        except (ValueError, TypeError):
            return False
        if state.source != "rpi.runtime_control":
            return False
        self.physical_seen = True
        self.physical_state, self.received_s = state, now_s
        return True

    def state(self, now_s: float) -> ManualControlState:
        if self.physical_seen:
            if (self.physical_state is not None and self.received_s is not None
                    and 0 <= now_s - self.received_s < self.stale_s):
                return self.physical_state
            return disabled_state("physical panel stream absent or stale")
        if not self.allow_panel_less:
            return disabled_state("physical panel required but not connected")
        if 0 <= now_s - self.started_s < self.startup_grace_s:
            return disabled_state("waiting for physical panel discovery")
        return armed_state()


class RoiSimAuthority:
    """Fresh Pi lease plus receiver acceptance; transforms only loopback SIM output.

    Physical output on Pi stays blocked. A missing lease after a managed session
    holds SIM disabled until another accepted lease or an ordered source switch. No
    synthetic fallback or unverified GPIO edge can restore automatic authority.
    """
    def __init__(self):
        self.lease = self.status = None
        self.lease_s = self.status_s = None
        self.engaged = False
        self.source = RoiSourceLease()

    def observe(self, payload, now_s, *, status=False):
        try:
            value = json.loads(payload)
        except (ValueError,TypeError):
            return False
        if not isinstance(value,dict):
            return False
        if status:
            if value.get('type') != 'visual_roi_status_v1': return False
            mode = value.get('managed_mode')
            if mode not in ('sim','sim_visual_lock') or not self.source.observe(value,now_s,managed_mode=mode):
                self.status = None
                return False
            self.status, self.status_s = value, now_s
        else:
            if value.get('type') != 'roi_sim_lease_v1' or type(value.get('enabled')) is not bool: return False
            self.lease, self.lease_s = value, now_s
            if value['enabled']: self.engaged = True
        return True

    def apply(self, panel, now_s):
        lease, status = self.lease or {}, self.status or {}
        fresh = (self.lease_s is not None and 0 <= now_s-self.lease_s <= .25
                 and self.status_s is not None and 0 <= now_s-self.status_s <= .25)
        accepted = (fresh and lease.get('enabled') is True
            and lease.get('managed_mode') == status.get('managed_mode')
            and status.get('managed_mode') in ('sim','sim_visual_lock')
            and status.get('source_mode') == 'sim' and status.get('source_ready') is True
            and status.get('control_disabled') is False
            and isinstance(lease.get('request_id'),str) and bool(lease['request_id'])
            and lease['request_id'] == status.get('active_request_id')
            and isinstance(lease.get('tracker_session'),str) and bool(lease['tracker_session'])
            and lease['tracker_session'] == status.get('tracker_session')
            and panel.source == 'rpi.runtime_control' and panel.control_cmd_valid and not panel.emergency)
        if accepted:
            return panel.model_copy(update=dict(active=False,control_cmd_enabled=True,
                joystick_rate_cmd=(0.,0.),fire=False,fire_arm=False,note='managed visual lock; SIM loopback only'))
        if self.engaged:
            return disabled_state('visual-lock lease absent, stale or released').model_copy(update={'emergency':panel.emergency})
        return panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-state-endpoint", required=True,
                        help="controller manual-state PULL endpoint; loopback only")
    parser.add_argument("--roi-control-endpoint",default=None,help="Pi managed SIM-only ROI lease")
    parser.add_argument("--roi-status-endpoint",default=None,help="Jetson exact ROI receiver acceptance")
    parser.add_argument("--publish-hz", type=float, default=20.0)
    parser.add_argument("--physical-panel-endpoint", default=None,
                        help="Pi ManualControlState PUB; omitted for panel-less SIM")
    parser.add_argument("--physical-stale-s", type=float, default=0.35)
    parser.add_argument("--discovery-grace-s", type=float, default=2.0)
    args = parser.parse_args()
    endpoint = require_simulation_loopback_endpoint(args.manual_state_endpoint, "--manual-state-endpoint")
    if not 1.0 <= args.publish_hz <= 100.0:
        parser.error("--publish-hz must be in [1, 100]")
    if args.physical_stale_s <= 0 or args.discovery_grace_s < 0:
        parser.error("--physical-stale-s must be positive and --discovery-grace-s nonnegative")
    if bool(args.roi_control_endpoint) != bool(args.roi_status_endpoint):
        parser.error("ROI lease and status endpoints must be supplied together")
    stop = install_signal_handlers()
    context = zmq.Context()
    push = context.socket(zmq.PUSH)
    push.setsockopt(zmq.LINGER, 0)
    push.setsockopt(zmq.SNDHWM, 1)
    push.connect(endpoint)
    roi = RoiSimAuthority()
    roi_sockets = []
    for index,address in enumerate((args.roi_control_endpoint,args.roi_status_endpoint)):
        if address:
            sub = context.socket(zmq.SUB)
            sub.setsockopt(zmq.SUBSCRIBE,b"")
            if index == 0: sub.setsockopt(zmq.CONFLATE,1)
            else: sub.setsockopt(zmq.RCVHWM,128)
            sub.setsockopt(zmq.LINGER,0)
            sub.connect(address)
            roi_sockets.append(sub)
    physical = monitor = None
    authority = PanelAuthority(started_s=time.monotonic(),
                               startup_grace_s=args.discovery_grace_s,
                               stale_s=args.physical_stale_s,
                               allow_panel_less=args.physical_panel_endpoint is None)
    if args.physical_panel_endpoint:
        physical = context.socket(zmq.SUB)
        physical.setsockopt(zmq.SUBSCRIBE, b"")
        physical.setsockopt(zmq.RCVHWM, 1)
        physical.setsockopt(zmq.CONFLATE, 1)
        physical.setsockopt(zmq.LINGER, 0)
        monitor = physical.get_monitor_socket(events=zmq.EVENT_CONNECTED)
        physical.connect(args.physical_panel_endpoint)
    print(f"[sim_panel] SIM-only output {endpoint}; physical panel {args.physical_panel_endpoint or 'absent'}", flush=True)
    try:
        while not stop.is_set():
            if monitor is not None:
                while True:
                    try:
                        recv_monitor_message(monitor, flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    authority.mark_connected()
            if physical is not None:
                while True:
                    try:
                        payload = physical.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    authority.observe(payload, now_s=time.monotonic())
            for index, sub in enumerate(roi_sockets):
                # Status shares its PUB with frame identities; never conflate it.
                for _ in range(128):
                    try: payload = sub.recv(flags=zmq.NOBLOCK)
                    except zmq.Again: break
                    roi.observe(payload,time.monotonic(),status=index == 1)
            try:
                now = time.monotonic()
                push.send_string(roi.apply(authority.state(now),now).model_dump_json(), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            stop.wait(1.0 / args.publish_hz)
    finally:
        if monitor is not None:
            monitor.close(0)
        if physical is not None:
            physical.close(0)
        for sub in roi_sockets: sub.close(0)
        push.close(0)
        context.destroy(linger=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
