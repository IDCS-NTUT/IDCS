"""Simulated safety panel: an armed auto-control state for the simulated mount.

The real panel is the Pi (``rpi.runtime_control``). In pure simulation there
is no panel, so this publishes ``ManualControlState`` with auto control
enabled, manual inactive and no emergency. It only accepts a loopback
endpoint, so it can never arm the controller that drives the real gimbal
(which binds its manual-state socket on the Jetson).
"""

from __future__ import annotations

import argparse
import time

import zmq

from common.schemas import ManualControlState
from common.shutdown import install_signal_handlers
from common.sim_mode import require_simulation_loopback_endpoint


def armed_state() -> ManualControlState:
    return ManualControlState(
        src_ts_ms=int(time.time() * 1000), source="sim_panel", active=False,
        emergency=False, control_cmd_enabled=True, joystick_raw=(0, 0),
        joystick_rate_cmd=(0.0, 0.0), note="simulated mount only",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-state-endpoint", required=True,
                        help="controller manual-state PULL endpoint; loopback only")
    parser.add_argument("--publish-hz", type=float, default=20.0)
    args = parser.parse_args()
    endpoint = require_simulation_loopback_endpoint(args.manual_state_endpoint, "--manual-state-endpoint")
    if not 1.0 <= args.publish_hz <= 100.0:
        parser.error("--publish-hz must be in [1, 100]")
    stop = install_signal_handlers()
    context = zmq.Context()
    push = context.socket(zmq.PUSH)
    push.setsockopt(zmq.LINGER, 0)
    push.setsockopt(zmq.SNDHWM, 1)
    push.connect(endpoint)
    print(f"[sim_panel] armed auto state -> {endpoint} at {args.publish_hz:.0f} Hz", flush=True)
    try:
        while not stop.is_set():
            try:
                push.send_string(armed_state().model_dump_json(), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            stop.wait(1.0 / args.publish_hz)
    finally:
        push.close(0)
        context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
