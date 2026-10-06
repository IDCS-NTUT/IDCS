"""Bounded live Camera/Simulator test through the Pi's real menu/client code.

Does not synthesize GPIO input or open another display. Navigates a menu
instance and confirms its action using the same command client as the running
display. Tests service/source transitions, not physical joystick operation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import zmq

from common.config import load_config_bundle, resolve_config_paths
from common.operator_commands import OperatorCommand
from common.perception import perception_snapshot_from_json
from rpi.display.commands import CommandClient
from rpi.display.menu import NavEvent, Settings
from rpi.display.status import SystemStatus
from rpi.operator_display import DisplayConfig, SETTINGS, build_menu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=100)
    parser.add_argument("--modes", default="camera,sim,camera")
    args = parser.parse_args()
    modes = args.modes.split(",")
    if any(m not in ("camera", "sim") for m in modes):
        parser.error("only control-free camera/sim modes are permitted")
    cfg = load_config_bundle(resolve_config_paths("configs/base")).data
    ctx = zmq.Context()
    client = CommandClient(ctx, cfg["net"]["zmq_operator_command"], status_period_s=0.25)
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(cfg["net"]["zmq_perception_v2"])
    status = SystemStatus()
    result = {"type": "MenuSwitchTrial", "physical_input_tested": False, "stages": [], "ok": False}

    def emit(event):
        print(json.dumps(event), flush=True)

    def poll():
        for response in client.poll(time.monotonic()):
            if response.state is not None:
                status.on_agent(response.state, time.monotonic())
            if response.command != "status":
                emit({"command": response.command, "ok": response.ok, "message": response.message})
                if not response.ok:
                    raise RuntimeError(response.message)
        try:
            snapshot = perception_snapshot_from_json(sub.recv(zmq.NOBLOCK))
        except zmq.Again:
            return None
        return snapshot

    try:
        deadline = time.monotonic() + 8
        while not status.agent_state and time.monotonic() < deadline:
            poll()
            time.sleep(0.02)
        state = status.agent_state
        if not state or state.get("motors_live") or state.get("busy") or not state.get("pc_online"):
            raise RuntimeError(f"unsafe/unavailable preflight state: {state}")
        for mode in modes:
            began = time.monotonic()
            deadline = began + args.timeout_s
            stage = {"requested": mode, "phases": [], "frames": 0, "clock_domains": [], "ok": False}
            result["stages"].append(stage)
            emit({"requested": mode, "event": "menu_confirm"})
            menu = build_menu(status, Settings(SETTINGS, None), DisplayConfig(cfg, None), client.submit)
            for event in (NavEvent.MENU, NavEvent.DOWN, NavEvent.RIGHT, NavEvent.RIGHT):
                menu.handle(event)  # Menu / System / Mode
            entries = list(menu._stack[-1].page.current_items())
            index = next(i for i, entry in enumerate(entries) if entry.key == ("mode", mode))
            for _ in range(index):
                menu.handle(NavEvent.DOWN)
            menu.handle(NavEvent.RIGHT)
            menu.handle(NavEvent.DOWN)  # Cancel -> Confirm
            menu.handle(NavEvent.RIGHT)
            frames = set()
            clocks = set()
            last_phase = None
            ready_at = None
            while time.monotonic() < deadline:
                snapshot = poll()
                state = status.agent_state
                if state.get("motors_live"):
                    raise RuntimeError("motor owner became active")
                phase = (state.get("busy"), state.get("message"))
                if phase != last_phase:
                    stage["phases"].append({"elapsed_s": round(time.monotonic()-began, 3),
                                            "busy": phase[0], "message": phase[1]})
                    emit({"requested": mode, "busy": phase[0], "message": phase[1]})
                    last_phase = phase
                expected = "pc_monotonic" if mode == "sim" else "jetson_monotonic"
                if snapshot is not None and snapshot.frame.source_clock_domain == expected:
                    frames.add((snapshot.frame.frame_id, snapshot.frame.source_time_ns))
                    clocks.add(snapshot.frame.source_clock_domain)
                if not state.get("busy") and "failed" in str(state.get("message")):
                    raise RuntimeError(str(state["message"]))
                if (state.get("mode") == mode and not state.get("busy")
                        and str(state.get("message")).endswith(" ready") and len(frames) >= 10):
                    ready_at = ready_at or time.monotonic()
                    if time.monotonic() - ready_at >= 3:
                        stage.update(ok=True, elapsed_s=round(time.monotonic()-began, 3),
                                     frames=len(frames), clock_domains=sorted(clocks), final_state=dict(state))
                        emit({"requested": mode, "event": "passed", "elapsed_s": stage["elapsed_s"], "frames": len(frames)})
                        break
                time.sleep(0.02)
            else:
                raise RuntimeError(f"{mode}: source transition timed out")
        result["ok"] = True
    except Exception as exc:
        result["error"] = str(exc)
        emit({"event": "failed", "error": str(exc)})
    finally:
        client.close()
        sub.close(0)
        ctx.term()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
