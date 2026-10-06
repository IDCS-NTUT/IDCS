"""Bounded panel-code target-picker check; no GPIO, fire, or motor commands.

Run on Pi during the software-only Simulator mode. Synthetic joystick
deflections exercise the actual navigator/menu, not the physical switch.
"""
import argparse
import json
from pathlib import Path
import time
import zmq
from common.config import load_config_bundle, resolve_config_paths
from rpi.display.commands import CommandClient
from rpi.display.inputs import PanelNavigator
from rpi.display.menu import Settings
from rpi.display.status import SystemStatus
from rpi.operator_display import build_menu, DisplayConfig, SETTINGS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=int)
    p.add_argument("--planner", action="store_true")
    p.add_argument("--expect-no-selection", action="store_true")
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    if sum((a.target is not None, a.planner, a.expect_no_selection)) != 1:
        p.error("choose target, planner or expect-no-selection")
    cfg = load_config_bundle(resolve_config_paths("configs/base")).data
    ctx = zmq.Context()
    client = CommandClient(ctx, cfg["net"]["zmq_operator_command"], status_period_s=.1)
    status = SystemStatus()
    sent, replies = [], []
    def send(command):
        sent.append(json.loads(command.to_json()))
        client.submit(command)
    menu = build_menu(status, Settings(SETTINGS, None), DisplayConfig(cfg, None), send)
    nav = PanelNavigator()
    result = {"physical_gpio_tested": False, "ok": False}
    def poll():
        for reply in client.poll(time.monotonic()):
            if reply.state is not None:
                status.on_agent(reply.state, time.monotonic())
            if reply.command != "status":
                replies.append({"command": reply.command, "ok": reply.ok, "message": reply.message})
                if not reply.ok:
                    raise RuntimeError(reply.message)
    def wait(predicate, timeout=8):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            poll()
            if predicate():
                return
            time.sleep(.02)
        raise RuntimeError("target-state condition timed out")
    def push(x, y):
        for value in ((0., 0.), (x, y)):
            for event in nav.update({"joystick": value, "manual_active": False, "emergency": False}, time.monotonic()):
                menu.handle(event)
    def catalog():
        return (status.agent_state.get("target_catalog") or {}).get("catalog") or {}
    try:
        wait(lambda: status.agent_state.get("mode") == "sim" and catalog().get("source") == "sim")
        if status.agent_state.get("motors_live") or status.agent_state.get("busy"):
            raise RuntimeError("physical owner/transition active")
        result["before"] = dict(status.agent_state)
        if a.expect_no_selection:
            wait(lambda: status.agent_state.get("selection_mode") == "operator" and catalog().get("selected_id") is None)
            result["remaining_targets"] = [t["track_id"] for t in catalog().get("tracks", ())]
            if not result["remaining_targets"]:
                raise RuntimeError("loss scenario needs another available target")
        else:
            push(-1, 0)
            if not menu.picker_active:
                raise RuntimeError("left did not open physical-panel target picker")
            items = list(menu._stack[-1].page.current_items())
            index = 0 if a.planner else next(i for i, item in enumerate(items) if item.key == ("track", a.target))
            for _ in range(index):
                push(0, -1)
            if sent:
                raise RuntimeError("browse changed target before confirmation")
            push(-1, 0)
            if sent or menu.is_open:
                raise RuntimeError("cancel changed target or failed to close")
            push(-1, 0)
            for _ in range(index):
                push(0, -1)
            push(1, 0)
            if len(sent) != 1 or menu.is_open:
                raise RuntimeError("right must confirm once and close picker")
            wait(lambda: bool(replies))
            wait(lambda: status.agent_state.get("selection_mode") == ("planner" if a.planner else "operator")
                 and (a.planner or (catalog().get("selected_id") == a.target and catalog().get("policy") == "operator_lock")))
        result.update(ok=True, commands=sent, replies=replies, after=dict(status.agent_state))
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        client.close()
        ctx.term()
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"ok": result["ok"], "error": result.get("error"), "commands": sent}), flush=True)
    return int(not result["ok"])

if __name__ == "__main__":
    raise SystemExit(main())
