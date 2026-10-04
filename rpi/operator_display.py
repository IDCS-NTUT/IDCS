"""Panel-screen operator display on the Pi.

Shows the Jetson's return video full screen on the panel's HDMI screen, with
a status bar, alert banner and an operator menu drawn over it. The menu is
driven by the panel joystick (while manual control is off), optional GPIO
menu buttons, or a USB keyboard; see ``rpi.display.inputs``.

    python -m rpi.operator_display --config configs/base [--check]

It subscribes to perception snapshots, controller diagnostics and the local
panel state. Changes the operator makes (target lock, target classes, mode,
recording) are requests to the Jetson's operator agent
(``jetson.operator_agent``), which owns that authority; the menu confirms
each system change first.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import gi
import zmq

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from common.config import ConfigError, load_config_bundle, resolve_config_paths  # noqa: E402
from common.operator_commands import OperatorCommand  # noqa: E402
from rpi.display.commands import CommandClient  # noqa: E402
from rpi.display.inputs import KeyboardInput, PanelNavigator  # noqa: E402
from rpi.display.menu import Action, Choice, Menu, Page, Settings, SettingSpec  # noqa: E402
from rpi.display.render import StatusBar, TargetBox, render  # noqa: E402
from rpi.display.status import SystemStatus  # noqa: E402
from rpi.display.video import ReturnVideo, missing_elements  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TICK_S = 0.05
DEFAULT_SETTINGS_PATH = Path.home() / ".config" / "idcs" / "operator_display.json"
SETTINGS = {
    "status_bar": SettingSpec("Status bar", ("on", "off"), "on"),
    "bar_detail": SettingSpec("Status detail", ("full", "mode only"), "full"),
}

log = logging.getLogger("rpi.operator_display")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base")
    parser.add_argument("--config-extra", default=None)
    parser.add_argument("--sink", default=None, help="override rpi.operator_display.sink")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS_PATH)
    parser.add_argument("--duration-s", type=float, default=None, help="exit after this long (tests)")
    parser.add_argument("--check", action="store_true", help="validate config and GStreamer elements, then exit")
    parser.add_argument("--debug", action="store_true")
    return parser


class DisplayConfig:
    def __init__(self, cfg: Mapping[str, Any], sink_override: str | None) -> None:
        net = cfg.get("net") or {}
        display = (cfg.get("rpi") or {}).get("operator_display") or {}
        try:
            self.port = int(net["rtp_return_port"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError("net.rtp_return_port is required") from exc
        self.jitter_ms = int(net.get("return_jitter_ms", 20))
        self.perception = str(net.get("zmq_perception_v2") or "")
        self.diagnostics = str(net.get("zmq_control_diagnostics") or "")
        self.panel = str(display.get("panel_state_endpoint") or "")
        self.command = str(net.get("zmq_operator_command") or "")
        self.sink = sink_override or str(display.get("sink") or "waylandsink fullscreen=true sync=false")
        self.session_env = {str(k): str(v) for k, v in (display.get("session_env") or {}).items()
                            if v is not None}
        if not self.panel:
            raise ConfigError("rpi.operator_display.panel_state_endpoint is required")


def _subscriber(ctx: zmq.Context, endpoint: str) -> zmq.Socket | None:
    if not endpoint:
        return None
    socket = ctx.socket(zmq.SUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.connect(endpoint)
    return socket


def _latest(socket: zmq.Socket | None) -> bytes | None:
    if socket is None:
        return None
    try:
        return socket.recv(flags=zmq.NOBLOCK)
    except zmq.Again:
        return None


def _version() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "describe", "--tags", "--always"],
                              capture_output=True, text=True, timeout=2).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_menu(status: SystemStatus, settings: Settings, config: DisplayConfig,
               send: Callable[[OperatorCommand], None] = lambda _command: None) -> Menu:
    version = _version()

    def track_items() -> list[Action]:
        if not status.perception.live(time.monotonic()):
            return []
        items = []
        for row in status.tracks:
            distance = f"  {row.distance_m:.0f} m" if row.distance_m is not None else ""
            value = ("LOCKED" if status.locked else "selected") if row.selected else None
            items.append(Action(f"#{row.track_id} {row.class_id} {row.confidence:.2f}{distance}",
                                lambda tid=row.track_id: send(OperatorCommand("lock", track_id=tid)),
                                key=("track", row.track_id), value=value))
        if status.locked:
            items.append(Action("Release lock", lambda: send(OperatorCommand("release"))))
        return items

    def target_hint() -> list[str]:
        if not status.perception.live(time.monotonic()):
            return ["No perception data."]
        return [] if status.tracks else ["No tracks."]

    def mode_items() -> list[Action]:
        state = status.agent_state
        current = state.get("mode")
        return [Action(m["label"], lambda name=m["name"]: send(OperatorCommand("set_mode", mode=name)),
                       confirm=f"Switch to {m['label']}?", key=("mode", m["name"]),
                       value="current" if m["name"] == current else None)
                for m in state.get("modes") or () if isinstance(m, dict) and "name" in m]

    def class_items() -> list[Action]:
        state = status.agent_state
        names = [str(n) for n in state.get("class_names") or ()]
        current = sorted(str(c) for c in state.get("target_classes") or ())
        options = [[n] for n in names] + ([names] if len(names) > 1 else [])
        return [Action(" + ".join(o), lambda o=o: send(OperatorCommand("target_classes", classes=tuple(o))),
                       confirm=f"Target {' + '.join(o)}?", value="current" if sorted(o) == current else None)
                for o in options]

    def system_items() -> list[Page | Action]:
        recording = bool(status.agent_state.get("recording"))
        return [
            Page("Mode", items=mode_items),
            Action("Recording", lambda: send(OperatorCommand("recording", on=not recording)),
                   confirm="Stop recording?" if recording else "Start recording?",
                   value="on" if recording else "off"),
            Page("Target type", items=class_items),
        ]

    def system_lines() -> list[str]:
        now = time.monotonic()
        state = status.agent_state
        if not status.agent.last_s:
            return ["Operator agent not answering."]
        lines = [
            f"Mode        {state.get('busy') or state.get('mode')}",
            f"Targets     {', '.join(state.get('target_classes') or ()) or '-'}",
            f"Lock        {('#' + str(state['lock_track_id'])) if state.get('lock_track_id') is not None else '-'}",
        ]
        if state.get("motors_live"):
            lines.append("Motor stack running: mode switching off")
        if not status._agent_recent(now):
            lines.append(f"(agent silent {status.agent.age_s(now):.0f} s)")
        return lines

    def status_lines() -> list[str]:
        now = time.monotonic()
        return [
            f"Panel       {status.panel_mode(now)}",
            f"Controller  {status.controller_reason if status.controller.live(now) else '--'}",
            "",
            *status.link_lines(now),
        ]

    def about_lines() -> list[str]:
        return [
            f"Version     {version}",
            f"Video port  {config.port}/udp",
            f"Perception  {config.perception or '-'}",
            f"Diagnostics {config.diagnostics or '-'}",
            f"Panel       {config.panel}",
            f"Commands    {config.command or '-'}",
        ]

    root = Page("Menu", items=(
        Page("Targets", items=track_items, lines=target_hint),
        Page("System", items=system_items, lines=system_lines),
        Page("Status", lines=status_lines),
        Page("Panel", lines=lambda: status.panel_lines(time.monotonic())),
        Page("Display", items=tuple(Choice(spec.label, key) for key, spec in SETTINGS.items())),
        Page("About", lines=about_lines),
    ))
    return Menu(root, settings)


class OperatorDisplay:
    def __init__(self, config: DisplayConfig, settings_path: Path) -> None:
        self.config = config
        self.status = SystemStatus()
        self.settings = Settings(SETTINGS, settings_path, log)
        self.ctx = zmq.Context()
        self.commands = CommandClient(self.ctx, config.command) if config.command else None
        self.menu = build_menu(self.status, self.settings, config, self._send)
        self.panel_nav = PanelNavigator()
        self.keyboard = KeyboardInput()
        self._agent_message: str | None = None
        self.perception = _subscriber(self.ctx, config.perception)
        self.diagnostics = _subscriber(self.ctx, config.diagnostics)
        self.panel = _subscriber(self.ctx, config.panel)
        self.video = ReturnVideo(port=config.port, jitter_ms=config.jitter_ms, sink=config.sink,
                                 on_frame=lambda: self.status.on_video_frame(time.monotonic()))
        self._drawn_key: object = None
        self._next_rescan_s = 0.0
        self._next_log_s = time.monotonic() + 10.0

    def tick(self) -> bool:
        now = time.monotonic()
        self._read_messages(now)
        if now >= self._next_rescan_s:
            self.keyboard.rescan()
            self._next_rescan_s = now + 2.0
        for event in self.keyboard.poll():
            self.menu.handle(event)
        if self.commands is not None:
            for reply in self.commands.poll(now):
                self._on_reply(reply, now)
        self.video.show_live(self.status.video.live(now))
        self._draw(now)
        if now >= self._next_log_s:
            self._next_log_s = now + 10.0
            log.info("video %.1f fps, perception %.1f Hz, panel %s",
                     self.status.video.rate_hz(now), self.status.perception.rate_hz(now),
                     self.status.panel_mode(now))
        return True

    def _read_messages(self, now: float) -> None:
        raw = _latest(self.perception)
        if raw is not None:
            message = _decode(raw, "PerceptionSnapshot")
            if message is not None:
                self.status.on_snapshot(message, now)
        raw = _latest(self.diagnostics)
        if raw is not None:
            message = _decode(raw, "ControlDiagnostics")
            if message is not None:
                self.status.on_diagnostics(message, now)
        raw = _latest(self.panel)
        if raw is not None:
            try:
                panel = _panel_state(raw)
            except ValueError as exc:
                log.debug("bad panel state: %s", exc)
            else:
                self.status.on_panel(panel, now)
                for event in self.panel_nav.update(panel, now):
                    self.menu.handle(event)

    def _send(self, command: OperatorCommand) -> None:
        if self.commands is None:
            self.status.show_notice("No operator agent configured", time.monotonic())
            return
        self.commands.submit(command)

    def _on_reply(self, reply, now: float) -> None:
        self.status.on_agent(reply.state, now)
        message = (reply.state or {}).get("message")
        if not reply.ok and reply.command != "status":
            self.status.show_notice(f"Refused: {reply.message}", now)
        elif message and message != self._agent_message:
            if self._agent_message is not None or reply.command != "status":
                self.status.show_notice(str(message), now)
        if message:
            self._agent_message = str(message)

    def _boxes(self) -> tuple[TargetBox, ...]:
        boxes = []
        rows = {row.track_id: row for row in self.status.tracks}
        if self.status.locked and self.status.selected_id in rows:
            row = rows[self.status.selected_id]
            if row.box is not None:
                boxes.append(TargetBox(row.box, f"LOCK #{row.track_id}", "lock"))
        item = self.menu.highlighted()
        key = getattr(item, "key", None)
        if isinstance(key, tuple) and key[0] == "track" and key[1] in rows:
            row = rows[key[1]]
            if row.box is not None and not (self.status.locked and row.selected):
                boxes.append(TargetBox(row.box, f"#{row.track_id}", "cursor"))
        return tuple(boxes)

    def _draw(self, now: float) -> None:
        bar = None
        if self.settings["status_bar"] == "on":
            fields: tuple[str, ...] = ()
            if self.settings["bar_detail"] == "full":
                fields = (
                    f"{self.status.video.rate_hz(now):.0f} fps",
                    self.status.targets_text(now),
                    self.status.controller_text(now),
                    self.status.agent_text(now),
                )
            bar = StatusBar(self.status.panel_mode(now), fields)
        alerts = self.status.alerts(now)
        view = self.menu.view()
        boxes = self._boxes()
        notice = self.status.current_notice(now)
        key = (self.video.frame_size, bar, tuple(alerts), view, boxes, notice)
        if key == self._drawn_key:
            return
        self._drawn_key = key
        width, height = self.video.frame_size
        self.video.set_overlay(render(width, height, bar=bar, alerts=alerts, menu=view,
                                      boxes=boxes, notice=notice))

    def close(self) -> None:
        self.video.stop()
        self.keyboard.close()
        if self.commands is not None:
            self.commands.close()
        self.ctx.destroy(linger=0)


def _decode(raw: bytes, kind: str) -> dict | None:
    try:
        message = json.loads(raw)
    except ValueError:
        return None
    return message if isinstance(message, dict) and message.get("type") == kind else None


def _panel_state(raw: bytes) -> dict:
    message = json.loads(raw)
    if not isinstance(message, dict) or message.get("type") != "PanelState":
        raise ValueError("not a PanelState message")
    return message


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    try:
        bundle = load_config_bundle(resolve_config_paths(args.config, args.config_extra))
        config = DisplayConfig(bundle.data, args.sink)
    except ConfigError as exc:
        raise SystemExit(f"config error: {exc}") from exc
    for key, value in config.session_env.items():
        os.environ.setdefault(key, value)
    Gst.init(None)
    missing = missing_elements(config.sink)
    if missing:
        raise SystemExit(f"missing GStreamer elements: {', '.join(missing)}")
    if args.check:
        print(f"operator display config ok: port {config.port}, sink {config.sink!r}")
        return 0

    display = OperatorDisplay(config, args.settings)
    loop = GLib.MainLoop()
    bus = display.video.pipeline.get_bus()
    bus.add_signal_watch()

    failed = []

    def on_error(_bus, message) -> None:
        error, debug = message.parse_error()
        failed.append(error.message)
        log.error("video pipeline error: %s (%s)", error.message, debug)
        loop.quit()

    bus.connect("message::error", on_error)
    GLib.timeout_add(int(TICK_S * 1000), display.tick)
    for signum in (2, 15):  # SIGINT, SIGTERM
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signum, lambda: (loop.quit(), False)[1])
    if args.duration_s is not None:
        GLib.timeout_add(int(args.duration_s * 1000), lambda: (loop.quit(), False)[1])
    display.video.start()
    log.info("showing return video from port %d on %s", config.port, config.sink)
    try:
        loop.run()
    finally:
        display.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
