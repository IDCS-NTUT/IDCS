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
from collections import deque
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
from common.hud_geometry import resolve_hud_fov  # noqa: E402
from common.hud_telemetry import MAX_BYTES  # noqa: E402
from rpi.display.hud import PanelHudState, PiHudRenderer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TICK_S = 1 / 30
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
    parser.add_argument('--report', type=Path, help='periodic read-only display metrics outside source tree')
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
        self.pose = str(net.get("zmq_gimbal_state") or "")
        self.hud_sim_bind = str(display.get("hud_sim_bind") or "")
        self.hud_fov = resolve_hud_fov(cfg, (1280, 720))
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
        items = [Action("Use planner", lambda: send(OperatorCommand("selection_mode", mode="planner")),
                        value="current" if status.agent_state.get("selection_mode") == "planner" else None)]
        envelope = status.agent_state.get("target_catalog") or {}
        catalog = envelope.get("catalog") or {}
        for row in status.target_rows(time.monotonic()):
            if row.missed_frames or row.confidence <= 0:
                continue
            distance = f"  {row.distance_m:.0f} m" if row.distance_m is not None else ""
            value = ("LOCKED" if catalog.get("policy") == "operator_lock" or status.locked else "selected") if row.selected else None
            items.append(Action(f"#{row.track_id} {row.class_id} {row.confidence:.2f}{distance}",
                                lambda tid=row.track_id, src=catalog.get("source"), session=catalog.get("session"):
                                    send(OperatorCommand("lock", track_id=tid, source=src, source_session=session)),
                                key=("track", row.track_id), value=value))
        return items

    def target_hint() -> list[str]:
        return ["Browse only; right confirms, left cancels."] + (
            [] if status.target_rows(time.monotonic()) else ["No fresh control-source targets."])

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
            f"PC agent    {'online' if state.get('pc_online') else 'offline'}",
            f"Targets     {', '.join(state.get('target_classes') or ()) or '-'}",
            f"Lock        {('#' + str(state['lock_track_id'])) if state.get('lock_track_id') is not None else '-'}",
        ]
        if state.get("pc_action_pending"):
            lines.append("PC acknowledgement pending")
        if state.get("message"):
            lines.append(str(state["message"]))
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
    return Menu(root, settings, picker_factory=lambda: Page("Select target", items=tuple(track_items()), lines=target_hint))


class OperatorDisplay:
    def __init__(self, config: DisplayConfig, settings_path: Path, report_path: Path | None = None) -> None:
        self.config = config
        self.status = SystemStatus()
        self.settings = Settings(SETTINGS, settings_path, log)
        self.ctx = zmq.Context()
        self.commands = CommandClient(self.ctx, config.command, status_period_s=.2) if config.command else None
        self.menu = build_menu(self.status, self.settings, config, self._send)
        self.panel_nav = PanelNavigator()
        self.keyboard = KeyboardInput()
        self._agent_message: str | None = None
        self.perception = _subscriber(self.ctx, config.perception)
        self.diagnostics = _subscriber(self.ctx, config.diagnostics)
        self.pose = _subscriber(self.ctx, config.pose)
        self.hud_state = PanelHudState()
        self.hud_renderer = PiHudRenderer(hfov_deg=config.hud_fov[0], vfov_deg=config.hud_fov[1])
        self.hud_sim = None
        if config.hud_sim_bind:
            self.hud_sim = self.ctx.socket(zmq.PULL)
            self.hud_sim.setsockopt(zmq.LINGER, 0)
            self.hud_sim.setsockopt(zmq.CONFLATE, 1)
            self.hud_sim.setsockopt(zmq.MAXMSGSIZE, MAX_BYTES)
            self.hud_sim.bind(config.hud_sim_bind)
        self.panel = _subscriber(self.ctx, config.panel)
        self.video = ReturnVideo(port=config.port, jitter_ms=config.jitter_ms, sink=config.sink,
                                 on_frame=lambda: self.status.on_video_frame(time.monotonic()))
        self._drawn_key: object = None
        self._next_rescan_s = 0.0
        self._next_log_s = time.monotonic() + 10.0
        self.report_path = report_path
        self.render_ms = deque(maxlen=300)
        self.hud_messages = 0
        self.last_hud_source = 'UNAVAILABLE'

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
            log.info("video %.1f fps, perception %.1f Hz, panel %s, hud %s, FF %s",
                     self.status.video.rate_hz(now), self.status.perception.rate_hz(now),
                     self.status.panel_mode(now), ','.join(self.hud_renderer.elements), self.hud_renderer.feedforward)
            if self.report_path is not None:
                samples = sorted(self.render_ms)
                self.report_path.parent.mkdir(parents=True, exist_ok=True)
                report = dict(pid=os.getpid(), mono_ns=time.monotonic_ns(),
                    video_fps=self.status.video.rate_hz(now), perception_hz=self.status.perception.rate_hz(now),
                    hud_source=self.last_hud_source, hud_elements=self.hud_renderer.elements,
                    feedforward_state=self.hud_renderer.feedforward, hud_messages=self.hud_messages,
                    overlay_build_p95_ms=None if not samples else samples[min(len(samples) - 1, int(.95 * len(samples)))],
                    overlay_build_max_ms=max(samples) if samples else None,
                    timing_qualification='latest telemetry only; return frame join/transit unqualified')
                temporary = self.report_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(report, indent=2))
                temporary.replace(self.report_path)
        return True

    def _read_messages(self, now: float) -> None:
        raw = _latest(self.perception)
        if raw is not None:
            message = _decode(raw, "PerceptionSnapshot")
            if message is not None:
                self.status.on_snapshot(message, now)
                self.hud_state.on_snapshot(message, now)
        raw = _latest(self.diagnostics)
        if raw is not None:
            message = _decode(raw, "ControlDiagnostics")
            if message is not None:
                self.status.on_diagnostics(message, now)
                self.hud_state.on_camera('diagnostics', message, now)
        raw = _latest(self.pose)
        if raw is not None:
            message = _decode(raw, 'CamState')
            if message is not None:
                self.hud_state.on_camera('pose', message, now)
        raw = _latest(self.hud_sim)
        if raw is not None:
            try:
                if self.hud_state.sim.accept(raw, now):
                    self.hud_messages += 1
            except (ValueError, TypeError, KeyError) as exc:
                log.debug('bad read-only SIM HUD: %s', exc)
        raw = _latest(self.panel)
        if raw is not None:
            try:
                panel = _panel_state(raw)
            except ValueError as exc:
                log.debug("bad panel state: %s", exc)
            else:
                self.status.on_panel(panel, now)
                if self.menu.picker_active and (panel.get("manual_active") or panel.get("emergency")):
                    self.menu._close()
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
        rows = {row.track_id: row for row in self.status.target_rows(time.monotonic())}
        catalog = ((self.status.agent_state.get("target_catalog") or {}).get("catalog") or {})
        locked_id = self.status.agent_state.get("lock_track_id")
        locked_id = self.status.selected_id if "target_catalog" not in self.status.agent_state else locked_id
        applied_lock = (self.status.locked if "target_catalog" not in self.status.agent_state else
                        catalog.get("policy") == "operator_lock" and catalog.get("selected_id") == locked_id)
        if applied_lock and locked_id in rows:
            row = rows[locked_id]
            if row.box is not None:
                boxes.append(TargetBox(row.box, f"LOCK {catalog.get('source', '')} #{row.track_id}", "lock"))
        item = self.menu.highlighted()
        key = getattr(item, "key", None)
        if isinstance(key, tuple) and key[0] == "track" and key[1] in rows:
            row = rows[key[1]]
            if row.box is not None and not (self.status.locked and row.selected):
                boxes.append(TargetBox(row.box, f"#{row.track_id}", "cursor"))
        return tuple(boxes)

    def _draw(self, now: float) -> None:
        catalog = ((self.status.agent_state.get('target_catalog') or {}).get('catalog') or {})
        hud_view = self.hud_state.view(now, mode=self.status.agent_state.get('mode'),
            source_session=catalog.get('session', ''), agent_fresh=self.status._agent_recent(now),
            video_live=self.status.video.live(now))
        self.last_hud_source = hud_view.source
        bar = None
        if self.settings["status_bar"] == "on":
            fields: tuple[str, ...] = ()
            if self.settings["bar_detail"] == "full":
                fields = (
                    f"{self.status.video.rate_hz(now):.0f} fps",
                    self.status.targets_text(now),
                    'ctrl ' + (hud_view.diagnostics or {}).get('reason', '--'),
                    self.status.agent_text(now),
                )
            bar = StatusBar(self.status.panel_mode(now), fields)
        alerts = self.status.alerts(now)
        fire_state = self.status.fire_text(now)
        view = self.menu.view()
        boxes = self._boxes()
        notice = self.status.current_notice(now)
        key = (self.video.frame_size, bar, tuple(alerts), fire_state, view, boxes, notice, hud_view)
        if key == self._drawn_key:
            return
        self._drawn_key = key
        began = time.perf_counter_ns()
        width, height = self.video.frame_size
        images = self.hud_renderer.render(width, height, hud_view)
        images.extend(render(width, height, bar=bar, alerts=alerts, menu=view,
                             boxes=boxes, notice=notice, fire_state=fire_state))
        self.video.set_overlay(images)
        self.render_ms.append((time.perf_counter_ns() - began) / 1e6)

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

    display = OperatorDisplay(config, args.settings, args.report)
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
