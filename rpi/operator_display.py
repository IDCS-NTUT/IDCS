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
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping

import gi
import zmq

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from common.live_config import runtime_bundle
from common.config import ConfigError, resolve_config_paths  # noqa: E402
from common.operator_commands import OperatorCommand  # noqa: E402
from rpi.display.config_menu import ConfigMenu, fire_control_enabled
from rpi.display.commands import CommandClient  # noqa: E402
from rpi.display.inputs import KeyboardInput, PanelNavigator  # noqa: E402
from rpi.display.menu import Action, Choice, Menu, NavEvent, Page, Settings  # noqa: E402
from rpi.display.roi_editor import RoiEditor, ROI_SETTINGS  # noqa: E402
from rpi.display.roi_panel import RoiPanelWorkflow  # noqa: E402
from rpi.display.render import StatusBar, TargetBox, render  # noqa: E402
from rpi.display.status import SystemStatus  # noqa: E402
from rpi.display.video import ReturnVideo, missing_elements  # noqa: E402
from rpi.display.return_identity import ReturnIdentityReceiver  # noqa: E402
from common.hud_geometry import resolve_hud_fov  # noqa: E402
from common.hud_telemetry import MAX_BYTES  # noqa: E402
from common.visual_roi_seed import VisualRoiSeed, VisualRoiRelease, VisualRoiSelect  # noqa: E402
from common.return_frame_identity import RoiSourceLease  # noqa: E402
from rpi.display.hud import PanelHudState, PiHudRenderer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TICK_S = 1 / 30
DEFAULT_SETTINGS_PATH = Path.home() / ".config" / "idcs" / "operator_display.json"
SETTINGS = ROI_SETTINGS  # Pi-local preferences; the safety strip cannot be hidden.

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
    parser.add_argument('--visual-roi-trial', action='store_true',
                        help='isolated SIM/camera display-only ROI trial; no automatic camera follow')
    return parser


class DisplayConfig:
    def __init__(self, cfg: Mapping[str, Any], sink_override: str | None) -> None:
        net = cfg.get("net") or {}
        display = (cfg.get("rpi") or {}).get("operator_display") or {}
        try:
            self.port = int(net["rtp_return_port"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError("net.rtp_return_port is required") from exc
        self.jitter_ms = int((cfg.get('video') or {}).get('return_jitter_ms',net.get('return_jitter_ms',20)))
        self.return_feedback_host = str(net.get('jetson_ip') or '') or None
        self.return_feedback_port = int(net.get('return_rtcp_feedback_port', 5003))
        self.return_sender_report_port = int(net.get('return_rtcp_sender_report_port', 5004))
        from common.return_rtp import validate_ports
        validate_ports(self.port,self.return_feedback_port,self.return_sender_report_port)
        self.perception = str(net.get("zmq_perception_v2") or "")
        self.diagnostics = str(net.get("zmq_control_diagnostics") or "")
        self.pose = str(net.get("zmq_gimbal_state") or "")
        self.hud_sim_bind = str(display.get("hud_sim_bind") or "")
        self.hud_fov = resolve_hud_fov(cfg, (1280, 720))
        self.panel = str(display.get("panel_state_endpoint") or "")
        self.command = str(net.get("zmq_operator_command") or "")
        self.return_frame_identity = str(net.get("zmq_return_frame_identity") or "")
        self.visual_roi_seed = str(net.get("zmq_visual_roi_seed") or "")
        self.visual_roi_perception = str(net.get("zmq_visual_roi_perception") or "")
        self.roi_lamp_endpoint = str(display.get("roi_lamp_endpoint") or "")
        self.roi_control_endpoint = str(display.get("roi_control_endpoint") or "")
        self.sink = sink_override or str(display.get("sink") or "waylandsink fullscreen=true sync=false")
        self.session_env = {str(k): str(v) for k, v in (display.get("session_env") or {}).items()
                            if v is not None}
        if not self.panel:
            raise ConfigError("rpi.operator_display.panel_state_endpoint is required")
        # Return-video recording follows the operator agent's Recording switch.
        raw = display.get("recording")
        self.recording = None
        if raw:
            try:
                self.recording = dict(dir=str(raw["dir"]), segment_s=int(raw.get("segment_s", 60)),
                                      max_total_gb=float(raw.get("max_total_gb", 15)))
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigError("rpi.operator_display.recording needs dir, segment_s, max_total_gb") from exc
            if not 10 <= self.recording["segment_s"] <= 600 or not 1 <= self.recording["max_total_gb"] <= 100:
                raise ConfigError("rpi.operator_display.recording: segment_s 10..600, max_total_gb 1..100")


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
               send: Callable[[OperatorCommand], None] = lambda _command: None,
               hud_details: Callable[[], list[str]] = lambda: ['No HUD telemetry'],
               roi_preview: Callable[[], None] = lambda: None,
               visual_roi_trial: bool = False,
               roi_available: Callable[[], bool] = lambda: False,
               roi_release: Callable[[], None] = lambda: None) -> Menu:
    version = _version()

    def visual_mode() -> bool:
        state = status.agent_state
        return (state.get('mode') in ('camera_visual_lock','sim_visual_lock')
                or state.get('roi_capable') is True and state.get('visual_lock') is True)

    def seamless_roi() -> bool:
        state = status.agent_state
        return state.get('mode') in ('camera','sim') and state.get('roi_capable') is True

    def picker_items() -> list[Action]:
        if visual_mode():
            return [Action('Select box',roi_preview), Action('Release',roi_release)]
        items = []
        envelope = status.agent_state.get("target_catalog") or {}
        catalog = envelope.get("catalog") or {}
        rows = sorted(status.target_rows(time.monotonic()),
                      key=lambda row: (row.track_id != status.agent_state.get('lock_track_id'),
                                       float('inf') if row.distance_m is None else row.distance_m,
                                       row.track_id))
        for row in rows:
            if row.missed_frames or row.confidence <= 0:
                continue
            distance = f"  {row.distance_m:.1f} m" if row.distance_m is not None else ""
            value = ("LOCKED" if catalog.get("policy") == "operator_lock" or status.locked else "selected") if row.selected else None
            items.append(Action(f"{row.display_name(len(rows)>1)} {row.confidence:.2f}{distance}",
                                lambda tid=row.track_id, src=catalog.get("source"), session=catalog.get("session"):
                                    send(OperatorCommand("lock", track_id=tid, source=src, source_session=session)),
                                key=("track", row.track_id), value=value))
        return items

    def target_items() -> list[Page | Action | Choice]:
        if visual_mode() or visual_roi_trial:
            return [Action('Select box',roi_preview), Action('Release',roi_release),
                    Choice('Box size','roi_box_size')]
        planner = status.agent_state.get('selection_mode') != 'operator'
        roi_items = ([Action('Select box',roi_preview), Action('Release',roi_release),
                      Choice('Box size','roi_box_size')] if seamless_roi() else [])
        return roi_items + [Action('Planner selection',
                       lambda: send(OperatorCommand('selection_mode',
                           mode='operator' if status.agent_state.get('selection_mode') != 'operator' else 'planner')),
                       value='on' if planner else 'off'),
                Page('Eligible classes', items=class_items)]

    def sim_scene_items() -> list[Action]:
        state = status.agent_state
        items = [Action(scene['label'],
                       lambda name=scene['name']: send(OperatorCommand('set_mode',mode='sim',scene=name)),
                       confirm=f"Switch to Simulator / {scene['label']}?",
                       key=('sim_scene',scene['name']),
                       value='current' if state.get('mode')=='sim' and state.get('sim_scene')==scene['name'] else None)
                for scene in state.get('sim_scenes') or ()
                if isinstance(scene,dict) and scene.get('name') in ('single','swarm') and 'label' in scene]
        if state.get('sim_visual_lock'):
            items.append(Action('Visual lock',lambda: send(OperatorCommand('set_mode',mode='sim_visual_lock')),
                confirm='Switch to Simulator / Visual lock?',key=('mode','sim_visual_lock'),
                value='current' if state.get('mode') == 'sim_visual_lock' else None))
        return items

    def mode_items() -> list[Page | Action]:
        state = status.agent_state
        current = state.get("mode")
        return [Page(m['label'],items=sim_scene_items) if m['name']=='sim' and (state.get('sim_scenes') or state.get('sim_visual_lock')) else
                Action(m["label"], lambda name=m["name"]: send(OperatorCommand("set_mode", mode=name)),
                       confirm=f"Switch to {m['label']}?", key=("mode", m["name"]),
                       value="current" if m["name"] == current else None)
                for m in state.get("modes") or () if isinstance(m, dict) and "name" in m and m["name"] != "sim_visual_lock"]

    def class_items() -> list[Action]:
        state = status.agent_state
        names = [str(n) for n in state.get("class_names") or ()]
        current = sorted(str(c) for c in state.get("target_classes") or ())
        options = [[n] for n in names] + ([names] if len(names) > 1 else [])
        return [Action(" + ".join(o), lambda o=o: send(OperatorCommand("target_classes", classes=tuple(o))),
                       confirm=f"Target {' + '.join(o)}?", value="current" if sorted(o) == current else None)
                for o in options]

    def operation_items() -> list[Page | Action]:
        recording = bool(status.agent_state.get("recording"))
        items = [
            Page("Source & scene", items=mode_items),
            Action("Recording", lambda: send(OperatorCommand("recording", on=not recording)),
                   confirm="Stop recording?" if recording else "Start recording?",
                   value="on" if recording else "off"),
        ]
        state = status.agent_state
        if state.get('mode') == 'sim' and state.get('sim_scene') == 'swarm' and not visual_mode():
            items.append(Action('Clear Swarm score',
                lambda session=state.get('sim_hud_session'): send(OperatorCommand('clear_swarm_score',source_session=session)),
                confirm='Clear Swarm score? Targets and recordings are kept.', key=('clear_swarm_score',)))
        if state.get('mode') == 'sim' and not seamless_roi() or (visual_roi_trial and roi_available()):
            items.append(Action('Visual ROI tracking (isolated)' if visual_roi_trial else 'ROI preview (no tracking)', roi_preview,
                                key=('roi_preview',)))
        return items

    def system_lines() -> list[str]:
        now = time.monotonic()
        state = status.agent_state
        if not status.agent.last_s:
            return ["Operator agent not answering."]
        lines = [
            *([f"Transition  {state['busy']}"] if state.get('busy') else []),
            f"PC agent    {'online' if state.get('pc_online') else 'offline'}",
        ]
        if state.get("pc_action_pending"):
            lines.append("PC acknowledgement pending")
        mode = next((m.get('label') for m in state.get('modes') or ()
                     if isinstance(m,dict) and m.get('name')==state.get('mode')),None)
        if state.get('mode')=='sim' and state.get('sim_scene')=='swarm':
            mode = f'{mode} / Swarm'
        if state.get("message") and state['message'] != f'mode {mode} ready':
            lines.append(str(state["message"]))
        if state.get("motors_live"):
            lines.append("Motor stack running: mode switching off")
        if not status._agent_recent(now):
            lines.append(f"(agent silent {status.agent.age_s(now):.0f} s)")
        return lines

    def connection_lines() -> list[str]:
        now = time.monotonic()
        return status.link_lines(now)

    def control_lines() -> list[str]:
        now = time.monotonic()
        return [f"Controller  {status.controller_reason if status.controller.live(now) else '--'}",
                "", *hud_details()]

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
        Page("Targets", items=target_items),
        Page("Operations", items=operation_items),
        Page("Information", items=(
            Page("System", lines=system_lines, columns=2),
            Page("Connections", lines=connection_lines, columns=2),
            Page("Control", lines=control_lines, columns=2),
            Page("Panel", lines=lambda: status.panel_lines(time.monotonic()), columns=2),
            Page("About", lines=about_lines, columns=2),
        )),
    ))
    return Menu(root, settings, picker_factory=lambda: Page("Select target", items=tuple(picker_items())))


class OperatorDisplay:
    def __init__(self, config: DisplayConfig, settings_path: Path, report_path: Path | None = None,
                 visual_roi_trial: bool = False) -> None:
        self.config = config
        self.visual_roi_trial = visual_roi_trial
        if visual_roi_trial and (not config.visual_roi_seed or not config.return_frame_identity
                                 or not config.roi_lamp_endpoint or not config.roi_control_endpoint
                                 or not config.visual_roi_perception):
            raise ConfigError('visual ROI trial requires seed, return identity, lamp and SIM lease endpoints')
        self.status = SystemStatus()
        self.settings = Settings(SETTINGS, settings_path, log)
        self.ctx = zmq.Context()
        self.commands = CommandClient(self.ctx, config.command, status_period_s=.2) if config.command else None
        self.roi_editor = RoiEditor()
        self.roi_workflow = RoiPanelWorkflow()
        self.roi_source = RoiSourceLease()
        self._roi_request_session: str | None = None
        self._roi_manual_seen = False
        self.menu = build_menu(self.status, self.settings, config, self._send,
                               self._hud_details, self._start_roi_preview, visual_roi_trial,
                               lambda: self._sim_roi_source_available(time.monotonic()), self._menu_release_roi)
        self.config_menu = ConfigMenu(self._send_config,
            lambda: fire_control_enabled(self.status, time.monotonic()))
        self.menu.config_menu = self.config_menu
        self.menu.root = Page(self.menu.root.title, items=(*self.menu.root.current_items(),
            Action("Settings", self.config_menu.open, value=">")))
        self.panel_nav = PanelNavigator()
        self.keyboard = KeyboardInput()
        self._agent_message: str | None = None
        self.perception = _subscriber(self.ctx, config.perception)
        self.roi_perception = (_subscriber(self.ctx, config.visual_roi_perception)
                               if visual_roi_trial else None)
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
        self.identity_receiver = None
        self.roi_seed = None
        self.roi_lamp_pub = None
        self.roi_control_pub = None
        self._pending_roi_request = None
        self._pending_roi_started_s = 0.0
        self._roi_select = None
        self._roi_select_sent_s = 0.0
        self._roi_open_cache = False
        if config.visual_roi_seed and config.roi_lamp_endpoint and config.roi_control_endpoint:
            self.roi_control_pub = self.ctx.socket(zmq.PUB)
            self.roi_control_pub.setsockopt(zmq.LINGER, 0)
            self.roi_control_pub.setsockopt(zmq.SNDHWM, 1)
            self.roi_control_pub.bind(config.roi_control_endpoint)
            self.roi_lamp_pub = self.ctx.socket(zmq.PUB)
            self.roi_lamp_pub.setsockopt(zmq.LINGER, 0)
            self.roi_lamp_pub.setsockopt(zmq.SNDHWM, 1)
            self.roi_lamp_pub.bind(config.roi_lamp_endpoint)
            self.roi_seed = self.ctx.socket(zmq.PUSH)
            self.roi_seed.setsockopt(zmq.LINGER, 0)
            self.roi_seed.setsockopt(zmq.SNDHWM, 1)
            self.roi_seed.setsockopt(zmq.IMMEDIATE, 1)
            self.roi_seed.connect(config.visual_roi_seed)
        self.video = ReturnVideo(port=config.port, jitter_ms=config.jitter_ms, sink=config.sink,
                                 on_frame=lambda: self.status.on_video_frame(time.monotonic()),
                                 feedback_host=config.return_feedback_host,
                                 feedback_port=config.return_feedback_port,
                                 sender_report_port=config.return_sender_report_port,
                                 recording=config.recording)
        if config.return_frame_identity:
            self.identity_receiver = ReturnIdentityReceiver(config.return_frame_identity,self.video.frame_identity)
        self._drawn_key: object = None
        self._next_rescan_s = 0.0
        self._next_log_s = time.monotonic() + 10.0
        self.report_path = report_path
        self.render_ms = deque(maxlen=300)
        self.hud_messages = 0
        self.last_hud_source = 'UNAVAILABLE'

    @property
    def roi_tracking(self) -> bool:
        state = self.status.agent_state
        return (self.visual_roi_trial or state.get('mode') in ('camera_visual_lock','sim_visual_lock')
                or state.get('mode') in ('camera','sim') and state.get('roi_capable') is True)

    def _roi_terminal(self, request_id: str, reason: str | None, now: float) -> None:
        if self.roi_workflow.result(request_id,False):
            self._cancel_roi_select()
            self._pending_roi_request = None
            self._roi_request_session = None
            self.status.show_notice(f'ROI ended: {reason or "lost"}; GPIO12 OFF selects again',now)

    def _roi_sim_lease(self, now: float) -> dict:
        panel = self.status.panel_state
        mode = self.status.agent_state.get('mode')
        enabled = (not self.visual_roi_trial and mode in ('sim','sim_visual_lock')
            and self.roi_workflow.state == 'tracking' and self.roi_workflow.request_id is not None
            and self._sim_roi_source_available(now) and self.status.panel.live(now) and self.status.video.live(now)
            and panel.get('manual_active') is True and panel.get('emergency') is False
            and (panel.get('inputs_valid') or {}).get('fire_control') is True
            and (panel.get('inputs') or {}).get('fire_control') is True)
        return dict(type='roi_sim_lease_v1',enabled=bool(enabled),managed_mode=mode,
                    request_id=self.roi_workflow.request_id if enabled else None,
                    tracker_session=self._roi_request_session if enabled else None)

    def _cancel_roi_select(self) -> None:
        previous = getattr(self, '_roi_select', None)
        self._roi_select = None
        if previous is not None and self.roi_seed is not None:
            try:
                self.roi_seed.send_json(VisualRoiSelect(*previous, False).as_message(), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass  # Receiver's 500ms lease also bounds lost cancellation.

    def _publish_roi_select(self, now: float) -> None:
        """Renew workflow liveness; only selecting permits cache copies."""
        if self.roi_seed is None:
            return
        # Normal mode starts its exact-identity probes on this lease. Fresh
        # source status can open the editor; only _request_roi may seed it.
        session = self.roi_source.session
        enabled = (not self.visual_roi_trial
            and self.status.agent_state.get('mode') in ('camera', 'sim')
            and self.roi_workflow.active and self.roi_editor.active
            and (self.roi_workflow.state in ('selecting', 'pending', 'tracking')
                 or self.roi_workflow.state == 'camera' and self._roi_open_cache)
            and session is not None and self._sim_roi_source_available(now)
            and self.status.video.live(now) and self.status.panel.live(now)
            and self.status.panel_state.get('emergency') is False)
        previous = self._roi_select
        if previous is not None and (not enabled or session != previous[1]):
            self._cancel_roi_select()
        if not enabled:
            return
        if self._roi_select is None:
            self._roi_select = (str(uuid.uuid4()), session)
            self._roi_select_sent_s = 0.0
        if now-self._roi_select_sent_s >= .1:
            try:
                self.roi_seed.send_json(VisualRoiSelect(*self._roi_select, True).as_message(), flags=zmq.NOBLOCK)
                self._roi_select_sent_s = now
            except zmq.Again:
                pass

    def _menu_release_roi(self) -> None:
        request = self.roi_workflow.disarm()
        self._release_roi(request)
        self.status.show_notice('ROI released; GPIO12 OFF selects again',time.monotonic())

    def tick(self) -> bool:
        now = time.monotonic()
        self._read_messages(now)
        if self.roi_workflow.active and not self.visual_roi_trial and (not self.roi_tracking or not self._sim_roi_source_available(now)):
            request = self.roi_workflow.request_id
            self.roi_workflow.close()
            self._release_roi(request)
            self.roi_editor.active = False
            self.status.show_notice('ROI closed: source mode changed or stale',now)
        if self.roi_lamp_pub is not None:
            panel = self.status.panel_state
            available = (self.roi_editor.active and self._sim_roi_source_available(now)
                         and self.status.video.live(now)
                         and self.status.panel.live(now)
                         and panel.get('manual_active') is True
                         and panel.get('emergency') is False)
            try:
                self.roi_lamp_pub.send_json(dict(type='roi_lamp_v1', available=bool(available),
                                                 roi_active=bool(self.roi_editor.active)),
                                            flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
        if self.roi_tracking and self.roi_editor.active:
            self._update_roi_panel(now)
        self._publish_roi_select(now)
        if self.roi_control_pub is not None:
            try:
                self.roi_control_pub.send_json(self._roi_sim_lease(now), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
        if self._pending_roi_request is not None and now - self._pending_roi_started_s > 2.0:
            request_id = self.roi_workflow.disarm()
            self._release_roi(request_id)
            self.status.show_notice('ROI response timed out; tracking status unknown', now)
        if self.roi_editor.active and not self.roi_tracking and not self._roi_preview_allowed(now):
            self.roi_editor.active = False
            self.status.show_notice('ROI preview closed: source or panel unavailable', now)
        if now >= self._next_rescan_s:
            self.keyboard.rescan()
            self._next_rescan_s = now + 2.0
        for event in self.keyboard.poll(text_entry=self.config_menu.active and self.config_menu.screen == "edit"):
            self._handle_navigation(event)
        if self.commands is not None:
            for reply in self.commands.poll(now):
                self._on_reply(reply, now)
        self.config_menu.tick()
        self.video.show_live(self.status.video.live(now))
        if self.video.recorder is not None and self.status.agent_state:
            # Last known agent state: a briefly silent agent does not cut footage.
            self.video.recorder.set_recording(bool(self.status.agent_state.get("recording")))
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
                    timing_qualification='exact ROI return-frame join; photon/transit latency unqualified')
                frame = self.video.frame_identity.latest()
                report['return_frame_identity'] = dict(matched=self.video.frame_identity.matched,
                    unmatched=self.video.frame_identity.unmatched,
                    late_matched=self.video.frame_identity.late_matched,
                    failure_reasons=dict(self.video.frame_identity.failure_reasons),
                    latest=None if frame is None else dict(tracker_session=frame.tracker_session,
                        frame_id=frame.frame_id,source_time_ns=frame.source_time_ns,
                        source_clock_domain=frame.source_clock_domain))
                report['return_receiver'] = self.video.receiver_stats()
                report['recording'] = None if self.video.recorder is None else self.video.recorder.report()
                report['roi'] = dict(enabled=self.roi_tracking,active=self.roi_editor.active,
                    state=self.roi_workflow.state,source_available=self._sim_roi_source_available(now),
                    request_id=self.roi_workflow.request_id)
                if self.identity_receiver is not None:
                    report['return_identity_receiver'] = dict(headers=self.identity_receiver.headers,
                        invalid=self.identity_receiver.invalid,result_overflow=self.identity_receiver.result_overflow,
                        error=self.identity_receiver.error)
                temporary = self.report_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(report, indent=2))
                temporary.replace(self.report_path)
        return True

    def _read_messages(self, now: float) -> None:
        if self.identity_receiver is not None:
            for identity,received_s in self.identity_receiver.drain():
                if not 0 <= time.monotonic()-received_s <= .5:
                    continue
                try:
                    if identity.get('type') == 'visual_roi_status_v1':
                        mode = self.status.agent_state.get('mode') if not self.visual_roi_trial else None
                        if not self.visual_roi_trial and not self.status._agent_recent(now):
                            mode = None
                        if self.roi_source.observe(identity, received_s, managed_mode=mode):
                            if (identity.get('terminal_request_id') == self.roi_workflow.request_id
                                    and self.roi_workflow.request_id is not None):
                                self._roi_terminal(identity['terminal_request_id'],identity.get('reason'),now)
                    elif identity.get('type') == 'visual_roi_result_v1':
                        if identity.get('accepted') is False and identity.get('request_id') == self.roi_workflow.request_id:
                            self._roi_terminal(identity['request_id'],identity.get('reason'),now)
                        elif identity.get('request_id') == self._pending_roi_request:
                            self._pending_roi_request = None
                            accepted = identity.get('accepted') is True
                            if self.roi_workflow.result(identity['request_id'], accepted):
                                notice = ('ROI locked' if accepted else
                                          f"ROI refused: {identity.get('reason', 'no reason')}")
                                if not accepted and 'texture' in str(identity.get('reason', '')):
                                    notice += '; OFF then Select/Enter to resize'
                                self.status.show_notice(notice, now)
                except (ValueError, TypeError):
                    pass
        raw = _latest(self.perception)
        roi_raw = _latest(self.roi_perception)
        if self.visual_roi_trial and self._sim_roi_source_available(now):
            raw = roi_raw
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
                if (panel.get("manual_active") or panel.get("emergency")) and not self.roi_tracking:
                    self.roi_editor.active = False
                if self.menu.picker_active and (panel.get("manual_active") or panel.get("emergency")):
                    self.menu._close()
                for event in self.panel_nav.update(panel, now, roi_editing=self.roi_editor.active):
                    self._handle_navigation(event)

    def _start_roi_preview(self) -> None:
        now = time.monotonic()
        if not self._roi_preview_allowed(now):
            self.status.show_notice('ROI preview needs live source video and panel', now)
            return
        normal_roi = not self.visual_roi_trial and self.status.agent_state.get('mode') in ('camera', 'sim')
        if self.roi_tracking and not normal_roi and self.video.frame_identity.latest() is None:
            self.status.show_notice('ROI refused: exact return frame identity unavailable', now)
            return
        self.menu._close()
        self.roi_editor.open(*self.video.frame_size, tracking_trial=self.roi_tracking,
                             direct_select=self.roi_tracking, box_size=self.settings["roi_box_size"])
        if self.roi_tracking:
            self.roi_workflow.open()
            self._roi_open_cache = True
            self._publish_roi_select(now)
            self._roi_manual_seen = False
            if self.roi_lamp_pub is not None:
                try:
                    self.roi_lamp_pub.send_json(dict(type='roi_lamp_v1', available=False,
                                                     roi_active=True), flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
            self.status.show_notice('ROI: Manual ON; OFF edits, Select/Enter move/size, ON locks', now)

    def _release_roi(self, request_id: str | None = None) -> None:
        self._roi_open_cache = False
        request_id = request_id or self.roi_workflow.request_id
        session = self._roi_request_session
        if request_id is not None and session is not None and self.roi_seed is not None:
            try:
                self.roi_seed.send_json(VisualRoiRelease(request_id, session).as_message(),
                                        flags=zmq.NOBLOCK)
            except zmq.Again:
                self.status.show_notice('ROI release transport full; tracking status unknown', time.monotonic())
        self._cancel_roi_select()
        self._pending_roi_request = None
        self._roi_request_session = None

    def _update_roi_panel(self, now: float) -> None:
        panel = self.status.panel_state
        if panel.get('manual_active') is True:
            self._roi_manual_seen = True
        elif panel.get('manual_active') is False and getattr(self,'_roi_manual_seen',False):
            request_id = self.roi_workflow.request_id
            self.roi_workflow.close()
            self._release_roi(request_id)
            self.roi_editor.active = False
            self.menu._close()
            self.status.show_notice('ROI closed: Manual off', now)
            return
        available = (self._sim_roi_source_available(now)
                     and self.status.video.live(now) and self.status.panel.live(now))
        request_id = self.roi_workflow.request_id
        action = self.roi_workflow.update(panel, available=available)
        if action == 'release':
            self._release_roi(request_id)
            self.status.show_notice('ROI released; fire-control OFF selects again', now)
        elif action == 'lock':
            self._request_roi(now)
        if (available and panel.get('manual_active') is True
                and (panel.get('inputs') or {}).get('fire_control') is False
                and self.roi_workflow.state == 'selecting'):
            x, y = panel.get('joystick') or (0.0, 0.0)
            self.roi_editor.move_joystick(float(x), float(y), now)
        else:
            self.roi_editor._last_move_s = None

    def _request_roi(self, now: float) -> None:
        frame = self.video.frame_identity.latest()
        if frame is None or not self.roi_source.live(time.monotonic(), frame):
            self.status.show_notice('ROI refused: displayed frame identity unavailable', now)
            return
        try:
            x,y,w,h = self.roi_editor.x,self.roi_editor.y,self.roi_editor.w,self.roi_editor.h
            width,height = self.video.frame_size
            request = VisualRoiSeed.from_mapping(dict(type='visual_roi_seed_v1',
                request_id=str(uuid.uuid4()),tracker_session=frame.tracker_session,
                frame_id=frame.frame_id,source_time_ns=frame.source_time_ns,
                source_clock_domain=frame.source_clock_domain,
                x=x,y=y,w=w,h=h,width=width,height=height))
            if self._pending_roi_request is not None or self.roi_seed is None:
                raise ValueError('ROI already pending or trial transport unavailable')
            self.roi_seed.send_json(request.as_message(),flags=zmq.NOBLOCK)
        except (ValueError,zmq.Again) as exc:
            self.status.show_notice(f'ROI refused: {exc}', now)
        else:
            self.roi_workflow.submitted(request.request_id)
            self._roi_open_cache = False
            self._roi_request_session = request.tracker_session
            self._pending_roi_request = request.request_id
            self._pending_roi_started_s = now
            self.status.show_notice('ROI lock requested', now)

    def _roi_preview_allowed(self, now: float) -> bool:
        panel = self.status.panel_state
        # The joystick can open the menu before Manual is selected, provided
        # auto control is already off. Manual may then be selected on the panel.
        permitted_panel = (panel.get('manual_active') is True or
                           panel.get('control_cmd_enabled') is False) if self.roi_tracking else (
                               panel.get('manual_active') is False)
        return (self._sim_roi_source_available(now)
                and self.status.video.live(now)
                and self.status.panel.live(now)
                and permitted_panel
                and not panel.get('emergency'))

    def _sim_roi_source_available(self, now: float) -> bool:
        if self.roi_tracking:
            now = time.monotonic()
            if not self.visual_roi_trial:
                state = self.status.agent_state
                if (not self.status._agent_recent(now) or state.get('busy')
                        or self.roi_source.source_mode != state.get('mode','').removesuffix('_visual_lock')):
                    return False
            # Exact displayed identity is required to START a lock. Once a
            # request names that session, a temporarily late image sideband
            # must not release it while the same verified receiver stays live.
            if self._roi_request_session is not None:
                return self.roi_source.live_session(now,self._roi_request_session)
            if not self.visual_roi_trial and self.status.agent_state.get('mode') in ('camera', 'sim'):
                return self.roi_source.live_session(now, self.roi_source.session)
            frame = self.video.frame_identity.latest()
            return self.roi_source.live(time.monotonic(), frame)
        state = self.status.agent_state
        return (self.status._agent_recent(now) and state.get('mode') == 'sim'
                and not state.get('busy'))

    def _handle_navigation(self, event) -> None:
        if self.roi_editor.active:
            if self.roi_tracking:
                if event is NavEvent.SELECT and self.roi_workflow.state != 'selecting':
                    return
                if self.roi_editor.handle(event) == 'cancel':
                    self._release_roi()
                    self.roi_workflow.close()
                return
            result = self.roi_editor.handle(event)
            if result == 'confirm':
                self.status.show_notice('ROI preview only; tracker not connected', time.monotonic())
            return
        self.menu.handle(event)

    def _send(self, command: OperatorCommand) -> None:
        if command.command == 'set_mode' and self.roi_workflow.active:
            request = self.roi_workflow.request_id
            self._release_roi(request)
            self.roi_workflow.close()
            self.roi_editor.active = False
        if self.commands is None:
            self.status.show_notice("No operator agent configured", time.monotonic())
            return
        self.commands.submit(command)

    def _send_config(self, request: dict) -> bool:
        return self.commands is not None and self.commands.submit_config(
            request, allowed=lambda: self.config_menu.request_allowed(request))

    def _on_reply(self, reply, now: float) -> None:
        if getattr(reply, "config", None) is not None:
            self.config_menu.on_reply(reply.config)
            return
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
        # Applied selection is already represented by Jetson's accepted OSD.
        # Only an uncommitted menu cursor may add a preview box.
        item = self.menu.highlighted()
        key = getattr(item, "key", None)
        if isinstance(key, tuple) and key[0] == "track" and key[1] in rows:
            row = rows[key[1]]
            if row.box is not None and not row.selected:
                boxes.append(TargetBox(row.box, row.display_name(len(rows)>1), "cursor"))
        if self.roi_editor.active and (not self.roi_tracking or
                (self.roi_workflow.state == 'selecting' and
                 self.status.panel_state.get('manual_active') is True and
                 (self.status.panel_state.get('inputs') or {}).get('fire_control') is False)):
            boxes.append(TargetBox(self.roi_editor.box(),
                                   ('ROI SIZE' if self.roi_editor.resizing else 'ROI SELECT')
                                   if self.roi_tracking else 'ROI PREVIEW', 'cursor'))
        return tuple(boxes)

    def _draw(self, now: float) -> None:
        hud_view = self.hud_state.view(now, mode=self.status.agent_state.get('mode'),
            source_session=self.status.hud_source_session(), agent_fresh=self.status._agent_recent(now),
            video_live=self.status.video.live(now))
        self.last_hud_source = hud_view.source
        state = self.status.agent_state
        roi_view = (self.visual_roi_trial or state.get('mode') in ('camera_visual_lock','sim_visual_lock')
                    or self.roi_editor.active or state.get('visual_lock') is True)
        roi_return = self.roi_tracking and roi_view and self._sim_roi_source_available(now)
        if roi_return:
            self.last_hud_source = 'ISOLATED ROI' if self.visual_roi_trial else 'VISUAL LOCK'
        view = self.roi_editor.view() if self.roi_editor.active else self.menu.view()
        information_open = getattr(view, 'kind', None) == 'info'
        # Information pages replace only the ordinary HUD. Keep source,
        # authority, recording, fire and E-stop in the foreground receiver
        # layer, even if the diagnostic panel occupies much of the frame.
        bar = StatusBar(self.status.panel_mode(now),
                        (None if roi_return and self.visual_roi_trial else self.status.agent_text(now, include_source=information_open),
                         *([f'ROI {self.roi_workflow.state.upper()}']
                           if self.roi_tracking and self.roi_editor.active
                           and self.roi_workflow.state in ('selecting','pending') else [])))
        alerts = self.status.alerts(now)
        fire_state = self.status.fire_text(now)
        boxes = self._boxes()
        notice = self.status.current_notice(now)
        key = (self.video.frame_size, bar, tuple(alerts), fire_state, view, boxes, notice,
               None if information_open or roi_return else hud_view, roi_return)
        if key == self._drawn_key:
            return
        self._drawn_key = key
        began = time.perf_counter_ns()
        width, height = self.video.frame_size
        images = [] if information_open or roi_return else self.hud_renderer.render(width, height, hud_view)
        if roi_return:
            # The alternative Jetson owner already burns in ROI geometry,
            # reticle and quality. Pi retains panel/menu/E-stop/fire only.
            self.hud_renderer.elements = ('gpu_return_overlay',)
            self.hud_renderer.feedforward = 'disabled'
        images.extend(render(width, height, bar=bar, alerts=alerts, menu=view,
                             boxes=boxes, notice=notice, fire_state=fire_state,
                             safety_state=self.status.estop_text(now)))
        self.video.set_overlay(images)
        self.render_ms.append((time.perf_counter_ns() - began) / 1e6)

    def _hud_details(self) -> list[str]:
        now = time.monotonic()
        catalog = ((self.status.agent_state.get('target_catalog') or {}).get('catalog') or {})
        view = self.hud_state.view(now, mode=self.status.agent_state.get('mode'),
            source_session=catalog.get('session',''),agent_fresh=self.status._agent_recent(now),
            video_live=self.status.video.live(now))
        from common.ui_style import hud_detail_lines
        return hud_detail_lines(view)

    def close(self) -> None:
        if self.identity_receiver is not None:
            self.identity_receiver.close()
        if self.roi_workflow.request_id is not None:
            self._release_roi()
        if self.roi_lamp_pub is not None:
            self.roi_lamp_pub.close(0)
        if self.roi_control_pub is not None:
            self.roi_control_pub.close(0)
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
        bundle = runtime_bundle(resolve_config_paths(args.config, args.config_extra))
        config = DisplayConfig(bundle.data, args.sink)
    except ConfigError as exc:
        raise SystemExit(f"config error: {exc}") from exc
    log.info("configuration %s pending_config_keys=[]", bundle.digest)
    for key, value in config.session_env.items():
        os.environ.setdefault(key, value)
    Gst.init(None)
    missing = missing_elements(config.sink, recording=config.recording is not None)
    if missing:
        raise SystemExit(f"missing GStreamer elements: {', '.join(missing)}")
    if args.check:
        if args.visual_roi_trial and (not config.visual_roi_seed or not config.return_frame_identity
                                      or not config.roi_lamp_endpoint or not config.roi_control_endpoint
                                      or not config.visual_roi_perception):
            raise SystemExit('visual ROI trial needs seed, return identity, lamp and SIM lease endpoints')
        print(f"operator display config ok: port {config.port}, sink {config.sink!r}")
        return 0

    display = OperatorDisplay(config, args.settings, args.report, args.visual_roi_trial)
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
