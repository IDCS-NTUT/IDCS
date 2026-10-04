"""What the panel display knows about the system, and how fresh it is.

Each input is a *link* (video frames, perception snapshots, controller
diagnostics, panel state) with a last-seen time and a rate. A link is live
while its last message is younger than ``STALE_S``. Nothing here talks to a
socket: the app feeds messages in, the renderer and menu pages read out.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Mapping

STALE_S = 1.0


@dataclass
class Link:
    last_s: float | None = None
    count: int = 0
    _times: deque = field(default_factory=lambda: deque(maxlen=120))

    def mark(self, now_s: float) -> None:
        self.last_s = now_s
        self.count += 1
        self._times.append(now_s)

    def live(self, now_s: float) -> bool:
        return self.last_s is not None and now_s - self.last_s <= STALE_S

    def age_s(self, now_s: float) -> float | None:
        return None if self.last_s is None else now_s - self.last_s

    def rate_hz(self, now_s: float, window_s: float = 1.0) -> float:
        recent = [t for t in self._times if now_s - t <= window_s]
        if len(recent) < 2:
            return 0.0
        return (len(recent) - 1) / max(recent[-1] - recent[0], 1e-6)


@dataclass(frozen=True)
class TrackRow:
    track_id: int
    class_id: str
    confidence: float
    missed_frames: int
    selected: bool
    distance_m: float | None
    box: tuple[float, float, float, float] | None = None  # normalized x, y, w, h


@dataclass
class SystemStatus:
    video: Link = field(default_factory=Link)
    perception: Link = field(default_factory=Link)
    controller: Link = field(default_factory=Link)
    panel: Link = field(default_factory=Link)
    video_size: tuple[int, int] | None = None
    tracks: tuple[TrackRow, ...] = ()
    selected_id: int | None = None
    controller_reason: str | None = None
    panel_state: Mapping = field(default_factory=dict)
    selection_policy: str | None = None
    agent: Link = field(default_factory=Link)
    agent_state: Mapping = field(default_factory=dict)
    notice: str | None = None
    notice_until_s: float = 0.0
    engaged_track_id: int | None = None
    engaged_until_s: float = 0.0
    _engaged_ns: int | None = None

    def on_video_frame(self, now_s: float) -> None:
        self.video.mark(now_s)

    def on_snapshot(self, snapshot: Mapping, now_s: float) -> None:
        """A PerceptionSnapshotV2 as decoded JSON (only the fields shown are read).

        Plain JSON, not the validated model: full validation at 60 snapshots/s
        held the interpreter long enough on the Pi to starve the per-frame
        video callbacks (13 fps displayed of 30).
        """
        self.perception.mark(now_s)
        selection = snapshot.get("selection") or {}
        selected = selection.get("track_id")
        distance = {a.get("track_id"): a.get("distance_m") for a in snapshot.get("assessments") or ()}
        tracks = [t for t in snapshot.get("tracks") or () if isinstance(t, Mapping)]
        self.selected_id = selected
        self.selection_policy = selection.get("policy")
        self.tracks = tuple(
            TrackRow(int(t.get("track_id", -1)), str(t.get("class_id", "")), float(t.get("confidence", 0.0)),
                     int(t.get("missed_frames", 0)), t.get("track_id") == selected, distance.get(t.get("track_id")),
                     _box(t.get("box")))
            # By id, so a list cursor stays on the same track as tracks come and go.
            for t in sorted(tracks, key=lambda t: t.get("track_id", -1))
        )

    @property
    def locked(self) -> bool:
        return self.selection_policy == "operator_lock"

    def on_diagnostics(self, diagnostics: Mapping, now_s: float) -> None:
        """ControlDiagnostics as decoded JSON."""
        self.controller.mark(now_s)
        self.controller_reason = str(diagnostics.get("reason") or "?")
        engaged_ns = diagnostics.get("engaged_monotonic_ns")
        if engaged_ns is not None and engaged_ns != self._engaged_ns:
            if self._engaged_ns is not None or self.controller.count > 1:
                self.engaged_track_id = diagnostics.get("engaged_track_id")
                self.engaged_until_s = now_s + 3.0
            self._engaged_ns = engaged_ns

    def on_agent(self, state: Mapping | None, now_s: float) -> None:
        if state is not None:
            self.agent.mark(now_s)
            self.agent_state = dict(state)

    def show_notice(self, text: str, now_s: float, duration_s: float = 4.0) -> None:
        self.notice = text
        self.notice_until_s = now_s + duration_s

    def current_notice(self, now_s: float) -> str | None:
        return self.notice if self.notice and now_s < self.notice_until_s else None

    def on_panel(self, panel: Mapping, now_s: float) -> None:
        self.panel.mark(now_s)
        self.panel_state = dict(panel)

    # --- derived state for the display -------------------------------------

    def panel_mode(self, now_s: float) -> str:
        """E-STOP > SAFE (master arm off) > MANUAL > ARMED > STANDBY, or NO PANEL."""
        if not self.panel.live(now_s):
            return "NO PANEL"
        panel = self.panel_state
        if panel.get("emergency"):
            return "E-STOP"
        if not panel.get("master_arm"):
            return "SAFE"
        if panel.get("manual_active"):
            return "MANUAL"
        if panel.get("control_cmd_enabled"):
            return "ARMED"
        return "STANDBY"

    def alerts(self, now_s: float) -> list[str]:
        """Conditions the operator must see without opening the menu."""
        alerts = []
        if self.panel.live(now_s) and self.panel_state.get("emergency"):
            alerts.append("EMERGENCY STOP")
        if not self.video.live(now_s):
            alerts.append("NO VIDEO" if self.video.last_s is None else "VIDEO LOST")
        if self.engaged_track_id is not None and now_s < self.engaged_until_s:
            alerts.append(f"ENGAGE #{self.engaged_track_id}")
        return alerts

    def targets_text(self, now_s: float) -> str:
        if not self.perception.live(now_s):
            return "no perception"
        text = f"{len(self.tracks)} trk"
        if self.selected_id is not None:
            text += f"  {'LOCK' if self.locked else 'sel'} #{self.selected_id}"
        return text

    def agent_text(self, now_s: float) -> str | None:
        """Short system state for the status bar: mode, REC."""
        if not self._agent_recent(now_s):
            return "agent --"
        state = self.agent_state
        parts = [str(state.get("busy") or state.get("mode") or "?")]
        if state.get("recording"):
            parts.append("REC")
        return " ".join(parts)

    def _agent_recent(self, now_s: float) -> bool:
        # The agent is polled every 2 s, so it is live for longer than a stream.
        return self.agent.last_s is not None and now_s - self.agent.last_s <= 5.0

    def controller_text(self, now_s: float) -> str:
        if not self.controller.live(now_s):
            return "ctrl --"
        return f"ctrl {self.controller_reason}"

    def link_lines(self, now_s: float) -> list[str]:
        def line(name: str, link: Link, unit: str = "Hz") -> str:
            age = link.age_s(now_s)
            if age is None:
                return f"{name:<12}never received"
            state = "ok" if link.live(now_s) else f"stale {age:.0f}s"
            return f"{name:<12}{state:<10}{link.rate_hz(now_s):5.1f} {unit}"

        agent_age = self.agent.age_s(now_s)
        agent = ("never answered" if agent_age is None
                 else "ok" if self._agent_recent(now_s) else f"stale {agent_age:.0f}s")
        return [
            line("Video", self.video, "fps"),
            line("Perception", self.perception),
            line("Controller", self.controller),
            line("Panel", self.panel),
            f"{'Agent':<12}{agent}",
        ]

    def panel_lines(self, now_s: float) -> list[str]:
        if not self.panel.live(now_s):
            return ["Panel silent."]
        panel = self.panel_state
        inputs = panel.get("inputs") or {}

        def on(value: object) -> str:
            return "on" if value else "off"

        x, y = panel.get("joystick") or (0.0, 0.0)
        return [
            f"Master arm  {on(panel.get('master_arm'))}",
            f"Auto        {on(panel.get('control_cmd_enabled'))}",
            f"Manual      {on(panel.get('manual_active'))}",
            f"E-stop      {on(panel.get('emergency'))}",
            f"Fire        {on(panel.get('fire'))}",
            f"Joystick    {float(x):+.2f} {float(y):+.2f}",
            *(f"  {role:<10}{on(value)}" for role, value in sorted(inputs.items())),
        ]

    def track_lines(self, now_s: float) -> list[str]:
        if not self.perception.live(now_s):
            return ["No perception data."]
        if not self.tracks:
            return ["No tracks."]
        lines = []
        for row in self.tracks:
            mark = ("L" if self.locked else ">") if row.selected else " "
            state = "coast" if row.missed_frames else "seen"
            distance = f"{row.distance_m:6.1f} m" if row.distance_m is not None else "      -"
            lines.append(f"{mark} #{row.track_id:<5}{row.class_id:<8}{row.confidence:4.2f}  {state:<6}{distance}")
        return lines


def _box(raw: object) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, Mapping):
        return None
    try:
        return (float(raw["x"]), float(raw["y"]), float(raw["w"]), float(raw["h"]))
    except (KeyError, TypeError, ValueError):
        return None
