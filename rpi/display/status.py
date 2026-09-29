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

from common.perception import PerceptionSnapshotV2
from common.schemas import ControlDiagnostics

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

    def on_video_frame(self, now_s: float) -> None:
        self.video.mark(now_s)

    def on_snapshot(self, snapshot: PerceptionSnapshotV2, now_s: float) -> None:
        self.perception.mark(now_s)
        selected = snapshot.selection.track_id if snapshot.selection is not None else None
        distance = {item.track_id: item.distance_m for item in snapshot.assessments}
        self.selected_id = selected
        self.tracks = tuple(
            TrackRow(t.track_id, t.class_id, t.confidence, t.missed_frames,
                     t.track_id == selected, distance.get(t.track_id))
            for t in sorted(snapshot.tracks, key=lambda t: (t.track_id != selected, t.track_id))
        )

    def on_diagnostics(self, diagnostics: ControlDiagnostics, now_s: float) -> None:
        self.controller.mark(now_s)
        self.controller_reason = diagnostics.reason

    def on_panel(self, panel: Mapping, now_s: float) -> None:
        self.panel.mark(now_s)
        self.panel_state = dict(panel)

    # --- derived state for the display -------------------------------------

    def panel_mode(self, now_s: float) -> str:
        """E-STOP > MANUAL > ARMED > SAFE, or NO PANEL when the panel is silent."""
        if not self.panel.live(now_s):
            return "NO PANEL"
        panel = self.panel_state
        if panel.get("emergency"):
            return "E-STOP"
        if panel.get("manual_active"):
            return "MANUAL"
        if panel.get("control_cmd_enabled"):
            return "ARMED"
        return "SAFE"

    def alerts(self, now_s: float) -> list[str]:
        """Conditions the operator must see without opening the menu."""
        alerts = []
        if self.panel.live(now_s) and self.panel_state.get("emergency"):
            alerts.append("EMERGENCY STOP")
        if not self.video.live(now_s):
            alerts.append("NO VIDEO" if self.video.last_s is None else "VIDEO LOST")
        return alerts

    def targets_text(self, now_s: float) -> str:
        if not self.perception.live(now_s):
            return "no perception"
        text = f"{len(self.tracks)} trk"
        if self.selected_id is not None:
            text += f"  sel #{self.selected_id}"
        return text

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

        return [
            line("Video", self.video, "fps"),
            line("Perception", self.perception),
            line("Controller", self.controller),
            line("Panel", self.panel),
        ]

    def track_lines(self, now_s: float) -> list[str]:
        if not self.perception.live(now_s):
            return ["No perception data."]
        if not self.tracks:
            return ["No tracks."]
        lines = []
        for row in self.tracks:
            mark = ">" if row.selected else " "
            state = "coast" if row.missed_frames else "seen"
            distance = f"{row.distance_m:6.1f} m" if row.distance_m is not None else "      -"
            lines.append(f"{mark} #{row.track_id:<5}{row.class_id:<8}{row.confidence:4.2f}  {state:<6}{distance}")
        return lines
