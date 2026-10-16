"""Four-way-joystick ROI draft for the software-only visual-lock console.

This module deliberately has no socket or authority to seed a tracker.  A
confirmed draft must be paired with an exact displayed-frame identity by its
caller before it can become a visual-lock request.
"""

from __future__ import annotations

from dataclasses import dataclass

from rpi.display.menu import MenuRow, MenuView, NavEvent


_FIELDS = ("X", "Y", "Width", "Height")


@dataclass
class RoiEditor:
    width: int = 1280
    height: int = 720
    x: int = 608
    y: int = 336
    w: int = 64
    h: int = 48
    step: int = 8
    cursor: int = 0
    active: bool = False
    tracking_trial: bool = False
    direct_select: bool = False
    resizing: bool = False
    _last_move_s: float | None = None
    _fractional_x: float = 0.0
    _fractional_y: float = 0.0

    def open(self, width: int, height: int, *, tracking_trial: bool = False,
             direct_select: bool = False) -> None:
        if width < 24 or height < 24:
            raise ValueError("ROI display dimensions are too small")
        self.width, self.height = int(width), int(height)
        self.w, self.h = min(64, self.width), min(48, self.height)
        self.x = (self.width - self.w) // 2
        self.y = (self.height - self.h) // 2
        self.cursor = 0
        self.active = True
        self.tracking_trial = tracking_trial
        self.direct_select = direct_select
        self.resizing = False
        self._last_move_s = None
        self._fractional_x = self._fractional_y = 0.0

    def move_joystick(self, x: float, y: float, now_s: float) -> None:
        """Move the box in image space; panel positive pitch is screen up."""
        if not self.active or not self.direct_select:
            return
        last, self._last_move_s = self._last_move_s, now_s
        if last is None or now_s <= last:
            return
        dt = min(now_s - last, 0.1)
        speed = min(self.width, self.height) * 0.6
        def axis(value: float) -> float:
            return 0.0 if abs(value) < 0.15 else max(-1.0, min(1.0, value))
        self._fractional_x += axis(x) * speed * dt
        self._fractional_y -= axis(y) * speed * dt
        dx, dy = int(self._fractional_x), int(self._fractional_y)
        self._fractional_x -= dx
        self._fractional_y -= dy
        if self.resizing:
            w = max(12, min(self.width, self.w + dx))
            h = max(12, min(self.height, self.h - dy))
            self.x = max(0, min(self.width-w, self.x+(self.w-w)//2))
            self.y = max(0, min(self.height-h, self.y+(self.h-h)//2))
            self.w, self.h = w, h
            return
        self.x = max(0, min(self.width - self.w, self.x + dx))
        self.y = max(0, min(self.height - self.h, self.y + dy))

    def box(self) -> tuple[float, float, float, float]:
        return (
            self.x / self.width,
            self.y / self.height,
            self.w / self.width,
            self.h / self.height,
        )

    def handle(self, event: NavEvent) -> str | None:
        """Return ``confirm`` or ``cancel``; no command is sent here."""
        if not self.active:
            return None
        if event in (NavEvent.MENU, NavEvent.BACK):
            self.active = False
            return "cancel"
        if self.direct_select:
            if event is NavEvent.SELECT:
                self.resizing = not self.resizing
                self._last_move_s = None
                self._fractional_x = self._fractional_y = 0.0
            return None
        if event is NavEvent.UP:
            self.cursor = (self.cursor - 1) % 6
        elif event is NavEvent.DOWN:
            self.cursor = (self.cursor + 1) % 6
        elif self.cursor == 5 and event in (NavEvent.RIGHT, NavEvent.SELECT):
            self.active = False
            return "cancel"
        elif self.cursor == 4 and event in (NavEvent.RIGHT, NavEvent.SELECT):
            self.active = False
            return "confirm"
        elif self.cursor < 4 and event in (NavEvent.LEFT, NavEvent.RIGHT):
            delta = self.step * (1 if event is NavEvent.RIGHT else -1)
            values = [self.x, self.y, self.w, self.h]
            values[self.cursor] += delta
            x, y, w, h = values
            w = max(12, min(w, self.width - x))
            h = max(12, min(h, self.height - y))
            x = max(0, min(x, self.width - w))
            y = max(0, min(y, self.height - h))
            self.x, self.y, self.w, self.h = x, y, w, h
        return None

    def view(self) -> MenuView | None:
        if not self.active or self.direct_select:
            return None
        values = (self.x, self.y, self.w, self.h)
        rows = tuple(MenuRow(name, f"{value} px") for name, value in zip(_FIELDS, values))
        rows += (MenuRow("Confirm ROI", None), MenuRow("Cancel", None))
        return MenuView(
            title="Simulator / Visual ROI",
            rows=rows,
            cursor=self.cursor,
            lines=("Up/down: field  Left/right: adjust",
                   "Confirm selects this exact return frame" if self.tracking_trial
                   else "Preview only; no tracker command"),
            depth=2,
        )
