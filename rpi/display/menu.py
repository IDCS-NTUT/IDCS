"""Operator menu: a small page tree driven by six navigation events.

The menu knows nothing about input devices or drawing. Inputs become
``NavEvent``s (``rpi.display.inputs``); the renderer draws ``Menu.view()``.
With a joystick alone the menu is fully usable: right enters or changes an
item (and opens the closed menu), left goes back (and closes it at the top).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, Sequence


class NavEvent(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    SELECT = "select"
    BACK = "back"
    MENU = "menu"  # open/close from anywhere


@dataclass(frozen=True)
class Choice:
    """A setting with a fixed set of values; entering it steps to the next."""

    label: str
    key: str


@dataclass(frozen=True)
class Action:
    """Runs ``run`` when entered; with ``confirm`` text, only after a confirm page.

    ``key`` identifies what the action is about (a track id, a mode), so the
    display can show it while the cursor is on it.
    """

    label: str
    run: Callable[[], None]
    confirm: str | None = None
    key: object = None
    value: str | None = None


Item = "Page | Choice | Action"


@dataclass(frozen=True)
class Page:
    """A submenu (``items``) or an information page (``lines``).

    ``items`` may be a callable for a live list (tracks, modes); ``lines`` is
    called on every draw, so information pages stay live.
    """

    title: str
    items: Sequence[Item] | Callable[[], Sequence[Item]] = ()
    lines: Callable[[], Sequence[str]] | None = None

    def current_items(self) -> Sequence[Item]:
        return self.items() if callable(self.items) else self.items


@dataclass(frozen=True)
class SettingSpec:
    label: str
    values: tuple[str, ...]
    default: str


class Settings:
    """Display settings, persisted as JSON; unknown or invalid values fall back."""

    def __init__(self, specs: Mapping[str, SettingSpec], path: Path | None,
                 log: logging.Logger | None = None) -> None:
        self.specs = dict(specs)
        self._path = path
        self._log = log or logging.getLogger(__name__)
        self._values = {key: spec.default for key, spec in self.specs.items()}
        if path is not None and path.is_file():
            try:
                stored = json.loads(path.read_text())
            except (OSError, ValueError) as exc:
                self._log.warning("ignoring unreadable settings %s: %s", path, exc)
                stored = {}
            for key, value in (stored.items() if isinstance(stored, dict) else ()):
                if key in self.specs and value in self.specs[key].values:
                    self._values[key] = value

    def __getitem__(self, key: str) -> str:
        return self._values[key]

    def step(self, key: str) -> str:
        values = self.specs[key].values
        self._values[key] = values[(values.index(self._values[key]) + 1) % len(values)]
        self._save()
        return self._values[key]

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._values, indent=1, sort_keys=True))
            tmp.replace(self._path)
        except OSError as exc:
            self._log.warning("could not save settings %s: %s", self._path, exc)


@dataclass(frozen=True)
class MenuRow:
    label: str
    value: str | None  # current value of a Choice, ">" for a page, None for an action


@dataclass(frozen=True)
class MenuView:
    title: str
    rows: tuple[MenuRow, ...]
    cursor: int | None  # None on an information page
    lines: tuple[str, ...]
    depth: int


@dataclass
class _Level:
    page: Page
    cursor: int = 0
    confirm: bool = False


def _noop() -> None:
    return None


@dataclass
class Menu:
    root: Page
    settings: Settings
    is_open: bool = False
    _stack: list[_Level] = field(default_factory=list)

    def handle(self, event: NavEvent) -> bool:
        """Apply one event; True when anything visible changed."""
        if event is NavEvent.MENU:
            return self._close() if self.is_open else self._open()
        if not self.is_open:
            return self._open() if event in (NavEvent.SELECT, NavEvent.RIGHT) else False
        level = self._stack[-1]
        items = level.page.current_items()
        if event in (NavEvent.BACK, NavEvent.LEFT):
            if len(self._stack) == 1:
                return self._close()
            self._stack.pop()
            return True
        if not items:
            return False
        level.cursor = min(level.cursor, len(items) - 1)
        if event is NavEvent.UP:
            level.cursor = (level.cursor - 1) % len(items)
            return True
        if event is NavEvent.DOWN:
            level.cursor = (level.cursor + 1) % len(items)
            return True
        item = items[level.cursor]
        if isinstance(item, Page):
            self._stack.append(_Level(item))
        elif isinstance(item, Choice):
            self.settings.step(item.key)
        elif item.confirm is not None and not level.confirm:
            # Cancel first, so a stray second push does not confirm.
            page = Page(item.confirm, items=(Action("Cancel", _noop), Action("Confirm", item.run)))
            self._stack.append(_Level(page, confirm=True))
        else:
            item.run()
            if level.confirm:
                self._stack.pop()
        return True

    def highlighted(self) -> "Page | Choice | Action | None":
        """The item under the cursor, or None when closed or on an empty page."""
        if not self.is_open:
            return None
        level = self._stack[-1]
        items = level.page.current_items()
        return items[min(level.cursor, len(items) - 1)] if items else None

    def view(self) -> MenuView | None:
        if not self.is_open:
            return None
        level = self._stack[-1]
        page = level.page
        rows = tuple(self._row(item) for item in page.current_items())
        lines = tuple(page.lines()) if page.lines is not None else ()
        return MenuView(
            title=page.title if level.confirm else " / ".join(lvl.page.title for lvl in self._stack),
            rows=rows,
            cursor=min(level.cursor, len(rows) - 1) if rows else None,
            lines=lines,
            depth=len(self._stack),
        )

    def _row(self, item: Page | Choice | Action) -> MenuRow:
        if isinstance(item, Page):
            return MenuRow(item.title, ">")
        if isinstance(item, Choice):
            return MenuRow(item.label, self.settings[item.key])
        return MenuRow(item.label, item.value)

    def _open(self) -> bool:
        self.is_open = True
        self._stack = [_Level(self.root)]
        return True

    def _close(self) -> bool:
        self.is_open = False
        self._stack = []
        return True
