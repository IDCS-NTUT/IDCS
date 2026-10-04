"""Draw the panel overlay: status bar, alert banner and menu.

Each element is its own small rectangle (premultiplied BGRA, cairo's native
ARGB32 on little-endian), so the video is only blended where something is
drawn. The caller re-renders when the content text changes, not per frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cairo

from rpi.display.menu import MenuView

SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

WHITE = (1.0, 1.0, 1.0)
DIM = (0.72, 0.75, 0.78)
PANEL_BG = (0.06, 0.07, 0.09, 0.82)
HIGHLIGHT = (0.20, 0.45, 0.85, 0.95)
MODE_COLOURS = {
    "SAFE": (0.25, 0.60, 0.30),
    "ARMED": (0.90, 0.55, 0.05),
    "STANDBY": (0.30, 0.50, 0.55),
    "MANUAL": (0.20, 0.45, 0.85),
    "E-STOP": (0.85, 0.12, 0.12),
    "NO PANEL": (0.45, 0.45, 0.45),
}
ALERT_BG = (0.80, 0.08, 0.08, 0.92)
BOX_COLOURS = {
    "cursor": (1.00, 0.85, 0.10),  # the track under the menu cursor
    "lock": (0.95, 0.15, 0.15),  # the operator's locked target
}


@dataclass(frozen=True)
class OverlayImage:
    x: int
    y: int
    width: int
    height: int
    stride: int
    data: bytes  # premultiplied BGRA rows of ``stride`` bytes


@dataclass(frozen=True)
class StatusBar:
    mode: str
    fields: tuple[str, ...]


@dataclass(frozen=True)
class TargetBox:
    box: tuple[float, float, float, float]  # normalized x, y, w, h
    label: str
    style: str  # key of BOX_COLOURS


def render(frame_w: int, frame_h: int, *, bar: StatusBar | None, alerts: list[str],
           menu: MenuView | None, boxes: tuple[TargetBox, ...] = (),
           notice: str | None = None) -> list[OverlayImage]:
    scale = frame_h / 720.0
    images = [_target_box(box, frame_w, frame_h, scale) for box in boxes]
    if bar is not None:
        images.append(_status_bar(bar, scale))
    if alerts:
        images.append(_alert(alerts, frame_w, scale))
    if menu is not None:
        images.append(_menu(menu, frame_w, frame_h, scale))
    if notice:
        images.append(_notice(notice, frame_w, frame_h, scale))
    return [_clip(image, frame_w, frame_h) for image in images]


def _target_box(target: TargetBox, frame_w: int, frame_h: int, scale: float) -> OverlayImage:
    """Outline around a track with its label above; transparent inside."""
    line = max(2.0, 3 * scale)
    margin = 4 * scale
    font = 15 * scale
    x, y, w, h = target.box
    bw, bh = max(w * frame_w, 8.0) + 2 * margin, max(h * frame_h, 8.0) + 2 * margin
    label_h = font * 1.5
    _, probe = _surface(1, 1)
    probe.select_font_face(SANS, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    probe.set_font_size(font)
    label_w = _text_width(probe, target.label) + 2 * margin
    surface, cr = _surface(max(bw, label_w), bh + label_h)
    colour = BOX_COLOURS.get(target.style, WHITE)
    cr.set_source_rgb(*colour)
    cr.set_line_width(line)
    cr.rectangle(line / 2, label_h + line / 2, bw - line, bh - line)
    cr.stroke()
    cr.rectangle(0, 0, label_w, label_h - 2 * scale)
    cr.fill()
    cr.select_font_face(SANS, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    cr.set_font_size(font)
    cr.set_source_rgb(0.0, 0.0, 0.0)
    cr.move_to(margin, font * 1.05)
    cr.show_text(target.label)
    return _image(surface, x * frame_w - margin, y * frame_h - margin - label_h)


def _notice(text: str, frame_w: int, frame_h: int, scale: float) -> OverlayImage:
    font = 18 * scale
    pad = 10 * scale
    _, probe = _surface(1, 1)
    probe.select_font_face(SANS)
    probe.set_font_size(font)
    width = _text_width(probe, text) + 2 * pad
    height = font + 2 * pad
    surface, cr = _surface(width, height)
    _rounded(cr, 0, 0, width, height, 6 * scale)
    cr.set_source_rgba(*PANEL_BG)
    cr.fill()
    cr.select_font_face(SANS)
    cr.set_font_size(font)
    cr.set_source_rgb(*WHITE)
    cr.move_to(pad, pad + font * 0.82)
    cr.show_text(text)
    return _image(surface, (frame_w - width) / 2, frame_h - height - 24 * scale)


def _surface(width: int, height: int) -> tuple[cairo.ImageSurface, cairo.Context]:
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, max(1, int(width)), max(1, int(height)))
    return surface, cairo.Context(surface)


def _image(surface: cairo.ImageSurface, x: float, y: float) -> OverlayImage:
    surface.flush()
    return OverlayImage(int(x), int(y), surface.get_width(), surface.get_height(),
                        surface.get_stride(), bytes(surface.get_data()))


def _text_width(cr: cairo.Context, text: str) -> float:
    return cr.text_extents(text).x_advance


def _rounded(cr: cairo.Context, x: float, y: float, w: float, h: float, r: float) -> None:
    cr.new_sub_path()
    cr.arc(x + w - r, y + r, r, -1.5708, 0)
    cr.arc(x + w - r, y + h - r, r, 0, 1.5708)
    cr.arc(x + r, y + h - r, r, 1.5708, 3.1416)
    cr.arc(x + r, y + r, r, 3.1416, 4.7124)
    cr.close_path()


def _status_bar(bar: StatusBar, scale: float) -> OverlayImage:
    font = 17 * scale
    pad = 8 * scale
    height = font + 2 * pad
    gap = 14 * scale
    _, probe = _surface(1, 1)
    probe.select_font_face(SANS, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    probe.set_font_size(font)
    chip_w = _text_width(probe, bar.mode) + 2 * pad
    probe.select_font_face(SANS)
    fields_w = sum(_text_width(probe, text) for text in bar.fields) + gap * len(bar.fields)
    surface, cr = _surface(chip_w + fields_w + 2 * pad, height)
    _rounded(cr, 0, 0, surface.get_width(), height, 6 * scale)
    cr.set_source_rgba(*PANEL_BG)
    cr.fill()
    _rounded(cr, 0, 0, chip_w, height, 6 * scale)
    cr.set_source_rgb(*MODE_COLOURS.get(bar.mode, MODE_COLOURS["NO PANEL"]))
    cr.fill()
    baseline = pad + font * 0.82
    cr.select_font_face(SANS, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    cr.set_font_size(font)
    cr.set_source_rgb(*WHITE)
    cr.move_to(pad, baseline)
    cr.show_text(bar.mode)
    cr.select_font_face(SANS)
    x = chip_w + gap
    for text in bar.fields:
        cr.set_source_rgb(*DIM)
        cr.move_to(x, baseline)
        cr.show_text(text)
        x += _text_width(cr, text) + gap
    return _image(surface, 12 * scale, 12 * scale)


def _alert(alerts: list[str], frame_w: int, scale: float) -> OverlayImage:
    font = 30 * scale
    pad = 12 * scale
    text = "   ".join(alerts)
    _, probe = _surface(1, 1)
    probe.select_font_face(SANS, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    probe.set_font_size(font)
    width = _text_width(probe, text) + 4 * pad
    height = font + 2 * pad
    surface, cr = _surface(width, height)
    _rounded(cr, 0, 0, width, height, 8 * scale)
    cr.set_source_rgba(*ALERT_BG)
    cr.fill()
    cr.select_font_face(SANS, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    cr.set_font_size(font)
    cr.set_source_rgb(*WHITE)
    cr.move_to(2 * pad, pad + font * 0.82)
    cr.show_text(text)
    return _image(surface, (frame_w - width) / 2, 64 * scale)


def _menu(view: MenuView, frame_w: int, frame_h: int, scale: float) -> OverlayImage:
    font = 20 * scale
    row_h = font * 1.7
    pad = 18 * scale
    width = min(frame_w - 40 * scale, 640 * scale)
    body_rows = len(view.rows) + len(view.lines) + (1 if view.rows and view.lines else 0)
    max_rows = int((frame_h * 0.8 - 3 * row_h) / row_h)
    visible = min(body_rows, max_rows)
    height = pad * 2 + row_h * (visible + 2)
    surface, cr = _surface(width, height)
    _rounded(cr, 0, 0, width, height, 10 * scale)
    cr.set_source_rgba(*PANEL_BG)
    cr.fill()

    cr.select_font_face(SANS, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    cr.set_font_size(font)
    cr.set_source_rgb(*WHITE)
    cr.move_to(pad, pad + font)
    cr.show_text(view.title)
    y = pad + row_h

    # Keep the cursor on screen when a list is longer than the panel.
    first = 0
    if view.cursor is not None and view.cursor >= visible:
        first = view.cursor - visible + 1
    cr.set_font_size(font)
    for index, row in list(enumerate(view.rows))[first:first + visible]:
        if index == view.cursor:
            _rounded(cr, pad * 0.5, y + row_h * 0.1, width - pad, row_h * 0.9, 5 * scale)
            cr.set_source_rgba(*HIGHLIGHT)
            cr.fill()
        cr.select_font_face(SANS)
        cr.set_source_rgb(*WHITE)
        cr.move_to(pad, y + row_h * 0.72)
        cr.show_text(row.label)
        if row.value is not None:
            cr.set_source_rgb(*(WHITE if index == view.cursor else DIM))
            cr.move_to(width - pad - _text_width(cr, row.value), y + row_h * 0.72)
            cr.show_text(row.value)
        y += row_h
    remaining = visible - min(len(view.rows), visible)
    if view.lines and remaining > 0:
        if view.rows:
            y += row_h
            remaining -= 1
        cr.select_font_face(MONO)
        cr.set_font_size(font * 0.9)
        cr.set_source_rgb(*DIM)
        for line in view.lines[:remaining]:
            cr.move_to(pad, y + row_h * 0.72)
            cr.show_text(line)
            y += row_h

    cr.select_font_face(SANS)
    cr.set_font_size(font * 0.75)
    cr.set_source_rgb(*DIM)
    hint = "up/down move   right select   left back"
    cr.move_to(pad, height - pad * 0.8)
    cr.show_text(hint)
    return _image(surface, (frame_w - width) / 2, (frame_h - height) / 2)


def _clip(image: OverlayImage, frame_w: int, frame_h: int) -> OverlayImage:
    """Keep rectangles inside the frame (the blender rejects ones that are not)."""
    x = min(max(image.x, 0), max(frame_w - image.width, 0))
    y = min(max(image.y, 0), max(frame_h - image.height, 0))
    return OverlayImage(x, y, image.width, image.height, image.stride, image.data)
