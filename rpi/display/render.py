"""Draw the panel overlay: status bar, alert banner and menu.

Each element is its own small rectangle (premultiplied BGRA, cairo's native
ARGB32 on little-endian), so the video is only blended where something is
drawn. The caller re-renders when the content text changes, not per frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cairo

from rpi.display.menu import MenuView
from common import ui_style as theme
from common.ui_fonts import configure_fonts

configure_fonts()  # Once, before Cairo creates any font faces.

SANS = theme.FONT_FAMILY
MONO = theme.MONO_FAMILY

WHITE = theme.unit_rgb(theme.WHITE)
DIM = theme.unit_rgb(theme.DIM)
PANEL_BG = theme.PANEL_RGBA
HIGHLIGHT = theme.HIGHLIGHT_RGBA
MODE_COLOURS = {name:theme.unit_rgb(rgb) for name,rgb in theme.MODE_RGB.items()}
ALERT_BG = theme.ALERT_RGBA
BOX_COLOURS = {
    "cursor": theme.unit_rgb(theme.AMBER),  # preview, not an accepted target
    "lock": theme.unit_rgb(theme.RED),
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
           notice: str | None = None, fire_state: str | None = None,
           safety_state: str | None = None) -> list[OverlayImage]:
    scale = theme.scale_for(frame_h)
    images = [_target_box(box, frame_w, frame_h, scale) for box in boxes]
    if menu is not None:
        images.append(_menu(menu, frame_w, frame_h, scale))
    if notice:
        images.append(_notice(notice, frame_w, frame_h, scale))
    # Safety state is mandatory and cannot be hidden behind a menu/notice.
    if safety_state and bar is None:
        bar = StatusBar('',())
    if bar is not None:
        # Alerts own emergency/fire-pressed messages; no duplicate chip below.
        mode = '' if bar.mode=='E-STOP' and 'EMERGENCY STOP' in alerts else bar.mode
        fields = tuple(value for value in bar.fields if value)
        if safety_state and 'EMERGENCY STOP' not in alerts:
            fields = (safety_state,) + fields
        if fire_state and fire_state not in alerts:
            fields = (fire_state,) + fields
        images.append(_status_bar(StatusBar(mode,fields), frame_w, frame_h, scale))
    if alerts:
        images.append(_alert(alerts, frame_w, scale))
    if fire_state and bar is None and fire_state not in alerts:
        images.append(_fire_status(fire_state, frame_w, frame_h, scale))
    return [_clip(image, frame_w, frame_h) for image in images]


def _target_box(target: TargetBox, frame_w: int, frame_h: int, scale: float) -> OverlayImage:
    """Outline around a track with its label above; transparent inside."""
    line = max(2.0, 3 * scale)
    margin = 4 * scale
    font = theme.TARGET_TEXT_PX * scale
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
    return _image(surface, (frame_w - width) / 2, frame_h - height - 64 * scale)


def _fire_status(text: str, frame_w: int, frame_h: int, scale: float) -> OverlayImage:
    """Fire-arm/button state stays visible without widening the top status bar."""
    font = theme.TEXT_PX * scale
    pad = 4 * scale
    _, probe = _surface(1, 1)
    probe.select_font_face(SANS, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    probe.set_font_size(font)
    width = _text_width(probe, text) + 2 * pad
    height = font + 2 * pad
    surface, cr = _surface(width, height)
    _rounded(cr, 0, 0, width, height, 6 * scale)
    colour = (theme.ALERT_RGBA if text == "FIRE BUTTON"
              else (*theme.unit_rgb(theme.AMBER), .92) if text == "fire ready"
              else PANEL_BG)
    cr.set_source_rgba(*colour)
    cr.fill()
    cr.select_font_face(SANS, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    cr.set_font_size(font)
    cr.set_source_rgb(*WHITE)
    cr.move_to(pad, pad + font * 0.82)
    cr.show_text(text)
    region = theme.HudLayout(frame_w,frame_h).receiver_status
    return _image(surface, region.x, region.y)


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


def _status_bar(bar: StatusBar, frame_w: int, frame_h: int, scale: float) -> OverlayImage:
    font = theme.TEXT_PX * scale
    pad = 4 * scale
    height = font + 2 * pad
    gap = 8 * scale
    _, probe = _surface(1, 1)
    probe.select_font_face(SANS, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    probe.set_font_size(font)
    chip_w = _text_width(probe, bar.mode) + 2 * pad if bar.mode else 0
    probe.select_font_face(SANS)
    region = theme.HudLayout(frame_w,frame_h).receiver_status
    fields_w = sum(_text_width(probe, text) for text in bar.fields) + gap * len(bar.fields)
    surface, cr = _surface(min(region.width,chip_w + fields_w + 2 * pad), height)
    _rounded(cr, 0, 0, surface.get_width(), height, 6 * scale)
    cr.set_source_rgba(*PANEL_BG)
    cr.fill()
    if chip_w:
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
        while text and _text_width(cr,text) > surface.get_width()-x-pad:
            text = text[:-1]
        cr.set_source_rgb(*DIM)
        cr.move_to(x, baseline)
        cr.show_text(text)
        x += _text_width(cr, text) + gap
    return _image(surface, region.x, region.y)


def _alert(alerts: list[str], frame_w: int, scale: float) -> OverlayImage:
    font = 30 * scale
    pad = 12 * scale
    # One fault per row prevents long combined alerts clipping E-stop/fire.
    lines = list(alerts)
    _, probe = _surface(1, 1)
    probe.select_font_face(SANS, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    probe.set_font_size(font)
    width = min(frame_w-40*scale,max(_text_width(probe,text) for text in lines)+4*pad)
    height = font*1.25*len(lines) + 2 * pad
    surface, cr = _surface(width, height)
    _rounded(cr, 0, 0, width, height, 8 * scale)
    cr.set_source_rgba(*ALERT_BG)
    cr.fill()
    cr.select_font_face(SANS, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    cr.set_font_size(font)
    cr.set_source_rgb(*WHITE)
    for index,text in enumerate(lines):
        while text and _text_width(cr,text) > width-4*pad:
            text = text[:-1]
        cr.move_to(2 * pad,pad+font*.82+index*font*1.25)
        cr.show_text(text)
    return _image(surface, (frame_w - width) / 2, 64 * scale)


def _menu(view: MenuView, frame_w: int, frame_h: int, scale: float) -> OverlayImage:
    if view.kind == "info":
        return _information_panel(view, frame_w, frame_h, scale)
    if view.kind == "picker":
        return _picker_panel(view, frame_w, frame_h, scale)
    return _action_menu(view, frame_w, frame_h, scale)


def _fit(cr: cairo.Context, value: str, width: float) -> str:
    if _text_width(cr, value) <= width:
        return value
    while value and _text_width(cr, value + "…") > width:
        value = value[:-1]
    return value + "…" if value else ""


def _action_menu(view: MenuView, frame_w: int, frame_h: int, scale: float) -> OverlayImage:
    font = theme.MENU_TEXT_PX * scale
    row_h = font * 1.7
    pad = 18 * scale
    width = min(frame_w - 40 * scale, (820 if view.kind == "config" else 440) * scale)
    body_rows = len(view.rows) + len(view.lines) + (1 if view.rows and view.lines else 0)
    max_rows = max(1, int((frame_h - 150 * scale - 2 * pad - 2 * row_h) / row_h))
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
    cr.show_text(_fit(cr, view.title, width - 2 * pad))
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
        cr.select_font_face(SANS, cairo.FONT_SLANT_NORMAL,
                            cairo.FONT_WEIGHT_BOLD if index == view.cursor else cairo.FONT_WEIGHT_NORMAL)
        cr.set_source_rgb(*WHITE)
        cr.move_to(pad, y + row_h * 0.72)
        value_width = _text_width(cr, row.value) + pad if row.value else 0
        cr.show_text(_fit(cr, row.label, width - 2 * pad - value_width))
        if row.value is not None:
            colour = (theme.CYAN if row.kind in ("page", "choice") else
                      theme.GREEN if row.value in ("current", "on") else theme.DIM)
            cr.set_source_rgb(*theme.unit_rgb(colour))
            cr.move_to(width - pad - _text_width(cr, row.value), y + row_h * 0.72)
            cr.show_text(_fit(cr, row.value, width - 2 * pad))
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
            cr.show_text(_fit(cr, line, width - 2 * pad))
            y += row_h

    cr.select_font_face(SANS)
    cr.set_font_size(font * 0.75)
    cr.set_source_rgb(*DIM)
    hint = "up/down move   right select   left back"
    cr.move_to(pad, height - pad * 0.8)
    cr.show_text(_fit(cr, hint, width - 2 * pad))
    return _image(surface, 20 * scale, 72 * scale)


def _information_panel(view: MenuView, frame_w: int, frame_h: int, scale: float) -> OverlayImage:
    """On-demand diagnostics; never a tiny unscrollable action menu."""
    pad, font = 24 * scale, 18 * scale
    width = min(frame_w - 40 * scale, 1000 * scale)
    height = min(frame_h - 150 * scale, 540 * scale)
    surface, cr = _surface(width, height)
    _rounded(cr, 0, 0, width, height, 10 * scale)
    cr.set_source_rgba(*PANEL_BG)
    cr.fill()
    cr.select_font_face(SANS, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    cr.set_font_size(theme.MENU_TEXT_PX * scale)
    cr.set_source_rgb(*WHITE)
    cr.move_to(pad, pad + theme.MENU_TEXT_PX * scale)
    cr.show_text(_fit(cr, view.title, width - 2 * pad))
    cr.set_source_rgb(*DIM)
    cr.set_line_width(scale)
    cr.move_to(pad, 2 * pad + font * .5)
    cr.line_to(width - pad, 2 * pad + font * .5)
    cr.stroke()
    body_y = 2 * pad + font
    row_h = font * 1.75
    rows_per_column = max(1, int((height - body_y - 2 * pad - font) / row_h))
    columns = 2 if view.columns > 1 and width >= 700 * scale else 1
    capacity = rows_per_column * columns
    start = min(view.scroll, max(0, len(view.lines) - capacity))
    lines = view.lines[start:start + capacity]
    per_column = min(rows_per_column, max(1, (len(lines) + columns - 1) // columns))
    col_gap = 24 * scale
    col_w = (width - 2 * pad - (columns - 1) * col_gap) / columns
    cr.select_font_face(MONO)
    cr.set_font_size(font)
    for index, line in enumerate(lines):
        col, row = divmod(index, per_column)
        x = pad + col * (col_w + col_gap)
        cr.set_source_rgb(*(WHITE if line.strip() else DIM))
        cr.move_to(x, body_y + (row + .8) * row_h)
        cr.show_text(_fit(cr, line, col_w))
    cr.select_font_face(SANS)
    cr.set_font_size(12 * scale)
    cr.set_source_rgb(*DIM)
    footer = (f"up/down scroll   left back   {start + 1}-{start + len(lines)} / {len(view.lines)}"
              if len(view.lines) > capacity else "left back")
    cr.move_to(pad, height - pad * .7)
    cr.show_text(footer)
    return _image(surface, (frame_w - width) / 2, 82 * scale)


def _picker_panel(view: MenuView, frame_w: int, frame_h: int, scale: float) -> OverlayImage:
    font, pad = 17 * scale, 14 * scale
    width, height = min(frame_w - 40 * scale, 430 * scale), 108 * scale
    surface, cr = _surface(width, height)
    _rounded(cr, 0, 0, width, height, 8 * scale)
    cr.set_source_rgba(*PANEL_BG)
    cr.fill()
    cr.select_font_face(SANS, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    cr.set_font_size(font)
    cr.set_source_rgb(*WHITE)
    cr.move_to(pad, pad + font)
    cr.show_text("Select target")
    current = view.rows[view.cursor] if view.cursor is not None and view.rows else None
    cr.move_to(pad, pad + font * 2.25)
    cr.show_text(_fit(cr, (f"{view.cursor + 1}/{len(view.rows)}  {current.label}"
                           if current is not None else "No fresh target"), width - 2 * pad))
    cr.select_font_face(SANS)
    cr.set_font_size(12 * scale)
    cr.set_source_rgb(*DIM)
    cr.move_to(pad, height - pad)
    cr.show_text(_fit(cr, "up/down cycle   right lock   left cancel", width - 2 * pad))
    return _image(surface, 20 * scale, frame_h - height - 64 * scale)


def _clip(image: OverlayImage, frame_w: int, frame_h: int) -> OverlayImage:
    """Keep rectangles inside the frame (the blender rejects ones that are not)."""
    x = min(max(image.x, 0), max(frame_w - image.width, 0))
    y = min(max(image.y, 0), max(frame_h - image.height, 0))
    return OverlayImage(x, y, image.width, image.height, image.stride, image.data)
