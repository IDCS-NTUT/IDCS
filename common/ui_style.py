"""Backend-independent operational UI theme and screen regions.

Display only: no clocks, sockets, configuration sync or control authority.
RGB bytes are canonical; adapters explicitly convert to BGR or unit RGB.
"""
from dataclasses import dataclass
import math

FONT_FAMILY = 'DejaVu Sans'
MONO_FAMILY = 'DejaVu Sans Mono'
REFERENCE_HEIGHT = 720
GREEN = (0, 255, 0)
AMBER = (255, 191, 0)
RED = (255, 51, 26)
WHITE = (255, 255, 255)
DIM = (184, 191, 199)
CYAN = (0, 217, 255)
BLUE = (51, 115, 217)
GRAY = (140, 140, 140)
PANEL = (15, 18, 23)
PANEL_ALPHA = .82
PANEL_RGBA = tuple(c / 255 for c in PANEL) + (PANEL_ALPHA,)
ALERT_RGBA = tuple(c / 255 for c in RED) + (.92,)
HIGHLIGHT_RGBA = tuple(c / 255 for c in BLUE) + (.95,)
MODE_RGB = {'SAFE': (64, 153, 77), 'ARMED': AMBER, 'MANUAL': BLUE,
            'E-STOP': RED, 'NO PANEL': GRAY}
TEXT_PX = 13
SMALL_TEXT_PX = 11
STATUS_TEXT_PX = 17
MENU_TEXT_PX = 20
TARGET_TEXT_PX = 14
CORNER_PX = 6
PADDING_PX = 4
STROKE_PX = 1


def control_rgb(reason, *, stale=False):
    return AMBER if stale else GREEN if reason=='tracking' else BLUE if reason=='manual' else AMBER


def source_summary(source, pose_available, selected_id=None, assessment=None):
    text=f'{source} | pose ' + ('OK' if pose_available else '--')
    if selected_id is not None:
        text+=f' | {"SIM target" if source=="SIM-ONLY" else "target"} #{selected_id}'
        assessment=assessment or {}
        distance=assessment.get('distance_m')
        if isinstance(distance,(int,float)) and math.isfinite(distance): text+=f' {distance:.1f}m'
        if assessment.get('threat_level'): text+=' '+str(assessment['threat_level'])
        if assessment.get('engagement_rank') is not None: text+=f' rank {assessment["engagement_rank"]}'
    return text


def control_summary(source, reason, mode, yaw, pitch, *, stale=False):
    state='STALE' if stale else str(reason).upper()
    return f'{source} | {state} | {mode} | Y {yaw:+.3f} P {pitch:+.3f} rad/s'


def parallax_summary(offset=None):
    return f'Parallax aim offset {offset:.1f}px' if offset is not None else 'Parallax aim -- (no fresh tracking geometry)'


def unit_rgb(rgb):
    return tuple(c / 255 for c in rgb)


def bgr(rgb):
    return tuple(reversed(rgb))


def scale_for(height):
    if not math.isfinite(height) or height <= 0:
        raise ValueError('positive finite frame height required')
    return height / REFERENCE_HEIGHT


def pixel_size(size, height):
    return max(1, round(size * scale_for(height)))


def track_rgb(*, selected=False, coasting=False):
    return RED if selected else AMBER if coasting else GREEN


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y + self.height

    def overlaps(self, other):
        return (self.x < other.right and self.right > other.x
                and self.y < other.bottom and self.bottom > other.y)


@dataclass(frozen=True)
class HudLayout:
    """Same logical pixels on PC/Pi/Jetson, scaled from a 720-high view."""
    width: int
    height: int

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError('positive frame dimensions required')

    @property
    def scale(self):
        return scale_for(self.height)

    @property
    def heading_y(self):
        return 146 * self.scale

    @property
    def elevation_top(self):
        return 158 * self.scale

    @property
    def elevation_bottom(self):
        return self.height - 198 * self.scale

    @property
    def feedforward(self):
        s = self.scale
        return Rect(12*s, self.height - 174*s, min(440*s, self.width - 24*s), 51*s)

    @property
    def status(self):
        s = self.scale
        return Rect(12*s, self.height - 119*s, min(930*s, self.width - 154*s), 110*s)

    @property
    def mode_badge(self):
        s = self.scale
        return Rect(self.width - 270*s, 58*s, 170*s, 26*s)

    @property
    def reserved(self):
        s = self.scale
        return (Rect(0, 0, self.width, 154*s),
                Rect(0, self.height-198*s, self.width, 198*s),
                Rect(self.width-92*s, 154*s, 92*s, self.height-352*s))

    def target_label(self, box, text, *, occupied=()):
        """Conservative GPU text bounds; leave boxes untouched and clamp labels.

        Glyph advances differ between GPU/Cairo/FreeType, so use a font-sized
        upper bound rather than pretend the exact GPU extent is known. Dense
        scenes may exhaust space; the caller must not infer collision freedom.
        """
        s = self.scale
        font = pixel_size(TARGET_TEXT_PX, self.height)
        left, top, width, height = box
        usable = max(1, self.width - 116*s)
        # A one-em per-character bound plus padding covers the configured font.
        limit = max(1, int((usable - 8*s) / font))
        label = str(text)[:256]
        if len(label) > limit:
            label = label[:max(0, limit-1)] + '…'
        w = min(usable, len(label)*font + 8*s)
        h = font + 8*s
        x = min(max(8*s, left), max(8*s, self.width - 100*s - w))
        low, high = 154*s, self.height - 198*s - h
        ys = [top-h-4*s, top+height+4*s, top, low]
        ys.extend(low + i*(h+4*s) for i in range(12))
        candidates = [Rect(x, max(low, min(high, y)), w, h) for y in ys]
        candidate = next((r for r in candidates if not any(r.overlaps(v) for v in (*self.reserved,*occupied))), candidates[0])
        return label, candidate
