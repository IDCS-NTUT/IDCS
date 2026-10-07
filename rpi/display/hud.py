"""Cairo operational HUD in bounded tiles; no video copies or control output."""
import math
from collections import deque

import cairo

from common.hud_geometry import elevation_tape_ticks, feedforward_state, heading_tape_ticks
from common.hud_telemetry import HudCache, HudView, STALE_S, finite, validate_data
from rpi.display.render import OverlayImage

GREEN = (0., 1., 0.)
AMBER = (1., .75, 0.)
WHITE = (1., 1., 1.)
CYAN = (0., .85, 1.)
RED = (1., .2, .2)


class PanelHudState:
    def __init__(self):
        self.sim = HudCache()
        self.camera = {}
        self.video_source = None
        self.video_received_s = None
        self.video_sequence = None

    def on_snapshot(self, value, now_s):
        frame = value.get('frame') or {}
        source = frame.get('source_clock_domain')
        identity = source, frame.get('tracker_session')
        sequence = value.get('sequence')
        if type(sequence) is not int or sequence < 0:
            return
        if identity == self.video_source and self.video_sequence is not None and sequence <= self.video_sequence:
            return
        if identity != self.video_source:
            self.camera.clear()
        self.video_source, self.video_received_s = identity, now_s
        self.video_sequence = sequence
        if source == 'jetson_monotonic':
            previous = self.camera.get('perception')
            if previous is None or value.get('sequence', -1) > previous[0].get('sequence', -1):
                tracks = list(value.get('tracks', ()))[:32]
                ids = {t.get('track_id') for t in tracks if isinstance(t, dict)}
                selected = (value.get('selection') or {}).get('track_id')
                data = dict(sequence=value.get('sequence'), frame_id=frame.get('frame_id'),
                    tracks=tracks, assessments=[a for a in value.get('assessments', ()) if a.get('track_id') in ids],
                    selected_id=selected if selected in ids else None)
                try:
                    validate_data('perception', data)
                except (ValueError, TypeError):
                    return
                self.camera['perception'] = data, now_s

    def on_camera(self, name, value, now_s):
        if name == 'diagnostics':
            if (value.get('timing') or {}).get('source_clock_domain') != 'jetson_monotonic':
                return
            seq = value.get('created_monotonic_ns')
            try:
                validate_data(name, value)
            except (ValueError, TypeError):
                return
        elif name == 'pose':
            # Do not invent camera motion when physical feedback is unavailable.
            if value.get('hardware_available') is not True:
                self.camera.pop('pose', None)
                return
            seq = value.get('state_monotonic_ns')
            try:
                validate_data(name, {k: value.get(k) for k in ('pan', 'tilt', 'home_pan', 'home_tilt')})
            except (ValueError, TypeError):
                return
        else:
            raise ValueError('unknown camera HUD sample')
        previous = self.camera.get(name)
        key = 'created_monotonic_ns' if name == 'diagnostics' else 'state_monotonic_ns'
        if type(seq) is int and (previous is None or seq > previous[0].get(key, -1)):
            self.camera[name] = value, now_s

    def view(self, now_s, *, mode, source_session='', agent_fresh=True, video_live=True):
        source = None if self.video_source is None else self.video_source[0]
        fresh_video = (video_live and self.video_received_s is not None
            and 0 <= now_s - self.video_received_s <= STALE_S)
        if not agent_fresh or not fresh_video:
            return HudView('UNAVAILABLE', '')
        if mode == 'sim' and source == 'pc_monotonic':
            return self.sim.view(now_s, source_session=source_session)
        if mode != 'camera' or source != 'jetson_monotonic':
            return HudView('UNAVAILABLE', '')
        values, ages = {}, {}
        for name in ('pose', 'diagnostics', 'perception'):
            entry = self.camera.get(name)
            age = None if entry is None else now_s - entry[1]
            if name == 'pose' and entry is not None:
                state = entry[0]
                stamps = [state.get(k) for k in ('pan_sample_monotonic_ns', 'tilt_sample_monotonic_ns')]
                pub = state.get('state_monotonic_ns')
                if not all(type(s) is int and 0 <= s <= pub for s in stamps):
                    age = None
                else:
                    age += (pub - min(stamps)) / 1e9
            ages[name + '_age_s'] = age
            values[name] = entry[0] if age is not None and 0 <= age <= STALE_S else None
        return HudView('CAMERA', str(self.video_source[1] or ''), **values, **ages)


class _Tile:
    def __init__(self, x, y, width, height):
        self.x, self.y = int(x), int(y)
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, max(1, int(width)), max(1, int(height)))
        self.cr = cairo.Context(self.surface)
        self.cr.translate(-self.x, -self.y)

    def line(self, x1, y1, x2, y2, colour=GREEN, width=1):
        cr = self.cr
        cr.set_source_rgb(*colour)
        cr.set_line_width(width)
        cr.move_to(x1, y1)
        cr.line_to(x2, y2)
        cr.stroke()

    def circle(self, x, y, radius, colour=GREEN, width=1):
        self.cr.set_source_rgb(*colour)
        self.cr.set_line_width(width)
        self.cr.arc(x, y, radius, 0, 2 * math.pi)
        self.cr.stroke()

    def text(self, text, x, y, size=13, colour=GREEN, background=True):
        cr = self.cr
        cr.select_font_face('DejaVu Sans', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(size)
        if background:
            extent = cr.text_extents(str(text))
            cr.set_source_rgba(0, 0, 0, .78)
            cr.rectangle(x - 3, y - size, extent.x_advance + 6, size + 5)
            cr.fill()
        cr.set_source_rgb(*colour)
        cr.move_to(x, y)
        cr.show_text(str(text))

    def image(self):
        self.surface.flush()
        return OverlayImage(self.x, self.y, self.surface.get_width(), self.surface.get_height(),
            self.surface.get_stride(), bytes(self.surface.get_data()))


class PiHudRenderer:
    def __init__(self, *, hfov_deg=135., vfov_deg=73.):
        self.hfov, self.vfov = hfov_deg, vfov_deg
        self.trails = {}
        self.identity = None
        self.last_sequence = None
        self.elements = ()
        self.feedforward = 'unavailable'
        self._cache = {}

    def cached(self, name, key, build):
        old = self._cache.get(name)
        if old is None or old[0] != key:
            self._cache[name] = key, build()
        return self._cache[name][1]

    def render(self, width, height, view):
        scale = height / 720
        elements, images = [], []
        identity = view.source, view.session
        if self.identity != identity:
            self.trails.clear()
            self.last_sequence = None
            self.identity = identity
        cx, cy = width / 2, height / 2
        def reticle():
            tile = _Tile(cx - 25 * scale, cy - 25 * scale, 50 * scale, 50 * scale)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                tile.line(cx + dx * 6 * scale, cy + dy * 6 * scale, cx + dx * 22 * scale, cy + dy * 22 * scale)
            tile.circle(cx, cy, 2 * scale)
            return tile.image()
        images.append(self.cached('reticle', (width, height), reticle))
        elements.append('center_reticle')
        pose = view.pose
        if pose and all(finite(pose.get(k)) for k in ('pan', 'tilt')):
            yaw = math.degrees(pose['pan'] - (pose.get('home_pan') or 0)) % 360
            pitch = math.degrees(pose['tilt'] - (pose.get('home_tilt') or 0))
            def heading():
                tile = _Tile(0, 108 * scale, width, 40 * scale)
                for tick in heading_tape_ticks(yaw, self.hfov, width):
                    x = tick.position_px
                    tile.line(x, 146 * scale, x, (128 if tick.major else 137) * scale)
                    if tick.major:
                        tile.text(str(int(round(tick.value_deg)) % 360), max(0, min(width - 28 * scale, x - 10 * scale)), 123 * scale, 11 * scale, background=False)
                tile.text(f'Y {yaw:05.1f}', cx - 28 * scale, 124 * scale, 12 * scale)
                return tile.image()
            images.append(self.cached('heading', (width, height, yaw, self.hfov), heading))
            def elevation():
                left = width - 92 * scale
                tile = _Tile(left, 154 * scale, 92 * scale, height - 350 * scale)
                for tick in elevation_tape_ticks(pitch, self.vfov, 158 * scale, height - 198 * scale):
                    y = tick.position_px
                    tile.line(width - 8 * scale, y, width - (30 if tick.major else 18) * scale, y)
                    if tick.major:
                        tile.text(f'{tick.value_deg:+.0f}', left + 3 * scale, y + 4 * scale, 11 * scale, background=False)
                tile.text(f'P {pitch:+.1f}', left + 3 * scale, height / 2, 12 * scale)
                return tile.image()
            images.append(self.cached('elevation', (width, height, pitch, self.vfov), elevation))
            elements.append('attitude')
        diag = view.diagnostics
        ff, fy, fp = feedforward_state(diag, view.diagnostics_age_s)
        self.feedforward = ff
        colour = CYAN if ff == 'active' else AMBER if ff != 'unavailable' else WHITE
        def feedforward():
            x, y = 12 * scale, height - 174 * scale
            tile = _Tile(x - 3 * scale, y, min(width - x, 440 * scale), 51 * scale)
            signed = lambda v: '--' if not finite(v) else f'{v:+.3f}'
            tile.text(f'FF {ff.upper()} | Y {signed(fy)} P {signed(fp)} rad/s', x, y + 15 * scale, 13 * scale, colour)
            for i, value in enumerate((fy, fp)):
                by = y + (29 + 14 * i) * scale
                tile.line(x, by, x + 330 * scale, by, (.3, .3, .3))
                tile.line(x + 165 * scale, by - 4 * scale, x + 165 * scale, by + 4 * scale, WHITE)
                if finite(value):
                    tile.line(x + 165 * scale, by, x + (165 + max(-1, min(1, value / .5)) * 165) * scale, by, colour, 3 * scale)
            return tile.image()
        images.append(self.cached('ff', (width, height, ff, fy, fp), feedforward))
        elements.append('feedforward_indicator')
        snapshot = view.perception
        if snapshot:
            tracks = [t for t in snapshot.get('tracks', ()) if t.get('missed_frames', 0) == 0][:32]
            ids = {t.get('track_id') for t in tracks}
            self.trails = {key: val for key, val in self.trails.items() if key in ids}
            seq = snapshot.get('sequence')
            if seq != self.last_sequence:
                for track in tracks:
                    box = track.get('box') or {}
                    if all(finite(box.get(k)) for k in ('x', 'y', 'w', 'h')):
                        point = (box['x'] + box['w'] / 2, box['y'] + box['h'] / 2)
                        self.trails.setdefault(track['track_id'], deque(maxlen=24)).append(point)
                self.last_sequence = seq
            selected = next((t for t in tracks if t.get('track_id') == snapshot.get('selected_id')), None)
            if tracks:
                elements.extend(('tracks', 'track_history'))
            for track in tracks[:8]:
                points = self.trails.get(track.get('track_id'), ())
                if len(points) > 1:
                    px = [(x * width, y * height) for x, y in points]
                    xs, ys = zip(*px)
                    x0, y0 = max(0, min(xs) - 3), max(0, min(ys) - 3)
                    tile = _Tile(x0, y0, min(width - x0, max(xs) - x0 + 4), min(height - y0, max(ys) - y0 + 4))
                    for a, b in zip(px, px[1:]):
                        tile.line(*a, *b, (.0, .65, .0))
                    images.append(tile.image())
                box = track.get('box') or {}
                if track.get('class_id', '').lower() == 'person' and all(finite(box.get(k)) for k in ('x', 'y', 'w', 'h')):
                    x0, y0 = box['x'] * width, box['y'] * height
                    bw, bh = box['w'] * width, box['h'] * height
                    tile = _Tile(x0, y0, min(width - x0, bw + 1), min(height - y0, bh + 1))
                    tile.line(x0, y0, x0 + bw, y0 + bh)
                    tile.line(x0, y0 + bh, x0 + bw, y0)
                    images.append(tile.image())
            if selected:
                box = selected.get('box') or {}
                if all(finite(box.get(k)) for k in ('x', 'y', 'w', 'h')):
                    tx = (box['x'] + box['w'] / 2) * width
                    ty = (box['y'] + box['h'] / 2) * height
                    assessments = {a.get('track_id'): a for a in snapshot.get('assessments', ())}
                    assessment = assessments.get(selected['track_id']) or {}
                    target_colour = {'benign': GREEN, 'suspicious': AMBER}.get(assessment.get('threat_level'), RED)
                    x0, y0 = max(0, min(cx, tx, box['x'] * width) - 12 * scale), max(0, min(cy, ty, box['y'] * height) - 12 * scale)
                    tile = _Tile(x0, y0, min(width - x0, max(cx, tx + box['w'] * width / 2) - x0 + 14 * scale),
                        min(height - y0, max(cy, ty + box['h'] * height / 2) - y0 + 14 * scale))
                    tile.line(cx, cy, tx, ty, AMBER)
                    tile.circle(tx, ty, 8 * scale, target_colour, 2 * scale)
                    elements.append('selection')
                    if assessment.get('distance_src') in ('height', 'average'):
                        tile.line(box['x'] * width, ty - 8 * scale, box['x'] * width, ty + 8 * scale, target_colour, 2)
                    if assessment.get('distance_src') in ('width', 'average'):
                        tile.line(tx - 8 * scale, box['y'] * height, tx + 8 * scale, box['y'] * height, target_colour, 2)
                    if assessment.get('distance_m') is not None:
                        elements.append('range')
                    if assessment.get('threat_level'):
                        elements.append('threat')
                    if assessment.get('engagement_rank') is not None:
                        elements.append('rank')
                    images.append(tile.image())
        else:
            self.trails.clear()
            self.last_sequence = None
        lines = [f'{view.source} | pose ' + ('OK' if 'attitude' in elements else '--')]
        if snapshot and view.source == 'SIM-ONLY':
            selected_id = snapshot.get('selected_id')
            assessment = next((a for a in snapshot.get('assessments', ()) if a.get('track_id') == selected_id), {})
            details = [f'SIM target #{selected_id}' if selected_id is not None else 'SIM target --']
            if finite(assessment.get('distance_m')):
                details.append(f'{assessment["distance_m"]:.1f}m')
            if assessment.get('threat_level'):
                details.append(str(assessment['threat_level']))
            if assessment.get('engagement_rank') is not None:
                details.append(f'rank {assessment["engagement_rank"]}')
            lines[0] += ' | ' + ' '.join(details)
        if diag:
            tracking = diag.get('reason') == 'tracking'
            mode = 'PID+FF' if any((diag.get(a) or {}).get('estimator_enabled') for a in ('yaw', 'pitch')) else 'PID'
            yaw, pitch = [(diag.get(a) or {}).get('final_rate_rad_s', 0) for a in ('yaw', 'pitch')]
            lines.append(f'{view.source} | {diag.get("reason", "--")} | {mode} | Y {yaw:+.3f} P {pitch:+.3f} rad/s')
            elements.append('control_status')
            aim, target = diag.get('aim_reference_norm'), diag.get('target_center_norm')
            if tracking and self.point(aim) and self.point(target):
                ax, ay = aim[0] * width, aim[1] * height
                tx, ty = target[0] * width, target[1] * height
                x0, y0 = max(0, min(ax, tx) - 12 * scale), max(0, min(ay, ty) - 12 * scale)
                tile = _Tile(x0, y0, min(width - x0, abs(ax - tx) + 24 * scale), min(height - y0, abs(ay - ty) + 24 * scale))
                arm = 6 * scale
                for a, b in (((ax, ay - arm), (ax + arm, ay)), ((ax + arm, ay), (ax, ay + arm)),
                             ((ax, ay + arm), (ax - arm, ay)), ((ax - arm, ay), (ax, ay - arm))):
                    tile.line(*a, *b, GREEN, 2)
                tile.line(ax, ay, tx, ty, GREEN)
                lines.append(f'Parallax aim offset {math.hypot(tx - ax, ty - ay):.1f}px')
                images.append(tile.image())
                elements.append('parallax_cue')
            else:
                lines.append('Parallax aim -- (no fresh tracking geometry)')
        else:
            lines.extend((f'{view.source} | control UNAVAILABLE/STALE', 'Parallax aim --'))
        elements.append('parallax_status')
        ages = [f'{name} {age * 1000:.0f}ms' for name, age in (('pose', view.pose_age_s), ('ctl', view.diagnostics_age_s), ('target', view.perception_age_s)) if age is not None]
        lines.append(view.age_basis + ': ' + (' | '.join(ages) if ages else '--'))
        lines.append('HUD latest telemetry; video frame join unavailable')
        def status():
            x, y = 12 * scale, height - 119 * scale
            tile = _Tile(x - 3 * scale, y, min(width - x - 130 * scale, 930 * scale), 110 * scale)
            for i, text in enumerate(lines):
                tile.text(text, x, y + (15 + 20 * i) * scale, (11 if i >= 3 else 13) * scale, WHITE)
            return tile.image()
        # Quantized text ages avoid regenerating on sub-ms clock jitter.
        images.append(self.cached('status', (width, height, tuple(lines)), status))
        elements.append('freshness')
        self.elements = tuple(dict.fromkeys(elements))
        return images

    @staticmethod
    def point(value):
        return isinstance(value, (list, tuple)) and len(value) == 2 and all(finite(v) and 0 <= v <= 1 for v in value)
