"""Sparse Cairo receiver HUD; Jetson exclusively owns target decorations."""
import math

import cairo

from common.hud_geometry import elevation_tape_ticks, feedforward_state, heading_tape_ticks
from common.hud_geometry import tracking_reticle_style
from common.hud_telemetry import HudCache, HudView, STALE_S, finite, validate_data
from rpi.display.render import OverlayImage, _rounded
from common import ui_style as theme
from common.ui_style import HudLayout

GREEN = theme.unit_rgb(theme.GREEN)
AMBER = theme.unit_rgb(theme.AMBER)
WHITE = theme.unit_rgb(theme.WHITE)
CYAN = theme.unit_rgb(theme.CYAN)


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
        self.x, self.y = math.floor(x), math.floor(y)
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
            max(1,math.ceil(x+width)-self.x), max(1,math.ceil(y+height)-self.y))
        self.cr = cairo.Context(self.surface)
        self.cr.translate(-self.x, -self.y)

    def line(self,x1,y1,x2,y2,colour=GREEN,width=1):
        self.cr.set_source_rgb(*colour)
        self.cr.set_line_width(width)
        self.cr.move_to(x1,y1)
        self.cr.line_to(x2,y2)
        self.cr.stroke()

    def circle(self,x,y,radius,colour=GREEN,width=1):
        self.cr.set_source_rgb(*colour)
        self.cr.set_line_width(width)
        self.cr.arc(x,y,radius,0,2*math.pi)
        self.cr.stroke()

    def text(self,value,x,y,size=13,colour=GREEN,background=True,centered=False):
        cr = self.cr
        cr.select_font_face(theme.FONT_FAMILY,cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(max(1,round(size)))
        text = str(value)
        if centered:
            x -= cr.text_extents(text).x_advance/2
        available = self.x+self.surface.get_width()-x-4
        while text and cr.text_extents(text).x_advance > available:
            text = text[:-1]
        if background:
            extent = cr.text_extents(text)
            cr.set_source_rgba(*theme.PANEL_RGBA)
            _rounded(cr,x-3,y-size,extent.x_advance+6,size+5,min(theme.CORNER_PX,size/2))
            cr.fill()
        cr.set_source_rgb(*colour)
        cr.move_to(x,y)
        cr.show_text(text)

    def image(self):
        self.surface.flush()
        return OverlayImage(self.x,self.y,self.surface.get_width(),self.surface.get_height(),
            self.surface.get_stride(),bytes(self.surface.get_data()))


class PiHudRenderer:
    def __init__(self, *, hfov_deg=135., vfov_deg=73.):
        self.hfov,self.vfov = hfov_deg,vfov_deg
        self.elements = ()
        self.feedforward = 'unavailable'
        self.reticle_mode,self.reticle_state = 'TRACK','idle'
        self._cache = {}

    def cached(self,name,key,build):
        old = self._cache.get(name)
        if old is None or old[0]!=key:
            self._cache[name] = key,build()
        return self._cache[name][1]

    def render(self,width,height,view):
        layout = HudLayout(width,height)
        s = layout.scale
        images,elements = [],['center_reticle','tracking_mode','feedforward_indicator']
        cx,cy = width/2,height/2
        style = tracking_reticle_style(view.perception,fresh=view.perception is not None,
            mode_hint='sim' if view.source=='SIM-ONLY' else None)
        self.reticle_mode,self.reticle_state = style.mode,style.state
        colour = theme.unit_rgb(style.rgb)
        def reticle():
            tile = _Tile(cx-22*s,cy-22*s,44*s,44*s)
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                tile.line(cx+dx*6*s,cy+dy*6*s,cx+dx*18*s,cy+dy*18*s,colour,width=theme.STROKE_PX*s)
            tile.circle(cx,cy,s,colour)
            return tile.image()
        reticle_image = self.cached('reticle',(width,height,style),reticle)
        pose = view.pose
        if pose and all(finite(pose.get(k)) for k in ('pan','tilt')):
            yaw = math.degrees(pose['pan']-(pose.get('home_pan') or 0))%360
            pitch = math.degrees(pose['tilt']-(pose.get('home_tilt') or 0))
            def heading():
                region = layout.heading
                tile = _Tile(region.x,region.y,region.width,region.height)
                for tick in heading_tape_ticks(yaw,self.hfov,width,margin_px=8*s,right_px=region.right-8*s):
                    x = tick.position_px
                    tile.line(x,layout.heading_y,x,layout.heading_y-(12 if tick.major else 6)*s,width=theme.STROKE_PX*s)
                    if tick.major and abs(x-cx)>28*s:
                        tile.text(str(round(tick.value_deg)%360),x,12*s,11*s,background=False,centered=True)
                tile.text(f'{yaw:05.1f}',cx,12*s,11*s,centered=True)
                return tile.image()
            images.append(self.cached('heading',(width,height,yaw,self.hfov),heading))
            def elevation():
                region = layout.elevation
                tile = _Tile(region.x,region.y,region.width,region.height)
                for tick in elevation_tape_ticks(pitch,self.vfov,layout.elevation_top,layout.elevation_bottom,frame_height=height):
                    y = tick.position_px
                    tile.line(width-6*s,y,width-(20 if tick.major else 13)*s,y,width=theme.STROKE_PX*s)
                    if tick.major and abs(y-cy)>13*s:
                        tile.text(f'{tick.value_deg:+.0f}',region.x+3*s,y+4*s,11*s,background=False)
                tile.text(f'{pitch:+.1f}',region.x+3*s,cy+4*s,11*s)
                return tile.image()
            images.append(self.cached('elevation',(width,height,pitch,self.vfov),elevation))
            elements.append('attitude')
        diag = view.diagnostics
        ff,fy,fp = feedforward_state(diag,view.diagnostics_age_s)
        self.feedforward = ff
        ff_colour = CYAN if ff=='active' else WHITE if ff=='unavailable' else AMBER
        def feedforward():
            r = layout.feedforward
            tile = _Tile(r.x,r.y,r.width,r.height)
            tile.text('FF '+ff.upper(),r.x,r.y+14*s,theme.SMALL_TEXT_PX*s,ff_colour)
            for i,value in enumerate((fy,fp)):
                y = r.y+(5+10*i)*s
                tile.line(r.x+105*s,y,r.x+165*s,y,theme.unit_rgb(theme.DIM))
                tile.line(r.x+135*s,y-3*s,r.x+135*s,y+3*s,WHITE)
                if finite(value) and ff in ('active','limited'):
                    tile.line(r.x+135*s,y,r.x+(135+max(-1,min(1,value/.5))*30)*s,y,ff_colour,2*s)
            return tile.image()
        images.append(self.cached('ff',(width,height,ff,fy,fp),feedforward))
        if diag:
            elements.append('control_status')
            aim,target = diag.get('aim_reference_norm'),diag.get('target_center_norm')
            if (diag.get('reason')=='tracking' and style.state=='tracking' and style.mode!='ROI'
                    and view.diagnostics_age_s is not None and 0<=view.diagnostics_age_s<=.25
                    and self.point(aim) and self.point(target)):
                ax,ay,tx,ty = aim[0]*width,aim[1]*height,target[0]*width,target[1]*height
                x0,y0 = max(0,min(ax,tx)-6*s),max(0,min(ay,ty)-6*s)
                tile = _Tile(x0,y0,min(width-x0,abs(ax-tx)+12*s),min(height-y0,abs(ay-ty)+12*s))
                arm = 4*s
                for a,b in (((ax,ay-arm),(ax+arm,ay)),((ax+arm,ay),(ax,ay+arm)),
                            ((ax,ay+arm),(ax-arm,ay)),((ax-arm,ay),(ax,ay-arm))):
                    tile.line(*a,*b,GREEN,s)
                tile.line(ax,ay,tx,ty,GREEN,s)
                images.append(tile.image())
                elements.append('parallax_cue')
        summary = theme.operational_summary(view.source,style,'attitude' in elements,diag,view.diagnostics_age_s)
        def status():
            r = layout.status
            tile = _Tile(r.x-3*s,r.y,r.width,r.height)
            tile.text(summary,r.x,r.y+16*s,theme.TEXT_PX*s,colour)
            return tile.image()
        images.append(self.cached('status',(width,height,summary,style),status))
        elements.extend(('source_status','freshness'))
        images.append(reticle_image)
        self.elements = tuple(elements)
        return images

    @staticmethod
    def point(value):
        return isinstance(value,(list,tuple)) and len(value)==2 and all(finite(v) and 0<=v<=1 for v in value)
