"""Sparse receiver HUD. Jetson alone owns target boxes and target facts.

Only latest-telemetry pose, aim, control and local display cues belong here.
No target trails, second selection rings, duplicated labels or engineering bars.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from common.perception import PerceptionSnapshotV2
from common.schemas import CamState, ControlCmd, ControlDiagnostics
from common.hud_geometry import (heading_tape_ticks, elevation_tape_ticks,
    feedforward_state, tracking_reticle_style, resolve_hud_fov)
from common import ui_style as theme
from common.ui_style import HudLayout
from pc.ui_text import text as ui_text, text_size

GREEN = theme.bgr(theme.GREEN)
AMBER = theme.bgr(theme.AMBER)
WHITE = theme.bgr(theme.WHITE)
CYAN = theme.bgr(theme.CYAN)
SUBPIXEL_SHIFT = 8
SUBPIXEL_SCALE = 1 << SUBPIXEL_SHIFT


@dataclass(frozen=True)
class HudRenderReport:
    elements: tuple[str, ...]
    feedforward_state: str = 'unavailable'
    feedforward_yaw_rad_s: Optional[float] = None
    feedforward_pitch_rad_s: Optional[float] = None
    reticle_mode: str = 'TRACK'
    reticle_state: str = 'idle'


def _subpixel_line(frame, start, end, colour, thickness=1):
    points = [tuple(round(v*SUBPIXEL_SCALE) for v in p) for p in (start, end)]
    cv2.line(frame, *points, colour, thickness, cv2.LINE_AA, shift=SUBPIXEL_SHIFT)


def _draw_center_reticle(frame, style):
    height, width = frame.shape[:2]
    s = height/720
    cx, cy = width/2, height/2
    colour = theme.bgr(style.rgb)
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        _subpixel_line(frame, (cx+dx*6*s,cy+dy*6*s),
                       (cx+dx*18*s,cy+dy*18*s), colour,
                       theme.pixel_size(1,height))
    cv2.circle(frame,(round(cx),round(cy)),theme.pixel_size(1,height),colour,-1,cv2.LINE_AA)


def _draw_attitude(frame, cam_state, hfov_deg, vfov_deg):
    height, width = frame.shape[:2]
    layout = HudLayout(width,height)
    s = layout.scale
    yaw = math.degrees(cam_state.pan-(cam_state.home_pan or 0)) % 360
    pitch = math.degrees(cam_state.tilt-(cam_state.home_tilt or 0))
    size = theme.pixel_size(theme.SMALL_TEXT_PX,height)
    for tick in heading_tape_ticks(yaw,hfov_deg,width,margin_px=8*s,
                                   right_px=layout.heading.right-8*s):
        x = tick.position_px
        _subpixel_line(frame,(x,layout.heading_y),
            (x,layout.heading_y-(12 if tick.major else 6)*s),GREEN,theme.pixel_size(1,height))
        if tick.major and abs(x-width/2)>28*s:
            label = str(round(tick.value_deg)%360)
            tw = text_size(label,size)[0][0]
            if 0 <= x-tw/2 and x+tw/2 < layout.heading.right:
                ui_text(frame,label,(x-tw/2,12*s),GREEN,size=size)
    label = f'{yaw:05.1f}'
    tw = text_size(label,size)[0][0]
    ui_text(frame,label,(width/2-tw/2,12*s),GREEN,size=size,background=True)
    right = width-6*s
    for tick in elevation_tape_ticks(pitch,vfov_deg,layout.elevation_top,
                                     layout.elevation_bottom,frame_height=height):
        y = tick.position_px
        _subpixel_line(frame,(right,y),(right-(14 if tick.major else 7)*s,y),
                       GREEN,theme.pixel_size(1,height))
        if tick.major and abs(y-height/2)>13*s:
            ui_text(frame,f'{tick.value_deg:+.0f}',(layout.elevation.x+3*s,y+4*s),GREEN,size=size)
    ui_text(frame,f'{pitch:+.1f}',(layout.elevation.x+3*s,height/2+4*s),
            GREEN,size=size,background=True)


def _draw_feedforward_indicator(frame, diagnostics, age):
    state,yaw,pitch = feedforward_state(diagnostics,age)
    height,width = frame.shape[:2]
    layout = HudLayout(width,height)
    r,s = layout.feedforward,layout.scale
    colour = CYAN if state=='active' else WHITE if state=='unavailable' else AMBER
    ui_text(frame,'FF '+state.upper(),(r.x,r.y+14*s),colour,
            size=theme.pixel_size(theme.SMALL_TEXT_PX,height),background=True,max_width=90*s)
    # A paired signed indicator, not numeric rates plus bars showing them again.
    centre = r.x+135*s
    for value,y in ((yaw,r.y+5*s),(pitch,r.y+15*s)):
        _subpixel_line(frame,(centre-30*s,y),(centre+30*s,y),theme.bgr(theme.DIM))
        _subpixel_line(frame,(centre,y-3*s),(centre,y+3*s),WHITE)
        if value is not None and state in ('active','limited'):
            _subpixel_line(frame,(centre,y),(centre+max(-1,min(1,value/.5))*30*s,y),
                           colour,theme.pixel_size(2,height))
    return state,yaw,pitch


def _valid_point(point):
    return (isinstance(point,(tuple,list)) and len(point)==2
            and all(isinstance(v,(float,int)) and math.isfinite(v) and 0<=v<=1 for v in point))


def _draw_aim(frame, aim, target):
    if not _valid_point(aim) or not _valid_point(target):
        return False
    height,width = frame.shape[:2]
    a = (round(aim[0]*width),round(aim[1]*height))
    b = (round(target[0]*width),round(target[1]*height))
    cv2.drawMarker(frame,a,GREEN,cv2.MARKER_DIAMOND,theme.pixel_size(8,height),
                   theme.pixel_size(1,height),cv2.LINE_AA)
    _subpixel_line(frame,a,b,GREEN,theme.pixel_size(1,height))
    return True


class V2HudRenderer:
    def __init__(self, *, hfov_deg, vfov_deg, authority_label='passive', trail_length=24):
        self.hfov_deg, self.vfov_deg = float(hfov_deg),float(vfov_deg)
        authority = str(authority_label or 'passive').strip()
        self.authority_label = 'SIM-ONLY' if authority.lower()=='sim' else authority
        # trail_length remains an API compatibility argument; receiver trails
        # have been removed, rather than running a second target-render owner.

    def render(self, frame, *, snapshot: Optional[PerceptionSnapshotV2],
               cam_state: Optional[CamState], control_cmd: Optional[ControlCmd],
               control_diagnostics: Optional[ControlDiagnostics]=None,
               cam_state_age_s=None, control_age_s=None, diagnostics_age_s=None,
               perception_age_s=None):
        height,width = frame.shape[:2]
        layout = HudLayout(width,height)
        fresh = snapshot is not None and (perception_age_s is None or 0<=perception_age_s<=.25)
        style = tracking_reticle_style(None if snapshot is None else snapshot.model_dump(mode='json'),
            fresh=fresh,mode_hint='sim' if self.authority_label=='SIM-ONLY' else None)
        elements = ['center_reticle','tracking_mode','feedforward_indicator']
        diag = None if control_diagnostics is None else control_diagnostics.model_dump(mode='json')
        ff,fy,fp = _draw_feedforward_indicator(frame,diag,diagnostics_age_s)
        pose_fresh = cam_state is not None and (cam_state_age_s is None or 0<=cam_state_age_s<=.25)
        if pose_fresh:
            _draw_attitude(frame,cam_state,self.hfov_deg,self.vfov_deg)
            elements.append('attitude')
        aim,target = None,None
        if style.state=='tracking' and style.mode!='ROI':
            if diag is not None and diag['reason']=='tracking' and diagnostics_age_s is not None and 0<=diagnostics_age_s<=.25:
                aim,target = diag.get('aim_reference_norm'),diag.get('target_center_norm')
            if (not _valid_point(aim) or not _valid_point(target)) and control_cmd is not None and control_cmd.target_ok and (control_age_s is None or 0<=control_age_s<=.25) and (diag is None or diagnostics_age_s is not None and 0<=diagnostics_age_s<=.25):
                origin,dot = control_cmd.laser_origin_px,control_cmd.laser_dot_px
                if origin is not None and dot is not None:
                    aim,target = (origin[0]/width,origin[1]/height),(dot[0]/width,dot[1]/height)
        if _draw_aim(frame,aim,target):
            elements.append('parallax_cue')
        # Compatibility commands contribute status only if diagnostics are absent.
        if diag is None and control_cmd is not None:
            diag = dict(reason='tracking' if control_cmd.target_ok else 'hold',yaw={},pitch={},
                        controller_mode=str(control_cmd.controller_mode or 'CONTROL').upper())
            diagnostics_age_s = 0 if control_age_s is None else control_age_s
        summary = theme.operational_summary(self.authority_label,style,pose_fresh,diag,diagnostics_age_s)
        ui_text(frame,summary,(layout.status.x,layout.status.y+16*layout.scale),
                theme.bgr(style.rgb),size=theme.pixel_size(theme.TEXT_PX,height),
                background=True,max_width=layout.status.width-8*layout.scale)
        elements.extend(('source_status','freshness'))
        if diag is not None:
            elements.append('control_status')
        _draw_center_reticle(frame,style)
        return HudRenderReport(tuple(elements),ff,fy,fp,style.mode,style.state)
