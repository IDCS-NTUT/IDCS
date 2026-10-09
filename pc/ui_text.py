"""Cached TrueType text and small-ROI blending for the OpenCV display.

No full-frame PIL conversions, network traffic or asynchronous render owners.
The mandatory font is bundled; never silently fall back to Hershey typography.
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common.ui_style import (PANEL, PANEL_ALPHA, CORNER_PX, PADDING_PX,
                             pixel_size, scale_for, bgr)

FONT_DIRECTORY = Path(__file__).resolve().parents[1] / 'assets/fonts'


@lru_cache(maxsize=64)
def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype(str(FONT_DIRECTORY/name), size,
                              layout_engine=ImageFont.Layout.BASIC)


@lru_cache(maxsize=512)
def glyph(text, size, bold=False):
    text = str(text)[:256]
    face = font(size, bold)
    left, top, right, bottom = face.getbbox(text, anchor='ls')
    mask = Image.new('L', (max(1,right-left), max(1,bottom-top)))
    ImageDraw.Draw(mask).text((-left,-top), text, font=face, fill=255, anchor='ls')
    alpha = np.asarray(mask, dtype=np.uint8).copy()
    alpha.setflags(write=False)
    return left, top, alpha


def text_size(text, size, bold=False):
    left, top, alpha = glyph(str(text)[:256], int(size), bold)
    return (alpha.shape[1] + max(0,left), max(1,-top)), max(0,top+alpha.shape[0])


def _blend(frame, x, y, alpha, colour):
    h, w = alpha.shape
    x, y = round(x), round(y)
    x0, y0 = max(0,x), max(0,y)
    x1, y1 = min(frame.shape[1],x+w), min(frame.shape[0],y+h)
    if x0 >= x1 or y0 >= y1:
        return
    a = alpha[y0-y:y1-y,x0-x:x1-x].astype(np.float32)[...,None] / 255
    roi = frame[y0:y1,x0:x1]
    roi[:] = np.rint(roi*(1-a) + np.asarray(colour,dtype=np.float32)*a).astype(np.uint8)


@lru_cache(maxsize=128)
def panel_mask(width, height, radius):
    mask = np.zeros((height,width),np.uint8)
    r = min(radius,width//2,height//2)
    cv2.rectangle(mask,(r,0),(width-r-1,height-1),round(PANEL_ALPHA*255),-1)
    cv2.rectangle(mask,(0,r),(width-1,height-r-1),round(PANEL_ALPHA*255),-1)
    for x,y in ((r,r),(width-r-1,r),(r,height-r-1),(width-r-1,height-r-1)):
        cv2.circle(mask,(x,y),r,round(PANEL_ALPHA*255),-1,cv2.LINE_AA)
    mask.setflags(write=False)
    return mask


def panel(frame, rect):
    x,y,width,height = (round(v) for v in rect)
    if width > 0 and height > 0:
        _blend(frame,x,y,panel_mask(width,height,pixel_size(CORNER_PX,frame.shape[0])),bgr(PANEL))


def text(frame, value, origin, colour, *, size=13, background=False,
         min_width=0, bold=False, max_width=None):
    size = max(1,round(size))
    value = str(value)[:256]
    if max_width is not None:
        while value and text_size(value,size,bold)[0][0] > max_width:
            value = value[:-2] + '…' if len(value)>1 else ''
    left,top,alpha = glyph(value,size,bold)
    x,y = origin
    if background:
        pad = max(1,pixel_size(PADDING_PX,frame.shape[0]))
        panel(frame,(x+left-pad,y+top-pad,max(min_width,alpha.shape[1])+2*pad,alpha.shape[0]+2*pad))
    _blend(frame,x+left,y+top,alpha,colour)


def put_text(frame, value, origin, _font, scale, colour, _thickness=1, _line=None):
    text(frame,value,origin,colour,size=pixel_size(scale*28,frame.shape[0]))


def get_text_size(value, _font, scale, _thickness=1, *, frame_height=720):
    return text_size(value,pixel_size(scale*28,frame_height))
