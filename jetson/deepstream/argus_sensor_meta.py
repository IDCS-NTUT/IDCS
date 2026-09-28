"""Sensor frame number and start-of-frame time from nvarguscamerasrc buffers.

nvarguscamerasrc attaches an ``AuxData`` record to each buffer as GstMiniObject
qdata named ``GstBufferMetaData``: the sensor's frame number and its
start-of-frame timestamp on CLOCK_MONOTONIC (the clock ``time.monotonic_ns``
reads). Probed on the Jetson (2026-09-28, IMX219 mode 4): present on every
frame, frame numbers consecutive, timestamps 6-8 ms before the buffer reaches
the camera src pad.

The record is read on the camera's src pad; nvstreammux builds new batch
buffers, so it is carried downstream keyed by the buffer PTS, which
``NvDsFrameMeta.buf_pts`` preserves.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from collections import OrderedDict
from dataclasses import dataclass


class _AuxData(ctypes.Structure):
    # Layout from nvarguscamerasrc (gstnvarguscamerasrc.hpp: AuxData).
    _fields_ = [("frame_num", ctypes.c_int64),
                ("timestamp", ctypes.c_int64),
                ("sensor_data", ctypes.c_void_p)]


@dataclass(frozen=True)
class SensorFrame:
    frame_number: int
    start_ns: int


class ArgusSensorMetaReader:
    """Reads the ``GstBufferMetaData`` qdata from a Gst.Buffer."""

    def __init__(self) -> None:
        gst = ctypes.CDLL(ctypes.util.find_library("gstreamer-1.0") or "libgstreamer-1.0.so.0")
        glib = ctypes.CDLL(ctypes.util.find_library("glib-2.0") or "libglib-2.0.so.0")
        glib.g_quark_from_string.restype = ctypes.c_uint32
        glib.g_quark_from_string.argtypes = [ctypes.c_char_p]
        self._get_qdata = gst.gst_mini_object_get_qdata
        self._get_qdata.restype = ctypes.c_void_p
        self._get_qdata.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self._quark = glib.g_quark_from_string(b"GstBufferMetaData")

    def read(self, buffer) -> SensorFrame | None:
        # hash() of a PyGObject Gst.Buffer is its C address (as pyds uses it).
        pointer = self._get_qdata(ctypes.c_void_p(hash(buffer)), self._quark)
        if not pointer:
            return None
        aux = _AuxData.from_address(pointer)
        if aux.frame_num < 0 or aux.timestamp <= 0:
            return None
        return SensorFrame(int(aux.frame_num), int(aux.timestamp))


class SensorFrameIndex:
    """Bounded PTS -> sensor frame map between the camera pad and the metadata probe."""

    def __init__(self, capacity: int = 256) -> None:
        self._frames: OrderedDict[int, SensorFrame] = OrderedDict()
        self._capacity = capacity
        self.recorded = 0
        self.missing = 0

    def record(self, pts_ns: int | None, frame: SensorFrame | None) -> None:
        if pts_ns is None or frame is None:
            self.missing += 1
            return
        self._frames[pts_ns] = frame
        self.recorded += 1
        while len(self._frames) > self._capacity:
            self._frames.popitem(last=False)

    def pop(self, pts_ns: int | None) -> SensorFrame | None:
        if pts_ns is None:
            return None
        return self._frames.pop(pts_ns, None)
