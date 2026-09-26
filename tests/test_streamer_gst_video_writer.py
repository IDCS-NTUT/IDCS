from __future__ import annotations

import numpy as np

from pc.streamer import Gst, GstVideoWriter


def test_appsrc_end_of_stream_uses_available_gobject_signal() -> None:
    Gst.init(None)
    writer = GstVideoWriter(
        "appsrc name=src is-live=true format=time "
        "caps=video/x-raw,format=BGR,width=2,height=2,framerate=30/1 ! "
        "fakesink sync=false",
        fps=30,
    )
    try:
        assert writer.write(np.zeros((2, 2, 3), dtype=np.uint8))
        writer.end_of_stream()
        message = writer._pipeline.get_bus().timed_pop_filtered(
            2 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR
        )
        assert message is not None and message.type == Gst.MessageType.EOS
    finally:
        writer.release()
