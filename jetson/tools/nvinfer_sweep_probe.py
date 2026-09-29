"""Run the detector sweep AVI through nvinfer with one config file (Jetson).

Writes analyzer JSONL (tools.analyze_detector_sim_sweep). Used to compare
nvinfer preprocessing (scaling-compute-hw, scaling-filter) against the
PyTorch reference (tools/sweep_reference_pytorch.py).

    python jetson/tools/nvinfer_sweep_probe.py sweep.avi nvinfer.txt out.jsonl
"""
import json, sys
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import pyds

avi, config, out_path = sys.argv[1:4]
Gst.init(None)
pipe = Gst.parse_launch(
    f"filesrc location={avi} ! avidemux ! jpegdec ! videoconvert ! video/x-raw,format=I420 ! "
    "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! m.sink_0 "
    "nvstreammux name=m batch-size=1 width=1280 height=720 ! "
    f"nvinfer name=infer config-file-path={config} ! fakesink sync=false")
out = open(out_path, "w")
count = [0]

def probe(pad, info):
    batch = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
    fl = batch.frame_meta_list
    while fl is not None:
        fm = pyds.NvDsFrameMeta.cast(fl.data)
        boxes = []
        ol = fm.obj_meta_list
        while ol is not None:
            o = pyds.NvDsObjectMeta.cast(ol.data)
            r = o.rect_params
            x, y = max(r.left, 0) / 1280, max(r.top, 0) / 720
            boxes.append({"cls": str(o.class_id), "x": x, "y": y,
                          "w": max(min(r.width / 1280, 1 - x), 1e-4), "h": max(min(r.height / 720, 1 - y), 1e-4),
                          "conf": float(o.confidence)})
            ol = ol.next
        out.write(json.dumps({"frame_id": fm.frame_num + 1, "boxes": boxes}) + "\n")
        count[0] += 1
        fl = fl.next
    return Gst.PadProbeReturn.OK

pipe.get_by_name("infer").get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe)
loop = GLib.MainLoop()
bus = pipe.get_bus(); bus.add_signal_watch()
bus.connect("message::eos", lambda *a: loop.quit())
bus.connect("message::error", lambda b, m: (print("ERR", m.parse_error()), loop.quit()))
pipe.set_state(Gst.State.PLAYING)
loop.run()
pipe.set_state(Gst.State.NULL)
out.close()
print("frames", count[0])
