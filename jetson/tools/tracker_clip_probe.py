"""Run a recorded clip through YOLO + NvDCF and log every box, ungated (Jetson).

Per frame (index = frame order in the clip) it writes one JSON line:

    det     YOLO's detections before the tracker: [class, conf, x, y, w, h]
    obj     the tracker's output objects: [id, class, detector_conf, tracker_conf, x, y, w, h]
            (detector_conf < 0 marks a tracker-only object)
    shadow  NvDCF's shadow-mode estimates: [id, class, conf, x, y, w, h]

Boxes are normalized to the frame. No coast gate is applied, so gate policies
can be compared offline (tools/tracker_clip_eval.py) on a single run.

    python jetson/tools/tracker_clip_probe.py clip.mp4 nvinfer.txt tracker.yml out.jsonl \
        [--tracker-width 960 --tracker-height 544]
"""
from __future__ import annotations

import argparse
import json
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

import pyds  # noqa: E402

from jetson.deepstream.pipeline import DS_ROOT, ShadowTrackPolicy, _shadow_observations  # noqa: E402

W, H = 1280, 720


def _norm(rect) -> list[float]:
    x, y = max(float(rect.left), 0.0) / W, max(float(rect.top), 0.0) / H
    w = max(min(float(rect.width) / W, 1.0 - x), 1e-4)
    h = max(min(float(rect.height) / H, 1.0 - y), 1e-4)
    return [round(x, 6), round(y, 6), round(w, 6), round(h, 6)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("infer_config")
    ap.add_argument("tracker_config")
    ap.add_argument("output")
    ap.add_argument("--tracker-width", type=int, default=960)
    ap.add_argument("--tracker-height", type=int, default=544)
    args = ap.parse_args()

    Gst.init(None)
    pipe = Gst.parse_launch(
        f"filesrc location={args.clip} ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 "
        f"nvstreammux name=m batch-size=1 width={W} height={H} ! "
        f"nvinfer name=infer config-file-path={args.infer_config} ! "
        f"nvtracker name=tracker ll-lib-file={DS_ROOT}/lib/libnvds_nvmultiobjecttracker.so "
        f"ll-config-file={args.tracker_config} tracker-width={args.tracker_width} "
        f"tracker-height={args.tracker_height} ! fakesink sync=false")
    detections: dict[int, list] = {}
    out = open(args.output, "w", encoding="utf-8")
    policy = ShadowTrackPolicy(min_confidence=0.0, max_age_frames=10**9)
    frames = [0]
    tracker_ns = []
    enter_ns: dict[int, int] = {}

    def frame_metas(info):
        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
        node = batch.frame_meta_list
        while node is not None:
            yield batch, pyds.NvDsFrameMeta.cast(node.data)
            node = node.next

    def objects(frame_meta):
        node = frame_meta.obj_meta_list
        while node is not None:
            yield pyds.NvDsObjectMeta.cast(node.data)
            node = node.next

    def on_infer(pad, info):
        for _, fm in frame_metas(info):
            detections[int(fm.frame_num)] = [
                [int(o.class_id), round(float(o.confidence), 4), *_norm(o.rect_params)] for o in objects(fm)]
            enter_ns[int(fm.frame_num)] = time.monotonic_ns()
        return Gst.PadProbeReturn.OK

    def on_tracker(pad, info):
        for batch, fm in frame_metas(info):
            n = int(fm.frame_num)
            if n in enter_ns:
                tracker_ns.append(time.monotonic_ns() - enter_ns.pop(n))
            objs = [[int(o.object_id), int(o.class_id), round(float(o.confidence), 4),
                     round(float(o.tracker_confidence), 4), *_norm(o.rect_params)] for o in objects(fm)]
            shadow = [[int(s.track_id), int(s.class_id), round(float(s.confidence), 4),
                       s.box.x, s.box.y, s.box.w, s.box.h]
                      for s in _shadow_observations(pyds, batch, fm, policy)]
            out.write(json.dumps({"i": n, "det": detections.pop(n, []), "obj": objs, "shadow": shadow}) + "\n")
            frames[0] += 1
        return Gst.PadProbeReturn.OK

    pipe.get_by_name("infer").get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, on_infer)
    pipe.get_by_name("tracker").get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, on_tracker)
    loop = GLib.MainLoop()
    bus = pipe.get_bus()
    bus.add_signal_watch()
    bus.connect("message::eos", lambda *_: loop.quit())
    bus.connect("message::error", lambda _b, m: (print("ERR", m.parse_error()), loop.quit()))
    pipe.set_state(Gst.State.PLAYING)
    loop.run()
    pipe.set_state(Gst.State.NULL)
    out.close()
    tracker_ns.sort()
    print(json.dumps({"frames": frames[0],
                      "tracker_ms_p50": tracker_ns[len(tracker_ns) // 2] / 1e6 if tracker_ns else None,
                      "tracker_ms_p95": tracker_ns[int(0.95 * len(tracker_ns))] / 1e6 if tracker_ns else None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
