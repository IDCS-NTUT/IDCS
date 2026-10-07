"""Deterministic pixels -> H264/NVDEC -> VPI seed -> native NvDCF fixtures.

CPU pixels/encoding are test-input generation, not a production readback path.
Truth supplies the initial test operator rectangle ONLY, then offline scoring.
Run under the exclusive software trial guard, with physical control inactive.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def fixture_score(rows, truth):
    if len(rows) != len(truth): raise ValueError("fixture frame join incomplete")
    errors,visible,selected,false_absent,false_after_grace = [],0,0,0,0
    absent_run = 0
    for i,(row,box) in enumerate(zip(rows,truth)):
        if row["frame_id"] != i: raise ValueError("fixture source frame order mismatch")
        absent_run = absent_run+1 if box is None else 0
        if box is None:
            false_absent += int(row["selected"])
            false_after_grace += int(row["selected"] and absent_run >= 3)
        elif i >= 5:
            visible += 1
            if row["selected"]:
                selected += 1
                x,y,w,h = row["box"]
                tx,ty,tw,th = box
                errors.append(float(np.hypot(x+w/2-tx-tw/2,y+h/2-ty-th/2)))
    return dict(visible=visible,selected=selected,coverage=selected/max(1,visible),
        centre_p95_px=float(np.percentile(errors,95)) if errors else None,
        centre_peak_px=max(errors) if errors else None,
        selected_absent_frames=false_absent,selected_absent_after_three_frames=false_after_grace,
        truth_input_after_seed=False,live_performance_qualification=False)


def capture_fixture(name, output):
    from jetson.visual_lock.capture import InputCapture
    from jetson.visual_lock.egl_lifetime import StandaloneEGL
    from tools.verify_visual_lock import synthetic
    import gi
    gi.require_version("Gst","1.0")
    from gi.repository import Gst,GLib
    frames = list(synthetic(name,width=1280,height=720,count=120))
    output.mkdir(parents=True,exist_ok=False)
    pixels = hashlib.sha256()
    for image,_ in frames: pixels.update(image.tobytes())
    truth = [box for _,box in frames]
    stamps = {1_000_000_000+round(i*1e9/60):i for i in range(len(frames))}
    records = []
    error = []
    with StandaloneEGL():
        Gst.init(None)
        pipeline = Gst.parse_launch(
            'appsrc name=input format=time is-live=false block=false max-bytes=6000000 '
            'caps="video/x-raw,format=BGR,width=1280,height=720,framerate=60/1" ! '
            'videoconvert ! video/x-raw,format=BGRx ! nvvideoconvert ! '
            'video/x-raw(memory:NVMM),format=NV12 ! '
            'nvv4l2h264enc maxperf-enable=1 bitrate=8000000 iframeinterval=30 idrinterval=30 '
            'num-B-Frames=0 insert-sps-pps=true insert-aud=true copy-timestamp=true ! '
            'h264parse ! video/x-h264,stream-format=byte-stream,alignment=au ! '
            'appsink name=encoded emit-signals=true sync=false max-buffers=2 drop=false')
        capture = InputCapture(output,max_bytes=64_000_000,max_duration_ns=3_000_000_000)
        source,sink = pipeline.get_by_name("input"),pipeline.get_by_name("encoded")
        loop = GLib.MainLoop()
        next_index = 0
        def fail(exc):
            error.append(str(exc)); GLib.idle_add(loop.quit)
        def feed(_source,_length):
            nonlocal next_index
            if next_index == len(frames): source.emit("end-of-stream"); return
            i = next_index; next_index += 1
            image = frames[i][0]
            buffer = Gst.Buffer.new_allocate(None,image.nbytes,None)
            buffer.fill(0,image.tobytes())
            buffer.pts = 1_000_000_000+round(i*1e9/60)
            buffer.duration = round(1e9/60)
            source.emit("push-buffer",buffer)
        def encoded(_sink):
            try:
                sample = sink.emit("pull-sample")
                buffer = sample.get_buffer()
                pts = int(buffer.pts)
                i = stamps[pts]
                if i != len(records): raise ValueError("fixture encode identity/order changed")
                capture.append(buffer.extract_dup(0,buffer.get_size()),pts)
                records.append(dict(frame_id=i,source_time_ns=pts,observed_time_ns=10_000_000_000+pts,
                    decoded_pts_ns=pts,seed=i==0,box=truth[0] if i==0 else None))
                return Gst.FlowReturn.OK
            except Exception as exc:
                fail(exc); return Gst.FlowReturn.ERROR
        feed_id = source.connect("need-data",feed)
        sink_id = sink.connect("new-sample",encoded)
        bus = pipeline.get_bus(); bus.add_signal_watch()
        def message(_bus,msg):
            if msg.type == Gst.MessageType.ERROR: fail(msg.parse_error())
            elif msg.type == Gst.MessageType.EOS: loop.quit()
        message_id = bus.connect("message",message)
        timeout = GLib.timeout_add(30000,lambda:(fail("fixture encode timeout"),False)[1])
        pipeline.set_state(Gst.State.PLAYING)
        try: loop.run()
        finally:
            pipeline.set_state(Gst.State.NULL); pipeline.get_state(5*Gst.SECOND)
            source.disconnect(feed_id); sink.disconnect(sink_id)
            bus.disconnect(message_id); bus.remove_signal_watch(); GLib.source_remove(timeout)
            capture.close()
            source = sink = bus = pipeline = sample = buffer = None
            import gc
            gc.collect()
    if error: raise ValueError("; ".join(error))
    if len(records) != len(frames) or capture.stop_reason: raise ValueError("incomplete fixture capture")
    (output/"frames.jsonl").write_text("".join(json.dumps(row)+"\n" for row in records))
    (output/"fixture.json").write_text(json.dumps(dict(name=name,pixel_sha256=pixels.hexdigest(),truth=truth,
        seed=truth[0],fps=60,resolution=[1280,720],clock_scope="synthetic fixture, not surveyed host timing"),indent=2)+"\n")
    return truth


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--provenance",required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    reports = []
    for name in ("translation","scale","distractor","loss","occlusion","textureless"):
        capture,replay = args.output/(name+"-input"),args.output/(name+"-replay")
        truth = capture_fixture(name,capture)
        child = subprocess.run([sys.executable,"-m","tools.replay_visual_lock_capture",
            "--capture",str(capture),"--library",args.library,"--output",str(replay),
            "--handoff-nvdcf","--max-wall-s","60","--provenance",args.provenance],
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
        (args.output/(name+"-child.log")).write_text(child.stdout)
        native = json.loads((replay/"report.json").read_text())
        rows = [json.loads(line) for line in (replay/"frames.jsonl").read_text().splitlines()]
        measured = fixture_score(rows,truth)
        passed = measured["selected_absent_after_three_frames"] == 0
        if name in ("translation","scale","distractor"):
            passed &= child.returncode == 0 and measured["coverage"] >= .95
            passed &= measured["centre_p95_px"] is not None and measured["centre_p95_px"] <= 3 and measured["centre_peak_px"] <= 8
        elif name == "textureless":
            passed &= child.returncode == 1 and measured["selected"] == 0 and native["handoff"]["seed_count"] == 0
        else:
            # Normal status1 is expected for a deliberately lost, latched trial.
            passed &= child.returncode == 1 and native["first_loss"] is not None
            passed &= all(not r["selected"] for r in rows[native["first_loss"]["index"]:])
        reports.append(dict(name=name,acceptance="pass" if passed else "fail",child_returncode=child.returncode,
            metrics=measured,handoff=native["handoff"],first_loss=native["first_loss"]))
        (args.output/"report.json").write_text(json.dumps(dict(provenance=args.provenance,cases=reports),indent=2)+"\n")
        print(json.dumps(reports[-1]),flush=True)
        if not passed: return 1
    return 0


if __name__ == "__main__": raise SystemExit(main())
