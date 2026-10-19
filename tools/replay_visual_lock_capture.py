"""Replay unannotated live evidence at the original processed-frame boundary.

Decode with original Gst PTS on Jetson NVDEC, join exact recorded PTS, and feed
ONLY frames the live tracker processed. No pacing or truth
feedback. This diagnoses tracking, not live throughput. Never reseeds on loss.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import threading
import re

from common.perception import NormalizedBoxV2,PerceptionFrameV2
from common.shutdown import install_signal_handlers


def annexb_nals(packet):
    """Annex-B delimiters cannot occur inside correctly escaped H264 RBSP."""
    return [nal for nal in re.split(rb"\x00{2,3}\x01",packet) if nal]


def operator_seed_box(seed):
    # New captures preserve the drawn ROI separately. Replaying the refined
    # box would apply tightening twice and lose the original operator evidence.
    report = seed.get("seed_report") or {}
    return report.get("operator_box") or seed["box"]


def load_manifest(capture):
    index = [json.loads(line) for line in (capture/"input.jsonl").read_text().splitlines()]
    if not index or len(index) > 120000:
        raise ValueError("capture must contain 1..120000 access units")
    offset,previous = 0,-1
    parameters = {}
    with (capture/"input.h264").open("rb") as video:
        for i,row in enumerate(index):
            if (row["encoded_index"] != i or row["offset"] != offset or row["size"] <= 0
                    or row["decoded_pts_ns"] <= previous):
                raise ValueError("invalid access-unit index/order")
            packet = video.read(row["size"])
            if hashlib.sha256(packet).hexdigest() != row["sha256"]:
                raise ValueError("input access-unit digest mismatch")
            nals = annexb_nals(packet)
            types = {nal[0] & 31 for nal in nals}
            for nal in nals:
                kind = nal[0] & 31
                if kind in (7,8): parameters[kind] = b"\x00\x00\x00\x01"+nal
            row["keyframe"] = 5 in types
            if row["keyframe"]:
                row["codec_prefix"] = b"".join(parameters[k] for k in (7,8) if k not in types and k in parameters)
            offset += row["size"]; previous = row["decoded_pts_ns"]
            if offset > 1_024_000_000: raise ValueError("capture exceeds compressed byte budget")
        if video.read(1): raise ValueError("unindexed input bytes")
    records = [json.loads(line) for line in (capture/"frames.jsonl").read_text().splitlines()]
    pts = {r["decoded_pts_ns"]:r for r in records}
    if len(pts) != len(records): raise ValueError("duplicate processed PTS")
    seed = next((r for r in records if r.get("seed")),None)
    if seed is None: raise ValueError("capture lacks explicit seed")
    captured_pts = {r["decoded_pts_ns"] for r in index}
    # A bounded input capture may finish before the live trial; clip explicitly.
    records = [r for r in records if r["decoded_pts_ns"] in captured_pts]
    if not records or records[0] is not seed:
        raise ValueError("seed missing from captured processed sequence")
    return index,{r["decoded_pts_ns"]:r for r in records},seed


def run(capture,library,output,provenance,max_wall_s=240,handoff_nvdcf=False):
    from jetson.visual_lock.egl_lifetime import StandaloneEGL
    from jetson.visual_lock.session import VisualLockSession
    index,records,seed = load_manifest(capture)
    first_idr = next((i for i,r in enumerate(index) if r["keyframe"]),None)
    if first_idr is None or index[first_idr]["decoded_pts_ns"] > seed["decoded_pts_ns"]:
        raise ValueError("seed precedes independently decodable keyframe")
    output.mkdir(parents=True,exist_ok=False)
    handoff = None
    if handoff_nvdcf:
        from jetson.visual_lock.nvdcf_handoff import NvDCFHandoff,write_operator_profile
        handoff = NvDCFHandoff()
        native_profile = output/"operator-nvdcf.yml"
        write_operator_profile(Path(__file__).resolve().parents[1]/"configs/deepstream/tracker_nvdcf.yml",native_profile)
    counts = Counter()
    first_loss = None
    decoded = processed = 0
    start = time.monotonic()
    error = []
    seen = set()
    progress = {"fed":0,"last_input_pts":None,"last_decoded_pts":None}
    with StandaloneEGL():
        from jetson.visual_lock.live_surface import BorrowedBackend,NVMMBorrower
        import gi
        gi.require_version("Gst","1.0")
        from gi.repository import Gst,GLib
        Gst.init(None)
        mux = ('mux.sink_0 nvstreammux name=mux batch-size=1 width=1280 height=720 '
               'live-source=false batched-push-timeout=16666 ! ') if handoff else ''
        tracker = ('nvtracker name=handoff_tracker '
            'll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so '
            f'll-config-file="{native_profile.resolve()}" tracker-width=960 tracker-height=544 ! ') if handoff else ''
        pipeline = Gst.parse_launch(
            'appsrc name=input format=time is-live=true block=false max-bytes=2000000 '
            'caps="video/x-h264,stream-format=byte-stream,alignment=au" ! h264parse ! '
            'nvv4l2decoder enable-max-performance=1 ! queue max-size-buffers=2 ! '
            + mux + 'nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA ! identity name=frames ! '
            + tracker + 'fakesink sync=false async=false')
        if handoff: import pyds
        session = VisualLockSession(lambda w,h:BorrowedBackend(library,w,h),seed_output=output)
        # The live surface helper has the same name beside either ABI library.
        borrower = NVMMBorrower(Path(library).resolve().parent/"libidcs_visual_lock_live_surface.so")
        loop = GLib.MainLoop()
        stopped = threading.Event()
        signal_stop = install_signal_handlers()
        log = (output/"frames.jsonl").open("x")
        def fail(exc):
            error.append(str(exc)); stopped.set(); GLib.idle_add(loop.quit)
        pending = {}

        def record_result(frame,pts,result):
            nonlocal processed,first_loss
            d,p = result.decision,result.proposal
            processed += 1
            counts[d.state] += 1
            counts["selected"] += int(result.snapshot.selection is not None)
            if d.state == "lost" and first_loss is None:
                first_loss = dict(index=processed-1,frame_id=frame.frame_id,reason=d.reason,
                    source_elapsed_s=(frame.source_time_ns-seed["source_time_ns"])/1e9)
            log.write(json.dumps(dict(frame_id=frame.frame_id,decoded_pts_ns=pts,
                source_time_ns=frame.source_time_ns,seed=processed==1,
                seed_report=result.seed_report,
                state=d.state,reason=d.reason,box=d.box,selected=result.snapshot.selection is not None,
                owner=handoff.stage if handoff else "vpi",native_track_id=handoff.native_id if handoff else None,
                inliers=d.inliers,psr=p.psr if p else None,features=len(p.before) if p else None,
                scale=d.scale,anchor_scale=getattr(d,"anchor_scale",None),
                geometry_confident=getattr(d,"geometry_confident",None),
                reference_features=len(p.reference_before) if p is not None and getattr(p,"reference_before",None) is not None else None,
                dcf_box=p.box if p else None,
                commit_features=getattr(session._backend,"commit_stats",None)))+"\n")
            if processed % 60 == 0: log.flush()

        def consume(_pad,info,_data):
            nonlocal decoded,processed,first_loss
            buffer = info.get_buffer()
            if buffer is None: return Gst.PadProbeReturn.OK
            try:
                # No stream mux or metadata reconstruction: decoder preserves
                # the original access-unit PTS directly on this GstBuffer.
                if buffer is not None:
                    pts = int(buffer.pts)
                    batch = meta = None
                    if handoff:
                        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
                        meta = pyds.NvDsFrameMeta.cast(batch.frame_meta_list.data)
                        pts = int(meta.buf_pts)
                    progress["last_decoded_pts"] = pts
                    decoded += 1
                    if decoded <= 3: print(f"decoded PTS {pts}; matched={pts in records}",flush=True)
                    row = records.get(pts)
                    if handoff and row is None: return Gst.PadProbeReturn.DROP
                    if row is not None:
                        if pts in seen: raise ValueError("duplicate replay decoded PTS")
                        seen.add(pts)
                        if len(seen) == 1 and row is not seed: raise ValueError("seed decode missing/reordered")
                        w,h = 1280,720
                        frame = PerceptionFrameV2(tracker_session="frozen-capture",frame_id=row["frame_id"],
                            source_time_ns=row["source_time_ns"],source_clock_domain="pc_monotonic",
                            source_identity_verified=True,observed_time_ns=row["observed_time_ns"],
                            observation_clock_domain="jetson_monotonic",width=w,height=h)
                        if handoff: handoff.begin(frame)
                        result = None
                        if handoff is None or handoff.needs_vpi:
                            with borrower.borrow(buffer) as image:
                                if counts["vpi_processed"] == 0:
                                    x,y,bw,bh = operator_seed_box(seed)
                                    result = session.seed(frame,image,NormalizedBoxV2(x=x/w,y=y/h,w=bw/w,h=bh/h))
                                else: result = session.update(frame,image)
                            counts["vpi_processed"] += 1
                        if handoff:
                            from jetson.visual_lock.nvdcf_handoff import inject_region
                            region = handoff.initialize(frame,result)
                            inject_region(pyds,batch,meta,region)
                            counts["native_seed_inputs"] += int(region is not None)
                            if region is not None: session.close()
                            if len(pending) >= 8: raise ValueError("native replay queue overflow")
                            pending[pts] = (frame,result)
                        else: record_result(frame,pts,result)
            except Exception as exc:
                session.close(); fail(exc)
                return Gst.PadProbeReturn.DROP
            return Gst.PadProbeReturn.OK
        pad = pipeline.get_by_name("frames").get_static_pad("src")
        probe_id = pad.add_probe(Gst.PadProbeType.BUFFER,consume,None)
        native_pad = native_probe_id = None
        if handoff:
            def native_output(_pad,info,_data):
                from jetson.visual_lock.nvdcf_handoff import native_tracks
                try:
                    buffer = info.get_buffer()
                    batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
                    meta = pyds.NvDsFrameMeta.cast(batch.frame_meta_list.data)
                    pts = int(meta.buf_pts)
                    frame,vpi = pending.pop(pts)
                    result = handoff.finish(frame,native_tracks(pyds,meta),vpi)
                    counts["nvdcf_processed"] += 1
                    record_result(frame,pts,result)
                except Exception as exc:
                    fail(exc)  # Pipeline quiescence precedes upstream VPI cleanup.
                    return Gst.PadProbeReturn.DROP
                return Gst.PadProbeReturn.OK
            native_pad = pipeline.get_by_name("handoff_tracker").get_static_pad("src")
            native_probe_id = native_pad.add_probe(Gst.PadProbeType.BUFFER,native_output,None)
        bus = pipeline.get_bus(); bus.add_signal_watch()
        def message(_bus,msg):
            if msg.type == Gst.MessageType.ERROR: fail(msg.parse_error())
            elif msg.type == Gst.MessageType.EOS: loop.quit()
        handler = bus.connect("message",message)
        def check_stop():
            (output/"health.json").write_text(json.dumps(dict(progress,decoded=decoded,processed=processed))+"\n")
            if signal_stop.is_set() or time.monotonic()-start > max_wall_s:
                fail("replay cancelled or bounded timeout")
                return False
            return True
        timeout_id = GLib.timeout_add(100,check_stop)
        source = pipeline.get_by_name("input")
        stream = (capture/"input.h264").open("rb")
        # A receiver capture may start on pre-IDR P pictures. They cannot be
        # decoded independently, and NVDEC can fill its input pool waiting for
        # missing references. Start at the first captured IDR, retaining all
        # processed frames and any captured parameter sets needed by that IDR.
        stream.seek(index[first_idr]["offset"])
        units = iter(index[first_idr:])
        # Never use a Python producer doing blocking GObject emit(): it can
        # hold the GIL while the downstream Python probe waits for that GIL.
        # need-data supplies one AU at a time, with a bounded appsrc queue.
        def feed(_source,_length):
            try:
                if stopped.is_set(): return
                unit = next(units,None)
                if unit is None:
                    source.emit("end-of-stream"); return
                packet = stream.read(unit["size"])
                if unit["encoded_index"] == first_idr:
                    packet = unit.get("codec_prefix",b"")+packet
                buffer = Gst.Buffer.new_allocate(None,len(packet),None)
                buffer.fill(0,packet); buffer.pts = unit["decoded_pts_ns"]
                buffer.dts = unit["decoded_pts_ns"]
                if not unit["keyframe"]: buffer.set_flags(Gst.BufferFlags.DELTA_UNIT)
                progress["fed"] += 1
                progress["last_input_pts"] = unit["decoded_pts_ns"]
                if progress["fed"] <= 3: print(f"feeding PTS {buffer.pts}",flush=True)
                if source.emit("push-buffer",buffer) != Gst.FlowReturn.OK:
                    if not stopped.is_set(): raise ValueError("replay input rejected")
            except Exception as exc: fail(exc)
        feed_handler = source.connect("need-data",feed)
        pipeline.set_state(Gst.State.PLAYING)
        try: loop.run()
        finally:
            stopped.set()
            pipeline.set_state(Gst.State.NULL); pipeline.get_state(5*Gst.SECOND)
            session.close(); log.close()
            source.disconnect(feed_handler); stream.close()
            pad.remove_probe(probe_id); bus.disconnect(handler); bus.remove_signal_watch()
            if native_pad is not None: native_pad.remove_probe(native_probe_id)
            GLib.source_remove(timeout_id)
            source = bus = pad = native_pad = pipeline = None
            import gc
            gc.collect()
    if error: raise ValueError("; ".join(error))
    if processed != len(records): raise ValueError("processed-frame join incomplete")
    report = dict(provenance=provenance,counts=dict(counts),decoded=decoded,processed=processed,
        first_loss=first_loss,tracking_sustained=counts["lost"] == 0 and counts["selected"] > 0,
        wall_duration_s=time.monotonic()-start,live_performance_qualification=False,
        input_sha256=file_digest(capture/"input.h264"),
        library_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest(),
        frame_identity="recorded exact PTS/access-unit index + original verified frame identity",
        decoder="Jetson NVDEC with preserved access-unit PTS; borrowed NVMM RGBA",
        skipped_pre_idr_access_units=first_idr,
        control_disabled=True,truth_input=False)
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    if handoff:
        report["handoff"] = handoff.report()
        report["nvdcf_profile_sha256"] = file_digest(native_profile)
        report["tracking_sustained"] &= handoff.stage == "nvdcf"
        (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report["tracking_sustained"] else 1


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1_048_576): digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture",type=Path,required=True)
    p.add_argument("--library",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--provenance",required=True)
    p.add_argument("--max-wall-s",type=float,default=240)
    p.add_argument("--handoff-nvdcf",action="store_true")
    args = p.parse_args()
    if not 5 <= args.max_wall_s <= 1800: p.error("wall limit must be 5..1800 seconds")
    return run(args.capture,args.library,args.output,args.provenance,args.max_wall_s,args.handoff_nvdcf)


if __name__ == "__main__": raise SystemExit(main())
