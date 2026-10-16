"""Bounded control-free RTP/Argus -> VPI CUDA -> GPU OSD -> return RTP trial.

Requires the only GPU video owner, exact verified source identity and a one-time
explicit pixel seed. No detector, truth subscriber, controller or actuator.
An opt-in panel session accepts exact displayed-frame seeds on an isolated port.
The simulated camera remains stationary.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from queue import Empty, Full, Queue
import time
import uuid

from common.perception import NormalizedBoxV2, PerceptionFrameV2, PerceptionSnapshotV2
from common.hud_geometry import tracking_reticle_style
from common.shutdown import install_signal_handlers
from common import ui_style as ui_theme
from common.ui_style import HudLayout
from jetson.deepstream.snapshot_transport import SnapshotTransport
from jetson.visual_lock.egl_lifetime import StandaloneEGL
from jetson.visual_lock.session import VisualLockSession
from jetson.visual_lock.return_identity import ReturnRtpMapper, SourceIdentity
from jetson.visual_lock.frame_cache import ExactFrameCache
from jetson.visual_lock.acquisition import acquire_cached
from common.visual_roi_seed import VisualRoiSeed, VisualRoiRelease
from common.gst_buffer_probe import probe_buffers
from jetson.visual_lock.camera_identity import CameraIdentityIndex
from jetson.visual_lock.gate_config import load_gates


class StartupCadence:
    """Wait before the one explicit trial seed; never re-acquire a lost lock."""
    def __init__(self, required=8, max_interval_ns=50_000_000):
        self.required, self.max_interval_ns = required, max_interval_ns
        self.previous = None
        self.consecutive = 0

    def observe(self, source_time_ns):
        if self.previous is not None and 0 < source_time_ns-self.previous <= self.max_interval_ns:
            self.consecutive += 1
        else:
            self.consecutive = 0
        self.previous = source_time_ns
        return self.consecutive >= self.required


class LiveDeadline:
    """Bound source startup independently from the requested live-video time."""
    def __init__(self, ready_s, duration_s, source_wait_s):
        self.ready_s, self.duration_s, self.source_wait_s = ready_s,duration_s,source_wait_s
        self.first_verified_s = None

    def observe(self, now):
        if self.first_verified_s is None: self.first_verified_s = now

    def reason(self, now):
        if self.first_verified_s is None:
            return "source startup timeout" if now-self.ready_s >= self.source_wait_s else None
        return "duration complete" if now-self.first_verified_s >= self.duration_s else None


def warm_kernels(library, image):
    """Warm NVMM interop on the owning thread; publish no dummy selection."""
    from jetson.visual_lock.live_surface import BorrowedBackend
    start = time.perf_counter()
    # This startup-only floor patch is discarded, not the target initializer.
    with BorrowedBackend(library,1280,720) as backend:
        backend.process(image,seed=(80,600,96,72))
        backend.process(image)
        backend.commit((80,600,96,72),update_template=True)
        backend.process(image)
        backend.discard()
    return (time.perf_counter()-start)*1000


def pipeline_description(input_port,return_hosts,return_port,return_fps,nvdcf_config=None,
                         return_keyframe_interval=1, feedback_port=5003, sender_report_port=5004,
                         *, source="sim", sensor_id=0, sensor_mode=4, flip_method=0, wb_mode=1):
    from common.return_rtp import sender_tail
    if not 1 <= input_port <= 65535 or not 1 <= return_port <= 65535:
        raise ValueError("invalid RTP port")
    if input_port==feedback_port:
        raise ValueError('return RTCP feedback must not share the input RTP port')
    if not 1 <= return_fps <= 60 or not return_hosts:
        raise ValueError("return hosts and 1..60 fps required")
    if type(return_keyframe_interval) is not int or not 1 <= return_keyframe_interval <= 60:
        raise ValueError('return keyframe interval must be 1..60')
    import ipaddress
    hosts = [str(ipaddress.ip_address(h)) for h in return_hosts]
    if source not in {"sim", "camera"}:
        raise ValueError("source must be sim or camera")
    if any(type(v) is not int or v < 0 for v in (sensor_id,sensor_mode,flip_method,wb_mode)) or flip_method > 7 or wb_mode > 9:
        raise ValueError("invalid Argus source settings")
    if source == "camera":
        input_chain = (
            f'nvarguscamerasrc name=camera sensor-id={sensor_id} sensor-mode={sensor_mode} wbmode={wb_mode} ! '
            'video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1,format=NV12 ! '
            + (f'nvvideoconvert name=camera_flip flip-method={flip_method} ! '
               'video/x-raw(memory:NVMM),format=NV12 ! ' if flip_method else '')
            + 'queue max-size-buffers=2 leaky=downstream ! mux.sink_0 ')
    else:
        input_chain = (
            f'udpsrc port={input_port} caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000" ! '
            'rtpjitterbuffer name=rtp_jitter latency=50 drop-on-latency=true ! rtph264depay ! '
            'h264parse name=input_parse ! video/x-h264,stream-format=byte-stream,alignment=au ! '
            'nvv4l2decoder name=decoder enable-max-performance=1 ! queue max-size-buffers=2 leaky=downstream ! mux.sink_0 ')
    tracker = ""
    if nvdcf_config is not None:
        path = str(Path(nvdcf_config).resolve()).replace('\\','/').replace('"','\\"')
        tracker = ('nvtracker name=handoff_tracker '
            'll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so '
            f'll-config-file="{path}" tracker-width=960 tracker-height=544 ! '
            'nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA ! ')
    return (
        input_chain +
        'nvstreammux name=mux batch-size=1 width=1280 height=720 live-source=true batched-push-timeout=16666 ! '
        'nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA ! identity name=visual_input ! '
        + tracker +
        'nvdsosd process-mode=1 ! nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12,width=1280,height=720 ! '
        'identity name=return_rate ! '
        f'capssetter name=return_caps caps="video/x-raw(memory:NVMM),framerate={return_fps}/1" replace=false ! '
        'queue max-size-buffers=1 leaky=downstream ! '
        # Visual-only return: independent IDR pictures avoid propagation of
        # reference-frame corruption while preserving original frame PTS.
        'nvv4l2h264enc name=return_encoder maxperf-enable=1 control-rate=1 bitrate=20000000 '
        f'iframeinterval={return_keyframe_interval} idrinterval={return_keyframe_interval} '
        'num-B-Frames=0 num-Ref-Frames=1 insert-sps-pps=true insert-aud=true insert-vui=true copy-timestamp=true preset-level=1 ! '
        'h264parse name=return_parse config-interval=-1 ! video/x-h264,stream-format=byte-stream,alignment=au ! '
        + sender_tail(hosts,return_port,pay_name='return_pay',feedback_port=feedback_port,
                      sender_report_port=sender_report_port)
    )


def decorate(pyds,batch,meta,result,frame_id,fps,owner="VPI",source="sim"):
    layout = HudLayout(1280,720)  # Trial caps remain fixed; no pipeline change.
    def _set_rgba(color,r,g,b,a=1.):
        color.red,color.green,color.blue,color.alpha = r,g,b,a
    display = pyds.nvds_acquire_display_meta_from_pool(batch)
    if display is None: return
    style = tracking_reticle_style(None if result is None else result.snapshot.model_dump(mode='json'),
        fresh=result is not None, mode_hint='roi', state_hint=result.decision.state if result else None)
    colour = tuple(c / 255 for c in style.rgb)
    display.num_labels = 2
    text = display.text_params[0]
    details = ""
    if result and result.proposal:
        details = f" PSR {result.proposal.psr:.1f} points {result.decision.inliers} {result.processing_ms:.1f}ms"
    text.display_text = f"{source.upper()} | {owner} | {fps:.1f}fps{details}"
    text.x_offset,text.y_offset = round(layout.roi_quality.x),round(layout.roi_quality.y)
    text.font_params.font_name,text.font_params.font_size = ui_theme.FONT_FAMILY,ui_theme.SMALL_TEXT_PX
    _set_rgba(text.font_params.font_color,1.,1.,1.)
    text.set_bg_clr = 1
    _set_rgba(text.text_bg_clr,*ui_theme.PANEL_RGBA)
    # Trial caps are fixed 1280x720. GPU OSD owns this reticle; viewers must
    # not invent a second ROI owner from simulator/controller telemetry.
    label = display.text_params[1]
    label.display_text = style.label + ' | CONTROL OFF'
    label.x_offset, label.y_offset = round(layout.mode_badge.x+4),round(layout.mode_badge.y)
    label.font_params.font_name, label.font_params.font_size = ui_theme.FONT_FAMILY,ui_theme.TEXT_PX
    _set_rgba(label.font_params.font_color, *colour)
    label.set_bg_clr = 1
    _set_rgba(label.text_bg_clr,*ui_theme.PANEL_RGBA)
    display.num_lines = 4
    for line, (x1,y1,x2,y2) in zip(display.line_params,
            ((618,360,634,360), (646,360,662,360), (640,338,640,354), (640,366,640,382))):
        line.x1,line.y1,line.x2,line.y2 = x1,y1,x2,y2
        line.line_width = 1
        _set_rgba(line.line_color, *colour)
    boxes = []
    if result and result.decision.box and result.decision.state not in ("lost","idle"):
        boxes.append((result.decision.box,colour,3))
        if result.proposal is not None:
            for x,y in result.proposal.after[:12]:
                if 0 <= x <= 1277 and 0 <= y <= 717:
                    boxes.append(((float(x)-1,float(y)-1,3.,3.),(0.,1.,1.),1))
    display.num_rects = len(boxes)
    for i,(box,color,border) in enumerate(boxes):
        rect = display.rect_params[i]
        rect.left,rect.top,rect.width,rect.height = box
        rect.border_width,rect.has_bg_color = border,0
        _set_rgba(rect.border_color,*color)
    pyds.nvds_add_display_meta_to_frame(meta,display)


def run(args):
    from common.ui_fonts import configure_fonts
    configure_fonts()  # Process-local typography; no system font installation.
    # New standalone pipeline owns EGL once. All wrappers and Gst/NVMM resources
    # are released before this owner terminates it; no _Exit or retained locals.
    with StandaloneEGL() as egl:
        from jetson.visual_lock.live_surface import NVMMBorrower,BorrowedBackend
        import gi
        import pyds
        gi.require_version("Gst","1.0")
        from gi.repository import Gst,GLib
        Gst.init(None)
        output = Path(args.output)
        output.mkdir(parents=True,exist_ok=False)
        handoff = None
        native_profile = None
        if getattr(args,"handoff_nvdcf",False):
            from jetson.visual_lock.nvdcf_handoff import NvDCFHandoff,write_operator_profile
            handoff = NvDCFHandoff()
            native_profile = output/"operator-nvdcf.yml"
            write_operator_profile(Path(args.nvdcf_profile),native_profile)
        source = getattr(args,"source","sim")
        source_domain = "jetson_monotonic" if source == "camera" else "pc_monotonic"
        pipeline = Gst.parse_launch(pipeline_description(args.input_port,args.return_host,args.return_port,args.return_fps,native_profile,
            getattr(args,'return_keyframe_interval',1),getattr(args,'return_feedback_port',5003),getattr(args,'return_sender_report_port',5004),
            source=source,sensor_id=getattr(args,'argus_sensor_id',0),sensor_mode=getattr(args,'argus_sensor_mode',4),
            flip_method=getattr(args,'argus_flip_method',0),wb_mode=getattr(args,'argus_wb_mode',1)))
        from common.return_rtp import NvidiaKeyframeBridge
        return_recovery=NvidiaKeyframeBridge(pipeline.get_by_name('return_encoder'))
        transport = SnapshotTransport(header_bind=args.header_bind if source == "sim" else None,
            snapshot_bind=args.snapshot_bind,verified_rtp_headers=source == "sim")
        return_mapper = ReturnRtpMapper()
        identity_queue = Queue(maxsize=128)
        identity_context = identity_socket = None
        if args.return_identity_bind:
            import zmq
            identity_context = zmq.Context()
            identity_socket = identity_context.socket(zmq.PUB)
            identity_socket.setsockopt(zmq.SNDHWM, 128)
            identity_socket.setsockopt(zmq.LINGER, 0)
            identity_socket.bind(args.return_identity_bind)
        roi_context = roi_socket = None
        roi_requests = Queue(maxsize=8)
        active_roi_request = None
        pending_roi_seed = None
        frame_cache = ExactFrameCache(capacity=16) if args.roi_bind else None
        if args.roi_bind:
            import zmq
            roi_context = zmq.Context()
            roi_socket = roi_context.socket(zmq.PULL)
            roi_socket.setsockopt(zmq.RCVHWM, 4)
            roi_socket.setsockopt(zmq.MAXMSGSIZE, 1024)
            roi_socket.setsockopt(zmq.LINGER, 0)
            roi_socket.bind(args.roi_bind)
        borrower = NVMMBorrower(args.surface_library)
        prepared = []
        def backend_factory(w,h):
            if (w,h) != (1280,720) or len(prepared) != 1:
                raise RuntimeError("initial live backend was not prepared on the frame owner")
            return prepared.pop()
        session = VisualLockSession(backend_factory,load_gates(getattr(args,"gates_json",None)))
        def close_session():
            session.close()
            while prepared: prepared.pop().close()
        tracker_session = uuid.uuid4().hex
        camera_frames = CameraIdentityIndex(tracker_session) if source == "camera" else None
        stop = install_signal_handlers()
        loop = GLib.MainLoop()
        counts = Counter()
        timings = deque(maxlen=6000)
        frame_work = deque(maxlen=6000)
        state_timings = defaultdict(lambda:deque(maxlen=6000))
        state_work = defaultdict(lambda:deque(maxlen=6000))
        selected_times = {"first":None,"last":None}
        arrivals = deque(maxlen=120)
        started = time.monotonic()
        deadline = LiveDeadline(started,args.duration_s,args.source_wait_s)
        last_verified_s = None
        outcome = {"error":None,"seeded":False,"closed_on_owner":False,"warmup_ms":None,"prepare_ms":None}
        cadence = StartupCadence()
        records = (output/"frames.jsonl").open("w",encoding="utf-8")
        acquisitions = (output/"roi-acquisitions.jsonl").open("w",encoding="utf-8")
        from jetson.visual_lock.capture import InputCapture
        input_capture = InputCapture(output,max_bytes=args.capture_max_mb*1_000_000,
                                     max_duration_ns=round((args.duration_s+5)*1e9))

        def roi_result(request,accepted,reason):
            message = dict(type="visual_roi_result_v1",request_id=request.request_id,
                           frame_id=request.frame_id,accepted=accepted,reason=reason)
            try: identity_queue.put_nowait(json.dumps(message,separators=(",",":")).encode())
            except Full: counts["return_identity_queue_full"] += 1

        def poll_roi_requests():
            if roi_socket is None: return False
            import zmq
            for _ in range(4):
                try: raw = roi_socket.recv(flags=zmq.NOBLOCK)
                except zmq.Again: break
                try:
                    message = json.loads(raw)
                    if not isinstance(message,dict): raise ValueError("ROI command must be a mapping")
                    request = (VisualRoiRelease.from_mapping(message)
                               if message.get('type') == 'visual_roi_release_v1'
                               else VisualRoiSeed.from_mapping(message))
                except (ValueError,TypeError):
                    counts["roi_invalid"] += 1
                    continue
                try: roi_requests.put_nowait((request,time.monotonic()))
                except Full:
                    if isinstance(request,VisualRoiSeed):
                        roi_result(request,False,"ROI command queue full")
                    else:
                        # An undeliverable release must quiesce the trial,
                        # never leave its prior selection publishing.
                        stop.set()
                    counts["roi_command_queue_full"] += 1
            return True

        if roi_socket is not None:
            GLib.timeout_add(10,poll_roi_requests)

        def iterate_meta(node):
            while node is not None:
                try: yield pyds.NvDsFrameMeta.cast(node.data)
                except StopIteration: return
                try: node = node.next
                except StopIteration: return

        def rtp_identity(_pad,info,_data):
            from common.rtp_identity import parse_rtp_identity
            buffer = info.get_buffer()
            if buffer is None: return Gst.PadProbeReturn.OK
            ok,mapping = buffer.map(Gst.MapFlags.READ)
            if not ok: return Gst.PadProbeReturn.OK
            try:
                packet = parse_rtp_identity(bytes(mapping.data[:12]))
                if packet.marker and 0 <= int(buffer.pts) < (1<<63)-1:
                    transport.push_rtp_marker(decoded_pts_ns=int(buffer.pts),key=packet.key)
            except ValueError: pass
            finally: buffer.unmap(mapping)
            return Gst.PadProbeReturn.OK

        next_return_s = None
        def return_gate(_pad,info,_data):
            nonlocal next_return_s
            now = time.monotonic()
            period = 1/args.return_fps
            if next_return_s is not None and now < next_return_s-period*.25:
                return Gst.PadProbeReturn.DROP
            base = now if next_return_s is None or now-next_return_s > period else next_return_s
            next_return_s = base+period
            return Gst.PadProbeReturn.OK

        pending = {}

        def finish_frame(batch,meta,header,frame,result,was_seed,work_start):
            if header is not None and identity_socket is not None:
                return_mapper.record_verified_frame(int(meta.frame_num), SourceIdentity(
                    tracker_session,header.frame_id,header.source_time_ns,source_domain))
            if result is not None:
                transport.publish(result.snapshot)
                work_ms = (time.perf_counter_ns()-work_start)/1e6
                frame_work.append(work_ms)
                counts[result.decision.state] += 1
                counts["selected"] += int(result.snapshot.selection is not None)
                if result.snapshot.selection is not None:
                    if selected_times["first"] is None:
                        selected_times["first"] = frame.observed_time_ns
                    selected_times["last"] = frame.observed_time_ns
                timings.append(result.processing_ms)
                state_timings[result.decision.state].append(result.processing_ms)
                state_work[result.decision.state].append(work_ms)
                records.write(json.dumps(dict(frame_id=header.frame_id,source_time_ns=header.source_time_ns,
                    result_frame=result.snapshot.frame.model_dump(mode="json"),
                    decoded_pts_ns=int(meta.buf_pts), seed=was_seed,
                    observed_time_ns=frame.observed_time_ns,state=result.decision.state,reason=result.decision.reason,
                    box=result.decision.box,selected=result.snapshot.selection is not None,
                    owner=(handoff.stage if handoff and handoff.seed_frame_id is not None and frame.frame_id >= handoff.seed_frame_id else "vpi"),
                    native_track_id=handoff.native_id if handoff else None,
                    processing_ms=result.processing_ms,psr=result.proposal.psr if result.proposal else None,
                    features=result.proposal.seed_features if result.proposal else None,
                    input_features=result.proposal.input_features if result.proposal else None,
                    forward_features=result.proposal.forward_features if result.proposal else None,
                    backward_features=result.proposal.backward_features if result.proposal else None,
                    feature_ids=result.proposal.point_ids.tolist() if result.proposal is not None and result.proposal.point_ids is not None else None,
                    reference_features=len(result.proposal.reference_before) if result.proposal is not None and result.proposal.reference_before is not None else None,
                    dcf_box=result.proposal.box if result.proposal else None,
                    inliers=result.decision.inliers,scale=result.decision.scale,
                    anchor_scale=result.decision.anchor_scale,anchor_residual_px=result.decision.anchor_residual_px,
                    geometry_confident=result.decision.geometry_confident,
                    template_updated=result.decision.update_template,
                    commit_features=getattr(session._backend,"commit_stats",None),
                    frame_work_ms=work_ms))+"\n")
                if result.decision.reason.startswith("backend fault"):
                    raise RuntimeError(result.decision.reason)
            elif frame is not None:
                if handoff:
                    handoff.sequence += 1
                    sequence = handoff.sequence
                else:
                    session.sequence += 1
                    sequence = session.sequence
                transport.publish(PerceptionSnapshotV2(sequence=sequence,frame=frame))
            decorate(pyds,batch,meta,result,header.frame_id if header else "?",
                     (len(arrivals)-1)/max(arrivals[-1]-arrivals[0],.001),
                     "NvDCF" if handoff and handoff.stage == "nvdcf" else "VPI->NvDCF" if handoff else "VPI",source)
            if counts["decoded"] % 30 == 0:
                records.flush()
                health = dict(counts=counts,state=result.decision.state if result else session.lock.state,
                              box=result.decision.box if result else session.lock.box,
                              monotonic_s=time.monotonic(),control_disabled=True,
                              handoff=handoff.report() if handoff else None)
                health['return_recovery'] = return_recovery.stats()
                health['return_identity'] = dict(mapped=return_mapper.mapped,
                    withheld=return_mapper.withheld,ambiguous=return_mapper.ambiguous,
                    source_pts=[return_mapper.source_first_pts,return_mapper.source_last_pts],
                    encoded_pts=[return_mapper.encoded_first_pts,return_mapper.encoded_last_pts])
                (output/"health.json").write_text(json.dumps(health)+"\n")

        def probe(_pad,info,_data):
            nonlocal active_roi_request, pending_roi_seed, last_verified_s
            buffer = info.get_buffer()
            if buffer is None: return Gst.PadProbeReturn.OK
            if stop.is_set():
                close_session()
                outcome["closed_on_owner"] = True
                GLib.idle_add(loop.quit)
                return Gst.PadProbeReturn.DROP
            try:
                batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
                if batch is None: raise RuntimeError("missing batch metadata")
                for meta in iterate_meta(batch.frame_meta_list):
                    counts["decoded"] += 1
                    arrival = time.monotonic()
                    arrivals.append(arrival)
                    while len(arrivals)>2 and arrival-arrivals[0]>1:
                        arrivals.popleft()
                    if camera_frames is not None:
                        header = camera_frames.pop(int(meta.buf_pts))
                        if header is None:
                            # A camera metadata hole cannot inherit a cached seed or
                            # continue publishing the previous selection. End visibly.
                            raise RuntimeError("camera sensor identity missing or ambiguous for exact PTS")
                    else:
                        transport.drain_headers()
                        header = transport.next_header(decoded_pts_ns=int(meta.buf_pts))
                    result = None
                    frame,was_seed,work_start = None,False,time.perf_counter_ns()
                    while True:
                        try: command,received_s = roi_requests.get_nowait()
                        except Empty: break
                        if isinstance(command,VisualRoiRelease):
                            if (pending_roi_seed is not None
                                    and pending_roi_seed[0].request_id == command.request_id
                                    and pending_roi_seed[0].tracker_session == command.tracker_session):
                                roi_result(pending_roi_seed[0],False,"ROI seed cancelled before application")
                                pending_roi_seed = None
                            if (active_roi_request is not None
                                    and active_roi_request.request_id == command.request_id
                                    and active_roi_request.tracker_session == command.tracker_session):
                                close_session()
                                if handoff:
                                    handoff.fail("operator released visual lock")
                                else:
                                    outcome["seeded"] = False
                                    if frame_cache is not None: frame_cache.clear()
                                    prepared.append(BorrowedBackend(args.library,1280,720))
                                active_roi_request = None
                                counts["roi_released"] += 1
                        elif outcome["seeded"] or pending_roi_seed is not None:
                            roi_result(command,False,"ROI is already locked or awaiting a seed")
                        else:
                            pending_roi_seed = (command,received_s)
                    if header is not None:
                        last_verified_s = time.monotonic()
                        deadline.observe(time.monotonic())
                        counts["verified"] += 1
                        frame = PerceptionFrameV2(tracker_session=tracker_session,frame_id=header.frame_id,
                            source_time_ns=header.source_time_ns,source_clock_domain=source_domain,
                            source_identity_verified=True,observed_time_ns=time.monotonic_ns(),
                            observation_clock_domain="jetson_monotonic",width=1280,height=720)
                        if outcome["warmup_ms"] is None:
                            with borrower.borrow(buffer) as image:
                                outcome["warmup_ms"] = warm_kernels(args.library,image)
                            # Payload allocation is per instance (~160ms here),
                            # not merely process-global kernel warmup. Allocate
                            # the actual target backend before its seed clock.
                            prepare_start = time.perf_counter()
                            prepared.append(BorrowedBackend(args.library,1280,720))
                            outcome["prepare_ms"] = (time.perf_counter()-prepare_start)*1000
                        ready = outcome["seeded"] or cadence.observe(frame.source_time_ns)
                        if ready:
                            work_start = time.perf_counter_ns()
                            if handoff: handoff.begin(frame)
                            if handoff and handoff.stage == "lost": close_session()
                            if handoff is None or handoff.needs_vpi:
                                if frame_cache is not None and not outcome["seeded"]:
                                    frame_cache.add(frame,borrower.clone(buffer),decoded_pts_ns=int(meta.buf_pts))
                                    if pending_roi_seed is not None:
                                        request,received_s = pending_roi_seed
                                        pending_roi_seed = None
                                        replay = frame_cache.replay_from(request)
                                        if time.monotonic()-received_s > .5 or replay is None:
                                            roi_result(request,False,"exact source frame absent or stale")
                                            acquisitions.write(json.dumps(dict(request=request.as_message(),
                                                accepted=False,reason="exact source frame absent or stale",
                                                received_jetson_s=received_s,processed_jetson_s=time.monotonic(),
                                                steps=[]))+"\n")
                                            acquisitions.flush()
                                            counts["roi_rejected"] += 1
                                        else:
                                            box = NormalizedBoxV2(x=request.x/1280,y=request.y/720,
                                                w=request.w/1280,h=request.h/720)
                                            from jetson.visual_lock.frame_cache import bounded_catchup
                                            counts['roi_replay_available'] += len(replay)
                                            replay = bounded_catchup(replay,session.lock.gates.max_gap_ns)
                                            counts['roi_replay_sampled'] += len(replay)
                                            # The cache ends at this exact current image. A failure
                                            # must not be overwritten by updating an already-lost
                                            # session ("explicit new seed required").
                                            acquisition = acquire_cached(session,replay,box)
                                            result = acquisition.result
                                            counts["vpi_processed"] += len(acquisition.steps)
                                            acquisitions.write(json.dumps(dict(request=request.as_message(),
                                                accepted=acquisition.accepted,reason=acquisition.reason,
                                                received_jetson_s=received_s,processed_jetson_s=time.monotonic(),
                                                gates=asdict(session.lock.gates),
                                                steps=acquisition.audit(frame_cache.decoded_pts)))+"\n")
                                            acquisitions.flush()
                                            was_seed = request.frame_id == frame.frame_id
                                            accepted = acquisition.accepted
                                            outcome["seeded"] = accepted
                                            outcome["roi_seed_frame_id"] = request.frame_id
                                            active_roi_request = request if accepted else None
                                            roi_result(request,accepted,acquisition.reason)
                                            counts["roi_accepted" if accepted else "roi_rejected"] += 1
                                            frame_cache.clear()
                                            if not accepted and not handoff:
                                                prepared.append(BorrowedBackend(args.library,1280,720))
                                elif not outcome["seeded"]:
                                    with borrower.borrow(buffer) as image:
                                        x,y,w,h = args.seed_xywh
                                        result = session.seed(frame,image,NormalizedBoxV2(x=x/1280,y=y/720,w=w/1280,h=h/720))
                                    was_seed = True
                                    outcome["seeded"] = True
                                    counts["vpi_processed"] += 1
                                else:
                                    with borrower.borrow(buffer) as image:
                                        result = session.update(frame,image)
                                    counts["vpi_processed"] += 1
                            if handoff:
                                from jetson.visual_lock.nvdcf_handoff import inject_region
                                seed_box = handoff.initialize(frame,result)
                                inject_region(pyds,batch,meta,seed_box)
                                counts["native_seed_inputs"] += int(seed_box is not None)
                                if seed_box is not None:
                                    close_session()  # VPI owner closes before async native processing.
                                    outcome["vpi_closed_after_handoff"] = True
                        else:
                            counts["startup_wait"] += 1
                    if handoff:
                        if header is None or not ready: meta.bInferDone = False
                        if len(pending) >= 8: raise RuntimeError("native frame metadata queue overflow")
                        pending[int(meta.buf_pts)] = (header,frame,result,was_seed,work_start,header is not None and ready)
                    else: finish_frame(batch,meta,header,frame,result,was_seed,work_start)
            except Exception as exc:
                outcome["error"] = str(exc)
                close_session()
                outcome["closed_on_owner"] = True
                GLib.idle_add(loop.quit)
                return Gst.PadProbeReturn.DROP
            return Gst.PadProbeReturn.OK

        def native_output(_pad,info,_data):
            from jetson.visual_lock.nvdcf_handoff import native_tracks
            buffer = info.get_buffer()
            if buffer is None: return Gst.PadProbeReturn.OK
            try:
                batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
                for meta in iterate_meta(batch.frame_meta_list):
                    header,frame,vpi,was_seed,work_start,ready = pending.pop(int(meta.buf_pts))
                    tracks = native_tracks(pyds,meta)
                    result = None
                    if ready:
                        result = handoff.finish(frame,tracks,vpi,(time.perf_counter_ns()-work_start)/1e6)
                        counts["nvdcf_processed"] += 1
                    finish_frame(batch,meta,header,frame,result,was_seed,work_start)
            except Exception as exc:
                outcome["error"] = f"native handoff fault: {exc}"
                # Nvtracker's output thread does not own the VPI backend.
                # Quiesce the pipeline before final fallback cleanup instead
                # of destroying a possibly in-flight upstream image operation.
                stop.set()
                GLib.idle_add(loop.quit)
                return Gst.PadProbeReturn.DROP
            return Gst.PadProbeReturn.OK

        probes = []
        def attach(name,pad_name,callback,data,probe_type=None):
            pad = pipeline.get_by_name(name).get_static_pad(pad_name)
            probes.append((pad,pad.add_probe(probe_type or Gst.PadProbeType.BUFFER,callback,data)))
        if camera_frames is not None:
            from jetson.deepstream.argus_sensor_meta import ArgusSensorMetaReader
            sensor_reader = ArgusSensorMetaReader()
            def camera_identity(_pad,info,_data):
                buffer = info.get_buffer()
                if buffer is not None:
                    try:
                        camera_frames.record(int(buffer.pts),sensor_reader.read(buffer))
                    except Exception as exc:
                        outcome["error"] = f"camera sensor metadata fault: {exc}"
                        stop.set(); GLib.idle_add(loop.quit)
                return Gst.PadProbeReturn.OK
            attach("camera","src",camera_identity,None)
        else:
            attach("rtp_jitter","src",rtp_identity,None)
        def capture_input(_pad,info,_data):
            buffer = info.get_buffer()
            if buffer is not None:
                try:
                    if input_capture.stop_reason is None:
                        input_capture.append(buffer.extract_dup(0,buffer.get_size()),int(buffer.pts))
                except Exception as exc:
                    outcome["error"] = f"input capture fault: {exc}"
                    stop.set(); GLib.idle_add(loop.quit)
            return Gst.PadProbeReturn.OK
        if camera_frames is None:
            attach("input_parse","src",capture_input,None)
        attach("visual_input","src",probe,None)
        if handoff: attach("handoff_tracker","src",native_output,None)
        attach("return_rate","sink",return_gate,None)
        pts_evidence = {name:[] for name in ('encoder_input','encoder_output','return_pay')}
        def encoder_input(_pad,info,_data):
            buffer = info.get_buffer()
            if buffer is None or identity_socket is None:
                return Gst.PadProbeReturn.OK
            pts = int(buffer.pts)
            batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
            metas = [] if batch is None else list(iterate_meta(batch.frame_meta_list))
            if len(pts_evidence['encoder_input']) < 32:
                pts_evidence['encoder_input'].append([pts, [[int(m.frame_num),int(m.buf_pts)] for m in metas]])
            if len(metas) == 1 and 0 <= pts < (1<<63)-1:
                mapped = return_mapper.record_encoder_input(pts,int(metas[0].frame_num))
                counts['encoder_input_identity_mapped' if mapped else 'encoder_input_identity_missing'] += 1
            else:
                counts['encoder_input_metadata_missing'] += 1
            return Gst.PadProbeReturn.OK
        def encoder_output(_pad,info,_data):
            buffer = info.get_buffer()
            if buffer is not None and len(pts_evidence['encoder_output']) < 32:
                pts_evidence['encoder_output'].append(int(buffer.pts))
            return Gst.PadProbeReturn.OK
        attach('return_encoder','sink',encoder_input,None)
        attach('return_encoder','src',encoder_output,None)
        encoded_file = (output/"return.h264").open("wb")
        encoded_bytes = 0
        def encoded(_pad,_info,_data):
            nonlocal encoded_bytes
            counts["return_encoded"] += 1
            if not _info.get_buffer().has_flags(Gst.BufferFlags.DELTA_UNIT):
                counts['return_keyframes'] += 1
            # Bounded compressed-video evidence, not a GPU pixel readback.
            if encoded_bytes < 20_000_000:
                buffer = _info.get_buffer()
                packet = buffer.extract_dup(0,buffer.get_size())
                encoded_file.write(packet)
                encoded_bytes += len(packet)
            return Gst.PadProbeReturn.OK
        attach("return_parse","src",encoded,None)
        def inspect_return_rtp(buffer):
            from common.rtp_identity import parse_rtp_identity
            ok,mapping = buffer.map(Gst.MapFlags.READ)
            if not ok: return
            try:
                packet = parse_rtp_identity(bytes(mapping.data[:12]))
                counts['return_rtp_markers'] += int(packet.marker)
                pts = int(buffer.pts)
                payload = return_mapper.encoded_packet(pts if 0 <= pts < (1<<63)-1 else None,packet)
                if payload is not None:
                    try: identity_queue.put_nowait(json.dumps(payload,separators=(",",":")).encode())
                    except Full: counts["return_identity_queue_full"] += 1
                if packet.marker and len(pts_evidence['return_pay']) < 32:
                    pts_evidence['return_pay'].append(pts)
            except ValueError: counts["return_identity_invalid_rtp"] += 1
            finally: buffer.unmap(mapping)
        def return_rtp_identity(_pad,info,_data):
            if identity_socket is not None:
                for buffer in probe_buffers(info,buffer_flag=Gst.PadProbeType.BUFFER,
                                            buffer_list_flag=Gst.PadProbeType.BUFFER_LIST):
                    inspect_return_rtp(buffer)
            return Gst.PadProbeReturn.OK
        attach("return_wire","src",return_rtp_identity,None,
               Gst.PadProbeType.BUFFER | Gst.PadProbeType.BUFFER_LIST)
        def send_return_identity():
            if identity_socket is None: return False
            import zmq
            for _ in range(64):
                try: data = identity_queue.get_nowait()
                except Empty: break
                try:
                    identity_socket.send(data,flags=zmq.NOBLOCK)
                    counts["return_identity_published"] += 1
                except zmq.Again: counts["return_identity_send_full"] += 1
            return True
        if identity_socket is not None:
            GLib.timeout_add(10,send_return_identity)
        status_sequence = 0
        def send_roi_status():
            nonlocal status_sequence
            if roi_socket is None or identity_socket is None:
                return False
            import zmq
            status_sequence += 1
            ready = last_verified_s is not None and 0 <= time.monotonic()-last_verified_s <= .25
            try:
                identity_socket.send_json(dict(type="visual_roi_status_v1",source_mode=source,
                    tracker_session=tracker_session,sequence=status_sequence,
                    source_ready=ready,control_disabled=True),flags=zmq.NOBLOCK)
            except zmq.Again:
                counts["return_identity_send_full"] += 1
            return True
        if roi_socket is not None:
            GLib.timeout_add(100,send_roi_status)
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        def message(_bus,msg,_data):
            if msg.type == Gst.MessageType.ERROR:
                err,debug = msg.parse_error()
                outcome["error"] = f"{err}: {debug}"
                stop.set(); loop.quit()
            elif msg.type == Gst.MessageType.EOS: loop.quit()
        bus_handler = bus.connect("message",message,None)
        def request_stop():
            stop.set()
            GLib.timeout_add(1500,lambda:(loop.quit(),False)[1])
            return False
        def check_deadline():
            reason = deadline.reason(time.monotonic())
            if reason == "source startup timeout": outcome["error"] = reason
            return request_stop() if stop.is_set() or reason else True
        GLib.timeout_add(100,check_deadline)
        pipeline.set_state(Gst.State.PLAYING)
        (output/"ready.json").write_text(json.dumps(dict(control_disabled=True,receiver_ready=True))+"\n")
        print("LIVE visual-lock receiver ready; control OFF",flush=True)
        try:
            loop.run()
        finally:
            pipeline.set_state(Gst.State.NULL)
            pipeline.get_state(5*Gst.SECOND)
            close_session()  # idempotent after owning streaming thread cleanup.
            bus.remove_signal_watch()
            bus.disconnect(bus_handler)
            for pad,probe_id in probes: pad.remove_probe(probe_id)
            probes.clear()
            transport.close()
            records.close()
            acquisitions.close()
            encoded_file.close()
            input_capture.close()
            if frame_cache is not None: frame_cache.clear()
            if roi_socket is not None:
                roi_socket.close(0)
                roi_context.term()
            if identity_socket is not None:
                identity_socket.close(0)
                identity_context.term()
            bus = pipeline = None
            import gc
            gc.collect()
        import numpy as np
        report = dict(source=args.provenance,counts=dict(counts),outcome=outcome,duration_s=time.monotonic()-started,
            tracker="NVIDIA VPI CUDA DCF/Harris/forward-backward LK",seed_xywh=args.seed_xywh,
            input=("Argus sensor frame/SOF joined by exact PTS" if camera_frames is not None else
                   "live H264 RTP96, exact PC SSRC/timestamp/decoded-PTS identity"),return_fps=args.return_fps,
            gpu_input="borrowed NVMM EGLImage; no host pixel readback",egl_vendor=egl.vendor,
            processing_ms_p50=float(np.percentile(timings,50)) if timings else None,
            processing_ms_p95=float(np.percentile(timings,95)) if timings else None,
            library_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
            surface_sha256=hashlib.sha256(Path(args.surface_library).read_bytes()).hexdigest(),
            transport=transport.report(),control_disabled=True,detector=None,truth_input=False)
        report["input_capture"] = input_capture.report()
        report["source_mode"] = source
        if camera_frames is not None:
            report["camera_identity"] = dict(recorded=camera_frames.recorded,missing=camera_frames.missing,
                conflicts=camera_frames.conflicts,sensor_id=args.argus_sensor_id,sensor_mode=args.argus_sensor_mode,
                flip_method=args.argus_flip_method,wb_mode=args.argus_wb_mode)
            report["input_capture"]["unsupported_reason"] = "raw Argus input: no pre-OSD compressed AU capture"
        report['return_recovery'] = return_recovery.stats()
        report["return_identity"] = dict(bind=args.return_identity_bind,
            mapped=return_mapper.mapped,withheld=return_mapper.withheld,
            ambiguous=return_mapper.ambiguous,
            published=counts["return_identity_published"],queue_full=counts["return_identity_queue_full"])
        report['return_identity']['source_pts'] = [return_mapper.source_first_pts,return_mapper.source_last_pts]
        report['return_identity']['encoded_pts'] = [return_mapper.encoded_first_pts,return_mapper.encoded_last_pts]
        report['return_identity']['pts_evidence'] = pts_evidence
        report["panel_roi"] = dict(bind=args.roi_bind,accepted=counts["roi_accepted"],
            rejected=counts["roi_rejected"],invalid=counts["roi_invalid"],
            exact_gpu_copy=True if args.roi_bind else False)
        report["handoff"] = handoff.report() if handoff else None
        if handoff:
            report["tracker"] = "VPI CUDA initialization -> native DeepStream NvDCF (one external region)"
            report["nvdcf_profile_sha256"] = hashlib.sha256(native_profile.read_bytes()).hexdigest()
            report["processing_ms_scope"] = "VPI when acquiring + native tracker pipeline elapsed; no simultaneous YOLO"
        updates = counts["verified"]-counts["startup_wait"]-counts["acquiring"]
        report["selection_fraction_after_acquisition"] = counts["selected"]/max(1,updates)
        report["tracking_sustained"] = (counts["lost"] == 0 and counts["selected"] > 0
                                       and report["selection_fraction_after_acquisition"] >= .95)
        import platform
        report["host"] = platform.node()
        report["python"] = platform.python_version()
        report["native_abi"] = 4
        report["resolution"] = [1280,720]
        report["gates"] = asdict(session.lock.gates)
        report["gates_sha256"] = hashlib.sha256(json.dumps(report["gates"],sort_keys=True).encode()).hexdigest()
        report["clock_domains"] = dict(source=source_domain,observation="jetson_monotonic")
        report["processing_ms_scope"] = "session VPI/fusion/commit/V2 construction; excludes NVMM borrow, network and OSD/encode/display"
        if handoff:
            report["processing_ms_scope"] = "VPI when acquiring + elapsed native tracker pipeline; excludes OSD/encode/display and network"
        report["frame_work_ms_p95"] = float(np.percentile(frame_work,95)) if frame_work else None
        report["frame_work_ms_scope"] = "NVMM borrow + session + publish; excludes OSD/encode/display and source/receiver startup"
        if handoff:
            report["frame_work_ms_scope"] = "VPI when acquiring + native conversion/tracker elapsed + publish; excludes OSD/encode/display and network"
        report["timing_by_state"] = {state:dict(samples=len(values),
            processing_ms_p50=float(np.percentile(values,50)),processing_ms_p95=float(np.percentile(values,95)),
            frame_work_ms_p95=float(np.percentile(state_work[state],95))) for state,values in state_timings.items()}
        span = (selected_times["last"]-selected_times["first"])/1e9 if counts["selected"]>1 else 0
        report["selected_span_s"] = span
        report["selected_observed_hz"] = (counts["selected"]-1)/span if span else None
        report["live_video_duration_s"] = time.monotonic()-deadline.first_verified_s if deadline.first_verified_s is not None else 0
        report["source_startup_timeout_s"] = args.source_wait_s
        (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2),flush=True)
        if outcome["error"] or not report["tracking_sustained"] or not counts["return_encoded"] or (handoff and handoff.stage != "nvdcf"):
            return 1
        return 0


def resolve_snapshot_bind(value: str | None, *, panel_roi: bool) -> str:
    endpoint = value or ("tcp://0.0.0.0:5597" if panel_roi else "tcp://0.0.0.0:5564")
    if panel_roi and endpoint.endswith(":5564"):
        raise ValueError("panel ROI snapshots cannot use the ordinary detector/controller port 5564")
    return endpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",required=True)
    p.add_argument("--surface-library",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--source",choices=("sim","camera"),default="sim",help="exclusive control-free source owner")
    p.add_argument("--argus-sensor-id",type=int,default=0)
    p.add_argument("--argus-sensor-mode",type=int,default=4)
    p.add_argument("--argus-flip-method",type=int,default=0)
    p.add_argument("--argus-wb-mode",type=int,default=1)
    p.add_argument("--gates-json",help="explicit Gates overrides, recorded with acquisition/report")
    p.add_argument("--seed-xywh",type=float,nargs=4,
                   help="one-time fixed trial seed; omit when --roi-bind accepts a panel ROI")
    p.add_argument("--duration-s",type=float,default=30)
    p.add_argument("--source-wait-s",type=float,default=60)
    p.add_argument("--capture-max-mb",type=int,default=256,help="bounded whole-AU unannotated input evidence (1..1024 MB)")
    p.add_argument("--input-port",type=int,default=5000)
    p.add_argument("--return-port",type=int,default=5002)
    p.add_argument("--return-fps",type=int,default=30)
    p.add_argument('--return-keyframe-interval',type=int,default=1,
                   help='1 retains all-IDR fallback; predictive intervals require receiver recovery qualification')
    p.add_argument('--return-feedback-port',type=int,default=5003)
    p.add_argument('--return-sender-report-port',type=int,default=5004)
    p.add_argument("--return-host",action="append",required=True)
    p.add_argument("--header-bind",default="tcp://0.0.0.0:5555")
    p.add_argument("--snapshot-bind",default=None,
                   help="V2 output; panel ROI defaults to isolated port 5597, fixed trial to 5564")
    p.add_argument("--return-identity-bind",default=None,
                   help="opt-in exact source/return RTP identity sideband for the Pi")
    p.add_argument("--roi-bind",default=None,
                   help="opt-in software-only one-shot panel ROI receiver; requires return identity sideband")
    p.add_argument("--provenance",required=True)
    p.add_argument("--handoff-nvdcf",action="store_true",help="one-shot VPI initialization, then native NvDCF only")
    p.add_argument("--nvdcf-profile",default=str(Path(__file__).resolve().parents[2]/"configs/deepstream/tracker_nvdcf.yml"))
    args = p.parse_args()
    import math
    if (args.seed_xywh is None) == (args.roi_bind is None):
        p.error("choose exactly one of --seed-xywh or --roi-bind")
    if args.roi_bind and not args.return_identity_bind:
        p.error("panel ROI requires --return-identity-bind")
    if args.roi_bind and args.handoff_nvdcf:
        p.error("panel release/reseed requires VPI-only mode; NvDCF handoff is a fixed-seed trial")
    try:
        args.snapshot_bind = resolve_snapshot_bind(args.snapshot_bind, panel_roi=bool(args.roi_bind) or args.source == "camera")
        load_gates(args.gates_json)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    if args.seed_xywh is not None:
        x,y,w,h = args.seed_xywh
        if not all(math.isfinite(v) for v in args.seed_xywh) or x<0 or y<0 or w<12 or h<12 or x+w>1280 or y+h>720:
            p.error("invalid seed rectangle")
    if not math.isfinite(args.duration_s) or not 5 <= args.duration_s <= 1800:
        p.error("duration must be 5..1800 seconds")
    if not math.isfinite(args.source_wait_s) or not 5 <= args.source_wait_s <= 300:
        p.error("source wait must be 5..300 seconds")
    if not 1 <= args.capture_max_mb <= 1024: p.error("capture budget must be 1..1024 MB")
    pipeline_description(args.input_port,args.return_host,args.return_port,args.return_fps,
        return_keyframe_interval=args.return_keyframe_interval,feedback_port=args.return_feedback_port,
        sender_report_port=args.return_sender_report_port,source=args.source,
        sensor_id=args.argus_sensor_id,sensor_mode=args.argus_sensor_mode,
        flip_method=args.argus_flip_method,wb_mode=args.argus_wb_mode)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
