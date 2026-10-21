"""Resident processing switch with video-independent, bounded ROI work.

One permanent NVMM path/encoder. GPU copies isolate tracker input from OSD;
neither initialization nor tracking is awaited by a video callback. Completed
observations keep their actual source frame/timestamps, never the display frame.
"""
import threading
import time
import os
from collections import deque

from common.perception import RoiStateV2
from common.visual_roi_seed import VisualRoiMode, VisualRoiSeed
from jetson.deepstream.roi_pipeline import RoiPipeline, roi_probe


class RoiWorker:
    """Single native owner, bounded pending images; no socket or GPIO access."""
    def __init__(self, roi, capacity=16):
        self.roi = roi
        self.pending = deque()
        self.results = deque(maxlen=4)
        self.condition = threading.Condition()
        self.capacity = capacity
        self.stopping = False
        self.busy = False
        self.error = None
        self.verify_count = int(os.environ.get('IDCS_RESIDENT_VERIFY_INPUT','0'))
        self.verifier = None
        self.submitted = self.processed = self.dropped = 0
        self.tracking_lag_ms=deque(maxlen=600)
        self.tracking_native_ms=deque(maxlen=600)
        self.thread = threading.Thread(target=self.run, name='resident-roi', daemon=True)
        self.thread.start()

    def submit(self, frame, image, pts):
        with self.condition:
            if self.stopping:
                image.close()
                return
            if len(self.pending) >= self.capacity:
                _, old, _ = self.pending.popleft()
                old.close()
                self.dropped += 1
            self.pending.append((frame,image,pts))
            self.submitted += 1
            self.condition.notify()

    def wake(self):
        with self.condition: self.condition.notify()

    def latest(self):
        with self.condition:
            value = self.results[-1] if self.results else None
            self.results.clear()
            return value

    def run(self):
        roi = self.roi
        try:
            while True:
                with self.condition:
                    if not self.pending and not self.stopping:
                        self.condition.wait(.01)
                    if self.stopping: break
                    self.busy = True
                    item = self.pending.popleft() if self.pending else None
                # Commands/expiry still progress with no source frames. Their
                # mutations and all VPI/cache lifetime stay on this thread.
                roi.begin_frame(allow_seed=True, count_frame=False)
                roi.service()
                if item is not None:
                    frame,image,pts = item
                    try:
                        if self.verify_count > 0:
                            if self.verifier is None:
                                from jetson.visual_lock.live_surface import CudaInputVerifier
                                self.verifier=CudaInputVerifier(roi.build/'libidcs_visual_lock.so')
                            if self.verifier.check(image) is not True:
                                raise RuntimeError('resident CUDA clone differs from fresh EGL pixels')
                            self.verify_count-=1
                        result = (roi.process(frame,None,decoded_pts_ns=pts,owned_image=image)
                                  if roi.processing_mode == 'roi' else None)
                        if roi.processing_mode == 'roi':
                            snapshot = roi.snapshot(frame,result)
                            completed = time.monotonic_ns()
                            if snapshot.selection is not None:
                                self.tracking_lag_ms.append((completed-frame.observed_time_ns)/1e6)
                                self.tracking_native_ms.append(result.processing_ms)
                            snapshot = snapshot.model_copy(update={
                                'frame':snapshot.frame.model_copy(update={'observed_time_ns':completed}),
                                'roi_state':RoiStateV2(capable=roi.capable,owns_selection=True,
                                    request_id=None if roi.active is None else roi.active.request_id)})
                            with self.condition:
                                self.results.append((roi.mode_revision,snapshot,result,completed))
                        self.processed += 1
                    finally:
                        if not any(cached is image for _,cached in roi.cache._frames):
                            image.close()
                with self.condition: self.busy = False
        except Exception as exc:
            self.error = str(exc)
            roi.fault('resident ROI worker: '+str(exc))
        finally:
            if self.verifier is not None:self.verifier.close()
            try: roi.close()
            except Exception as exc: self.error = 'ROI worker cleanup: '+str(exc)
            with self.condition:
                while self.pending:
                    _,image,_ = self.pending.popleft()
                    image.close()
                self.busy = False

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify()
        # Never destroy EGL while a native call still owns it.
        self.thread.join()
        if self.error: raise RuntimeError(self.error)


class ResidentRoiPipeline(RoiPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.roi.managed_window = False
        self.roi.explicit_modes = True
        self.owner = self.pipeline.get_by_name('roi_owner')
        if self.owner is None: raise RuntimeError('resident ROI queue missing')
        self.metadata_pad.remove_probe(self.metadata_id)
        self.active = True
        from jetson.visual_lock.live_surface import NVMMClonePool
        try:self.pool=NVMMClonePool(self.roi.borrower) if self.roi.capable else None
        except Exception as exc:
            self.pool=None
            self.roi.fault('resident native pool unavailable: '+str(exc))
        self.worker = RoiWorker(self.roi)
        self.desired_mode = 'yolo'
        self.pending_mode = None
        self.completed = None
        self.last_draw = None
        self.last_revision = 0
        self.attach('osd','sink',self.metadata_probe,data=self.metadata_data)
        self.attach('mux','src',self.route)
        self.attach('primary','sink',self.gpu_entry)
        if self.pipeline.get_by_name('tracker') is not None:
            self.attach('tracker','sink',self.gpu_entry)
        self.attach('encoder','sink',self.encoder_input)
        self.attach('return_wire','src',self.return_rtp,
                    self.gst.PadProbeType.BUFFER | self.gst.PadProbeType.BUFFER_LIST)

    def sync_identity(self):
        # Fixed probes, cheap disabled branch. Never relink or remove an owner
        # while the video is running. History is still per selecting lease.
        wanted = self.roi.cache_enabled
        if wanted != self.identity_live:
            from jetson.visual_lock.return_identity import ReturnRtpMapper
            with self.frames_lock:
                self.identity_live = wanted
                self.identity_generation += 1
                self.identity_frames.clear()
                self.mapper = ReturnRtpMapper()

    def lifecycle(self):
        if not hasattr(self,'worker'): return
        if (self.pending_mode is not None
                and self.roi.mode_result.get('mode_command_id')==self.pending_mode):
            self.desired_mode=self.roi.processing_mode
            self.pending_mode=None
        if self.roi.mode_revision != self.last_revision:
            self.last_revision = self.roi.mode_revision
            self.completed = self.last_draw = None
        self.sync_identity()
        self.metadata_data[1].roi_idle_state = RoiStateV2(capable=self.roi.capable,
            owns_selection=self.roi.processing_mode=='roi')
        self.roi.probe_status = self.probe_state()

    def receive(self, message):
        if message.get('type') == 'visual_roi_mode_v1':
            command = VisualRoiMode.from_mapping(message)
            # Route away from inference immediately on entry, but do not reopen
            # YOLO until the worker acknowledges leaving ROI after native work.
            if (command.tracker_session == self.roi.tracker_session
                    and command.expected_revision == self.roi.mode_revision
                    and (command.mode=='yolo' or self.roi.capable)):
                self.desired_mode = command.mode
                self.pending_mode = command.command_id
            self.roi.enqueue(message)
        else:
            if message.get('type') == 'visual_roi_seed_v1':
                request = VisualRoiSeed.from_mapping(message)
                from jetson.visual_lock.return_identity import SourceIdentity
                key=SourceIdentity(request.tracker_session,request.frame_id,
                                   request.source_time_ns,request.source_clock_domain)
                with self.frames_lock: ready=key in self.identity_frames
                if not ready or not self.identity_live:
                    self.roi._notify(request,False,'exact displayed frame absent; wait for ROI ready')
                    return
            self.roi.enqueue(message)
        self.worker.wake()

    @roi_probe
    def route(self, _pad, info, _data):
        buffer = info.get_buffer()
        if buffer is None: return self.gst.PadProbeReturn.OK
        skipped = self.desired_mode=='roi' or self.roi.processing_mode=='roi'
        pts = int(buffer.pts)
        with self.frames_lock: self.frames[pts]=(None,skipped)
        self.roi.yolo_skipped = skipped
        if self.selector is not None and self.selector_paused != skipped:
            self.selector.pause_for_roi() if skipped else self.selector.resume_from_roi()
            self.selector_paused = skipped
        if skipped:
            self.stage_clock.record_infer_input(time.monotonic(),pts_ns=pts)
            self.roi.counts['yolo_skipped'] += 1
            if self.owner.get_static_pad('sink').chain(buffer) != self.gst.FlowReturn.OK:
                self.fault('resident ROI video bypass failed')
            return self.gst.PadProbeReturn.DROP
        self.roi.counts['yolo_frames'] += 1
        return self.gst.PadProbeReturn.OK

    def metadata_probe(self, pad, info, data):
        from jetson.deepstream.pipeline import _metadata_probe
        buffer = info.get_buffer()
        pts = None if buffer is None else int(buffer.pts)
        with self.frames_lock: self.metadata_skipped=self.frames.get(pts,(None,False))[1]
        try:
            return _metadata_probe(pad,info,data)
        except Exception as exc:
            self.fault('resident metadata: '+str(exc))
            return self.gst.PadProbeReturn.OK
        finally:
            with self.frames_lock: self.frames.pop(pts,None)

    def process_frame(self, frame, buffer, pts):
        if self.stopping:return None
        if self.pool is not None and self.roi.processing_mode=='roi' and (self.roi.cache_enabled or self.roi.owns_selection):
            # Independent NVMM copy, before nvdsosd modifies the video surface.
            # A descriptor/ref alone would alias reused/OSD-mutated pixels.
            image=self.pool.acquire(buffer)
            if image is not None:self.worker.submit(frame,image,pts)
            else:self.roi.counts['pool_exhaustion_drops']+=1
        value=self.worker.latest()
        self.completed = value if value is not None and value[0]==self.roi.mode_revision else None
        if self.completed is not None: self.last_draw=self.completed
        if self.last_draw is not None and time.monotonic_ns()-self.last_draw[3] > 150_000_000:
            self.last_draw=None
        return None if self.last_draw is None else self.last_draw[2]

    def publication(self):
        return None if self.completed is None else self.completed[1]

    def probe_state(self):
        value=super().probe_state()
        if hasattr(self,'worker'):
            value.update(resident=True,roi_worker_pending=len(self.worker.pending),
                roi_worker_processed=self.worker.processed,roi_worker_dropped=self.worker.dropped,
                roi_worker_error=self.worker.error)
            if self.worker.verifier is not None:
                value['roi_input_verification']=self.worker.verifier.report()
            for name,samples in (('roi_tracking_lag_ms',self.worker.tracking_lag_ms),
                                 ('roi_tracking_native_ms',self.worker.tracking_native_ms)):
                values=sorted(samples)
                value[name]=None if not values else dict(count=len(values),median=values[len(values)//2],
                    p95=values[min(len(values)-1,int(.95*len(values)))],maximum=values[-1])
        if getattr(self,'pool',None) is not None:value.update(roi_pool_images=len(self.pool.images),roi_pool_capacity=self.pool.capacity)
        return value

    def close(self):
        self.stopping=True
        errors=[]
        def cleanup(action):
            try:action()
            except Exception as exc:errors.append(str(exc))
        # close() joins the native owner before reporting its error. A failed
        # tracker must not skip pool/socket/probe teardown during shutdown.
        if hasattr(self,'worker'):cleanup(self.worker.close)
        if getattr(self,'pool',None) is not None:cleanup(self.pool.close)
        cleanup(lambda:self.glib.source_remove(self.timer))
        for sock in (self.pull,self.pub):
            if sock is not None:cleanup(lambda sock=sock:sock.close(0))
        cleanup(self.context.term)
        for name,pad,_ in list(self.probes):cleanup(lambda name=name,pad=pad:self.detach(name,pad))
        self.active=self.identity_live=False
        if errors:raise RuntimeError('resident ROI cleanup: '+'; '.join(errors))
