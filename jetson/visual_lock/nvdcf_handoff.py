"""One explicit VPI image-region initialization, then native NvDCF ownership.

No YOLO, truth, repeated detector injection, automatic reseed or motion control.
The native track ID is pinned after initialization; the public visual-lock ID
stays stable within the source/session. No internal filter state is transferred.
"""
from dataclasses import dataclass
from functools import wraps
import math
from threading import RLock

from common.perception import (NormalizedBoxV2, PerceptionSnapshotV2,
                               PerceptionTrackV2, TargetSelectionV2)
from .fusion import Decision
from .session import Output


VISUAL_CLASS_ID = 65535
VISUAL_COMPONENT_ID = 9001
UNTRACKED_ID = (1 << 64) - 1


def serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._mutex:
            return method(self,*args,**kwargs)
    return call


@dataclass(frozen=True)
class NativeTrack:
    track_id: int
    box: tuple[float, float, float, float]
    confidence: float
    class_id: int = VISUAL_CLASS_ID


class NvDCFHandoff:
    def __init__(self, confirm_frames=3, min_confidence=.2, max_gap_ns=150_000_000):
        self._mutex = RLock()  # Upstream input and native output are different threads.
        if type(confirm_frames) is not int or confirm_frames < 1:
            raise ValueError("positive confirmation frame count required")
        if not math.isfinite(min_confidence) or not 0 < min_confidence <= 1:
            raise ValueError("invalid native confidence gate")
        if type(max_gap_ns) is not int or max_gap_ns <= 0:
            raise ValueError("positive source gap required")
        self.confirm_frames, self.min_confidence, self.max_gap_ns = confirm_frames,min_confidence,max_gap_ns
        self.stage = "vpi"
        self.reason = "waiting for validated VPI selection"
        self.previous = self.last_output = None
        self.pending = {}
        self.native_id = None
        self.box = None
        self.good = self.bad = self.sequence = self.age = self.seed_count = 0
        self.public_id = 1
        self.seed_frame_id = self.completed_frame_id = None
        self.confirm_age = 0

    @property
    def needs_vpi(self):
        return self.stage == "vpi"

    @serialized
    def fail(self, reason):
        self.stage, self.reason = "lost",reason

    @serialized
    def begin(self, frame):
        """Validate before image work; return at most one current-frame seed."""
        if len(self.pending) >= 8:
            self.fail("native handoff queue overflow")
            return None
        self.pending[frame.frame_id] = frame
        if self.stage == "lost": return None
        if not frame.source_identity_verified or not frame.tracker_session:
            self.fail("unverified handoff identity"); return None
        previous = self.previous
        if previous is not None:
            identity = lambda f: (f.tracker_session,f.source_clock_domain,f.observation_clock_domain,f.width,f.height)
            if identity(previous) != identity(frame):
                self.fail("handoff source/session/geometry changed"); return None
            if frame.frame_id <= previous.frame_id or frame.source_time_ns <= previous.source_time_ns:
                self.fail("handoff frame order invalid"); return None
            if frame.observed_time_ns < previous.observed_time_ns:
                self.fail("handoff observation clock reversed"); return None
            if frame.source_time_ns-previous.source_time_ns > self.max_gap_ns:
                self.fail("handoff source gap"); return None
        self.previous = frame
        return None

    @serialized
    def initialize(self, frame, vpi):
        """After VPI processed THIS frame, supply a one-shot external region."""
        if self.stage != "vpi" or self.pending.get(frame.frame_id) != frame or vpi is None:
            return None
        if vpi.snapshot.frame != frame:
            self.fail("VPI seed frame mismatch"); return None
        if vpi.decision.state == "lost":
            self.fail("VPI initialization lost"); return None
        if vpi.snapshot.selection is None: return None
        self.stage,self.reason = "confirming","native initialization pending agreement"
        self.public_id = vpi.snapshot.selection.track_id
        self.seed_count += 1
        self.seed_frame_id = frame.frame_id
        return vpi.decision.box

    @serialized
    def finish(self, frame, candidates, vpi=None, processing_ms=0.):
        if self.pending.pop(frame.frame_id,None) != frame:
            self.fail("native output frame mismatch")
        if self.last_output is not None and frame.frame_id <= self.last_output.frame_id:
            self.fail("native output order invalid")
        self.last_output = frame
        self.sequence += 1
        self.age += 1
        proposal = vpi.proposal if vpi else None
        if self.stage == "vpi" or (self.stage != "lost" and self.seed_frame_id is not None and frame.frame_id < self.seed_frame_id):
            if vpi is not None and vpi.decision.state == "lost": self.fail("VPI initialization lost")
            decision = vpi.decision if vpi and self.stage != "lost" else Decision("lost",self.reason)
            return Output(decision,PerceptionSnapshotV2(sequence=self.sequence,frame=frame),proposal,processing_ms)
        if self.stage != "lost":
            if self.stage == "confirming":
                self.confirm_age += 1
                if self.confirm_age > 12: self.fail("native confirmation timeout")
        if self.stage != "lost":
            reason = self._check_candidates(frame,candidates,vpi)
            if reason:
                self.bad += 1; self.good = 0
                # A changed identity/ambiguous native object is not recoverable
                # by guessing. Ordinary confidence loss gets three gated frames.
                if self.bad >= 3 or reason in ("native identity changed","ambiguous native tracks","invalid native geometry"):
                    self.fail(reason)
                else: self.reason = reason
            else:
                native = candidates[0]
                if self.native_id is None: self.native_id = native.track_id
                self.box = native.box
                self.bad = 0; self.good += 1
                if self.stage == "confirming" and self.good >= self.confirm_frames:
                    self.stage = "nvdcf"
                    self.completed_frame_id = frame.frame_id
                self.reason = "native visual tracking" if self.stage == "nvdcf" else "native/VPI agreement confirming"
        selected = self.stage == "nvdcf" and self.bad == 0 and self.good >= self.confirm_frames
        state = "tracking" if selected else "lost" if self.stage == "lost" else "uncertain" if self.bad else "acquiring"
        decision = Decision(state,self.reason,self.box,accepted=selected)
        tracks,selection = (),None
        if selected:
            x,y,w,h = self.box
            tracks = (PerceptionTrackV2(track_id=self.public_id,
                box=NormalizedBoxV2(x=x/frame.width,y=y/frame.height,w=w/frame.width,h=h/frame.height),
                class_id="visual",confidence=candidates[0].confidence,age_frames=self.age,missed_frames=0),)
            selection = TargetSelectionV2(track_id=self.public_id,source_frame_id=frame.frame_id,
                applied_frame_id=frame.frame_id,selected_time_ns=frame.observed_time_ns,
                selection_clock_domain=frame.observation_clock_domain,policy="operator_vpi_to_nvdcf")
        return Output(decision,PerceptionSnapshotV2(sequence=self.sequence,frame=frame,tracks=tracks,selection=selection),
                      proposal,processing_ms)

    def _check_candidates(self, frame, candidates, vpi):
        if len(candidates) > 1: return "ambiguous native tracks"
        if not candidates: return "native target unavailable"
        native = candidates[0]
        if native.class_id != VISUAL_CLASS_ID or type(native.track_id) is not int or not 0 <= native.track_id < UNTRACKED_ID:
            return "native identity changed"
        if self.native_id is not None and native.track_id != self.native_id: return "native identity changed"
        if not math.isfinite(native.confidence) or not self.min_confidence <= native.confidence <= 1:
            return "weak native confidence"
        x,y,w,h = native.box
        if not all(math.isfinite(v) for v in native.box) or min(w,h)<12 or min(x,y)<0 or x+w>frame.width or y+h>frame.height:
            return "invalid native geometry"
        if self.stage == "confirming" and frame.frame_id == self.seed_frame_id:
            if vpi is None or vpi.snapshot.frame != frame or vpi.snapshot.selection is None:
                return "VPI agreement unavailable"
            px,py,pw,ph = vpi.decision.box
            distance = math.hypot(x+w/2-px-pw/2,y+h/2-py-ph/2)
            if distance > max(3.,.15*min(pw,ph)) or not .8 <= w/pw <= 1.2 or not .8 <= h/ph <= 1.2:
                return "native/VPI disagreement"
        elif self.stage == "confirming" and self.native_id is None:
            return "VPI agreement unavailable"
        if self.box is not None:
            px,py,pw,ph = self.box
            if math.hypot(x+w/2-px-pw/2,y+h/2-py-ph/2) > max(12.,.5*max(pw,ph)):
                return "native position jump"
            if not .9 <= w/pw <= 1.1 or not .9 <= h/ph <= 1.1:
                return "native scale jump"
        return None

    @serialized
    def report(self):
        return dict(stage=self.stage,reason=self.reason,seed_count=self.seed_count,
                    seed_frame_id=self.seed_frame_id,completed_frame_id=self.completed_frame_id,
                    native_track_id=self.native_id,confirm_frames=self.confirm_frames,
                    min_confidence=self.min_confidence,max_gap_ns=self.max_gap_ns,
                    recurrent_seed=False,filter_state_transferred=False)


def write_operator_profile(source, destination):
    """Derive a dedicated one-target profile; never rewrite production config."""
    import yaml
    text = source.read_text(encoding="utf-8")
    profile = yaml.safe_load("\n".join(line for line in text.splitlines() if not line.startswith("%YAML:")))
    if profile["VisualTracker"]["visualTrackerType"] != 2:
        raise ValueError("VPI-backed native NvDCF profile required")
    profile["TargetManagement"].update(maxTargetsPerStream=1,probationAge=0,outputShadowTracks=0)
    destination.write_text("%YAML:1.0\n---\n"+yaml.safe_dump(profile,sort_keys=False),encoding="utf-8")
    return profile


def inject_region(pyds, batch, meta, box):
    """Explicit external-ROI metadata, never labeled a YOLO detection."""
    meta.bInferDone = False
    if box is None: return
    obj = pyds.nvds_acquire_obj_meta_from_pool(batch)
    if obj is None: raise RuntimeError("native initialization metadata unavailable")
    obj.object_id,obj.class_id,obj.unique_component_id = UNTRACKED_ID,VISUAL_CLASS_ID,VISUAL_COMPONENT_ID
    obj.confidence = 1.  # Validated input-region weight, not classifier probability.
    obj.obj_label = "visual"
    obj.rect_params.left,obj.rect_params.top,obj.rect_params.width,obj.rect_params.height = box
    coords = obj.detector_bbox_info.org_bbox_coords
    coords.left,coords.top,coords.width,coords.height = box
    pyds.nvds_add_obj_meta_to_frame(meta,obj,None)
    meta.bInferDone = True  # The only frame with an external region registration.


def native_tracks(pyds, meta):
    objects = []
    node = meta.obj_meta_list
    while node is not None:
        obj = pyds.NvDsObjectMeta.cast(node.data)
        rect = obj.rect_params
        objects.append(NativeTrack(int(obj.object_id),(float(rect.left),float(rect.top),float(rect.width),float(rect.height)),
                                   float(obj.tracker_confidence),int(obj.class_id)))
        # Native metadata is hidden from OSD: only coordinator-accepted boxes
        # are drawn. A weak/old native estimate must not look like selection.
        rect.border_width = 0
        obj.text_params.display_text = ""
        try: node = node.next
        except StopIteration: break
    return objects
