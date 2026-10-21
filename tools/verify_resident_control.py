"""Bounded native ROI -> real controller -> stable software mount test.

No sockets, services, GPIO or motor APIs. Run alone under the software GPU
guard. CPU-rendered deterministic texture is uploaded to CUDA as test input;
truth is used for the initial operator seed and scoring, never tracker updates.
This qualifies a component loop, not real-camera or physical-mount dynamics.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np

from common.control import AxisPair,ControlConfig,LaserAimingControlConfig
from common.perception import PerceptionFrameV2
from common.schemas import CamState,ManualControlState
from common.visual_roi_seed import VisualRoiMode,VisualRoiSelect,VisualRoiSeed,VisualRoiRelease
from jetson.control.pid import AxisPIDConfig,BasicPID
from jetson.control.timing import ClockBounds
from jetson.control.video_controller import VideoControllerCore,VideoControllerPolicy
from jetson.control_observation import ControlObservationAssembler
from jetson.deepstream.resident_roi import RoiWorker
from jetson.deepstream.roi_runtime import RoiRuntime
from jetson.visual_lock.egl_lifetime import StandaloneEGL
from tools.visual_lock_replay_gpu import ReplayGPUBackend


class Pixels:
    def __init__(self,array):self.array=array
    def close(self):self.array=None


class StandaloneWorker(RoiWorker):
    def run(self):
        # StandaloneEGL deliberately checks thread ownership. Bootstrap and
        # terminate it on the same native owner that constructs/uses VPI.
        with StandaloneEGL() as self.egl:
            super().run()


def run(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--duration-s',type=float,default=18.)
    args=p.parse_args(argv)
    if not 16 <= args.duration_s <= 30:p.error('duration must be 16..30 seconds')
    args.output.mkdir(parents=True,exist_ok=False)
    config=ControlConfig(fx_px=1000.,fy_px=1000.,cx_px=640.,cy_px=360.,aim_mode='camera_center',
        rate_limits=AxisPair(.5,.5),accel_limits=AxisPair(2.,2.),yaw_sign=1,pitch_sign=-1,
        frame_size=(1280,720),fov_deg=None,laser=LaserAimingControlConfig(1,'known_size',10))
    axis=AxisPIDConfig(2.5,0.,0.,0.,.5,2.)
    core=VideoControllerCore(BasicPID(axis,axis),VideoControllerPolicy(live_authorized=True,
        source_clock_domain='jetson_monotonic',max_capture_age_ns=150_000_000,max_travel_rad=.5))
    assembler=ControlObservationAssembler(config)
    report=dict(passed=False,library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        source_clock='one Jetson monotonic clock; exact zero clock offset, no LAN assumption',
        plant='stable software rate integrator with 50ms first-order rate response; no hardware gains',
        seed=736,frames=0,selected=0,safety_zero_failures=0,queue_drops=0)
    rows=[];reasons=Counter();errors=[];localization=[]
    rng=np.random.default_rng(736)
    texture=np.repeat(np.repeat(rng.integers(30,240,(27,36,3),dtype=np.uint8),4,0),4,1)
    session=uuid.uuid4().hex
    def exercise():
        started=time.monotonic();last=started;rates=np.zeros(2);pose=np.zeros(2);demand=np.zeros(2)
        class Backend(ReplayGPUBackend):
            def __init__(self,w,h):super().__init__(args.library,w,h,kind='cuda',egl_owner=worker.egl)
            def process(self,image,**kwargs):return super().process(image.array,**kwargs)
        roi=RoiRuntime('camera',session,build=args.library.parent,evidence_root=args.output/'native',
            factory=lambda _:(object(),Backend))
        roi.explicit_modes=True
        worker=StandaloneWorker(roi)
        request=None
        try:
            roi.enqueue(VisualRoiMode(str(uuid.uuid4()),session,'roi',0).as_message());worker.wake()
            end=time.monotonic()+2
            while roi.processing_mode!='roi' and time.monotonic()<end:time.sleep(.001)
            if roi.processing_mode!='roi':raise RuntimeError('ROI mode did not apply')
            lease=VisualRoiSelect(str(uuid.uuid4()),session,True)
            roi.enqueue(lease.as_message());worker.wake()
            started=last=time.monotonic()
            first=None
            for i in range(round(args.duration_s*60)):
                deadline=started+i/60
                remaining=deadline-time.monotonic()
                if remaining>0:time.sleep(remaining)
                now=time.monotonic();dt=min(.1,now-last);last=now;t=now-started
                rates+=(demand-rates)*(1-math.exp(-dt/.05));pose+=rates*dt
                target=np.array((.07+.1*math.sin(.45*t),.04+.05*math.sin(.63*t)))
                cx=640+1000*math.tan(target[0]-pose[0]);cy=360-1000*math.tan(target[1]-pose[1])
                x,y=round(cx-72),round(cy-54)
                if not (0<=x<=1136 and 0<=y<=612):raise RuntimeError('fixture target left image')
                image=np.full((720,1280,4),32,np.uint8);image[:,:,3]=255
                image[y:y+108,x:x+144,:3]=texture
                stamp=time.monotonic_ns()
                frame=PerceptionFrameV2(tracker_session=session,frame_id=i,source_time_ns=stamp,
                    received_time_ns=stamp,observed_time_ns=stamp,source_clock_domain='jetson_monotonic',
                    receive_clock_domain='jetson_monotonic',observation_clock_domain='jetson_monotonic',
                    source_identity_verified=True,width=1280,height=720)
                if first is None:first=(frame,x,y)
                # Renewal uses the source's own fixture clock; no periodic truth
                # box is ever supplied after the one explicit seed below.
                roi.enqueue(lease.as_message())
                worker.submit(frame,Pixels(image),i)
                if request is None and roi.cache._frames:
                    f,sx,sy=first
                    request=VisualRoiSeed(str(uuid.uuid4()),session,f.frame_id,f.source_time_ns,
                                         f.source_clock_domain,sx,sy,144,108)
                    roi.enqueue(request.as_message())
                value=worker.latest()
                if value is not None:
                    snap=value[1]
                    assembler.update_perception_snapshot(snap,received_at=time.monotonic())
                    if snap.selection is not None:
                        report['selected']+=1
                        selected=snap.tracks[0].box
                        # Score only the true source frame, not current pose.
                        old=rows[snap.frame.frame_id] if snap.frame.frame_id<len(rows) else None
                        if old is not None:localization.append(math.hypot(
                            (selected.x+selected.w/2)*1280-old['cx'],
                            (selected.y+selected.h/2)*720-old['cy']))
                emergency=7<=t<8
                auto=not (11<=t<12)
                manual=14<=t<15
                assembler.update_manual_state(ManualControlState(src_ts_ms=0,source='isolated-fixture',
                    active=manual,emergency=emergency,control_cmd_enabled=auto,control_cmd_valid=True,
                    joystick_raw=(0,0),joystick_rate_cmd=(0.,0.)),received_at=time.monotonic())
                assembler.update_cam_state(CamState(frame_id=i,src_ts_ms=0,pan=float(pose[0]),tilt=float(pose[1]),
                    pan_rate=float(rates[0]),tilt_rate=float(rates[1])),received_at=time.monotonic())
                obs=assembler.build(now=time.monotonic())
                decision=core.decide(obs,ClockBounds(0,0,obs.created_monotonic_ns,0.))
                demand=np.array((decision.intent.yaw_rate_rad_s,decision.intent.pitch_rate_rad_s))
                reasons[decision.intent.reason]+=1
                if (emergency or not auto or manual) and np.any(demand!=0):report['safety_zero_failures']+=1
                error=float(np.linalg.norm(target-pose))
                if t>3 and not emergency and auto and not manual:errors.append(error)
                rows.append(dict(i=i,t=t,cx=cx,cy=cy,error_rad=error,pose=pose.tolist(),rate=demand.tolist(),
                    reason=decision.intent.reason,selected=obs.target.valid))
                report['frames']+=1
                if worker.error:raise RuntimeError(worker.error)
                if request is not None and roi.terminal.get('request_id')==request.request_id:
                    raise RuntimeError('ROI ended: '+str(roi.terminal.get('reason')))
            report['elapsed_s']=time.monotonic()-started
            report['input_fps']=report['frames']/report['elapsed_s']
            report['queue_drops']=worker.dropped
            report['tracking_lag_ms']=list(worker.tracking_lag_ms)
            report['tracking_native_ms']=list(worker.tracking_native_ms)
            report['reasons']=dict(reasons)
            report['bearing_error_p95_rad']=float(np.percentile(errors,95)) if errors else None
            report['localization_p95_px']=float(np.percentile(localization,95)) if localization else None
            report['passed']=(report['safety_zero_failures']==0 and report['input_fps']>=55
                and reasons.get('tracking',0)>report['frames']*.6 and errors
                and report['bearing_error_p95_rad']<.06 and localization
                and report['localization_p95_px']<4)
        except Exception as exc:report['error']=repr(exc)
        finally:
            if request is not None:roi.enqueue(VisualRoiRelease(request.request_id,session).as_message())
            worker.close()
    exercise()
    (args.output/'trace.json').write_text(json.dumps(rows)+'\n')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('tracking_lag_ms','tracking_native_ms')}),flush=True)
    return int(not report['passed'])


if __name__=='__main__':raise SystemExit(run())
