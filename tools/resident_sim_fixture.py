"""Deterministic textured target and stable software mount for video trials.

Run instead of pc.streamer, with its normal verified-RTP arguments. Uses only
the streamer's explicitly selected loopback control socket, never hardware.
Truth output is seed/scoring evidence; the controller must consume Jetson V2.
No repeated truth boxes are supplied to the visual tracker.
"""
import math
import time
import uuid
from types import SimpleNamespace

import numpy as np

from common.control import ControlConfig
from common.perception import (NormalizedBoxV2,PerceptionFrameV2,
    PerceptionSnapshotV2,PerceptionTrackV2)
from common.schemas import ControlIntent


class Capture:
    def __init__(self,spec,w,h,fps,cfg=None,**kwargs):
        if spec!='sim' or (w,h)!=(1280,720) or not kwargs.get('sim_control_enabled'):
            raise ValueError('fixture requires software mount, 1280x720 and explicit loopback control')
        if (cfg.get('sim') or {}).get('use_jetson_cam_state') is not False:
            raise ValueError('fixture prohibits hardware-in-loop')
        self.config=ControlConfig.from_raw_config(cfg,(w,h))
        self.period=1/fps;self.deadline=self.last=time.monotonic();self.started=self.last
        self.pose=np.zeros(2);self.rates=np.zeros(2);self.intent=None
        self.gen=SimpleNamespace(drain_planner_events=lambda:[],planner_eval_enabled=lambda:False)
        rng=np.random.default_rng(736)
        self.texture=np.repeat(np.repeat(rng.integers(30,240,(24,32,3),dtype=np.uint8),4,0),4,1)
        self.session=uuid.uuid4().hex
        self._control_activity=SimpleNamespace(session=self.session)

    def isOpened(self):return True
    def release(self):pass
    def handle_control_cmd(self,payload):
        # Same-host short lease, strict ControlIntent only; never ControlCmd.
        if payload.get('type')=='ControlIntent':self.intent=ControlIntent.model_validate(payload)

    def read(self):
        time.sleep(max(0,self.deadline-time.monotonic()))
        now=time.monotonic();self.deadline=max(self.deadline+self.period,now)
        dt=min(.1,now-self.last);self.last=now
        intent=self.intent
        demand=np.zeros(2)
        if intent is not None and intent.mode=='live' and time.monotonic_ns()<=intent.valid_until_monotonic_ns:
            demand=np.clip([intent.yaw_rate_rad_s,intent.pitch_rate_rad_s],-.5,.5)
        self.rates+=(demand-self.rates)*(1-math.exp(-dt/.05));self.pose+=self.rates*dt
        self.target=np.array((.08+.05*math.sin(.3*(now-self.started)),
                              .03+.025*math.sin(.5*(now-self.started))))
        c=self.config
        cx=c.cx_px+c.fx_px*math.tan(self.target[0]-self.pose[0])
        cy=c.cy_px-c.fy_px*math.tan(self.target[1]-self.pose[1])
        x,y=round(cx-64),round(cy-48)
        if not (0<=x<=1152 and 0<=y<=624):raise RuntimeError('fixture target left image')
        self.box=NormalizedBoxV2(x=x/1280,y=y/720,w=.1,h=96/720)
        self.last_frame_source_ns=time.monotonic_ns()
        image=np.full((720,1280,3),32,np.uint8);image[y:y+96,x:x+128]=self.texture
        return True,image

    def build_cam_state(self,frame_id,src_ts_ms):
        return dict(type='CamState',frame_id=frame_id,src_ts_ms=src_ts_ms,
            pan=float(self.pose[0]),tilt=float(self.pose[1]),pan_rate=float(self.rates[0]),tilt_rate=float(self.rates[1]),
            home_pan=0.,home_tilt=0.,pan_sample_monotonic_ns=self.last_frame_source_ns,
            tilt_sample_monotonic_ns=self.last_frame_source_ns,state_monotonic_ns=time.monotonic_ns())

    def build_ground_truth_snapshot(self,frame_id,source_time_ns):
        stamp=time.monotonic_ns()
        return PerceptionSnapshotV2(sequence=frame_id,frame=PerceptionFrameV2(
            tracker_session=self.session,frame_id=frame_id,source_time_ns=source_time_ns,
            received_time_ns=stamp,observed_time_ns=stamp,source_clock_domain='pc_monotonic',
            receive_clock_domain='pc_monotonic',observation_clock_domain='pc_monotonic',
            source_identity_verified=True,width=1280,height=720),tracks=(PerceptionTrackV2(
                track_id=1,class_id='fixture',confidence=1.,missed_frames=0,box=self.box),))


def main():
    from pc import streamer
    streamer.open_source=Capture
    streamer.main()


if __name__=='__main__':main()
