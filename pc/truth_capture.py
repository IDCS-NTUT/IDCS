"""Bounded simulator-only truth evidence; never a tracking/control input."""
import json
from pathlib import Path


class TruthCapture:
    def __init__(self,path,max_frames=120000):
        self.path = Path(path)
        if type(max_frames) is not int or not 1 <= max_frames <= 120000:
            raise ValueError("truth capture limit must be 1..120000 frames")
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.file = self.path.open("x",encoding="utf-8")
        self.max_frames = max_frames
        self.count = 0
        self.last_id = self.last_time = -1

    def append(self,frame_id,source_time_ns,width,height,snapshot):
        if self.count >= self.max_frames: return False
        if frame_id <= self.last_id or source_time_ns <= self.last_time:
            raise ValueError("truth capture identity/order violation")
        if snapshot is not None and (snapshot.frame.frame_id != frame_id or
                                     snapshot.frame.source_time_ns != source_time_ns):
            raise ValueError("truth snapshot does not describe sent frame")
        row = dict(frame_id=frame_id,source_time_ns=source_time_ns,width=width,height=height,
                   source_clock_domain="pc_monotonic",truth=[] if snapshot is None else [
                       dict(id=t.track_id,x=t.box.x,y=t.box.y,w=t.box.w,h=t.box.h) for t in snapshot.tracks])
        self.file.write(json.dumps(row)+"\n")
        self.count += 1
        self.last_id,self.last_time = frame_id,source_time_ns
        if self.count % 60 == 0: self.file.flush()
        return True

    def close(self):
        self.file.close()
        return dict(frames=self.count,max_frames=self.max_frames,limit_reached=self.count>=self.max_frames,
                    path=str(self.path),pixels_recorded=False,tracking_input=False)
