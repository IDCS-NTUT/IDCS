"""CPU-only, bounded native RTP/RTCP fault replay on loopback.

Input is a retained Annex-B H264 fixture. No camera, GPIO, GPU, controller or
production port is used. Native depay/rtpbin handle recovery; the UDP relay only
injects declared faults. Source time is synthetic fixture time, not live timing.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import socket
import threading
import time

from common.return_rtp import receiver_head, sender_tail, ReceiverRecovery


def vcl(data):
    return [n.rstrip(b'\0') for n in re.split(b'\x00\x00\x00?\x01',data)
            if n and n[0]&31 in (1,5)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port-base',type=int,default=25000)
    a=p.parse_args()
    if not 1024<=a.port_base<=65000:p.error('invalid dedicated test ports')
    import gi
    gi.require_version('Gst','1.0')
    from gi.repository import Gst
    Gst.init(None)
    a.output.mkdir(parents=True,exist_ok=False)
    source=Gst.parse_launch(f'filesrc location="{a.input}" ! h264parse ! '
        'video/x-h264,stream-format=byte-stream,alignment=au ! appsink name=read sync=false')
    source.set_state(Gst.State.PLAYING)
    aus=[]
    deadline=time.monotonic()+15
    try:
        while len(aus)<180 and time.monotonic()<deadline:
            sample=source.get_by_name('read').emit('try-pull-sample',Gst.SECOND)
            if sample is None:
                if source.get_by_name('read').get_property('eos'):break
                continue
            buf=sample.get_buffer();aus.append(buf.extract_dup(0,buf.get_size()))
    finally:source.set_state(Gst.State.NULL)
    if len(aus)!=180:raise RuntimeError('fixture must contain180 complete AUs')
    keys={tuple(hashlib.sha256(n).hexdigest() for n in vcl(raw)):i for i,raw in enumerate(aus)}
    if len(keys)!=len(aus):raise RuntimeError('fixture VCL identities must be unique')
    stop=threading.Event()
    relay=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    relay.bind(('127.0.0.1',a.port_base+5));relay.settimeout(.1)
    forward=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    received=[];events=[];errors=[];injections=Counter()
    def proxy():
        previous=None;frame=-1;single=False
        while not stop.is_set():
            try:raw,_=relay.recvfrom(65535)
            except socket.timeout:continue
            timestamp=int.from_bytes(raw[4:8],'big')
            if timestamp!=previous:previous=timestamp;frame+=1
            drop=(frame==35 and not single) or 70<=frame<=72
            if frame==35 and not single:single=True;injections['single_packet']+=1
            if 70<=frame<=72:injections['burst_packets']+=1
            if not drop:forward.sendto(raw,('127.0.0.1',a.port_base+2))
    worker=threading.Thread(target=proxy,daemon=True)
    worker.start()
    rx=Gst.parse_launch(receiver_head(a.port_base+2,20,feedback_host='127.0.0.1',
        feedback_port=a.port_base+3,sender_report_port=a.port_base+4)+
        'identity name=au ! h264parse ! avdec_h264 ! fakesink name=decoded sync=false')
    recovery=ReceiverRecovery(rx)
    tx=Gst.parse_launch('appsrc name=feed is-live=true format=time block=true '
        'caps="video/x-h264,stream-format=byte-stream,alignment=au,framerate=30/1" ! h264parse ! '+
        sender_tail(['127.0.0.1'],a.port_base+5,pay_name='pay',feedback_port=a.port_base+3,
                    sender_report_port=a.port_base+4))
    decoded=0;stalled=False
    def output(_pad,info):
        nonlocal stalled
        buf=info.get_buffer();raw=buf.extract_dup(0,buf.get_size())
        nals=vcl(raw)
        key=tuple(hashlib.sha256(n).hexdigest() for n in nals)
        index=keys.get(key)
        if index is None:errors.append('unknown/incomplete output VCL')
        received.append(dict(index=index,idr=any(n[0]&31==5 for n in nals),local_ns=time.monotonic_ns()))
        if index==110 and not stalled:
            stalled=True;injections['downstream_stall']+=1
            time.sleep(.2)
        return Gst.PadProbeReturn.OK
    def frame(_pad,_info):
        nonlocal decoded
        decoded+=1
        return Gst.PadProbeReturn.OK
    def upstream(_pad,info):
        event=info.get_event();s=event.get_structure() if event else None
        if s and s.get_name()=='GstForceKeyUnit':events.append(dict(local_ns=time.monotonic_ns()))
        return Gst.PadProbeReturn.OK
    rx.get_by_name('au').get_static_pad('src').add_probe(Gst.PadProbeType.BUFFER,output)
    rx.get_by_name('decoded').get_static_pad('sink').add_probe(Gst.PadProbeType.BUFFER,frame)
    tx.get_by_name('feed').get_static_pad('src').add_probe(Gst.PadProbeType.EVENT_UPSTREAM,upstream)
    rx.set_state(Gst.State.PLAYING);tx.set_state(Gst.State.PLAYING)
    began=time.monotonic()
    def drain():
        for pipeline in (rx,tx):
            bus=pipeline.get_bus()
            while True:
                msg=bus.pop_filtered(Gst.MessageType.ERROR|Gst.MessageType.ELEMENT)
                if msg is None:break
                if msg.type==Gst.MessageType.ERROR:errors.append(str(msg.parse_error()))
                elif pipeline is rx:recovery.on_message(bus,msg)
    try:
        for i,raw in enumerate(aus):
            delay=began+i/30-time.monotonic()
            if delay>0:time.sleep(delay)
            buf=Gst.Buffer.new_wrapped(raw);buf.pts=buf.dts=i*Gst.SECOND//30;buf.duration=Gst.SECOND//30
            if tx.get_by_name('feed').emit('push-buffer',buf)!=Gst.FlowReturn.OK:
                errors.append('sender flow failure');break
            drain()
        tx.get_by_name('feed').emit('end-of-stream')
        time.sleep(.5);drain()
    finally:
        tx.set_state(Gst.State.NULL);rx.set_state(Gst.State.NULL)
        tx.get_state(3*Gst.SECOND);rx.get_state(3*Gst.SECOND)
        stop.set();worker.join(timeout=2);relay.close();forward.close()
    gaps=[dict(before=x['index'],after=y['index'],idr=y['idr']) for x,y in zip(received,received[1:])
          if x['index'] is not None and y['index'] is not None and y['index']!=x['index']+1]
    passed=(not errors and all(g['idr'] for g in gaps) and len(gaps)>=2 and len(events)>0
            and recovery.keyframe_requests>0 and injections['downstream_stall']==1 and decoded>60
            and received[-1]['index']>=170)
    result=dict(passed=passed,input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
        fixture_timestamps=True,control_disabled=True,frames_supplied=len(aus),decoded=decoded,
        received=received,gaps=gaps,sender_keyframe_requests=events,receiver=recovery.stats(),
        injections=dict(injections),errors=errors,duration_s=time.monotonic()-began)
    (a.output/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('received','sender_keyframe_requests')},indent=2))
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
