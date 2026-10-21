"""Exclusive Pi display receiver test using controlled loopback RTP, never motors.

Stop the ordinary display owner first. This tool does not change services,
source modes, GPIO or configuration. Every wait is bounded; all Gst owners
are stopped in finally. The output directory must be new.
"""
from __future__ import annotations

import argparse
import json
import os
import hashlib
from pathlib import Path
import time


def run(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sink',default='waylandsink fullscreen=true sync=false')
    parser.add_argument('--port',type=int,default=15082)
    parser.add_argument('--record',action='store_true')
    args=parser.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=False)
    import gi
    gi.require_version('Gst','1.0')
    from gi.repository import GLib,Gst
    from rpi.display.video import ReturnVideo
    from rpi.display.render import StatusBar,render
    Gst.init(None)
    context=GLib.MainContext.default()
    frames=[];outputs=[];faults=[];fatal=[];cases=[]
    video=sender=None
    started=time.monotonic()
    source=Path(__file__).resolve().parents[1]
    report=dict(pid=os.getpid(),cases=cases,passed=False,recording=args.record,
        source=str(source),gstreamer=Gst.version_string(),clock_domain='pi_monotonic',
        fixture=dict(width=640,height=360,fps=30,encoder='x264enc',decoder='v4l2h264dec'),
        source_sha256={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in (
            'rpi/display/video.py','rpi/display/recording.py','rpi/operator_display.py',
            'common/return_rtp.py','tools/verify_pi_return_recovery.py')})
    try:
        video=ReturnVideo(port=args.port,jitter_ms=20,sink=args.sink,width=640,height=360,
            feedback_host='127.0.0.1',feedback_port=args.port+1,sender_report_port=args.port+2,
            recording=None if not args.record else dict(dir=str(args.output/'recordings'),segment_s=10,max_total_gb=1),
            on_frame=lambda:frames.append(time.monotonic()),
            on_receiver_lost=lambda reason:faults.append(dict(at=time.monotonic(),reason=reason)))
        original_pipeline=video.pipeline
        bus=video.pipeline.get_bus();bus.add_signal_watch()
        def message(_bus,msg):
            if msg.type==Gst.MessageType.ERROR and not video.handle_message(msg):
                err,debug=msg.parse_error();fatal.append(f'{err.message}: {debug}')
            elif msg.type==Gst.MessageType.EOS:
                fatal.append('EOS escaped receiver to display pipeline')
        bus.connect('message',message)
        video.pipeline.get_by_name('overlay').get_static_pad('src').add_probe(
            Gst.PadProbeType.BUFFER,
            lambda *_:(outputs.append(time.monotonic()),Gst.PadProbeReturn.OK)[1])
        video.start()
        def pump(until,timeout):
            end=time.monotonic()+timeout
            while time.monotonic()<end:
                for _ in range(30):
                    if not context.pending():break
                    context.iteration(False)
                if fatal:raise RuntimeError(fatal[-1])
                now=time.monotonic();video.service(now)
                video.show_live(bool(frames and now-frames[-1]<.5))
                if video.recorder is not None:video.recorder.set_recording(args.record)
                video.set_overlay(render(*video.frame_size,bar=StatusBar('TEST',('RECEIVER RECOVERY',)),
                    alerts=[],menu=None,fire_state='fire unknown',safety_state='E-stop unknown'))
                if video.pipeline is not original_pipeline:raise RuntimeError('display pipeline replaced')
                if until():return
                time.sleep(.01)
            raise RuntimeError('bounded wait expired')
        def source(ssrc):
            return Gst.parse_launch('videotestsrc is-live=true pattern=ball ! '
                'video/x-raw,format=I420,width=640,height=360,framerate=30/1 ! '
                'x264enc tune=zerolatency speed-preset=ultrafast threads=1 bitrate=800 key-int-max=15 ! '
                'video/x-h264,profile=baseline ! '
                f'rtph264pay pt=97 config-interval=1 ssrc={ssrc} ! '
                f'udpsink host=127.0.0.1 port={args.port} sync=false async=false')
        pump(lambda:len(outputs)>=8,3.)
        for index in range(3):
            if sender is not None:sender.set_state(Gst.State.NULL)
            before=len(outputs);gap_start=time.monotonic()
            pump(lambda:time.monotonic()-gap_start>=1.2,2.)
            cases.append(dict(case='source_gap',output_frames=len(outputs)-before,
                showing_live=video._showing_live,decoded_age_s=None if not frames else time.monotonic()-frames[-1],
                last_output_age_s=None if not outputs else time.monotonic()-outputs[-1]))
            if len(outputs)-before<5:raise RuntimeError('fallback/UI stopped during source gap')
            sender=source(70001+index);sender.set_state(Gst.State.PLAYING)
            before=len(frames);began=time.monotonic()
            pump(lambda:len(frames)-before>=30,12.)
            cases.append(dict(case='sender_change',ssrc=70001+index,
                elapsed_s=time.monotonic()-began,receiver=video.recovery_status()))
        for kind in ('eos','receiver_error'):
            before=len(frames);generation=video._generation;began=time.monotonic()
            if kind=='eos':
                video.receiver.get_by_name('decoded').get_static_pad('src').push_event(Gst.Event.new_eos())
            else:
                error=GLib.Error.new_literal(Gst.StreamError.quark(),'controlled receiver failure',Gst.StreamError.FAILED)
                video.pipeline.post_message(Gst.Message.new_error(video.receiver.get_by_name('return_input'),error,'test'))
            pump(lambda:video._generation>generation and len(frames)-before>=30,12.)
            cases.append(dict(case=kind,elapsed_s=time.monotonic()-began,receiver=video.recovery_status()))
        report.update(passed=True,frames=len(frames),output_frames=len(outputs),faults=faults,
            receiver=video.recovery_status(),output_gap_max_ms=max((b-a)*1000 for a,b in zip(outputs,outputs[1:])))
    except Exception as exc:
        report['error']=str(exc)
    finally:
        if sender is not None:sender.set_state(Gst.State.NULL)
        if video is not None:video.stop()
        report['elapsed_s']=time.monotonic()-started
        report['fatal']=fatal
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)
    return int(not report['passed'])


if __name__=='__main__':
    raise SystemExit(run())
