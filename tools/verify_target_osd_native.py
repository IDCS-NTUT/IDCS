"""Bounded native GPU OSD pixel test; no inference, network, control or camera.

Run only with the operational DeepStream/visual-lock GPU owners stopped.
Synthetic accepted tracks exercise the actual production metadata renderer.
Artifacts are written outside source. A passing mock is not native acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_owners():
    prohibited = ('jetson.deepstream.runtime', 'jetson.deepstream.verify_pipeline',
                  'jetson.visual_lock.live_trial', 'tools.verify_target_osd_native')
    import os
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (any(argv[i] == '-m' and argv[i+1] in prohibited for i in range(len(argv)-1))
                or any(a.endswith('/verify_target_osd_native.py') for a in argv)):
            raise RuntimeError('GPU owner must be stopped before native test: '+entry.name)


def run(output, border_override, width, height):
    check_owners()
    output.mkdir(parents=True, exist_ok=False)
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstVideo', '1.0')
    from gi.repository import Gst, GstVideo
    import numpy as np
    from common.ui_fonts import configure_fonts
    from jetson.visual_lock.egl_lifetime import StandaloneEGL
    configure_fonts()
    records, errors = [], []
    started = time.monotonic()
    with StandaloneEGL():
        import pyds
        from common.perception import NormalizedBoxV2, PerceptionTrackV2, TrackAssessmentV2
        from jetson.deepstream.pipeline import _decorate_osd_metadata
        Gst.init(None)
        pipeline = Gst.parse_launch(
            'videotestsrc num-buffers=4 pattern=solid-color foreground-color=0xff404040 ! '
            f'video/x-raw,width={width},height={height},framerate=10/1 ! nvvideoconvert ! '
            'video/x-raw(memory:NVMM),format=RGBA ! mux.sink_0 '
            f'nvstreammux name=mux batch-size=1 width={width} height={height} batched-push-timeout=40000 ! '
            'nvdsosd name=osd process-mode=1 display-text=1 display-bbox=1 ! '
            'nvvideoconvert ! video/x-raw,format=RGBA ! appsink name=result sync=false max-buffers=4 drop=false')
        tracks = tuple(PerceptionTrackV2(track_id=90000+i, class_id='drone',confidence=.8,
            box=NormalizedBoxV2(x=.1+i*.25,y=.25,w=.15,h=.2),missed_frames=0) for i in range(3))
        assessments = tuple(TrackAssessmentV2(track_id=t.track_id, engagement_rank=i+1,
            threat_level=('benign','suspicious','threatening')[i]) for i,t in enumerate(tracks))

        def attach(frame, meta):
            for i in range(meta.num_rects):
                rect = meta.rect_params[i]
                if rect.has_bg_color:
                    if border_override is not None:
                        rect.border_width = border_override
                        rect.border_color.set(rect.bg_color.red,rect.bg_color.green,rect.bg_color.blue,1.)
                    if len(records) < 3:
                        records.append(dict(x=rect.left,y=rect.top,w=rect.width,h=rect.height,
                            rgb=[round(v*255) for v in (rect.bg_color.red,rect.bg_color.green,rect.bg_color.blue)]))
            pyds.nvds_add_display_meta_to_frame(frame,meta)

        adapter = SimpleNamespace(nvds_acquire_display_meta_from_pool=pyds.nvds_acquire_display_meta_from_pool,
                                  nvds_add_display_meta_to_frame=attach)

        def probe(pad, info):
            try:
                batch = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
                frame = pyds.NvDsFrameMeta.cast(batch.frame_meta_list.data)
                _decorate_osd_metadata(adapter,batch,frame,[],accepted_tracks=tracks,
                    assessments=assessments,target_track_id=tracks[0].track_id,target_assessment=assessments[0])
            except Exception as exc:
                errors.append(str(exc))
            return Gst.PadProbeReturn.OK

        pad = pipeline.get_by_name('osd').get_static_pad('sink')
        probe_id = pad.add_probe(Gst.PadProbeType.BUFFER,probe)
        image = None
        try:
            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError('native pipeline failed to start')
            sink = pipeline.get_by_name('result')
            deadline = time.monotonic()+10
            while time.monotonic() < deadline:
                sample = sink.emit('try-pull-sample',Gst.SECOND)
                if sample is not None:
                    video = GstVideo.VideoInfo.new_from_caps(sample.get_caps())
                    raw = sample.get_buffer().extract_dup(0,sample.get_buffer().get_size())
                    image = np.ndarray((video.height,video.width,4),np.uint8,buffer=raw,
                                       strides=(video.stride[0],4,1)).copy()
                    break
                message = pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
                if message is not None:
                    raise RuntimeError(str(message.parse_error()))
            if image is None or errors:
                raise RuntimeError('native output missing: '+str(errors))
        finally:
            pipeline.set_state(Gst.State.NULL)
            pipeline.get_state(5*Gst.SECOND)
            pad.remove_probe(probe_id)
            sample = None
            sink = None
            pipeline = None
        import cv2
        cv2.imwrite(str(output/'gpu-osd.png'),image[:,:,:3][:,:,::-1])
        results = []
        for record in records:
            x,y,w,h = (round(record[k]) for k in ('x','y','w','h'))
            roi = image[y+2:y+h-2,x+2:x+w-2,:3].astype(int)
            fraction = float(np.mean(np.max(np.abs(roi-np.array(record['rgb'])),axis=2) <= 6))
            # The rank must be INSIDE the fill, not merely present near it.
            x0,y0 = max(0,x-8),max(0,y-8)
            around = image[y0:y+h+8,x0:x+w+8,:3]
            gy,gx = np.where(np.max(around,axis=2)<35)
            bounds = None if len(gx)==0 else [int(gx.min()+x0),int(gy.min()+y0),int(gx.max()+x0),int(gy.max()+y0)]
            inside = bounds is not None and bounds[0]>=x and bounds[1]>=y and bounds[2]<x+w and bounds[3]<y+h
            # At the minimum 10px badge, the glyph/antialiasing can occupy
            # almost half the inner ROI. Require a majority to be solid colour
            # AND independently require the dark glyph wholly inside the fill.
            results.append(dict(**record,solid_colour_fraction=fraction,glyph_bounds=bounds,
                                glyph_inside_fill=inside,passed=fraction>=.50 and inside))
        report = dict(passed=len(results)==3 and all(r['passed'] for r in results),
            mode='GPU',width=width,height=height,border_override=border_override,measurements=results,
            duration_s=time.monotonic()-started,hardware_qualified=False,
            renderer_sha256=hashlib.sha256((ROOT/'jetson/deepstream/pipeline.py').read_bytes()).hexdigest(),
            scope='Actual native OSD pixels on deterministic boxes; not operational return/transit acceptance')
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--border-override',type=int,choices=(0,1),help='Diagnostic only; never changes production renderer')
    parser.add_argument('--width',type=int,choices=(640,800,1280,1920),default=1280)
    parser.add_argument('--height',type=int,choices=(360,600,720,1080),default=720)
    args = parser.parse_args()
    return run(args.output,args.border_override,args.width,args.height)


if __name__ == '__main__':
    raise SystemExit(main())
