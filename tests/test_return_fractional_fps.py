"""59.94fps file returns preserve real images/PTS at nominal encoder cadence."""
from pathlib import Path

import pytest

from jetson.deepstream.pipeline import ReturnRateGate, _pipeline_description, _return_rate_description


@pytest.mark.parametrize('fps',[30,60])
@pytest.mark.parametrize('source',['file','argus','rtp'])
def test_fractional_file_return_sets_nominal_caps_without_rate_negotiation(source,fps):
    text = _pipeline_description(input_file=Path('fixture.mp4') if source == 'file' else None,
        live_argus=source == 'argus',rtp_input_port=5000 if source == 'rtp' else None,
        argus_sensor_id=0,argus_sensor_mode=4,argus_width=1280,argus_height=720,argus_fps=60,
        nvinfer_config=Path('fixture.txt'),paced=True,tracker='none',gpu_osd=True,
        return_h264=True,return_udp_host=None,return_udp_port=None,return_h264_file=None,return_fps=fps)
    if source == 'file':
        assert 'identity name=return_rate' in text
        assert 'capssetter name=return_caps' in text
        assert 'videorate' not in text
    else:
        assert f'videorate name=return_rate drop-only=true max-rate={fps}' in text
        assert 'capssetter' not in text
    assert f'framerate={fps}/1' in text
    assert 'copy-timestamp=true' in text
    assert text.count('nvv4l2h264enc') == 1


@pytest.mark.parametrize('fps,expected',[(30,60),(60,120)])
def test_fractional_input_arrival_gate_remains_bounded(fps,expected):
    gate = ReturnRateGate(fps)
    admitted = [i for i in range(120) if gate.admit(1.+i*1001/60000)]
    assert len(admitted) == expected
    assert len(set(admitted)) == len(admitted)


@pytest.mark.parametrize('fps,expected',[(30,60),(60,120)])
def test_gst_fractional_file_caps_preserve_unique_source_pts_and_metadata(fps,expected):
    gi = pytest.importorskip('gi')
    try:
        gi.require_version('Gst','1.0')
    except ValueError:
        pytest.skip('Gst typelib unavailable')
    from gi.repository import Gst
    Gst.init(None)
    if not Gst.ElementFactory.find('capssetter'):
        pytest.skip('capssetter unavailable')
    # CPU-only caps/metadata fixture: NVMM is a caps feature, no GPU allocation.
    pipeline = Gst.parse_launch('appsrc name=source format=time ! '
        + _return_rate_description(fps,file_replay=True) + 'appsink name=sink sync=false')
    source = pipeline.get_by_name('source')
    source.set_property('caps',Gst.Caps.from_string(
        'video/x-raw(memory:NVMM),format=GRAY8,width=1,height=1,framerate=60000/1001'))
    gate = ReturnRateGate(fps)
    def admit(_pad,info):
        # Deterministic injection into the same gate used by the runtime;
        # timestamp identity is checked separately below, not inferred by time.
        return Gst.PadProbeReturn.OK if gate.admit(info.get_buffer().pts/1e9) else Gst.PadProbeReturn.DROP
    pipeline.get_by_name('return_rate').get_static_pad('sink').add_probe(Gst.PadProbeType.BUFFER,admit)
    reference = Gst.Caps.from_string('timestamp/x-idcs-source')
    originals = {}
    try:
        pipeline.set_state(Gst.State.PLAYING)
        for i in range(120):
            buffer = Gst.Buffer.new_wrapped(b'\x00')
            buffer.pts = 1_000_000_000+round(i*1_000_000_000*1001/60000)
            buffer.duration = round(1_000_000_000*1001/60000)
            originals[i+1] = buffer.pts
            buffer.add_reference_timestamp_meta(reference,i+1,buffer.duration)
            assert source.emit('push-buffer',buffer) == Gst.FlowReturn.OK
        source.emit('end-of-stream')
        seen = []
        while True:
            sample = pipeline.get_by_name('sink').emit('try-pull-sample',Gst.SECOND)
            if sample is None: break
            buffer = sample.get_buffer()
            metadata = buffer.get_reference_timestamp_meta(reference)
            assert metadata is not None
            assert buffer.pts == originals[metadata.timestamp]
            seen.append(metadata.timestamp)
            caps = sample.get_caps()
            assert caps.get_features(0).contains('memory:NVMM')
            assert tuple(caps.get_structure(0).get_fraction('framerate'))[-2:] == (fps,1)
        assert len(seen) == expected and len(set(seen)) == expected
        assert pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR) is None
    finally:
        pipeline.set_state(Gst.State.NULL)
