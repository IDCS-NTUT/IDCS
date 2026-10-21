"""Native GStreamer RTP/RTCP return-video transport (no motion authority).

The depayloader owns H264 loss/keyframe handling. RTCP carries its upstream
GstForceKeyUnit requests to the actual sender; no custom codec recovery exists.
"""
from __future__ import annotations

import re
import time

RTP_CAPS = ('application/x-rtp,media=video,encoding-name=H264,payload=97,'
            'clock-rate=90000,rtcp-fb-nack-pli=(boolean)true')


def validate_ports(rtp_port: int, feedback_port: int, sender_report_port: int) -> None:
    if any(type(p) is not int or not 1 <= p <= 65535
           for p in (rtp_port, feedback_port, sender_report_port)):
        raise ValueError('return RTP/RTCP ports must be integers in 1..65535')
    if len({rtp_port, feedback_port, sender_report_port}) != 3:
        raise ValueError('return RTP/RTCP ports must be distinct')


def _host(host: str) -> str:
    if not isinstance(host, str) or not re.fullmatch(r'[A-Za-z0-9._:-]+', host):
        raise ValueError('invalid return host')
    return host


def sender_tail(hosts, rtp_port: int, *, pay_name: str, feedback_port: int = 5003,
                sender_report_port: int = 5004) -> str:
    """One encoder, mirrored RTP/SR, multiple receivers' native PLI feedback."""
    validate_ports(rtp_port, feedback_port, sender_report_port)
    hosts = tuple(_host(h) for h in hosts)
    if not hosts:
        raise ValueError('return hosts required')
    def clients(port):
        return ','.join(f'[{h}]:{port}' if ':' in h else f'{h}:{port}' for h in hosts)
    return (
        f'rtph264pay name={pay_name} pt=97 config-interval=1 ! '
        f'{RTP_CAPS} ! return_session.send_rtp_sink_0 '
        'rtpbin name=return_session rtp-profile=avpf '
        'return_session.send_rtp_src_0 ! identity name=return_wire ! '
        f'multiudpsink name=return_udp clients="{clients(rtp_port)}" sync=false async=false '
        f'udpsrc name=return_feedback port={feedback_port} caps="application/x-rtcp" ! '
        'return_session.recv_rtcp_sink_0 '
        'return_session.send_rtcp_src_0 ! '
        f'multiudpsink name=return_reports clients="{clients(sender_report_port)}" sync=false async=false'
    )


def receiver_head(rtp_port: int, jitter_ms: int, *, feedback_host: str | None = None,
                  feedback_port: int = 5003, sender_report_port: int = 5004,
                  bind_ip: str | None = None) -> str:
    """RTP to complete, independently recoverable H264 AUs.

    Standalone mode still waits for periodic keyframes, but cannot request one
    remotely. Queue/latency bounds remain enabled; no unbounded buffering fix.
    """
    validate_ports(rtp_port, feedback_port, sender_report_port)
    if type(jitter_ms) is not int or not 0 <= jitter_ms <= 2000:
        raise ValueError('return jitter must be 0..2000 ms')
    bind = f'address="{_host(bind_ip)}" ' if bind_ip else ''
    if feedback_host:
        prefix = (
            f'udpsrc name=return_input {bind}port={rtp_port} buffer-size=4000000 caps="{RTP_CAPS}" ! '
            'return_session.recv_rtp_sink_0 '
            f'udpsrc {bind}port={sender_report_port} caps="application/x-rtcp" ! return_session.recv_rtcp_sink_0 '
            'return_session.send_rtcp_src_0 ! '
            f'udpsink name=return_feedback host="{_host(feedback_host)}" port={feedback_port} sync=false async=false '
            f'rtpbin name=return_session rtp-profile=avpf latency={jitter_ms} '
            'drop-on-latency=true do-lost=true autoremove=true '
            'return_session. ! application/x-rtp,media=video,encoding-name=H264,payload=97 ! '
        )
    else:
        prefix = (
            f'udpsrc name=return_input {bind}port={rtp_port} buffer-size=4000000 caps="{RTP_CAPS}" ! '
            f'rtpjitterbuffer name=return_jitter latency={jitter_ms} drop-on-latency=true do-lost=true '
            'post-drop-messages=true drop-messages-interval=100 ! '
        )
    return prefix + (
        'rtph264depay name=return_depay wait-for-keyframe=true request-keyframe=true ! '
        'video/x-h264,stream-format=byte-stream,alignment=au ! '
    )


class ReceiverRecovery:
    """Wire native jitterbuffer observability; never modify codec data or PTS."""

    def __init__(self, pipeline, *, on_rtp_packet=None, bus=None):
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        self._Gst = Gst
        self._jitters = []
        self._on_rtp_packet = on_rtp_packet
        self._closed = False
        self.drops = {'num-drop-on-latency': 0, 'num-too-late-drops': 0}
        self.keyframe_requests = 0
        session = pipeline.get_by_name('return_session')
        if session is not None:
            session.connect('new-jitterbuffer', self._new_jitter)
        else:
            self._configure(pipeline.get_by_name('return_jitter'))
        pipeline.get_by_name('return_depay').get_static_pad('sink').add_probe(
            Gst.PadProbeType.EVENT_UPSTREAM, self._on_upstream)
        # Owner already runs a bus watch (or drains messages in its read loop).
        self._bus = bus if bus is not None else pipeline.get_bus()
        self._bus_handler = self._bus.connect('message::element', self.on_message)

    def close(self):
        if not self._closed:
            self._closed = True
            self._bus.disconnect(self._bus_handler)

    def _new_jitter(self, _session, jitter, _index, _ssrc):
        self._configure(jitter)

    def _configure(self, jitter):
        if jitter is None:
            raise RuntimeError('return jitterbuffer missing')
        jitter.set_property('post-drop-messages', True)
        jitter.set_property('drop-messages-interval', 100)
        jitter.set_property('do-lost', True)
        self._jitters.append(jitter)
        self._jitters = self._jitters[-2:]
        if self._on_rtp_packet is not None:
            jitter.get_static_pad('src').add_probe(self._Gst.PadProbeType.BUFFER, self._on_rtp_packet)

    def _on_upstream(self, _pad, info):
        event = info.get_event()
        structure = event.get_structure() if event is not None else None
        if structure is not None and structure.get_name() == 'GstForceKeyUnit':
            self.keyframe_requests += 1
        return self._Gst.PadProbeReturn.OK

    def on_message(self, _bus, message):
        stats = message.get_structure()
        if stats is None or stats.get_name() != 'drop-msg':
            return
        for field, counter in (('num-drop-on-latency', 'num-drop-on-latency'),
                               ('num-too-late', 'num-too-late-drops')):
            if stats.has_field(field):
                self.drops[counter] += int(stats.get_value(field))

    def stats(self):
        result = dict(self.drops, keyframe_requests=self.keyframe_requests)
        for jitter in self._jitters:
            stats = jitter.get_property('stats')
            for key in ('num-pushed', 'num-lost', 'num-late', 'num-duplicates'):
                if stats.has_field(key):
                    result[key] = result.get(key, 0) + int(stats.get_value(key))
            if stats.has_field('avg-jitter'):
                result['avg-jitter'] = int(stats.get_value('avg-jitter'))
        return result


class NvidiaKeyframeBridge:
    """Adapt native GstForceKeyUnit feedback to NVIDIA's native force-IDR action.

    Some installed NVIDIA versions expose the action without implementing the
    standard event. Only act after encoder output has started; coalesce storms.
    No bitstream manipulation, artificial frame identity or decoder reseeding.
    """
    def __init__(self, encoder):
        import gi
        gi.require_version('Gst','1.0')
        from gi.repository import Gst, GObject
        self._Gst=Gst
        self.encoder=encoder
        self.enabled=bool(GObject.signal_lookup('force-IDR',encoder.__gtype__))
        self.started=False
        self.pending=False
        self.last_force_s=None
        self.requests=self.forced=self.coalesced=0
        pad=encoder.get_static_pad('src')
        pad.add_probe(Gst.PadProbeType.BUFFER,self._started)
        pad.add_probe(Gst.PadProbeType.EVENT_UPSTREAM,self._request)
        encoder.get_static_pad('sink').add_probe(Gst.PadProbeType.BUFFER,self._before_input)

    def _started(self,_pad,_info):
        self.started=True
        return self._Gst.PadProbeReturn.OK

    def _request(self,_pad,info):
        event=info.get_event()
        structure=event.get_structure() if event is not None else None
        if structure is None or structure.get_name()!='GstForceKeyUnit':
            return self._Gst.PadProbeReturn.OK
        self.requests+=1
        if not self.enabled:
            return self._Gst.PadProbeReturn.OK
        if self.pending:self.coalesced+=1
        self.pending=True
        return self._Gst.PadProbeReturn.DROP

    def _before_input(self,_pad,_info):
        # Apply on the encoder input owner, after its first encoded output.
        # No action can fire from RTCP while the encoder is starting/stopping.
        now=time.monotonic()
        if self.pending and self.started and (self.last_force_s is None or now-self.last_force_s>=.1):
            self.encoder.emit('force-IDR')
            self.last_force_s=now;self.forced+=1;self.pending=False
        return self._Gst.PadProbeReturn.OK

    def stats(self):
        return dict(native_action=self.enabled,requests=self.requests,forced=self.forced,coalesced=self.coalesced)
