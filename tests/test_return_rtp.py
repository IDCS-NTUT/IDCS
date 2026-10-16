from types import SimpleNamespace as NS

import pytest

from common.return_rtp import receiver_head, sender_tail, validate_ports, ReceiverRecovery, NvidiaKeyframeBridge


def test_native_receiver_requires_complete_aus_and_explicit_feedback():
    text=receiver_head(5002,20,feedback_host='192.168.0.5')
    for part in ('rtp-profile=avpf','do-lost=true','drop-on-latency=true',
                 'wait-for-keyframe=true request-keyframe=true','alignment=au',
                 'port=5003','port=5004','rtcp-fb-nack-pli=(boolean)true'):
        assert part in text
    assert 'latency=20' in text
    assert 'leaky' not in text


def test_native_sender_mirrors_one_encode_and_exposes_final_wire_identity():
    text=sender_tail(['192.168.0.1','192.168.0.3'],5002,pay_name='return_pay')
    assert '192.168.0.1:5002,192.168.0.3:5002' in text
    assert '192.168.0.1:5004,192.168.0.3:5004' in text
    assert text.index('send_rtp_src_0') < text.index('name=return_wire') < text.index('name=return_udp')
    assert 'recv_rtcp_sink_0' in text and 'port=5003' in text


def test_standalone_receiver_still_waits_for_keyframes_without_claiming_feedback():
    text=receiver_head(5002,20)
    assert 'do-lost=true' in text and 'wait-for-keyframe=true' in text
    assert 'rtpbin' not in text and 'udpsink' not in text


@pytest.mark.parametrize('ports',[(5002,5002,5004),(0,5003,5004),(65536,5003,5004),(True,5003,5004)])
def test_invalid_ports_rejected(ports):
    with pytest.raises(ValueError):validate_ports(*ports)


@pytest.mark.parametrize('host',['bad host','a!b','x\nport=123',''])
def test_pipeline_injection_rejected(host):
    with pytest.raises(ValueError):sender_tail([host],5002,pay_name='return_pay')


def test_latency_discard_counts_are_not_confused_with_network_loss():
    recovery=ReceiverRecovery.__new__(ReceiverRecovery)
    recovery.drops={'num-drop-on-latency':0,'num-too-late-drops':0}
    def message(name,**values):
        return NS(get_structure=lambda:NS(get_name=lambda:name,has_field=lambda k:k in values,get_value=lambda k:values[k]))
    recovery.on_message(None,NS(get_structure=lambda:None))
    recovery.on_message(None,message('unrelated',**{'num-drop-on-latency':100}))
    recovery.on_message(None,message('drop-msg',**{'num-drop-on-latency':2,'num-too-late':1}))
    recovery.on_message(None,message('drop-msg',**{'num-drop-on-latency':3}))
    assert recovery.drops=={'num-drop-on-latency':5,'num-too-late-drops':1}


def test_nvidia_action_bridge_never_forces_before_output_and_coalesces_storms(monkeypatch):
    bridge=NvidiaKeyframeBridge.__new__(NvidiaKeyframeBridge)
    bridge._Gst=NS(PadProbeReturn=NS(OK='ok',DROP='drop'))
    actions=[]
    bridge.encoder=NS(emit=lambda name:actions.append(name))
    bridge.enabled=True;bridge.started=False;bridge.pending=False;bridge.last_force_s=None
    bridge.requests=bridge.forced=bridge.coalesced=0
    clock=[1.0]
    monkeypatch.setattr('common.return_rtp.time.monotonic',lambda:clock[0])
    info=NS(get_event=lambda:NS(get_structure=lambda:NS(get_name=lambda:'GstForceKeyUnit')))
    assert bridge._request(None,info)=='drop' and actions==[]
    bridge._before_input(None,None)
    assert actions==[]
    bridge._started(None,None)
    bridge._before_input(None,None)
    assert actions==['force-IDR']
    assert bridge._request(None,info)=='drop'
    bridge._request(None,info)
    bridge._before_input(None,None)
    assert actions==['force-IDR']
    clock[0]=1.2;bridge._before_input(None,None)
    assert actions==['force-IDR','force-IDR']
    assert bridge.stats()==dict(native_action=True,requests=3,forced=2,coalesced=1)
    bridge.enabled=False
    assert bridge._request(None,info)=='ok'  # standard encoder handles its own event
