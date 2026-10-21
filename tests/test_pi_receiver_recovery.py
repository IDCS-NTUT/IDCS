"""Receiver lifecycle contracts with fake Gst; no GPU, sockets, GPIO or sleeps."""
import importlib.util
from collections import deque
from pathlib import Path
from threading import Lock
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock
import sys

import pytest


@pytest.fixture
def recovery(monkeypatch):
    gi=ModuleType('gi');gi.require_version=lambda *_:None
    repository=ModuleType('gi.repository')
    gst=NS(State=NS(NULL='NULL',PLAYING='PLAYING'),StateChangeReturn=NS(FAILURE='failure'),
           PadLinkReturn=NS(OK='ok'),MessageType=NS(ERROR='error'),
           PadProbeReturn=NS(OK='ok',DROP='drop'),
           PadProbeType=NS(BUFFER='buffer',EVENT_DOWNSTREAM='event'),EventType=NS(EOS='eos'))
    repository.Gst=gst;repository.GstVideo=NS();repository.GLib=NS(idle_add=Mock())
    monkeypatch.setitem(sys.modules,'gi',gi);monkeypatch.setitem(sys.modules,'gi.repository',repository)
    render=ModuleType('rpi.display.render');render.OverlayImage=object
    monkeypatch.setitem(sys.modules,'rpi.display.render',render)
    spec=importlib.util.spec_from_file_location('pi_receiver_contract',
        Path(__file__).resolve().parents[1]/'rpi/display/video.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    clock=NS(now=100.)
    monkeypatch.setattr(module.time,'monotonic',lambda:clock.now)
    def receiver():
        pad=NS(is_linked=lambda:True,unlink=Mock(),link=Mock(return_value='ok'),add_probe=Mock())
        return NS(get_static_pad=lambda _:pad,set_state=Mock(),get_state=Mock(return_value=(None,'NULL',None)),
                  set_name=Mock(),set_locked_state=Mock(),add_pad=Mock(return_value=True),sync_state_with_parent=Mock(return_value=True),
                  get_by_name=lambda _:NS(get_static_pad=lambda _:pad))
    video=module.ReturnVideo.__new__(module.ReturnVideo)
    video._closed=False;video._generation=0;video._receiver_failed=False
    video._restart_at=None;video._attempts=video._restarts=0;video._last_error=None
    video._live_since=None;video._state='waiting';video._retired=deque(maxlen=2)
    video._retiring_recorder=None;video._stopping_since=None;video._receiver_text='receiver'
    video._lock=Lock();video._showing_live=True;video.frame_identity=NS(clear=Mock())
    video._last_rtp_sample='old';video._on_receiver_lost=Mock()
    video._select=NS(set_property=Mock());video._live_pad=object();video._fallback_pad=object()
    video.recorder=None;video.recovery=NS(close=Mock())
    video.receiver=receiver();video.pipeline=NS(remove=Mock(),add=Mock(),set_state=Mock())
    video._wire_receiver=Mock();video._recording_config=None
    gst.parse_bin_from_description=Mock(side_effect=lambda *_:receiver())
    gst.GhostPad=NS(new=Mock(side_effect=lambda _name,pad:pad))
    return NS(video=video,module=module,gst=gst,clock=clock,receiver=receiver)


def test_fault_switches_to_fallback_invalidates_identity_and_notifies_only_once(recovery):
    v=recovery.video;old=v.receiver
    v._receiver_fault('source changed',0)
    v._receiver_fault('duplicate',0)
    assert v._state=='recovering' and v._attempts==1 and v._restart_at==100.25
    assert not v._showing_live and v._last_rtp_sample is None
    v._on_receiver_lost.assert_called_once_with('source changed')
    old.set_state.assert_not_called()  # Callback does not mutate the graph.
    v.show_live(True)
    assert not v._showing_live


def test_receiver_start_failure_does_not_stop_successful_fallback_output(recovery):
    v=recovery.video;v.receiver.sync_state_with_parent.return_value=False
    v.start()
    v.pipeline.set_state.assert_called_once_with('PLAYING')
    v.receiver.set_locked_state.assert_called_once_with(False)
    assert v._state=='recovering'


def test_rebuild_only_receiver_after_null_ack_keeps_display_pipeline(recovery):
    v=recovery.video;old=v.receiver;pipeline=v.pipeline
    v._receiver_fault('error',0)
    v.service(100.2);recovery.gst.parse_bin_from_description.assert_not_called()
    v.service(100.3)
    old.set_state.assert_called_once_with('NULL');pipeline.remove.assert_called_once_with(old)
    assert v.pipeline is pipeline and v.receiver is not old
    pipeline.set_state.assert_not_called()
    assert v._generation==1 and v._restarts==1 and v._state=='waiting'
    recovery.gst.parse_bin_from_description.assert_called_once_with('receiver',False)
    v._receiver_fault('stale callback',0)
    assert not v._receiver_failed


def test_unstopped_decoder_blocks_replacement_and_requires_explicit_retry(recovery):
    v=recovery.video;v.receiver.get_state.return_value=(None,'PLAYING',None)
    v._receiver_fault('error',0);v.service(100.3);v.service(102.4)
    assert v._state=='failed' and v._restart_at is None
    recovery.gst.parse_bin_from_description.assert_not_called()
    v.retry_receiver();v.service(103.)
    recovery.gst.parse_bin_from_description.assert_not_called()
    v.receiver.get_state.return_value=(None,'NULL',None)
    v.retry_receiver();v.service(104.)
    assert v._restarts==1


def test_rebuild_failures_back_off_then_stop_and_manual_retry_rearms(recovery):
    v=recovery.video
    recovery.gst.parse_bin_from_description.side_effect=RuntimeError('missing decoder')
    v._receiver_fault('first',0)
    for _ in range(5):
        due=v._restart_at;assert due is not None
        recovery.clock.now=due;v.service(due)
    assert v._attempts==6 and v._state=='failed'
    v.service(v._restart_at)
    assert v._restart_at is None
    assert recovery.gst.parse_bin_from_description.call_count==5
    recovery.gst.parse_bin_from_description.side_effect=lambda *_:recovery.receiver()
    v.retry_receiver();v.service(recovery.clock.now)
    assert v._state=='waiting' and v._restarts==1


def test_recording_finalization_is_polled_without_sleep_or_display_teardown(recovery):
    v=recovery.video;rec=NS(begin_close=Mock(),finish_close=Mock(return_value=False))
    v.recorder=rec;v._receiver_fault('error',0)
    rec.begin_close.assert_called_once();assert v.recorder is None
    v.service(100.3);recovery.gst.parse_bin_from_description.assert_not_called()
    rec.finish_close.return_value=True;v.service(102.3)
    assert v._restarts==1


def test_receiver_errors_are_local_but_output_errors_remain_fatal(recovery):
    v=recovery.video
    message=NS(type='error',src=NS(has_as_ancestor=lambda parent:parent is v.receiver),
               parse_error=lambda:(NS(message='lost'),None))
    assert v.handle_message(message)
    message.src=NS(has_as_ancestor=lambda _:False)
    assert not v.handle_message(message)
    old=v.receiver;v.service(100.3)
    message.src=NS(has_as_ancestor=lambda parent:parent is old)
    assert v.handle_message(message) and not v._receiver_failed


def test_closed_or_stale_generation_never_restarts(recovery):
    v=recovery.video
    v._closed=True;v._receiver_fault('late',0);v.service(200.)
    assert v._attempts==0
    recovery.gst.parse_bin_from_description.assert_not_called()


def test_receiver_eos_is_dropped_and_recovery_dispatched_on_ui_thread(recovery):
    v=recovery.video
    del v._wire_receiver
    v.pipeline.get_bus=lambda:object()
    recovery.module.ReceiverRecovery=Mock(return_value=v.recovery)
    v._wire_receiver()
    event_callback=v.receiver.get_static_pad('src').add_probe.call_args_list[-1].args[1]
    assert event_callback(None,NS(get_event=lambda:NS(type='eos'))) == 'drop'
    recovery.module.GLib.idle_add.assert_called_once_with(v._receiver_fault,'return stream EOS',0)
    assert v._attempts==0
