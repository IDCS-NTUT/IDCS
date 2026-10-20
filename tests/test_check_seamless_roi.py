"""Bounded helper transport contracts; all sockets and clocks are faked."""
import json
from types import SimpleNamespace as NS

import pytest
import zmq

from tools import check_seamless_roi as helper


@pytest.mark.parametrize("failure", (None, "select_send", "after_select", "seed_send",
                                   "after_seed", "release_send", "interrupt", "cleanup"))
def test_helper_waits_for_peer_and_cancels_on_every_exit(tmp_path, monkeypatch, failure):
    clock = NS(now=100.)
    peer = NS(select=None, request=None, released=False, result_sent=False, commands=[], pending=[],
              delayed=False, closed=[], sends=0)
    class Socket:
        def __init__(self, kind):
            self.kind, self.options = kind, {}
        def setsockopt(self, option, value):
            self.options[option] = value
        def connect(self, _endpoint):
            pass
        def send_json(self, message, flags=0):
            assert flags == 0  # A first pre-handshake NOBLOCK send is a regression.
            assert self.options[zmq.IMMEDIATE] == 1 and self.options[zmq.SNDTIMEO] == 2000
            peer.commands.append(message)
            kind = message["type"]
            if not peer.delayed:
                clock.now += .3  # Peer handshake takes longer than the reported 0.2 s failure.
                peer.delayed = True
            if kind == "visual_roi_select_v1":
                peer.select = message
                if failure == "select_send" and message["enabled"]:
                    raise zmq.Again()
                if failure == "cleanup" and not message["enabled"]:
                    raise zmq.ZMQError("peer gone")
            elif kind == "visual_roi_seed_v1":
                peer.request = message
                if failure == "seed_send":
                    raise zmq.Again()
            else:
                if failure == "release_send" and not peer.released:
                    peer.released = True
                    raise zmq.Again()
                peer.released = True
        def recv_json(self, flags=0):
            if not peer.pending:
                raise zmq.Again()
            return peer.pending.pop(0)
        def recv(self):
            if failure == "after_seed":
                raise ValueError("injected perception error")
            if failure == "interrupt":
                raise KeyboardInterrupt()
            return b"snapshot"
        def close(self, linger):
            peer.closed.append((self.kind, linger))
    identity, perception, push = Socket("identity"), Socket("perception"), Socket("push")
    class Poller:
        def register(self, *_):
            pass
        def poll(self, _timeout):
            clock.now += .02
            capable = not (failure == "after_select" and peer.select is not None)
            peer.pending = [dict(type="visual_roi_status_v1", source_mode="sim", managed_mode="sim",
                roi_capable=capable, source_ready=True, tracker_session="session", roi_cache_enabled=True,
                roi_cache_frames=3), dict(rtp_ssrc=1, tracker_session="session", frame_id=10,
                source_time_ns=1_000_000, source_clock_domain="pc_monotonic")]
            if peer.request is not None and not peer.result_sent:
                peer.result_sent = True
                peer.pending.append(dict(type="visual_roi_result_v1", accepted=True,
                                         request_id=peer.request["request_id"]))
            return [(identity, zmq.POLLIN)] + ([(perception, zmq.POLLIN)] if peer.request else [])
    sockets = iter((identity, perception, push))
    context = NS(socket=lambda _kind: next(sockets), term=lambda: None)
    monkeypatch.setattr(helper.zmq, "Context", lambda: context)
    monkeypatch.setattr(helper.zmq, "Poller", Poller)
    monkeypatch.setattr(helper.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(helper, "perception_snapshot_from_json", lambda _raw:
        NS(frame=NS(tracker_session="session"), roi_state=NS(owns_selection=not peer.released),
           selection=None if peer.released else NS(policy="operator_visual_lock")))
    output = tmp_path/"helper.json"
    result = helper.run(["--box", "530", "247", "96", "72", "--duration-s", "1",
                         "--output", str(output)])
    report = json.loads(output.read_text())
    assert result == int(failure is not None and failure != "cleanup")
    assert peer.delayed and report["elapsed_s"] >= .3
    assert peer.commands[-1]["type"] == "visual_roi_select_v1"
    assert peer.commands[-1]["enabled"] is False
    if peer.request is not None:
        assert any(command["type"] == "visual_roi_release_v1" for command in peer.commands)
    if failure is None:
        assert report["accepted"] and report["selected_frames"] and report["detector_returned"]
        # Renew the original lease throughout lock; no new lease/cache is opened.
        assert sum(command == peer.commands[0] for command in peer.commands) > 2
    if failure == "cleanup":
        assert report["selection_cancel_delivery_failed"]
    assert ("push", 2000) in peer.closed
    assert ("identity", 0) in peer.closed and ("perception", 0) in peer.closed
