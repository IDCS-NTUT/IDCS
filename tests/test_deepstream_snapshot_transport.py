from pathlib import Path

import pytest

from common.perception import perception_snapshot_from_json
from common.synthetic_perception import load_synthetic_scenario, snapshot_at
from jetson.deepstream import snapshot_transport
from jetson.deepstream.snapshot_transport import SnapshotTransport
from common.rtp_identity import RtpFrameKey


class _FakeSocket:
    def __init__(self) -> None:
        self.bound = None
        self.sent: list[str] = []
        self.received: list[dict] = []

    def setsockopt(self, *_args) -> None:
        pass

    def bind(self, endpoint: str) -> None:
        self.bound = endpoint

    def send_string(self, payload: str, *, flags: int) -> None:
        assert flags == snapshot_transport.zmq.NOBLOCK
        self.sent.append(payload)

    def recv_json(self, *, flags: int):
        assert flags == snapshot_transport.zmq.NOBLOCK
        if not self.received:
            raise snapshot_transport.zmq.Again()
        return self.received.pop(0)

    def close(self) -> None:
        pass


class _FakeContext:
    def __init__(self) -> None:
        self.sockets: list[_FakeSocket] = []

    def socket(self, _kind) -> _FakeSocket:
        socket = _FakeSocket()
        self.sockets.append(socket)
        return socket

    def destroy(self, linger=None) -> None:
        pass


def _snapshot():
    scenario = load_synthetic_scenario(
        Path("tests/fixtures/synthetic_tracking_v1.json")
    )
    return snapshot_at(scenario, 2)


def test_transport_publishes_lossless_v2(monkeypatch):
    context = _FakeContext()
    monkeypatch.setattr(snapshot_transport.zmq, "Context", lambda: context)
    transport = SnapshotTransport(
        header_bind=None,
        snapshot_bind="tcp://127.0.0.1:6102",
    )

    source = _snapshot()
    assert transport.publish(source)

    socket = context.sockets[0]
    assert socket.bound == "tcp://127.0.0.1:6102"
    assert perception_snapshot_from_json(socket.sent[0]) == source
    assert transport.report()["published"] == 1


def test_transport_requires_snapshot_endpoint():
    with pytest.raises(ValueError, match="snapshot_bind"):
        SnapshotTransport(header_bind=None, snapshot_bind="")


def test_verified_transport_joins_by_rtp_key_and_decoded_pts(monkeypatch):
    context = _FakeContext()
    monkeypatch.setattr(snapshot_transport.zmq, "Context", lambda: context)
    transport = SnapshotTransport(
        header_bind="tcp://127.0.0.1:6101",
        snapshot_bind="tcp://127.0.0.1:6102",
        verified_rtp_headers=True,
    )
    context.sockets[0].received.extend([
        {"frame_id": 2, "source_time_ns": 1_033_333_333,
         "source_clock_domain": "pc_monotonic", "rtp_ssrc": 9, "rtp_timestamp": 6000},
        {"frame_id": 1, "source_time_ns": 1_000_000_000,
         "source_clock_domain": "pc_monotonic", "rtp_ssrc": 9, "rtp_timestamp": 3000},
    ])
    transport.drain_headers()
    transport.push_rtp_marker(decoded_pts_ns=200, key=RtpFrameKey(9, 3000))
    transport.push_rtp_marker(decoded_pts_ns=300, key=RtpFrameKey(9, 6000))
    first = transport.next_header(decoded_pts_ns=200)
    second = transport.next_header(decoded_pts_ns=300)
    assert first is not None and first.frame_id == 1
    assert first.source_time_ns == 1_000_000_000
    assert first.source_identity_verified
    assert second is not None and second.frame_id == 2
    assert transport.next_header(decoded_pts_ns=200) is None
    transport.close()
