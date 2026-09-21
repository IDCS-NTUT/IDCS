from pathlib import Path

import pytest

from common.perception import perception_snapshot_from_json
from common.synthetic_perception import load_synthetic_scenario, snapshot_at
from jetson.deepstream import snapshot_transport
from jetson.deepstream.snapshot_transport import SnapshotTransport


class _FakeSocket:
    def __init__(self) -> None:
        self.bound = None
        self.sent: list[str] = []

    def setsockopt(self, *_args) -> None:
        pass

    def bind(self, endpoint: str) -> None:
        self.bound = endpoint

    def send_string(self, payload: str, *, flags: int) -> None:
        assert flags == snapshot_transport.zmq.NOBLOCK
        self.sent.append(payload)

    def close(self) -> None:
        pass


class _FakeContext:
    def __init__(self) -> None:
        self.sockets: list[_FakeSocket] = []

    def socket(self, _kind) -> _FakeSocket:
        socket = _FakeSocket()
        self.sockets.append(socket)
        return socket

    def term(self) -> None:
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
