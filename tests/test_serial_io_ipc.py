from __future__ import annotations

import threading
import uuid

import pytest
import zmq

from common.serial_io import SerialCommandClient


@pytest.fixture
def serial_responder():
    context = zmq.Context()
    endpoint = f"inproc://serial-command-{uuid.uuid4().hex}"
    observed: list[dict[str, object]] = []
    ready = threading.Event()
    stop = threading.Event()
    errors: list[BaseException] = []

    def respond() -> None:
        # Only this thread creates, operates and closes its responder socket.
        server = context.socket(zmq.REP)
        try:
            server.setsockopt(zmq.LINGER, 0)
            server.setsockopt(zmq.SNDTIMEO, 1000)
            server.bind(endpoint)
            ready.set()
            for _ in range(100):
                if stop.is_set():
                    return
                if server.poll(10, zmq.POLLIN):
                    observed.append(server.recv_json(flags=zmq.NOBLOCK))
                    server.send_json({"type": "SerialCommandAck", "cmd_id": "test:1",
                                      "accepted": True, "queued": True})
                    return
            errors.append(TimeoutError("test responder received no command"))
        except BaseException as exc:
            errors.append(exc)
        finally:
            ready.set()
            server.close(linger=0)

    thread = threading.Thread(target=respond)
    try:
        thread.start()
        assert ready.wait(timeout=2.0), "test responder startup timed out"
        assert not errors, errors
        yield context, endpoint, observed
    finally:
        stop.set()
        if thread.ident is not None:
            thread.join(timeout=3.0)
        # Context destruction must not race a different thread's live socket.
        assert not thread.is_alive(), "test responder did not stop"
        context.term()
    assert not errors, errors


def test_serial_command_client_sends_typed_request_and_receives_ack(serial_responder) -> None:
    context, endpoint, observed = serial_responder
    client = SerialCommandClient(endpoint, timeout_ms=100, ctx=context)
    try:
        reply = client.send_command({"cmd_id": "test:1", "type": "wrong"})
    finally:
        client.close()

    assert reply is not None
    assert reply["accepted"] is True
    assert observed == [{"cmd_id": "test:1", "type": "SerialCommandRequest"}]


def test_serial_responder_cleanup_without_a_command(serial_responder) -> None:
    # Fixture teardown must stop its worker even if no client sends a command.
    _context, _endpoint, observed = serial_responder
    assert observed == []


def test_serial_command_client_timeout_resets_req_socket() -> None:
    context = zmq.Context()
    endpoint = f"inproc://serial-command-timeout-{uuid.uuid4().hex}"
    client = SerialCommandClient(endpoint, timeout_ms=1, ctx=context)
    try:
        assert client.send_command({"cmd_id": "test:timeout"}) is None
        assert client.send_command({"cmd_id": "test:timeout-again"}) is None
    finally:
        client.close()
        context.destroy(linger=0)
