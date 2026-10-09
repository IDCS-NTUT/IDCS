"""Bounded loopback request to the already-running simulator; no motion path."""
import zmq

from common.sim_mode import require_simulation_loopback_endpoint
from common.sim_score import DEFAULT_SCORE_ENDPOINT, ScoreResetRequest


def clear_score(task, client_session, *, endpoint=DEFAULT_SCORE_ENDPOINT):
    endpoint = require_simulation_loopback_endpoint(endpoint, 'operator.sim_score_endpoint')
    request = ScoreResetRequest(source_session=task.source_session,
                               client_session=client_session, request_id=task.task_id)
    # Socket/context belong to this worker thread, never the polling thread.
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    try:
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, 2000)
        socket.setsockopt(zmq.RCVTIMEO, 2000)
        socket.setsockopt(zmq.MAXMSGSIZE, 4096)
        socket.connect(endpoint)
        socket.send_string(request.model_dump_json())
        reply = socket.recv_json()
        if not isinstance(reply, dict) or reply.get('type') != 'SimScoreResetReply':
            return 1, 'invalid simulator score acknowledgement'
        if reply.get('ok') is not True:
            return 1, str(reply.get('message', 'score reset refused'))[:300]
        if (reply.get('request_id') != request.request_id or reply.get('source_session') != request.source_session
                or type(reply.get('epoch')) is not int or reply['epoch'] < 1):
            return 1, 'score acknowledgement identity mismatch'
        return 0, f"Swarm score cleared (epoch {reply['epoch']})"
    except zmq.Again:
        return 1, 'score reset acknowledgement timed out; outcome unconfirmed'
    finally:
        socket.close(0)
        ctx.term()
