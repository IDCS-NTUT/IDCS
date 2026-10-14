import time

import zmq

from common.return_frame_identity import ReturnFrameIdentity
from common.rtp_identity import RtpFrameKey
from rpi.display.return_identity import ReturnIdentityReceiver


def test_headers_join_current_image_without_a_display_tick_and_worker_closes():
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER,0)
    port = publisher.bind_to_random_port('tcp://127.0.0.1')
    identity = ReturnFrameIdentity()
    receiver = ReturnIdentityReceiver(f'tcp://127.0.0.1:{port}',identity)
    header = dict(rtp_ssrc=123,rtp_timestamp=456,tracker_session='run',frame_id=42,
                  source_time_ns=100,source_clock_domain='pc_monotonic')
    try:
        assert receiver.ready.wait(1)
        end = time.monotonic()+2
        while receiver.headers == 0 and time.monotonic() < end:
            publisher.send_json(header)
            time.sleep(.01)
        assert receiver.headers > 0 and receiver.error is None
        current = dict(header,rtp_timestamp=457,frame_id=43)
        displayed_s = time.monotonic()
        identity.push_marker(1000,RtpFrameKey(123,457))
        identity.push_decoded(1000)
        assert identity.displayed(1000,displayed_s) is None
        publisher.send_json(current)
        frame = None
        while frame is None and time.monotonic()-displayed_s < .25:
            frame = identity.latest(time.monotonic())
            time.sleep(.002)
        assert frame is not None and frame.frame_id == 43
        assert frame.displayed_local_s == displayed_s  # receive time cannot rejuvenate image age
        assert identity.late_matched == 1
    finally:
        receiver.close()
        publisher.close(0)
        context.term()
    assert not receiver.thread.is_alive()
