"""Receive exact-frame headers independently of menu/HUD rendering.

The worker owns its ZMQ socket/context and feeds the thread-safe video join.
Status and seed results retain their actual local receive times for the UI.
"""
import json
from queue import Empty, Full, Queue
import threading
import time

import zmq
from common.visual_roi_seed import ROI_SIDEBAND_MAX_BYTES


class ReturnIdentityReceiver:
    def __init__(self, endpoint, identity):
        self.endpoint, self.identity = endpoint, identity
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.results = Queue(maxsize=8)
        self._status = None
        self._lock = threading.Lock()
        self.headers = self.invalid = self.result_overflow = 0
        self.error = None
        self.thread = threading.Thread(target=self._run,name='return-frame-identity',daemon=True)
        self.thread.start()

    def _run(self):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        try:
            socket.setsockopt(zmq.LINGER,0)
            socket.setsockopt(zmq.RCVHWM,128)
            socket.setsockopt(zmq.MAXMSGSIZE,ROI_SIDEBAND_MAX_BYTES)
            socket.setsockopt(zmq.SUBSCRIBE,b'')
            socket.connect(self.endpoint)
            self.ready.set()
            while not self.stop.is_set():
                if not socket.poll(10):
                    continue
                received_s = time.monotonic()
                try:
                    value = json.loads(socket.recv())
                    if not isinstance(value,dict):
                        raise ValueError('sideband must be an object')
                except (ValueError,TypeError):
                    self.invalid += 1
                    continue
                if value.get('type') == 'visual_roi_status_v1':
                    with self._lock:
                        self._status = value,received_s
                elif value.get('type') == 'visual_roi_result_v1':
                    try:
                        self.results.put_nowait((value,received_s))
                    except Full:
                        self.result_overflow += 1
                elif self.identity.push_header(value):
                    self.headers += 1
                else:
                    self.invalid += 1
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.ready.set()
            socket.close(0)
            context.term()

    def drain(self):
        with self._lock:
            status, self._status = self._status, None
        if status is not None:
            yield status
        for _ in range(8):
            try:
                yield self.results.get_nowait()
            except Empty:
                break

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError('return identity receiver did not stop')
