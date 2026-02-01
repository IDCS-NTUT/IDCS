# common/shutdown.py
import os
import signal
import sys
import threading


def install_signal_handlers(overtime_s: float | None = 5.0) -> threading.Event:
    """
    Returns a threading.Event that is set when SIGINT or SIGTERM is received.
    Works on Linux and Windows (Ctrl+C -> SIGINT).
    """
    stop_event = threading.Event()
    timer_lock = threading.Lock()
    force_timer: threading.Timer | None = None

    def _handler(signum, _):
        nonlocal force_timer
        stop_event.set()
        if overtime_s is None or overtime_s <= 0:
            return
        with timer_lock:
            if force_timer is None:
                force_timer = threading.Timer(overtime_s, _force_exit)
                force_timer.daemon = True
                force_timer.start()

    def _force_exit():
        sys.stderr.write(
            f"\nShutdown overtime ({overtime_s}s) exceeded; forcing exit.\n"
        )
        sys.stderr.flush()
        os._exit(1)

    # Python delivers signals to the main thread only.
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass  # some platforms may not allow setting all

    return stop_event
