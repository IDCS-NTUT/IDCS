# common/shutdown.py
import signal, threading

def install_signal_handlers() -> threading.Event:
    """
    Returns a threading.Event that is set when SIGINT or SIGTERM is received.
    Works on Linux and Windows (Ctrl+C -> SIGINT).
    """
    stop_event = threading.Event()

    def _handler(signum, _):
        stop_event.set()

    # Python delivers signals to the main thread only.
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass  # some platforms may not allow setting all

    return stop_event