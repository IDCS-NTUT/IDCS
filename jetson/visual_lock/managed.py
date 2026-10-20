"""Managed receiver policy and bounded evidence; no devices or sockets."""
from pathlib import Path
import uuid

MODES = {"camera_visual_lock": "camera", "sim_visual_lock": "sim"}


def validate_managed(mode, source, *, roi_bind, identity_bind, seed, handoff):
    if mode is None:
        return
    if (MODES.get(mode) != source or not roi_bind or not identity_bind
            or seed is not None or handoff):
        raise ValueError("managed mode requires matching source, panel ROI and exact return identity; VPI only")


def evidence_directory(root, *, max_sessions=8):
    """Retain at most eight sessions, never prune recordings or rollback data.

    Each managed run caps logs/capture/seed diagnostics below 512 MB. Once all
    slots are occupied, refuse startup until the operator archives evidence.
    The exclusive systemd source owner serializes allocation.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if len(list(root.iterdir())) >= max_sessions:
        raise ValueError(f"visual-lock evidence retention full: archive sessions from {root}")
    return root / ("session-" + uuid.uuid4().hex)


class BoundedLog:
    """Keep complete initial JSONL records; a full budget never stops tracking."""
    def __init__(self, path, max_bytes):
        self.stream = Path(path).open("x", encoding="utf-8")
        self.max_bytes, self.bytes, self.dropped = max_bytes, 0, 0

    def write(self, row):
        size = len(row.encode("utf-8"))
        if self.bytes + size > self.max_bytes:
            self.dropped += 1
            return
        self.stream.write(row)
        self.bytes += size

    def flush(self):
        self.stream.flush()

    def close(self):
        self.stream.close()

    def report(self):
        return dict(bytes=self.bytes, max_bytes=self.max_bytes, dropped=self.dropped)


def loss_returns_to_selection(request, result):
    return request is not None and result is not None and result.decision.state == "lost"


def recycle_lost_roi(request, result, *, close, clear_cache, prepare, notify):
    """Discard lost authority and prepare resources, without ever calling seed."""
    if not loss_returns_to_selection(request,result):
        return False
    notify(request,False,result.decision.reason)
    close()
    clear_cache()
    prepare()
    return True
