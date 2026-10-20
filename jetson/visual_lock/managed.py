"""Managed receiver policy and bounded evidence; no devices or sockets."""
from pathlib import Path
import re
import shutil
import uuid

MODES = {"camera_visual_lock": "camera", "sim_visual_lock": "sim"}


def validate_managed(mode, source, *, roi_bind, identity_bind, seed, handoff):
    if mode is None:
        return
    if (MODES.get(mode) != source or not roi_bind or not identity_bind
            or seed is not None or handoff):
        raise ValueError("managed mode requires matching source, panel ROI and exact return identity; VPI only")


SESSION = re.compile(r"session-[0-9a-f]{32}")


def _size(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


def evidence_directory(root, *, budget_bytes=4_000_000_000, reserve_bytes=512_000_000,
                       min_free_bytes=5_000_000_000):
    """Rotate managed sessions within a size budget per source.

    Each managed run caps its logs/capture/seed diagnostics below 512 MB
    (``reserve_bytes``). Before a new session the oldest managed
    ``session-<hex>`` directories in this root are deleted until the kept
    sessions plus the reserve fit ``budget_bytes``. Nothing else is touched:
    other names in the root, trials, recordings and rollback data are never
    pruned. Startup is refused only if the disk itself has less than
    ``min_free_bytes`` free. The exclusive systemd source owner serializes this.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sessions = sorted((p for p in root.iterdir() if p.is_dir() and not p.is_symlink()
                       and SESSION.fullmatch(p.name)), key=lambda p: p.stat().st_mtime)
    sizes = {p: _size(p) for p in sessions}
    while sessions and sum(sizes.values()) + reserve_bytes > budget_bytes:
        oldest = sessions.pop(0)
        shutil.rmtree(oldest)
        del sizes[oldest]
    if shutil.disk_usage(root).free < min_free_bytes:
        raise ValueError(f"visual-lock evidence: less than {min_free_bytes // 1_000_000_000} GB free under {root}")
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
