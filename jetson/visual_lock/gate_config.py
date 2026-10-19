"""Explicit replay/live gate overrides; defaults remain the recorded ABI4 gates."""
import json
from pathlib import Path

from .fusion import Gates


def load_gates(path=None):
    if path is None:
        return Gates()
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("gate config must be a JSON object")
    if any(type(v) not in (int, float) for v in values.values()):
        raise ValueError("gate values must be numbers, not booleans or strings")
    try:
        return Gates(**values)
    except TypeError as exc:
        raise ValueError(f"invalid gate config: {exc}") from exc
