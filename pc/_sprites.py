"""Helpers for loading billboard sprites used by the simulation renderer.

The CPU renderer and the simulation camera both need to resolve sprite
references (aliases or paths), load the backing images, and reason about their
dimensions.  Centralising the logic here keeps the behaviour consistent across
callers and avoids repeated disk access by using a small in-memory cache.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

__all__ = ["load_sprite_image", "get_sprite_aspect_ratio", "resolve_sprite_path"]

_ROOT_DIR = Path(__file__).resolve().parents[1]
_ASSETS_DIR = _ROOT_DIR / "assets"

# Basic alias map so billboards can reference sprites by short names.
_SPRITE_ALIASES: Dict[str, Path] = {
    "person": _ASSETS_DIR / "sprites" / "person.png",
    "drone": _ASSETS_DIR / "sprites" / "drone.png",
}


def resolve_sprite_path(sprite_ref: Any) -> Path:
    """Resolve ``sprite_ref`` to an absolute :class:`Path`.

    Aliases defined in :data:`_SPRITE_ALIASES` take precedence.  Otherwise the
    reference is interpreted as a filesystem path relative to the repository
    root.
    """

    key = str(sprite_ref)
    alias = _SPRITE_ALIASES.get(key)
    if alias is not None:
        return alias

    candidate = Path(key)
    if not candidate.is_absolute():
        candidate = _ROOT_DIR / candidate
    return candidate


@lru_cache(maxsize=32)
def _load_sprite_cached(key: str) -> Tuple[np.ndarray, np.ndarray]:
    path = resolve_sprite_path(key)
    if not path.exists():
        raise ValueError(f"billboard sprite '{key}' does not exist")

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"billboard sprite '{key}' could not be loaded")

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
    elif image.shape[2] == 4:
        bgr = image[..., :3]
        alpha = image[..., 3]
    elif image.shape[2] == 3:
        bgr = image
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
    else:
        raise ValueError(f"billboard sprite '{key}' has unsupported channel count")

    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    else:
        bgr = np.ascontiguousarray(bgr)

    if alpha.dtype != np.uint8:
        alpha = np.clip(alpha, 0, 255).astype(np.uint8)
    else:
        alpha = np.ascontiguousarray(alpha)

    return bgr, alpha


def load_sprite_image(sprite_ref: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return the BGR image and alpha mask for ``sprite_ref``.

    The returned arrays are cached in-memory, so callers must treat them as
    read-only.
    """

    key = str(sprite_ref)
    return _load_sprite_cached(key)


def get_sprite_aspect_ratio(sprite_ref: Any) -> float:
    """Return ``width / height`` for the resolved sprite image."""

    sprite_bgr, _ = load_sprite_image(sprite_ref)
    height, width = sprite_bgr.shape[:2]
    if height <= 0:
        raise ValueError(f"billboard sprite '{sprite_ref}' has invalid dimensions")
    return float(width) / float(height)

