"""OpenGL renderer package placeholder."""

from __future__ import annotations

from typing import Any

from .. import register_renderer


class GLRenderer:
    """Placeholder OpenGL renderer stub.

    The actual implementation will be provided in subsequent steps.  The class
    is registered with the renderer plug-in system so that configuration wiring
    can be exercised ahead of the full renderer bring-up.
    """

    def __init__(self, *, context: Any, **_: Any) -> None:
        if context is None:  # pragma: no cover - defensive only
            raise ValueError("SimCamera context must be provided")
        self._context = context

    def render(self, frame: Any, /, **_: Any) -> None:  # pragma: no cover - stub
        raise NotImplementedError("OpenGL renderer is not implemented yet")


register_renderer("gl", lambda **kwargs: GLRenderer(**kwargs))

__all__ = ["GLRenderer"]
