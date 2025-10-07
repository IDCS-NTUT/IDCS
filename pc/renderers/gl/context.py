"""Utilities for creating off-screen OpenGL contexts.

This module provides a thin abstraction over the platform specific context
creation back-ends so the renderer core can work with a single
`GLContext` object regardless of whether the program is running on a desktop
machine or a Jetson class device that relies on EGL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping
import os
import sys


ContextBackend = Literal["desktop", "egl"]


class GLContextError(RuntimeError):
    """Raised when an OpenGL context cannot be created or used."""


@dataclass
class ContextConfig:
    """Configuration values used for OpenGL context creation."""

    width: int = 1280
    height: int = 720
    backend: str = "auto"
    gl_version: tuple[int, int] = (3, 3)
    debug: bool = False
    share: Any | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Context width and height must be positive")

        if len(self.gl_version) != 2:
            raise ValueError("gl_version must be a (major, minor) tuple")

        major, minor = self.gl_version
        if major < 1 or minor < 0:
            raise ValueError("gl_version must be >= (1, 0)")

        normalized = (self.backend or "auto").strip().lower()
        if normalized == "":
            normalized = "auto"
        self.backend = normalized

    def resolve_backend(self) -> ContextBackend:
        """Return the concrete backend that should be used."""

        backend = self.backend
        if backend == "auto":
            return _auto_backend()
        if backend in {"desktop", "egl"}:
            return backend  # type: ignore[return-value]
        raise ValueError(f"Unsupported OpenGL backend '{backend}'")


class GLContext:
    """Wrapper around the native OpenGL context handle."""

    def __init__(
        self,
        *,
        backend: ContextBackend,
        handle: Any,
        config: ContextConfig,
        destroy: Callable[[], None],
    ) -> None:
        self.backend = backend
        self.handle = handle
        self.config = config
        self._destroy_cb = destroy
        self._destroyed = False

    def destroy(self) -> None:
        """Tear down the context and free the underlying resources."""

        if self._destroyed:
            return
        self._destroy_cb()
        self._destroyed = True

    def __enter__(self) -> "GLContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        self.destroy()


def create_gl_context(
    config: ContextConfig | Mapping[str, Any] | None = None,
) -> GLContext:
    """Create an OpenGL context based on the supplied configuration."""

    resolved_config = _ensure_config(config)
    backend = resolved_config.resolve_backend()
    if backend == "desktop":
        return _create_standalone_context("desktop", resolved_config)
    return _create_standalone_context("egl", resolved_config)


def _ensure_config(
    config: ContextConfig | Mapping[str, Any] | None,
) -> ContextConfig:
    if config is None:
        return ContextConfig()
    if isinstance(config, ContextConfig):
        return config
    if isinstance(config, Mapping):
        return ContextConfig(**config)
    raise TypeError("context configuration must be a ContextConfig or mapping")


def _auto_backend() -> ContextBackend:
    if sys.platform.startswith("linux"):
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            return "desktop"
        return "egl"
    return "desktop"


def _create_standalone_context(
    backend: ContextBackend,
    config: ContextConfig,
) -> GLContext:
    moderngl = _import_moderngl()
    require = _gl_version_to_requirement(config.gl_version)
    kwargs: dict[str, Any] = {"require": require}
    if backend == "egl":
        kwargs["backend"] = "egl"

    share_handle = _unwrap_share_handle(config.share)
    if share_handle is not None:
        kwargs["share"] = share_handle

    try:
        ctx = moderngl.create_standalone_context(**kwargs)
    except Exception as exc:  # pragma: no cover - exercised in integration tests
        raise GLContextError(
            f"Failed to create {backend} OpenGL context: {exc!s}"
        ) from exc

    _configure_context(ctx, config)

    return GLContext(
        backend=backend,
        handle=ctx,
        config=config,
        destroy=lambda: _release_handle(ctx),
    )


def _import_moderngl():
    try:
        import moderngl  # type: ignore[import]
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise GLContextError(
            "moderngl is required to use the OpenGL renderer backend"
        ) from exc
    return moderngl


def _gl_version_to_requirement(version: tuple[int, int]) -> int:
    major, minor = version
    return major * 100 + minor * 10


def _unwrap_share_handle(share: Any | None) -> Any | None:
    if share is None:
        return None
    if isinstance(share, GLContext):
        return share.handle
    return share


def _configure_context(ctx: Any, config: ContextConfig) -> None:
    try:
        ctx.viewport = (0, 0, config.width, config.height)
    except AttributeError:  # pragma: no cover - depends on moderngl version
        pass

    try:
        ctx.error_checking = config.debug
    except AttributeError:  # pragma: no cover - depends on moderngl version
        pass


def _release_handle(handle: Any) -> None:
    release = getattr(handle, "release", None)
    if callable(release):
        release()


__all__ = [
    "ContextBackend",
    "ContextConfig",
    "GLContext",
    "GLContextError",
    "create_gl_context",
]
