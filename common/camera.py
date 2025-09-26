"""Helpers for camera intrinsics shared across Jetson and PC components."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional, Tuple


class CameraIntrinsicsConfigError(ValueError):
    """Raised when the camera intrinsics configuration is invalid."""


@dataclass(frozen=True)
class CameraIntrinsics:
    """Calibrated camera intrinsics expressed in pixel units."""

    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    fov_deg: Optional[Tuple[float, float]]

    @classmethod
    def from_raw_config(
        cls, cfg: Mapping[str, Any], frame_size: Tuple[int, int]
    ) -> "CameraIntrinsics":
        camera_section: MutableMapping[str, Any] = dict(cfg.get("camera", {}))
        if not camera_section:
            raise CameraIntrinsicsConfigError("config is missing 'camera' section")

        intrinsics_cfg = camera_section.get("intrinsics")
        if not isinstance(intrinsics_cfg, Mapping):
            raise CameraIntrinsicsConfigError(
                "camera.intrinsics section is required and must be a mapping"
            )

        width, height = frame_size
        if width <= 0 or height <= 0:
            raise CameraIntrinsicsConfigError("frame dimensions must be positive")

        source = str(intrinsics_cfg.get("source", "fov")).strip().lower()

        cx_px = _parse_principal_component(intrinsics_cfg.get("cx_px"), width / 2.0, "cx_px")
        cy_px = _parse_principal_component(intrinsics_cfg.get("cy_px"), height / 2.0, "cy_px")

        if source == "fov":
            fov_cfg = intrinsics_cfg.get("fov_deg")
            if not isinstance(fov_cfg, Mapping):
                raise CameraIntrinsicsConfigError(
                    "camera.intrinsics.fov_deg must be a mapping with 'h' and 'v'"
                )
            try:
                hfov = float(fov_cfg["h"])
                vfov = float(fov_cfg["v"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CameraIntrinsicsConfigError(
                    "camera.intrinsics.fov_deg must include numeric h and v"
                ) from exc

            fx_px, fy_px = focal_lengths_from_fov(width, height, hfov, vfov)
            return cls(fx_px=fx_px, fy_px=fy_px, cx_px=cx_px, cy_px=cy_px, fov_deg=(hfov, vfov))

        if source == "direct":
            try:
                fx_px = float(intrinsics_cfg["fx_px"])
                fy_px = float(intrinsics_cfg["fy_px"])
            except KeyError as exc:
                raise CameraIntrinsicsConfigError(
                    "camera.intrinsics.fx_px and fy_px are required when source='direct'"
                ) from exc
            except (TypeError, ValueError) as exc:
                raise CameraIntrinsicsConfigError(
                    "camera.intrinsics.fx_px and fy_px must be numeric"
                ) from exc

            if fx_px <= 0.0 or fy_px <= 0.0:
                raise CameraIntrinsicsConfigError("focal lengths must be positive")

            return cls(fx_px=fx_px, fy_px=fy_px, cx_px=cx_px, cy_px=cy_px, fov_deg=None)

        raise CameraIntrinsicsConfigError(
            "camera.intrinsics.source must be either 'fov' or 'direct'"
        )


def focal_lengths_from_fov(
    width_px: int, height_px: int, hfov_deg: float, vfov_deg: float
) -> Tuple[float, float]:
    """Compute focal lengths in pixels from the provided FOV angles."""

    if width_px <= 0 or height_px <= 0:
        raise CameraIntrinsicsConfigError("frame dimensions must be positive")

    if not (0.0 < hfov_deg < 180.0) or not (0.0 < vfov_deg < 180.0):
        raise CameraIntrinsicsConfigError("FOV degrees must lie in (0, 180)")

    fx_px = (width_px / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy_px = (height_px / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    return fx_px, fy_px


def _parse_principal_component(value: Any, default: float, key: str) -> float:
    if value is None:
        return default
    try:
        val = float(value)
    except (TypeError, ValueError) as exc:
        raise CameraIntrinsicsConfigError(
            f"camera.intrinsics.{key} must be numeric if provided"
        ) from exc
    return val
