"""One-shot software-only ROI request tied to an exact displayed source frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import uuid

# Shared upper bound for return identity, expanded ROI status and seed results.
# Status carries bounded counters/terminal diagnostics, not just a frame header.
ROI_SIDEBAND_MAX_BYTES = 16_384


@dataclass(frozen=True)
class VisualRoiMode:
    """Source-session-scoped compare-and-set; never a source/service command."""
    command_id: str
    tracker_session: str
    mode: str
    expected_revision: int

    @classmethod
    def from_mapping(cls, raw: Mapping) -> "VisualRoiMode":
        if raw.get('type') != 'visual_roi_mode_v1':
            raise ValueError('unexpected processing-mode type')
        try:
            command_id = str(uuid.UUID(str(raw['command_id'])))
            session, mode, revision = raw['tracker_session'], raw['mode'], raw['expected_revision']
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('invalid processing-mode request') from exc
        if (not isinstance(session,str) or not 1 <= len(session) <= 80
                or mode not in ('yolo','roi') or type(revision) is not int or revision < 0):
            raise ValueError('invalid processing-mode request')
        return cls(command_id,session,mode,revision)

    def as_message(self) -> dict:
        return dict(type='visual_roi_mode_v1', **self.__dict__)


@dataclass(frozen=True)
class VisualRoiSelect:
    """Short, session-scoped cache lease on the existing seed socket."""

    selection_id: str
    tracker_session: str
    enabled: bool

    @classmethod
    def from_mapping(cls, raw: Mapping) -> "VisualRoiSelect":
        if raw.get("type") != "visual_roi_select_v1":
            raise ValueError("unexpected ROI selection type")
        try:
            selection_id = str(uuid.UUID(str(raw["selection_id"])))
            session, enabled = raw["tracker_session"], raw["enabled"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid ROI selection lease") from exc
        if not isinstance(session, str) or not 0 < len(session) <= 80 or type(enabled) is not bool:
            raise ValueError("invalid ROI selection lease")
        return cls(selection_id, session, enabled)

    def as_message(self) -> dict:
        return {"type": "visual_roi_select_v1", **self.__dict__}


@dataclass(frozen=True)
class VisualRoiSeed:
    request_id: str
    tracker_session: str
    frame_id: int
    source_time_ns: int
    source_clock_domain: str
    x: int
    y: int
    w: int
    h: int
    width: int = 1280
    height: int = 720

    @classmethod
    def from_mapping(cls, raw: Mapping) -> "VisualRoiSeed":
        if raw.get("type") != "visual_roi_seed_v1":
            raise ValueError("unexpected ROI request type")
        try:
            request_id = str(uuid.UUID(str(raw["request_id"])))
            session = str(raw["tracker_session"])
            domain = str(raw["source_clock_domain"])
            numbers = {key: raw[key] for key in (
                "frame_id", "source_time_ns", "x", "y", "w", "h", "width", "height")}
            if any(type(value) is not int for value in numbers.values()):
                raise ValueError("ROI geometry and identity must be integers")
            instance = cls(request_id, session, numbers["frame_id"],
                numbers["source_time_ns"], domain, numbers["x"], numbers["y"],
                numbers["w"], numbers["h"], numbers["width"], numbers["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid ROI request") from exc
        if (not session or len(session) > 80 or domain not in {"pc_monotonic", "jetson_monotonic"}
                or instance.frame_id < 0 or instance.source_time_ns < 0
                or (instance.width, instance.height) != (1280, 720)
                or min(instance.x, instance.y) < 0 or min(instance.w, instance.h) < 12
                or instance.x + instance.w > instance.width
                or instance.y + instance.h > instance.height):
            raise ValueError("out-of-bounds or unverified ROI request")
        return instance

    def as_message(self) -> dict:
        return {"type": "visual_roi_seed_v1", **self.__dict__}


@dataclass(frozen=True)
class VisualRoiRelease:
    """Cancel one exact ROI seed; never a generic target or motion command."""

    request_id: str
    tracker_session: str

    @classmethod
    def from_mapping(cls, raw: Mapping) -> "VisualRoiRelease":
        if raw.get("type") != "visual_roi_release_v1":
            raise ValueError("unexpected ROI release type")
        try:
            request_id = str(uuid.UUID(str(raw["request_id"])))
            session = raw["tracker_session"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid ROI release") from exc
        if not isinstance(session, str) or not session or len(session) > 80:
            raise ValueError("invalid ROI release session")
        return cls(request_id, session)

    def as_message(self) -> dict:
        return {"type": "visual_roi_release_v1", **self.__dict__}
