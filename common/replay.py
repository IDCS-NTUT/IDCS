"""Replay helpers for deterministic debug/step-mode runs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2

from common.clock import MonotonicClock
from common.schemas import DetectionMsg


_LOG = logging.getLogger("common.replay")


class FrameReplayCapture:
    """``VideoCapture``-like adapter that replays frames from disk."""

    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(self, path: Path, *, loop: bool = True) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"frame replay path does not exist: {self._path}")
        self._loop = bool(loop)
        self._cap = None
        self._files: List[Path] = []
        self._index = 0
        self._exhausted = False

        if self._path.is_dir():
            self._files = sorted(
                [
                    p
                    for p in self._path.iterdir()
                    if p.is_file() and p.suffix.lower() in self._IMAGE_EXTS
                ]
            )
            if not self._files:
                raise ValueError(
                    f"no replayable image files found under {self._path!s}"
                )
        else:
            cap = cv2.VideoCapture(str(self._path))
            if not cap.isOpened():
                raise ValueError(f"failed to open replay video file: {self._path}")
            self._cap = cap

    # OpenCV-style API -------------------------------------------------
    def isOpened(self) -> bool:  # pragma: no cover - simple forwarding method
        if self._cap is not None:
            return bool(self._cap.isOpened())
        return bool(self._files)

    def read(self) -> Tuple[bool, Optional["cv2.Mat"]]:
        if self._cap is not None:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                return True, frame
            if not self._loop:
                return False, None
            # restart and try once more
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok or frame is None:
                return False, None
            return True, frame

        if self._exhausted:
            return False, None

        frame_path = self._files[self._index]
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"failed to decode replay frame: {frame_path}")

        self._index += 1
        if self._index >= len(self._files):
            if self._loop:
                self._index = 0
            else:
                self._exhausted = True

        return True, frame

    def release(self) -> None:  # pragma: no cover - trivial wrapper
        if self._cap is not None:
            self._cap.release()
            self._cap = None


@dataclass
class _ReplayEntry:
    msg: DetectionMsg


class DetectionReplay:
    """Replay cache for deterministic detection messages."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"detection replay path does not exist: {self._path}")
        self._entries: Dict[int, _ReplayEntry] = {}
        self._load_path(self._path)
        if not self._entries:
            raise ValueError(f"no detection replay entries found in {self._path}")
        _LOG.info(
            "loaded %d detection replay frames from %s",
            len(self._entries),
            self._path,
        )

    # Loading helpers --------------------------------------------------
    def _load_path(self, path: Path) -> None:
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_dir():
                    self._load_path(child)
                elif child.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
                    self._load_file(child)
            return
        self._load_file(path)

    def _load_file(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    self._ingest_payload(payload, source=f"{path}:{line_no}")
            return

        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self._ingest_payload(payload, source=str(path))

    def _ingest_payload(self, payload: object, *, source: str) -> None:
        if isinstance(payload, Mapping):
            if "detections" in payload and isinstance(payload["detections"], Sequence):
                for item in payload["detections"]:
                    self._ingest_payload(item, source=source)
                return
            self._store_detection(payload, source=source)
            return

        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            for item in payload:
                self._ingest_payload(item, source=source)
            return

        raise ValueError(f"unsupported detection replay payload in {source}: {type(payload)!r}")

    def _store_detection(self, payload: Mapping[str, object], *, source: str) -> None:
        try:
            msg = DetectionMsg(**payload)
        except Exception as exc:  # pragma: no cover - defensive path
            raise ValueError(f"invalid DetectionMsg in {source}: {exc}") from exc
        frame_id = int(msg.frame_id)
        self._entries[frame_id] = _ReplayEntry(msg=msg)

    # Public API -------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._entries)

    def has_frame(self, frame_id: int) -> bool:
        return frame_id in self._entries

    def materialize(
        self,
        frame_id: int,
        *,
        header: Mapping[str, object],
        clock: MonotonicClock,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[DetectionMsg]:
        entry = self._entries.get(int(frame_id))
        if entry is None:
            return None

        msg = entry.msg.model_copy(deep=True)
        header_frame = int(header.get("frame_id", frame_id))
        header_src = int(header.get("src_ts_ms", msg.src_ts_ms))
        delta = header_src - msg.src_ts_ms

        msg.frame_id = header_frame
        msg.src_ts_ms = header_src

        if msg.rx_ts_ms:
            msg.rx_ts_ms = int(msg.rx_ts_ms + delta)
        else:
            msg.rx_ts_ms = header_src

        if msg.infer_ts_ms:
            msg.infer_ts_ms = int(msg.infer_ts_ms + delta)
        else:
            msg.infer_ts_ms = max(msg.rx_ts_ms, header_src)

        if msg.infer_ts_ms < msg.rx_ts_ms:
            msg.infer_ts_ms = msg.rx_ts_ms

        if image_size is not None:
            width, height = image_size
            if width:
                msg.img_w = int(width)
            if height:
                msg.img_h = int(height)

        return msg


__all__ = ["FrameReplayCapture", "DetectionReplay"]

