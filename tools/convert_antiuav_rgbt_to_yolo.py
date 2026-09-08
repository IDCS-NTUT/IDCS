#!/usr/bin/env python3
"""Convert Anti-UAV-RGBT visible videos to a sequence-safe YOLO dataset.

The Anti-UAV-RGBT release stores one annotated target per video frame in
``visible.json``.  It is a tracking corpus, so extracting every frame would
both create a very large dataset and leak near-identical images across splits.
This utility keeps sequences together, samples frames at a defined cadence, and
uses only the visible/RGB modality used by the current camera pipeline.

Example (temporary validation only, until the official validation sequences are
downloaded)::

    python tools/convert_antiuav_rgbt_to_yolo.py \
        --source-root ../train/Anti-UAV-RGBT \
        --output-root ../train/yolo-antiuav-rgb \
        --temporary-val-fraction 0.15 --frame-stride 20

Use the official ``label_new/val.json`` split automatically once the matching
sequence directories are present.  The script otherwise refuses to produce a
dataset unless ``--temporary-val-fraction`` is supplied explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import cv2
except ImportError as exc:  # pragma: no cover - exercised on the target host
    raise SystemExit("OpenCV is required; install opencv-python to run this converter") from exc


LOGGER = logging.getLogger(__name__)
DRONE_CLASS_ID = 0
DRONE_CLASS_NAME = "drone"


@dataclass(frozen=True)
class SequenceSource:
    """A single RGB Anti-UAV sequence and its annotation."""

    name: str
    directory: Path
    tags: tuple[str, ...]

    @property
    def video_path(self) -> Path:
        return self.directory / "visible.mp4"

    @property
    def annotation_path(self) -> Path:
        return self.directory / "visible.json"


@dataclass
class ConversionStats:
    """Counters included in the conversion report."""

    sequences: int = 0
    frames_read: int = 0
    images_written: int = 0
    positive_images: int = 0
    negative_images: int = 0
    clipped_boxes: int = 0
    discarded_boxes: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "sequences": self.sequences,
            "frames_read": self.frames_read,
            "images_written": self.images_written,
            "positive_images": self.positive_images,
            "negative_images": self.negative_images,
            "clipped_boxes": self.clipped_boxes,
            "discarded_boxes": self.discarded_boxes,
        }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Anti-UAV-RGBT root containing train/, test/, and label_new/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination directory for images/, labels/, dataset.yaml, and report.",
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("label_new/train.json"),
        help="Path relative to --source-root unless absolute (default: %(default)s).",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path("label_new/val.json"),
        help="Path relative to --source-root unless absolute (default: %(default)s).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=20,
        help="Keep every Nth target-present frame (default: %(default)s).",
    )
    parser.add_argument(
        "--negative-stride",
        type=int,
        default=60,
        help=(
            "Keep every Nth target-absent frame as a YOLO background image; "
            "set 0 to omit negatives (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--temporary-val-fraction",
        type=float,
        default=0.0,
        help=(
            "Deterministically reserve this fraction of *present train sequences* "
            "as validation only when no official validation sequence is available."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=20260816,
        help="Seed for the temporary sequence-level split (default: %(default)s).",
    )
    parser.add_argument(
        "--max-sequences-per-split",
        type=int,
        default=0,
        help="For smoke tests: process at most N sequences in each split; 0 means all.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="OpenCV JPEG quality from 1 to 100 (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifests and report planned samples without decoding or writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output root. Required to delete prior conversion output.",
    )
    return parser.parse_args(argv)


def _resolve_source_path(source_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else source_root / path


def _load_manifest(path: Path) -> dict[str, tuple[str, ...]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"manifest does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest is not valid JSON: {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"manifest must be a JSON object keyed by sequence: {path}")

    manifest: dict[str, tuple[str, ...]] = {}
    for name, tags in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"manifest has an invalid sequence name: {name!r}")
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"manifest tags for {name!r} must be a list of strings")
        manifest[name] = tuple(tags)
    return manifest


def _discover_sequences(source_root: Path) -> dict[str, Path]:
    """Find complete sequences across supplied split folders without trusting names."""
    discovered: dict[str, Path] = {}
    for annotation_path in source_root.glob("*/*/visible.json"):
        directory = annotation_path.parent
        if not (directory / "visible.mp4").is_file():
            LOGGER.warning("Skipping incomplete sequence without visible.mp4: %s", directory)
            continue
        existing = discovered.get(directory.name)
        if existing is not None and existing != directory:
            raise ValueError(
                f"duplicate sequence name {directory.name!r}: {existing} and {directory}"
            )
        discovered[directory.name] = directory
    if not discovered:
        raise ValueError(f"no complete visible sequences found below {source_root}")
    return discovered


def _materialize_sources(
    manifest: Mapping[str, tuple[str, ...]],
    discovered: Mapping[str, Path],
) -> tuple[list[SequenceSource], list[str]]:
    present: list[SequenceSource] = []
    missing: list[str] = []
    for name in sorted(manifest):
        directory = discovered.get(name)
        if directory is None:
            missing.append(name)
        else:
            present.append(SequenceSource(name=name, directory=directory, tags=manifest[name]))
    return present, missing


def _temporary_split(
    sources: Sequence[SequenceSource], fraction: float, seed: int
) -> tuple[list[SequenceSource], list[SequenceSource]]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("--temporary-val-fraction must be greater than 0 and less than 1")
    if len(sources) < 2:
        raise ValueError("at least two present training sequences are required for a temporary split")

    ranked = sorted(
        sources,
        key=lambda source: hashlib.sha256(
            f"{seed}:{source.name}".encode("utf-8")
        ).hexdigest(),
    )
    val_count = min(len(ranked) - 1, max(1, round(len(ranked) * fraction)))
    validation_names = {source.name for source in ranked[:val_count]}
    train = [source for source in sources if source.name not in validation_names]
    val = [source for source in sources if source.name in validation_names]
    return train, val


def _read_annotation(path: Path) -> tuple[list[int], list[list[float]]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid annotation JSON: {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"annotation must be an object: {path}")
    exist = raw.get("exist")
    boxes = raw.get("gt_rect")
    if not isinstance(exist, list) or not isinstance(boxes, list) or len(exist) != len(boxes):
        raise ValueError(f"annotation requires same-length exist and gt_rect arrays: {path}")

    parsed_exist: list[int] = []
    parsed_boxes: list[list[float]] = []
    for index, (is_present, box) in enumerate(zip(exist, boxes)):
        if is_present not in (0, 1):
            raise ValueError(f"exist[{index}] must be 0 or 1 in {path}")
        valid_box = isinstance(box, list) and len(box) == 4 and all(
            isinstance(value, (int, float)) for value in box
        )
        if is_present and not valid_box:
            raise ValueError(
                f"gt_rect[{index}] must be [x, y, w, h] when exist[{index}] is 1 in {path}"
            )
        if not is_present and not (valid_box or box == []):
            raise ValueError(
                f"gt_rect[{index}] must be empty or [x, y, w, h] when exist[{index}] is 0 in {path}"
            )
        parsed_exist.append(int(is_present))
        parsed_boxes.append([float(value) for value in box] if valid_box else [])
    return parsed_exist, parsed_boxes


def _frame_is_selected(frame_index: int, is_present: int, args: argparse.Namespace) -> bool:
    if is_present:
        return frame_index % args.frame_stride == 0
    return args.negative_stride > 0 and frame_index % args.negative_stride == 0


def _clip_yolo_box(
    box: Sequence[float], image_width: int, image_height: int
) -> tuple[str | None, bool]:
    left, top, width, height = box
    right = left + width
    bottom = top + height
    clipped_left = min(max(left, 0.0), float(image_width))
    clipped_top = min(max(top, 0.0), float(image_height))
    clipped_right = min(max(right, 0.0), float(image_width))
    clipped_bottom = min(max(bottom, 0.0), float(image_height))
    clipped = (clipped_left, clipped_top, clipped_right, clipped_bottom) != (
        left,
        top,
        right,
        bottom,
    )
    clipped_width = clipped_right - clipped_left
    clipped_height = clipped_bottom - clipped_top
    if clipped_width <= 0.0 or clipped_height <= 0.0:
        return None, clipped
    center_x = (clipped_left + clipped_right) / 2.0 / image_width
    center_y = (clipped_top + clipped_bottom) / 2.0 / image_height
    normalized_width = clipped_width / image_width
    normalized_height = clipped_height / image_height
    label = (
        f"{DRONE_CLASS_ID} {center_x:.8f} {center_y:.8f} "
        f"{normalized_width:.8f} {normalized_height:.8f}\n"
    )
    return label, clipped


def _planned_sample_count(source: SequenceSource, args: argparse.Namespace) -> tuple[int, int]:
    exist, _ = _read_annotation(source.annotation_path)
    positives = sum(
        1
        for index, is_present in enumerate(exist)
        if is_present and _frame_is_selected(index, is_present, args)
    )
    negatives = sum(
        1
        for index, is_present in enumerate(exist)
        if not is_present and _frame_is_selected(index, is_present, args)
    )
    return positives, negatives


def _write_sequence(
    source: SequenceSource,
    split: str,
    output_root: Path,
    args: argparse.Namespace,
) -> ConversionStats:
    exist, boxes = _read_annotation(source.annotation_path)
    capture = cv2.VideoCapture(str(source.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {source.video_path}")

    images_dir = output_root / "images" / split
    labels_dir = output_root / "labels" / split
    stats = ConversionStats(sequences=1)
    try:
        for index, (is_present, box) in enumerate(zip(exist, boxes)):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"video ended at frame {index}, but annotation has {len(exist)} frames: "
                    f"{source.video_path}"
                )
            stats.frames_read += 1
            if not _frame_is_selected(index, is_present, args):
                continue

            label_text = ""
            if is_present:
                image_height, image_width = frame.shape[:2]
                label_text, was_clipped = _clip_yolo_box(box, image_width, image_height)
                if was_clipped:
                    stats.clipped_boxes += 1
                if label_text is None:
                    stats.discarded_boxes += 1
                    continue
                stats.positive_images += 1
            else:
                stats.negative_images += 1

            stem = f"{source.name}_{index:06d}"
            image_path = images_dir / f"{stem}.jpg"
            label_path = labels_dir / f"{stem}.txt"
            if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
                raise RuntimeError(f"failed to write image: {image_path}")
            label_path.write_text(label_text, encoding="utf-8")
            stats.images_written += 1
    finally:
        capture.release()
    return stats


def _add_stats(total: ConversionStats, addition: ConversionStats) -> None:
    total.sequences += addition.sequences
    total.frames_read += addition.frames_read
    total.images_written += addition.images_written
    total.positive_images += addition.positive_images
    total.negative_images += addition.negative_images
    total.clipped_boxes += addition.clipped_boxes
    total.discarded_boxes += addition.discarded_boxes


def _limit_sources(sources: Sequence[SequenceSource], limit: int) -> list[SequenceSource]:
    return list(sources[:limit]) if limit else list(sources)


def _prepare_output_root(output_root: Path, overwrite: bool) -> None:
    resolved = output_root.resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError("refusing to use a filesystem root as --output-root")
    if output_root.exists():
        if not overwrite:
            raise ValueError(
                f"output already exists: {output_root}; choose a new directory or pass --overwrite"
            )
        LOGGER.warning("Removing requested prior output: %s", output_root)
        shutil.rmtree(output_root)
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=False)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=False)


def _write_dataset_yaml(output_root: Path) -> None:
    yaml_text = (
        f"path: {output_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  {DRONE_CLASS_ID}: {DRONE_CLASS_NAME}\n"
    )
    (output_root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


def _report_plan(split_sources: Mapping[str, Iterable[SequenceSource]], args: argparse.Namespace) -> None:
    for split, sources in split_sources.items():
        source_list = list(sources)
        positives = 0
        negatives = 0
        for source in source_list:
            planned_positives, planned_negatives = _planned_sample_count(source, args)
            positives += planned_positives
            negatives += planned_negatives
        LOGGER.info(
            "%s: %d sequences; planned samples: %d positive + %d negative = %d",
            split,
            len(source_list),
            positives,
            negatives,
            positives + negatives,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")
    if args.negative_stride < 0:
        raise ValueError("--negative-stride must be non-negative")
    if args.max_sequences_per_split < 0:
        raise ValueError("--max-sequences-per-split must be non-negative")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")

    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        raise ValueError(f"source root does not exist: {source_root}")
    train_manifest_path = _resolve_source_path(source_root, args.train_manifest)
    val_manifest_path = _resolve_source_path(source_root, args.val_manifest)
    train_manifest = _load_manifest(train_manifest_path)
    val_manifest = _load_manifest(val_manifest_path)
    discovered = _discover_sequences(source_root)

    train_sources, missing_train = _materialize_sources(train_manifest, discovered)
    val_sources, missing_val = _materialize_sources(val_manifest, discovered)
    LOGGER.info(
        "Found %d complete sequences; manifests resolve to %d/%d train and %d/%d validation sequences.",
        len(discovered),
        len(train_sources),
        len(train_manifest),
        len(val_sources),
        len(val_manifest),
    )
    if missing_train:
        LOGGER.warning("Missing %d declared training sequences", len(missing_train))
    if missing_val:
        LOGGER.warning("Missing %d declared validation sequences", len(missing_val))

    split_kind = "official"
    if not val_sources:
        train_sources, val_sources = _temporary_split(
            train_sources, args.temporary_val_fraction, args.split_seed
        )
        split_kind = "temporary_sequence_holdout"
        LOGGER.warning(
            "Using a temporary %d/%d sequence validation holdout; do not report it as the official validation split.",
            len(val_sources),
            len(train_sources) + len(val_sources),
        )
    elif args.temporary_val_fraction:
        LOGGER.warning("Ignoring --temporary-val-fraction because official validation sequences are available")

    split_sources = {
        "train": _limit_sources(train_sources, args.max_sequences_per_split),
        "val": _limit_sources(val_sources, args.max_sequences_per_split),
    }
    _report_plan(split_sources, args)
    if args.dry_run:
        return 0

    output_root = args.output_root.resolve()
    _prepare_output_root(output_root, args.overwrite)
    conversion_stats: dict[str, dict[str, int]] = {}
    for split, sources in split_sources.items():
        total = ConversionStats()
        for source_index, source in enumerate(sources, start=1):
            LOGGER.info("[%s %d/%d] %s", split, source_index, len(sources), source.name)
            _add_stats(total, _write_sequence(source, split, output_root, args))
        conversion_stats[split] = total.as_dict()

    _write_dataset_yaml(output_root)
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_root": str(source_root),
        "class_map": {str(DRONE_CLASS_ID): DRONE_CLASS_NAME},
        "split_kind": split_kind,
        "sampling": {
            "frame_stride": args.frame_stride,
            "negative_stride": args.negative_stride,
            "jpeg_quality": args.jpeg_quality,
        },
        "manifest_sequences": {"train": len(train_manifest), "val": len(val_manifest)},
        "available_sequences": {"train": len(train_sources), "val": len(val_sources)},
        "missing_sequences": {"train": missing_train, "val": missing_val},
        "written": conversion_stats,
    }
    (output_root / "conversion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info("Wrote YOLO dataset to %s", output_root)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc
