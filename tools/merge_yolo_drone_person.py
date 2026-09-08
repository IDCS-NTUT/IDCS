#!/usr/bin/env python3
"""Merge verified one-class YOLO sources into an IDCS drone/person corpus.

The source datasets remain untouched. Images are hard-linked when the source and
destination are on the same filesystem (copied otherwise); labels are always
written afresh so their class IDs can be checked and remapped.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Sequence


SPLITS = ("train", "val")
SOURCES = (
    ("anti_uav", "drone", 0),
    ("openimages_person", "person", 1),
)


@dataclass
class SplitStats:
    images: int = 0
    labels: int = 0
    instances: int = 0
    hardlinks: int = 0
    copies: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "images": self.images,
            "labels": self.labels,
            "instances": self.instances,
            "hardlinks": self.hardlinks,
            "copies": self.copies,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone-root", type=Path, required=True)
    parser.add_argument("--person-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        default=0,
        help="Smoke-test limit per source and split; 0 means all images.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output root. Required before deleting it.",
    )
    return parser.parse_args(argv)


def source_pairs(root: Path, split: str, limit: int) -> list[tuple[Path, Path]]:
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError(f"source lacks images/{split} or labels/{split}: {root}")
    images = sorted(path for path in images_dir.iterdir() if path.is_file())
    if limit:
        images = images[:limit]
    pairs: list[tuple[Path, Path]] = []
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise ValueError(f"missing label for image: {image}")
        pairs.append((image, label))
    return pairs


def remap_label(label_path: Path, target_class: int) -> tuple[str, int]:
    rows: list[str] = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_no} must contain five YOLO fields")
        try:
            source_class = int(fields[0])
            x, y, width, height = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_no} has non-numeric YOLO fields") from exc
        if source_class != 0:
            raise ValueError(f"{label_path}:{line_no} expected source class 0, got {source_class}")
        if not (
            0.0 <= x <= 1.0
            and 0.0 <= y <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
            and x - width / 2 >= -1e-6
            and x + width / 2 <= 1 + 1e-6
            and y - height / 2 >= -1e-6
            and y + height / 2 <= 1 + 1e-6
        ):
            raise ValueError(f"{label_path}:{line_no} has an out-of-bounds box")
        rows.append(f"{target_class} {x:.8f} {y:.8f} {width:.8f} {height:.8f}")
    return ("\n".join(rows) + "\n") if rows else "", len(rows)


def prepare_output(root: Path, overwrite: bool) -> None:
    resolved = root.resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError("refusing to use a filesystem root as --output-root")
    if root.exists():
        if not overwrite:
            raise ValueError(f"output already exists: {root}; use --overwrite to replace it")
        shutil.rmtree(root)
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True, exist_ok=False)
        (root / "labels" / split).mkdir(parents=True, exist_ok=False)


def link_or_copy(source: Path, destination: Path) -> bool:
    try:
        os.link(source, destination)
        return True
    except OSError:
        shutil.copy2(source, destination)
        return False


def write_dataset_yaml(root: Path) -> None:
    (root / "dataset.yaml").write_text(
        "\n".join(
            (
                f"path: {root.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: drone",
                "  1: person",
                "",
            )
        ),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_images_per_split < 0:
        raise ValueError("--max-images-per-split must be non-negative")
    roots = {"anti_uav": args.drone_root.resolve(), "openimages_person": args.person_root.resolve()}
    for name, root in roots.items():
        if not root.is_dir():
            raise ValueError(f"{name} root does not exist: {root}")

    all_pairs: dict[tuple[str, str], list[tuple[Path, Path]]] = {}
    for source_name, _, _ in SOURCES:
        for split in SPLITS:
            all_pairs[(source_name, split)] = source_pairs(
                roots[source_name], split, args.max_images_per_split
            )
    for source_name, _, _ in SOURCES:
        print(
            f"{source_name}: "
            + ", ".join(
                f"{split}={len(all_pairs[(source_name, split)])}" for split in SPLITS
            )
        )
    if args.dry_run:
        return 0

    output_root = args.output_root.resolve()
    prepare_output(output_root, args.overwrite)
    report: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "class_map": {"0": "drone", "1": "person"},
        "sources": {name: str(path) for name, path in roots.items()},
        "splits": {},
    }

    for split in SPLITS:
        split_report: dict[str, dict[str, int]] = {}
        for source_name, _, target_class in SOURCES:
            stats = SplitStats()
            for image, label in all_pairs[(source_name, split)]:
                destination_stem = f"{source_name}_{image.stem}"
                destination_image = output_root / "images" / split / f"{destination_stem}{image.suffix.lower()}"
                destination_label = output_root / "labels" / split / f"{destination_stem}.txt"
                if destination_image.exists() or destination_label.exists():
                    raise RuntimeError(f"destination collision: {destination_stem}")
                label_text, instances = remap_label(label, target_class)
                hardlinked = link_or_copy(image, destination_image)
                destination_label.write_text(label_text, encoding="utf-8")
                stats.images += 1
                stats.labels += 1
                stats.instances += instances
                if hardlinked:
                    stats.hardlinks += 1
                else:
                    stats.copies += 1
            split_report[source_name] = stats.as_dict()
        report["splits"][split] = split_report

    write_dataset_yaml(output_root)
    (output_root / "merge_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote merged dataset: {output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2) from exc
