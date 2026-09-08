#!/usr/bin/env python3
"""Build a deterministic, cache-free class-balanced YOLO subset using hardlinks.

The source dataset is never changed.  Every validation image is retained; for
training, all drone-containing and negative images are retained and a seeded
subset of person-only images is selected to meet a requested person:drone
instance ratio.  Images and labels are hard-linked, so no image pixels or
Ultralytics cache files are duplicated.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val")


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path
    counts: Counter[int]

    @property
    def instances(self) -> int:
        return sum(self.counts.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--person-per-drone",
        type=float,
        default=1.5,
        help="Target maximum person instances per drone instance in train (default: 1.5).",
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_sample(image: Path, labels_dir: Path) -> Sample:
    label = labels_dir / f"{image.stem}.txt"
    if not label.is_file():
        raise ValueError(f"missing label for image: {image}")
    counts: Counter[int] = Counter()
    for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label}:{line_number}: expected five YOLO fields")
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"{label}:{line_number}: invalid YOLO row") from exc
        if class_id not in (0, 1):
            raise ValueError(f"{label}:{line_number}: unexpected class {class_id}")
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"{label}:{line_number}: normalized coordinates out of range")
        counts[class_id] += 1
    return Sample(image=image, label=label, counts=counts)


def split_samples(root: Path, split: str) -> list[Sample]:
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError(f"source lacks images/{split} or labels/{split}: {root}")
    return [
        read_sample(image, labels_dir)
        for image in sorted(images_dir.iterdir())
        if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES
    ]


def choose_train_samples(samples: Iterable[Sample], ratio: float, seed: int) -> tuple[list[Sample], dict[str, int]]:
    samples = list(samples)
    drones = sum(sample.counts[0] for sample in samples)
    if drones == 0:
        raise ValueError("training split has no drone instances")
    target_people = int(drones * ratio)
    base = [sample for sample in samples if sample.counts[0] or not sample.counts[1]]
    person_only = [sample for sample in samples if sample.counts[1] and not sample.counts[0]]
    random.Random(seed).shuffle(person_only)

    selected = list(base)
    people = sum(sample.counts[1] for sample in base)
    for sample in person_only:
        count = sample.counts[1]
        if people + count <= target_people:
            selected.append(sample)
            people += count

    # A one-person image almost always permits an exact target; if it does not,
    # keep the closest additional image rather than silently returning a much
    # more imbalanced subset.
    if people < target_people:
        remaining = [sample for sample in person_only if sample not in selected]
        if remaining:
            closest = min(remaining, key=lambda sample: abs((people + sample.counts[1]) - target_people))
            selected.append(closest)
            people += closest.counts[1]

    selected.sort(key=lambda sample: sample.image.name)
    return selected, {
        "drone_instances": drones,
        "target_person_instances": target_people,
        "selected_person_instances": people,
        "person_only_candidates": len(person_only),
        "person_only_selected": sum(sample.counts[1] > 0 and sample.counts[0] == 0 for sample in selected),
    }


def count_instances(samples: Iterable[Sample]) -> dict[str, int]:
    samples = list(samples)
    return {
        "images": len(samples),
        "drone_instances": sum(sample.counts[0] for sample in samples),
        "person_instances": sum(sample.counts[1] for sample in samples),
        "empty_images": sum(not sample.instances for sample in samples),
    }


def materialize(samples: Iterable[Sample], output_root: Path, split: str) -> None:
    for sample in samples:
        target_image = output_root / "images" / split / sample.image.name
        target_label = output_root / "labels" / split / sample.label.name
        os.link(sample.image, target_image)
        os.link(sample.label, target_label)


def main() -> int:
    args = parse_args()
    if not 0 < args.person_per_drone:
        raise ValueError("--person-per-drone must be positive")
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    if not source.is_dir():
        raise ValueError(f"source root does not exist: {source}")
    if output.exists():
        raise ValueError(f"output already exists; refusing to replace it: {output}")

    original = {split: split_samples(source, split) for split in SPLITS}
    selected_train, selection = choose_train_samples(original["train"], args.person_per_drone, args.seed)
    selected = {"train": selected_train, "val": original["val"]}
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_root": str(source),
        "selection": {"person_per_drone": args.person_per_drone, "seed": args.seed, **selection},
        "original": {split: count_instances(samples) for split, samples in original.items()},
        "result": {split: count_instances(samples) for split, samples in selected.items()},
        "storage": "hardlinked images and labels; no cache files included",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
        materialize(selected[split], output, split)
    (output / "dataset.yaml").write_text(
        f"path: {output}\ntrain: images/train\nval: images/val\nnames:\n  0: drone\n  1: person\n",
        encoding="utf-8",
    )
    (output / "balance_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}")
        raise SystemExit(2) from exc
