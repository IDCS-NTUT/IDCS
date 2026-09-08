#!/usr/bin/env python3
"""Train one YOLO epoch and save a checkpoint without final validation.

This is intentionally only for architecture/input-throughput smoke checkpoints.
Ultralytics validates at the final epoch even with ``val=False``; on the target
Jetson that validation path has been unstable.  The wrapper bypasses validation
after the training epoch while leaving the normal optimizer, EMA, and save
path intact.  It must not be used for model-quality evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def bypass_final_validation(trainer) -> None:
    """Replace final-epoch validation with neutral metrics for this smoke run."""

    trainer.validate = lambda: ({}, 0.0)
    trainer.final_eval = lambda: None


def main() -> int:
    args = parse_args()
    model = YOLO(args.model)
    model.add_callback("on_train_start", bypass_final_validation)
    model.train(
        data=str(args.data),
        epochs=1,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=False,
        val=False,
        plots=False,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        save=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
