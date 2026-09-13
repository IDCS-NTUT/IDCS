"""Generate a deterministic rendered-simulator detector qualification replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from pc.sim_camera import SimCamera


CLASS_CASES = {
    "person": {
        "depths": (-6.0, -10.0, -14.0),
        "lateral": (-2.0, 0.0, 2.0),
        "target": {"sprite": "person", "height": 1.7, "ground_y": 0.0},
    },
    "drone": {
        "depths": (-2.5, -5.0, -8.0),
        "lateral": (-1.2, 0.0, 1.2),
        "target": {"sprite": "drone", "width": 0.7, "ground_y": 2.0},
    },
}


def _background(index: int) -> list[dict[str, Any]]:
    variants = (
        [],
        [{"base_centre": [5.0, -12.0], "footprint": [5.0, 4.0], "height": 9.0,
          "color": [175, 185, 205]}],
        [{"base_centre": [-5.0, -9.0], "footprint": [4.0, 3.0], "height": 6.0,
          "color": [150, 170, 180]}],
    )
    return variants[index % len(variants)]


def _render(width: int, height: int, target: dict[str, Any] | None,
            buildings: list[dict[str, Any]]) -> np.ndarray:
    scene = {
        "mode": "static_targets",
        "targets": [] if target is None else [target],
        "buildings": buildings,
        "cubes": [],
    }
    camera = SimCamera(
        width=width,
        height=height,
        renderer_name="cpu",
        renderer_opts={"draw_ground_grid": False},
        debug=False,
        scene=scene,
        fps_hz=60.0,
    )
    ok, frame = camera.next_frame()
    if not ok:
        raise RuntimeError("simulator failed to render validation frame")
    return frame.copy()


def _target_box(frame: np.ndarray, blank: np.ndarray) -> tuple[list[float], int]:
    delta = np.max(cv2.absdiff(frame, blank), axis=2)
    ys, xs = np.nonzero(delta > 8)
    if len(xs) < 64:
        raise RuntimeError("validation target is not visibly rendered")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    height, width = frame.shape[:2]
    return [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height], int(len(xs))


def generate(output: Path, manifest_path: Path, *, width: int, height: int,
             fps: int, case_frames: int, blank_frames: int) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open validation video writer: {output}")
    cases: list[dict[str, Any]] = []
    blanks: list[dict[str, Any]] = []
    frame_id = 0
    try:
        for class_name, spec in CLASS_CASES.items():
            for depth_index, depth in enumerate(spec["depths"]):
                for position_index, lateral in enumerate(spec["lateral"]):
                    background_id = (depth_index + position_index) % 3
                    buildings = _background(background_id)
                    blank = _render(width, height, None, buildings)
                    blank_start_frame = frame_id + 1
                    for _ in range(blank_frames):
                        writer.write(blank)
                        frame_id += 1
                    blanks.append({
                        "blank_id": f"blank-{class_name}-d{depth_index}-x{position_index}-b{background_id}",
                        "start_frame": blank_start_frame,
                        "end_frame": frame_id,
                        "background_id": background_id,
                    })
                    target = dict(spec["target"])
                    target["ground"] = [float(lateral), float(depth)]
                    rendered = _render(width, height, target, buildings)
                    box, visible_pixels = _target_box(rendered, blank)
                    start_frame = frame_id + 1
                    for _ in range(case_frames):
                        writer.write(rendered)
                        frame_id += 1
                    cases.append({
                        "case_id": f"{class_name}-d{depth_index}-x{position_index}-b{background_id}",
                        "expected_class": class_name,
                        "start_frame": start_frame,
                        "end_frame": frame_id,
                        "expected_box": box,
                        "visible_pixels": visible_pixels,
                        "depth_m": abs(float(depth)),
                        "lateral_m": float(lateral),
                        "background_id": background_id,
                    })
    finally:
        writer.release()
    manifest = {
        "schema": "idcs.detector_sim_sweep",
        "version": 1,
        "video": {"path": str(output), "width": width, "height": height, "fps": fps,
                  "frames": frame_id},
        "case_frames": case_frames,
        "blank_frames": blank_frames,
        "blanks": blanks,
        "cases": cases,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--case-frames", type=int, default=45)
    parser.add_argument("--blank-frames", type=int, default=30)
    args = parser.parse_args(argv)
    if min(args.width, args.height, args.fps, args.case_frames, args.blank_frames) <= 0:
        parser.error("dimensions, rates, and frame counts must be positive")
    manifest = generate(
        args.output, args.manifest, width=args.width, height=args.height, fps=args.fps,
        case_frames=args.case_frames, blank_frames=args.blank_frames,
    )
    print(json.dumps({"output": str(args.output), "manifest": str(args.manifest),
                      "frames": manifest["video"]["frames"], "cases": len(manifest["cases"])},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
