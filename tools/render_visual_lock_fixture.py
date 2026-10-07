"""Record real OpenGL simulator pixels for an isolated tracker replay.

This does not run a streamer/controller or change active services/configuration.
Ground-truth rectangles are a separate scoring artifact, not tracker feedback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import yaml

from pc.sim_camera import SimCamera


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--frames",type=int,default=180)
    ap.add_argument("--scene",default="configs/sim/v2_scene.yaml")
    ap.add_argument("--provenance",default="unspecified")
    args = ap.parse_args()
    if not 3 <= args.frames <= 600: ap.error("frames must be 3..600")
    config_path = Path(args.scene)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["sim"]
    args.output.mkdir(parents=True,exist_ok=False)
    camera = SimCamera(1280,720,renderer_name="opengl",renderer_opts=config["renderer_opts"],
                       scene=config["scene"],camera=config["camera"],fps_hz=60.)
    writer = cv2.VideoWriter(str(args.output/"pixels.avi"),cv2.VideoWriter_fourcc(*"FFV1"),60.,(1280,720))
    if not writer.isOpened(): raise RuntimeError("lossless FFV1 writer unavailable")
    rows = []
    pixels = hashlib.sha256()
    try:
        for i in range(args.frames):
            ok,image = camera.next_frame()
            if not ok or camera._renderer._gl is None:
                raise RuntimeError("OpenGL rendering unavailable; no CPU fallback accepted")
            writer.write(image)
            pixels.update(image.tobytes())
            truth = camera.build_ground_truth_snapshot(i,1_000_000_000+round(i*1e9/60))
            rows.append({"index":i,"source_time_ns":truth.frame.source_time_ns,
                         "truth":[t.box.model_dump() for t in truth.tracks]})
            if i in (0,60,120): cv2.imwrite(str(args.output/f"pixels-{i:03}.png"),image)
    finally:
        writer.release()
    (args.output/"truth.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
    info = {"source":args.provenance,"scene_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "pixels_sha256":pixels.hexdigest(),"frames":args.frames,"fps":60,"resolution":[1280,720],
            "gl":dict(camera._renderer._gl.info),"target_render_mode":config["renderer_opts"]["target_render_mode"],
            "plant":"ideal, stationary camera; no motion controller"}
    (args.output/"provenance.json").write_text(json.dumps(info,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(info,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
