"""Bounded GPU pixel replay. No sockets, services, motor or truth-driven tracking.

Run as a module: python -m tools.verify_visual_lock --library ... --output ...
OpenCV only decodes/renders/annotates pixels. DCF, Harris and LK run in VPI CUDA.
Truth is used once for a simulated operator seed, then only by the scorer.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import socket
import time
import uuid

import cv2
import numpy as np

from common.perception import NormalizedBoxV2, PerceptionFrameV2
from jetson.visual_lock.backend import VPIBackend
from jetson.visual_lock.egl_lifetime import StandaloneEGL
from jetson.visual_lock.session import VisualLockSession
from jetson.visual_lock.gate_config import load_gates


def synthetic(name: str, width=640, height=360, count=120):
    rng = np.random.default_rng(736)
    # Low-frequency, strong texture so LK has stable support at pyramid levels.
    texture = cv2.resize(rng.integers(30,240,(18,24,3),dtype=np.uint8),(96,72),interpolation=cv2.INTER_NEAREST)
    other = cv2.resize(rng.integers(30,240,(18,24,3),dtype=np.uint8),(96,72),interpolation=cv2.INTER_NEAREST)
    for i in range(count):
        image = np.full((height,width,3),32,np.uint8)
        scale = 1 + .0015*i if name == "scale" else 1
        w,h = round(96*scale),round(72*scale)
        x,y = round(width*.23+i*.8),round(height*.3+12*np.sin(i*.04))
        truth = (x,y,w,h)
        if name == "distractor":
            image[y:y+72,width-x-96:width-x] = other
        absent = name in {"occlusion","loss"} and i >= 55 and (name == "loss" or i < 67)
        if not absent:
            patch = np.full((h,w,3),140,np.uint8) if name == "textureless" else cv2.resize(texture,(w,h))
            image[y:y+h,x:x+w] = patch
        yield image, None if absent else truth


def replay(video: str, truth_path: str | None):
    labels = {}
    if truth_path:
        for line in Path(truth_path).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            labels[r["index"]] = r.get("truth", [])
    cap = cv2.VideoCapture(video)
    if not cap.isOpened(): raise ValueError("cannot decode replay video")
    try:
        i = 0
        while True:
            ok,image = cap.read()
            if not ok: break
            entries = labels.get(i, [])
            t = entries[0] if entries else None
            truth = (t["x"]*image.shape[1],t["y"]*image.shape[0],
                     t["w"]*image.shape[1],t["h"]*image.shape[0]) if t else None
            yield image,truth
            i += 1
    finally:
        cap.release()


def preload_frames(frames,*,max_bytes=512*1024*1024):
    """Bounded test input cache; keeps file decode/render outside paced delivery."""
    prepared,used = [],0
    for i,(image,truth) in enumerate(frames):
        if i >= 600: break
        used += image.nbytes
        if used > max_bytes: raise ValueError("preload exceeds 512 MiB test-input budget")
        prepared.append((image,truth))
    return prepared


def evaluate(name, frames, library, output: Path, explicit_seed=None, fps=60., *,
             backend_factory=None, pace=False, gates=None):
    session_tracker = VisualLockSession(backend_factory or (lambda w,h: VPIBackend(library,w,h)),
                                        gates=load_gates() if gates is None else gates)
    lock = session_tracker.lock
    session = str(uuid.uuid4())
    rows, errors, elapsed, psr_values = [], [], [], []
    pixel_digest = hashlib.sha256()
    saved = 0
    previews = []
    snapshots = []
    actual_seed = None
    work_ms,deadline_lateness_ms = [],[]
    schedule = None
    prehashed = isinstance(frames,list)
    if prehashed:
        for bgr,_ in frames: pixel_digest.update(bgr.tobytes())
    try:
        for i,(bgr,truth) in enumerate(frames):
            if i >= 600: break
            if i == 0:
                # Acquisition is measured separately; steady-state cadence starts
                # after resource allocation/seed, not with accumulated seed debt.
                deadline = time.monotonic()
            elif schedule is not None:
                deadline = schedule+(i-1)/fps
                wait = deadline-time.monotonic()
                if pace and wait > 0: time.sleep(wait)
            else:
                schedule = time.monotonic()
                deadline = schedule
            deadline_lateness_ms.append(max(0.,time.monotonic()-deadline)*1000.)
            h,w = bgr.shape[:2]
            if not prehashed: pixel_digest.update(bgr.tobytes())
            frame = PerceptionFrameV2(tracker_session=session,frame_id=i,
                source_time_ns=1_000_000_000+round(i*1e9/fps),source_clock_domain="fixture_monotonic",
                source_identity_verified=True, observed_time_ns=time.monotonic_ns(),
                observation_clock_domain="jetson_monotonic",width=w,height=h)
            work_start = time.perf_counter_ns()
            rgba = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGBA)
            start = time.perf_counter_ns()
            if i == 0:
                seed = explicit_seed or truth
                if seed is None: raise ValueError("first frame needs explicit operator seed")
                actual_seed = tuple(seed)
                box = NormalizedBoxV2(x=seed[0]/w,y=seed[1]/h,w=seed[2]/w,h=seed[3]/h)
                result = session_tracker.seed(frame,rgba,box)
            else:
                result = session_tracker.update(frame,rgba)
            decision, p = result.decision, result.proposal
            total_ms = (time.perf_counter_ns()-start)/1e6
            work_ms.append((time.perf_counter_ns()-work_start)/1e6)
            if i and lock.state != "lost": elapsed.append(total_ms)
            # Contract output is tested even for absent/uncertain observations.
            snapshot = result.snapshot
            snapshots.append(snapshot.model_dump_json(exclude_none=True))
            centre_error = None
            if snapshot.selection and truth:
                x,y,bw,bh = decision.box
                tx,ty,tw,th = truth
                centre_error = float(np.hypot(x+bw/2-tx-tw/2,y+bh/2-ty-th/2))
                errors.append(centre_error)
            rows.append({"i":i,"state":decision.state,"reason":decision.reason,
                         "box":decision.box,"inliers":decision.inliers,"scale":decision.scale,
                         "disagreement_px":decision.disagreement_px,"psr":float(p.psr) if p else None,
                         "peak":float(p.peak) if p else None,
                         "native_ms":p.native_ms if p else None,"total_ms":total_ms,
                         "selected":snapshot.selection is not None,"truth":truth,"error_px":centre_error})
            if i and decision.accepted: psr_values.append(p.psr)
            if saved < 5 and (i in (0,30,54,58,80)):
                previews.append((i,bgr.copy(),decision.box,decision.state))
                saved += 1
    finally:
        final_state = lock.state
        session_tracker.close()
    for i,image,box,state in previews:
        if box:
            x,y,bw,bh = box
            cv2.rectangle(image,(round(x),round(y)),(round(x+bw),round(y+bh)),(0,255,0),1)
        cv2.putText(image,state,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
        cv2.imwrite(str(output/f"{name}-{i:03}.png"),image)
    if not rows: raise ValueError("empty replay")
    visible = [r for r in rows[3:] if r["truth"]]
    tracking = [r for r in visible if r["selected"]]
    false_tracking = sum(r["selected"] for r in rows if r["truth"] is None)
    percentile = lambda values,p: float(np.percentile(values,p)) if values else None
    report = {"name":name,"frames":len(rows),"visible_frames_after_acquisition":len(visible),
        "seed_box_px":actual_seed,"fps":fps,"resolution":[w,h],
        "tracking_coverage":len(tracking)/len(visible) if visible else 0,
        "false_tracking_absent":false_tracking,"final_state":final_state,
        "centre_error_px_median":percentile(errors,50),"centre_error_px_p95":percentile(errors,95),
        "processing_ms_median":percentile(elapsed,50),"processing_ms_p95":percentile(elapsed,95),
        "frame_work_ms_p95":percentile(work_ms[3:],95),
        "frame_work_over_period":sum(v > 1000./fps for v in work_ms[3:]),
        "paced":pace,"start_lateness_ms_p95":percentile(deadline_lateness_ms[3:],95) if pace else None,
        "accepted_psr_min":min(psr_values) if psr_values else None,
        "pixels_sha256":pixel_digest.hexdigest(),"gates":asdict(lock.gates)}
    report["gates_sha256"] = hashlib.sha256(json.dumps(report["gates"],sort_keys=True).encode()).hexdigest()
    (output/f"{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
    (output/f"{name}-perception-v2.jsonl").write_text("\n".join(snapshots)+"\n",encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--video")
    ap.add_argument("--truth")
    ap.add_argument("--gates-json",help="explicit Gates overrides; resolved values/digest recorded in report")
    ap.add_argument("--seed-xywh",type=float,nargs=4)
    ap.add_argument("--fps",type=float,default=60.)
    ap.add_argument("--provenance",default="unspecified")
    ap.add_argument("--input-memory",choices=("host","cuda","egl"),default="host")
    ap.add_argument("--surface-library",help="test-only NVMM upload library, needed for EGL replay")
    ap.add_argument("--standalone-egl",action="store_true",help="own EGL in this fresh process; never embed in a live owner")
    ap.add_argument("--pace",action="store_true",help="pace source delivery at --fps (no network/display qualification)")
    ap.add_argument("--preload",action="store_true",help="bounded input cache before pacing; separates codec/render overhead")
    ap.add_argument("--opencv-threads",type=int,default=1,help="replay decode/colour-conversion threads, not tracking")
    args = ap.parse_args()
    try:
        gates = load_gates(args.gates_json)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))
    if not np.isfinite(args.fps) or args.fps <= 0: ap.error("invalid fps")
    if not 1 <= args.opencv_threads <= 8: ap.error("opencv threads must be 1..8")
    cv2.setNumThreads(args.opencv_threads)
    if args.input_memory != "host" and not args.standalone_egl:
        ap.error("GPU replay requires --standalone-egl")
    if args.input_memory == "egl" and not args.surface_library:
        ap.error("EGL replay requires --surface-library")
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    reports = []
    with StandaloneEGL() if args.standalone_egl else nullcontext() as egl:
        factory = None
        if args.input_memory != "host":
            from tools.visual_lock_replay_gpu import ReplayGPUBackend
            factory = lambda w,h: ReplayGPUBackend(args.library,w,h,kind=args.input_memory,
                egl_owner=egl,surface_library=args.surface_library)
        kwargs = dict(backend_factory=factory,pace=args.pace,gates=gates)
        prepare = preload_frames if args.preload else lambda frames:frames
        if args.video:
            reports.append(evaluate("replay",prepare(replay(args.video,args.truth)),args.library,output,args.seed_xywh,args.fps,**kwargs))
        else:
            for name in ("translation","scale","distractor","occlusion","loss","textureless"):
                reports.append(evaluate(name,prepare(synthetic(name)),args.library,output,fps=args.fps,**kwargs))
    provenance = {"host":socket.gethostname(),"platform":platform.platform(),"source":args.provenance,
        "numpy_version":np.__version__,"opencv_version":cv2.__version__,"library_abi":4,
        "detector_model":None,"plant_model":None,
        "random_seed":736,"backend":"NVIDIA VPI CUDA DCF + Harris + forward/backward PyrLK",
        "library_sha256":hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
        "input_video_sha256":hashlib.sha256(Path(args.video).read_bytes()).hexdigest() if args.video else None,
        "tracking_inputs":"pixels and initial seed only; truth used only for scoring thereafter",
        "input_memory":args.input_memory,"egl_bootstrap":args.standalone_egl,
        "input_preloaded":args.preload,"opencv_threads":cv2.getNumThreads(),
        "egl_vendor":egl.vendor if egl else None,"egl_version":egl.version if egl else None,
        "test_surface_library_sha256":hashlib.sha256(Path(args.surface_library).read_bytes()).hexdigest() if args.surface_library else None,
        "timing_scope":"processing includes replay upload + tracking/fusion/commit/V2; frame_work also includes colour conversion, not provenance hashing/decoding/rendering/annotation/network/display; preloaded input hash and previews are outside paced delivery",
        "report":reports}
    if not args.video:
        good = all(r["tracking_coverage"] >= .95 and r["centre_error_px_p95"] <= 3.
                   for r in reports[:3])
        safe = all(r["false_tracking_absent"] == 0 and r["final_state"] == "lost" for r in reports[3:])
        provenance["acceptance"] = "pass" if good and safe else "fail"
    else:
        provenance["acceptance"] = "measurement only; representative real data still required"
    (output/"report.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(provenance,indent=2))
    return 1 if provenance["acceptance"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
