"""Read-only, bounded actual simulator controller/pose/selection evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import zmq

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=int)
    p.add_argument("--no-selection", action="store_true")
    p.add_argument("--duration-s", type=float, default=5.)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    if not 3 <= a.duration_s <= 20 or (a.target is None) == (not a.no_selection):
        p.error("bounded duration and exactly one target/no-selection required")
    ctx, sockets = zmq.Context(), []
    for port in (5574, 15565, 5572):
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.CONFLATE, 1)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        s.connect(f"tcp://127.0.0.1:{port}")
        sockets.append(s)
    poller = zmq.Poller()
    for s in sockets:
        poller.register(s, zmq.POLLIN)
    counts, tracks, reasons, poses, errors = Counter(), Counter(), Counter(), [], []
    last = {}
    start = time.monotonic()
    try:
        while time.monotonic() - start < a.duration_s:
            events = dict(poller.poll(100))
            for index, s in enumerate(sockets):
                if s not in events:
                    continue
                msg = json.loads(s.recv())
                name = ("truth", "controller", "pose")[index]
                counts[name] += 1
                last[name] = msg
                if index == 0:
                    selected = (msg.get("selection") or {}).get("track_id")
                    if (a.no_selection and selected is not None) or (not a.no_selection and selected != a.target):
                        errors.append(f"truth selected unexpected target {selected}")
                elif index == 1:
                    track = msg.get("track_id")
                    tracks[str(track)] += 1
                    reasons[msg.get("reason")] += 1
                    if (a.no_selection and msg.get("reason") == "tracking") or (not a.no_selection and track != a.target):
                        errors.append(f"controller unexpected track/reason: {track}/{msg.get('reason')}")
                else:
                    poses.append([msg["pan"], msg["tilt"]])
    finally:
        for s in sockets:
            s.close(0)
        ctx.term()
    if min(counts.values(), default=0) < a.duration_s * 10 or len(counts) != 3:
        errors.append("insufficient live samples")
    if not a.no_selection and reasons["tracking"] < a.duration_s * 10:
        errors.append("controller did not actively track")
    result = dict(passed=not errors, counts=dict(counts), controller_tracks=dict(tracks),
        controller_reasons=dict(reasons), first_pose=poses[0] if poses else None,
        last_pose=poses[-1] if poses else None, last=last, errors=sorted(set(errors)))
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "last"}), flush=True)
    return int(bool(errors))

if __name__ == "__main__":
    raise SystemExit(main())
