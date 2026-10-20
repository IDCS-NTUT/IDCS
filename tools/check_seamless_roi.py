"""Bounded software-SIM ROI seed/release check; no panel, service or motor APIs.

Run on the Jetson after the reviewer establishes one normal Simulator owner.
The supplied pixel box must cover texture on the displayed SIM object. This
checks transport/selection, not detector quality or physical control.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import zmq

from common.perception import perception_snapshot_from_json
from common.visual_roi_seed import VisualRoiSeed, VisualRoiRelease, VisualRoiSelect


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", default="tcp://127.0.0.1:5595")
    parser.add_argument("--seed", default="tcp://127.0.0.1:5596")
    parser.add_argument("--perception", default="tcp://127.0.0.1:5564")
    parser.add_argument("--box", nargs=4, type=int, required=True, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--duration-s", type=float, default=5.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.duration_s <= 30:
        parser.error("duration must be 1..30 seconds")
    if args.output.exists():
        parser.error("choose a new evidence report path")
    report = dict(started_wall_s=time.time(), box=args.box, selected_frames=0,
                  owned_frames=0, statuses=[], accepted=False, detector_returned=False)
    request = None
    selecting = None
    select_sent_s = 0.0
    context = zmq.Context()
    identity = context.socket(zmq.SUB)
    perception = context.socket(zmq.SUB)
    push = context.socket(zmq.PUSH)
    for sock, endpoint in ((identity, args.identity), (perception, args.perception)):
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.RCVHWM, 128)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(endpoint)
    push.setsockopt(zmq.LINGER, 0)
    push.setsockopt(zmq.SNDHWM, 4)
    push.setsockopt(zmq.IMMEDIATE, 1)
    push.connect(args.seed)
    poller = zmq.Poller()
    poller.register(identity, zmq.POLLIN)
    poller.register(perception, zmq.POLLIN)
    status = latest_frame = None
    started = time.monotonic()
    release_s = None
    release_required = False
    error = None
    try:
        while time.monotonic()-started < 15 + args.duration_s:
            now = time.monotonic()
            events = dict(poller.poll(20))
            if identity in events:
                for _ in range(64):
                    try:
                        message = identity.recv_json(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    if message.get("type") == "visual_roi_status_v1":
                        # Refuse camera, managed fallback and isolated trials.
                        if (message.get("source_mode") != "sim" or message.get("managed_mode") != "sim"
                                or message.get("roi_capable") is not True):
                            raise RuntimeError("normal Simulator ROI capability required")
                        status = (message, now)
                        if (request is not None and release_s is None
                                and message.get("terminal_request_id") == request.request_id):
                            raise RuntimeError(f"ROI ended before release: {message.get('reason')}")
                        report["statuses"].append(message)
                        report["statuses"] = report["statuses"][-400:]
                    elif "rtp_ssrc" in message:
                        latest_frame = (message, now)
                    elif request is not None and message.get("request_id") == request.request_id:
                        if message.get("type") == "visual_roi_result_v1":
                            report["accepted"] |= message.get("accepted") is True
                            if message.get("accepted") is False:
                                report["terminal_reason"] = message.get("reason")
                                if release_s is None:
                                    raise RuntimeError(f"ROI ended before release: {message.get('reason')}")
            if perception in events:
                snapshot = perception_snapshot_from_json(perception.recv())
                if request is not None and snapshot.frame.tracker_session == request.tracker_session:
                    state = snapshot.roi_state
                    if (release_s is None and report["owned_frames"] and state is not None
                            and not state.owns_selection):
                        raise RuntimeError("ROI returned to detector before explicit release")
                    report["owned_frames"] += int(state is not None and state.owns_selection)
                    report["selected_frames"] += int(snapshot.selection is not None
                        and snapshot.selection.policy == "operator_visual_lock")
                    if release_s is not None and state is not None and not state.owns_selection:
                        report["detector_returned"] = True
                        break
            if request is None and status is not None:
                source, source_s = status
                if selecting is None and source.get("source_ready") is True:
                    selecting = VisualRoiSelect(str(uuid.uuid4()), source["tracker_session"], True)
                if selecting is not None and now-select_sent_s >= .1:
                    push.send_json(selecting.as_message(), flags=zmq.NOBLOCK)
                    select_sent_s = now
                current, seen_s = latest_frame if latest_frame is not None else ({}, 0.)
                if (now-source_s <= .25 and now-seen_s <= .1 and source.get("source_ready") is True
                        and source.get("roi_cache_enabled") is True and source.get("roi_cache_frames", 0) >= 3
                        and current.get("tracker_session") == source.get("tracker_session")):
                    x, y, w, h = args.box
                    request = VisualRoiSeed.from_mapping(dict(type="visual_roi_seed_v1",
                        request_id=str(uuid.uuid4()), tracker_session=current["tracker_session"],
                        frame_id=current["frame_id"], source_time_ns=current["source_time_ns"],
                        source_clock_domain=current["source_clock_domain"],
                        width=1280, height=720, x=x, y=y, w=w, h=h))
                    push.send_json(request.as_message(), flags=zmq.NOBLOCK)
                    release_required = True
                    report["request"] = request.as_message()
                    report["seed_at_s"] = now
            if request is not None and release_s is None and now-report["seed_at_s"] >= args.duration_s:
                push.send_json(VisualRoiRelease(request.request_id, request.tracker_session).as_message(), flags=zmq.NOBLOCK)
                release_s = now
                release_required = False
        if not (report["accepted"] and report["selected_frames"] and report["detector_returned"]):
            raise RuntimeError("seed/tracking/release contract not completed; inspect report and texture")
    except (RuntimeError, ValueError, zmq.ZMQError) as exc:
        error = str(exc)
    finally:
        if selecting is not None:
            try:
                push.send_json(VisualRoiSelect(selecting.selection_id, selecting.tracker_session, False).as_message(),
                               flags=zmq.NOBLOCK)
            except zmq.Again:
                report["selection_cancel_delivery_failed"] = True
        if release_required and request is not None:
            try:
                push.send_json(VisualRoiRelease(request.request_id, request.tracker_session).as_message(), flags=zmq.NOBLOCK)
            except zmq.Again:
                report["release_delivery_failed"] = True
        report["error"] = error
        report["elapsed_s"] = time.monotonic()-started
        for sock in (identity, perception, push):
            sock.close(0)
        context.term()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return int(error is not None)


if __name__ == "__main__":
    raise SystemExit(run())
