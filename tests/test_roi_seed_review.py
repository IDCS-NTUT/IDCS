import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import review_roi_seeds as review
from tools.run_visual_lock_trial_guard import video_owners


def test_manifest_requires_six_local_inputs_explicit_boxes_and_unique_names(tmp_path):
    video = tmp_path/"clip.mp4"
    video.write_bytes(b"fixture")
    entries = [dict(name=str(i), video="clip.mp4", seed_xywh=[1,2,12,12]) for i in range(6)]
    manifest = tmp_path/"clips.json"
    manifest.write_text(json.dumps(entries))
    clips = review.load_clips(manifest)
    assert len(clips) == 6 and Path(clips[0]["video"]) == video
    for invalid in (entries[:5], [dict(e, name="duplicate") for e in entries],
                    [dict(e, name="../escape") for e in entries],
                    [dict(e, seed_xywh=[]) for e in entries]):
        manifest.write_text(json.dumps(invalid))
        with pytest.raises(ValueError): review.load_clips(manifest)


def test_comparison_handles_old_reports_and_does_not_score_unlabeled_video(tmp_path):
    report = dict(name="replay", tracking_coverage=.5, centre_error_px_median=1.,
                  centre_error_px_p95=2., false_tracking_absent=7, seed_box_px=[1,2,12,12])
    (tmp_path/"report.json").write_text(json.dumps(dict(report=[report])))
    rows = [dict(i=0, state="acquiring", box=[1,2,12,12]),
            dict(i=1, state="lost", reason="weak DCF response")]
    (tmp_path/"replay.jsonl").write_text("\n".join(map(json.dumps, rows)))
    result = review.summarize(tmp_path, "replay")
    assert result["seed_accepted"] and result["refined_box"] == [1,2,12,12]
    assert result["first_loss"] == dict(frame=1, reason="weak DCF response")
    unlabeled = review.summarize(tmp_path, "replay", labeled=False)
    assert unlabeled["error_p50"] is None and unlabeled["error_p95"] is None
    assert unlabeled["false_tracking"] is None
    assert review.summarize(tmp_path/"absent")["missing"]
    table = review.table([dict(name="case", baseline=result, candidate=unlabeled)])
    assert "baseline" in table and "candidate" in table and "n/a" in table


def test_outer_runner_calls_one_guard_and_no_per_replay_guard(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(review, "load_clips", lambda _: [])
    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(review.subprocess, "run", fake_run)
    monkeypatch.setattr(review.sys, "argv", ["review_roi_seeds", "--library", "lib.so",
        "--output", str(tmp_path/"out"), "--baseline", "baseline", "--clips", "clips.json",
        "--provenance", "test"])
    assert review.main() == 0
    assert len(calls) == 1
    assert calls[0].count("tools.run_visual_lock_trial_guard") == 1
    assert "--inside-guard" in calls[0]


def test_batch_runner_is_detected_as_an_existing_owner(tmp_path):
    folder = tmp_path/"900008"
    folder.mkdir()
    (folder/"cmdline").write_bytes(b"python\0-m\0tools.review_roi_seeds\0")
    assert video_owners("jetson", tmp_path) == [dict(pid=900008, module="tools.review_roi_seeds")]


def test_guard_excludes_only_invoking_review_parent(monkeypatch, tmp_path):
    from tools import run_visual_lock_trial_guard as guard
    monkeypatch.setattr(guard.os, "getppid", lambda: 900008)
    for pid, module in ((900008, "tools.review_roi_seeds"),
                        (900009, "tools.review_roi_seeds"),
                        (900010, "tools.verify_visual_lock")):
        folder = tmp_path/str(pid)
        folder.mkdir()
        (folder/"cmdline").write_bytes(f"python\0-m\0{module}\0".encode())
    assert sorted(video_owners("jetson", tmp_path), key=lambda row: row["pid"]) == [
        dict(pid=900009, module="tools.review_roi_seeds"),
        dict(pid=900010, module="tools.verify_visual_lock")]
    monkeypatch.setattr(guard.os, "getppid", lambda: 900010)
    assert dict(pid=900010, module="tools.verify_visual_lock") in video_owners("jetson", tmp_path)


def test_child_command_has_bound_and_records_nonzero_status(tmp_path):
    def runner(command, **kwargs):
        assert command == ["python", "fixture"] and kwargs["timeout"] == 30
        return SimpleNamespace(returncode=1)
    assert review.run_child(["python", "fixture"], tmp_path/"log", timeout=30, runner=runner) == 1


def test_suite_runs_every_replay_serially_without_nested_guards(monkeypatch,tmp_path):
    from contextlib import nullcontext
    import numpy as np
    commands = []
    def report(name):
        return dict(name=name,seed_accepted=name != "wall", selected_frames=117 if name != "wall" else 0,
                    first_loss=dict(frame=0,reason="seed lacks texture") if name == "wall" else None)
    def child(command, log, **kwargs):
        commands.append(command)
        output = Path(command[command.index("--output")+1])
        output.mkdir()
        names = ["replay"] if "--video" in command else review.FIXTURES
        (output/"report.json").write_text(json.dumps(dict(report=[report(n) for n in names],acceptance="pass")))
        return 0
    monkeypatch.setattr(review,"run_child",child)
    monkeypatch.setattr(review,"StandaloneEGL",nullcontext)
    monkeypatch.setattr(review.cv2,"imread",lambda _: np.zeros((1,1,3),np.uint8))
    monkeypatch.setattr(review,"evaluate",lambda name,*a,**kw: report(name))
    hands = tmp_path/"hands"; hands.mkdir()
    for _, filename, _ in review.HAND_CASES: (hands/filename).write_bytes(b"fixture")
    output = tmp_path/"missing-parent"/"out"
    args = SimpleNamespace(output=str(output),baseline=str(tmp_path/"missing-baseline"),
        library="library",provenance="test",hand_frames=str(hands))
    clips = [dict(name=str(i),video="clip.avi",seed_xywh=[1,2,12,12]) for i in range(6)]
    assert review.run_suite(args,clips) == 0
    assert len(commands) == 7
    assert all("tools.run_visual_lock_trial_guard" not in c for c in commands)
    result = json.loads((output/"comparison.json").read_text())
    assert len(result["comparison"]) == 17 and result["baseline_complete"] is False
