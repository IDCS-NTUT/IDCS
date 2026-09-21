import json

from tools.runtime_process_guard import (
    command_runs_module,
    find_module_owners,
    main,
)


def _process(proc_root, pid: int, argv: list[str]):
    directory = proc_root / str(pid)
    directory.mkdir()
    (directory / "cmdline").write_bytes(b"\0".join(item.encode() for item in argv) + b"\0")


def test_command_module_match_is_structural_not_substring():
    assert command_runs_module(["python", "-u", "-m", "pc.streamer", "--config", "x"], "pc.streamer")
    assert not command_runs_module(["bash", "-c", "look for pc.streamer"], "pc.streamer")
    assert not command_runs_module(["python", "pc.streamer"], "pc.streamer")


def test_find_module_owners_reports_only_exact_python_module(tmp_path):
    _process(tmp_path, 101, ["python", "-u", "-m", "pc.streamer", "--config", "x"])
    _process(tmp_path, 102, ["python", "-m", "pc.ui"])
    _process(tmp_path, 103, ["grep", "pc.streamer"])

    assert find_module_owners(["pc.streamer"], proc_root=tmp_path, self_pid=999) == [{
        "pid": 101,
        "modules": ["pc.streamer"],
        "argv": ["python", "-u", "-m", "pc.streamer", "--config", "x"],
    }]


def test_main_blocks_duplicate_and_reports_owner(tmp_path, capsys):
    _process(tmp_path, 201, ["python", "-m", "jetson.deepstream.runtime"])

    assert main([
        "--module", "jetson.deepstream.runtime", "--proc-root", str(tmp_path)
    ]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["safe_to_start"] is False
    assert result["owners"][0]["pid"] == 201


def test_main_allows_zero_owner(tmp_path, capsys):
    assert main(["--module", "pc.ui", "--proc-root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["safe_to_start"] is True
