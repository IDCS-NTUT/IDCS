from pathlib import Path
from subprocess import CompletedProcess

from jetson.deepstream import preflight


def test_engine_check_distinguishes_unavailable_gpu(tmp_path, monkeypatch):
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"plan")
    trtexec = tmp_path / "trtexec"
    trtexec.write_text("", encoding="utf-8")
    monkeypatch.setattr(preflight.shutil, "which", lambda _name: str(trtexec))
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(
            args[0], 1, stdout="", stderr="Cuda failure: no CUDA-capable device is detected"
        ),
    )
    errors: list[str] = []

    preflight._check_engine(engine, errors)

    assert errors == [
        "Jetson GPU runtime is unavailable; TensorRT engine compatibility was not evaluated"
    ]


def test_engine_check_keeps_engine_failure_distinct(tmp_path, monkeypatch):
    engine = Path(tmp_path / "model.engine")
    engine.write_bytes(b"plan")
    trtexec = tmp_path / "trtexec"
    trtexec.write_text("", encoding="utf-8")
    monkeypatch.setattr(preflight.shutil, "which", lambda _name: str(trtexec))
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(
            args[0], 1, stdout="", stderr="Error Code 1: Serialization assertion failed"
        ),
    )
    errors: list[str] = []

    preflight._check_engine(engine, errors)

    assert errors == ["TensorRT cannot deserialize the configured engine"]
