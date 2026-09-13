import ast
import dis
from pathlib import Path

from jetson.deepstream import pipeline, verify_pipeline
from jetson.deepstream import runtime


def test_verification_cli_delegates_to_shared_pipeline_core():
    assert verify_pipeline.run is pipeline.run


def test_production_runtime_names_shared_pipeline_not_verifier():
    imports = {
        instruction.argval
        for instruction in dis.get_instructions(runtime.run)
        if instruction.opname == "IMPORT_NAME"
    }

    assert "jetson.deepstream.pipeline" in imports
    assert "jetson.deepstream.verify_pipeline" not in imports


def test_shared_pipeline_core_has_no_legacy_perception_dependency():
    source = Path(pipeline.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "common.schemas" not in imported_modules
    assert "common.perception_compat" not in imported_modules
    assert "detection_msg_from_snapshot" not in source


def test_controller_and_trace_consumers_use_only_v2_perception_transport():
    paths = (
        Path("jetson/deepstream/shadow_controller.py"),
        Path("jetson/tools/shadow_fixed_rate_controller.py"),
        Path("tools/record_control_protocol_trace.py"),
        Path("tools/record_control_trace.py"),
        Path("tools/analyze_control_trace.py"),
    )

    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "DetectionMsg" not in source, path
        assert "detection_msg_from_json" not in source, path
        assert "zmq_results" not in source, path
        assert "perception" in source.lower(), path
        assert "snapshot" in source.lower(), path

    scheduler_source = Path("jetson/fixed_rate_controller.py").read_text(
        encoding="utf-8"
    )
    assert "DetectionMsg" not in scheduler_source
    assert "update_detection" not in scheduler_source
    assert "ControlObservation" in scheduler_source


def test_host_video_consumers_use_only_v2_perception_transport():
    for path in (
        Path("pc/streamer.py"),
        Path("pc/ui.py"),
        Path("pc/metadata_monitor.py"),
        Path("pc/sim_camera.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert "DetectionMsg" not in source, path
        assert "detection_msg_from_json" not in source, path
        assert "zmq_results" not in source, path
        assert "PerceptionSnapshot" in source or "perception" in source.lower(), path
