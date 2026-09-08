import dis

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
