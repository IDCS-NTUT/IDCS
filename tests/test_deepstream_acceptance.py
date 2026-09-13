from jetson.deepstream.acceptance import evaluate_report


def _report() -> dict:
    return {"frames": 100, "gpu_osd_enabled": True, "h264_return_enabled": True,
            "encoded_h264_buffers": 100, "steady_pipeline_fps": 60.1,
            "return_output": {"control_disabled": True},
            "shadow_transport": {"published": 98, "snapshot_published": 98,
                                 "legacy_published": 98, "invalid_headers": 0,
                                 "dropped_nonmonotonic": 0}}


def test_acceptance_passes_control_free_correlated_return_video():
    assert evaluate_report(_report()) == {"failures": [], "warnings": []}


def test_acceptance_rejects_control_or_header_contract_regression():
    report = _report()
    report["return_output"]["control_disabled"] = False
    report["shadow_transport"]["dropped_nonmonotonic"] = 1
    outcome = evaluate_report(report)
    assert "runtime report does not prove control-disabled return output" in outcome["failures"]
    assert "non-monotonic PC frame headers observed" in outcome["failures"]


def test_acceptance_rejects_receiver_side_order_regression():
    outcome = evaluate_report(_report(), receiver_report={"messages": 10, "invalid": 0,
                                                           "nonmonotonic_frame_ids": 1,
                                                           "nonmonotonic_source_timestamps": 0})
    assert "PC receiver observed non-monotonic frame IDs" in outcome["failures"]


def test_acceptance_requires_native_v2_but_legacy_display_is_opt_in():
    report = _report()
    report["shadow_transport"]["snapshot_published"] = 0
    outcome = evaluate_report(report)
    assert "no PerceptionSnapshot V2 records published" in outcome["failures"]

    report = _report()
    report["shadow_transport"]["legacy_published"] = 0
    outcome = evaluate_report(report)
    assert outcome == {"failures": [], "warnings": []}
    outcome = evaluate_report(report, require_legacy_display=True)
    assert "no legacy display records published" in outcome["failures"]


def test_acceptance_accepts_headerless_argus_metadata_contract():
    report = _report()
    report["shadow_transport"]["header_correlation"] = False
    report["shadow_transport"]["invalid_headers"] = 4
    report["shadow_transport"]["dropped_nonmonotonic"] = 2
    assert evaluate_report(report) == {"failures": [], "warnings": []}
