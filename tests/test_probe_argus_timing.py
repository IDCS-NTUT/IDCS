from tools.probe_argus_timing import choose_pts_age_ns, summarize


def test_pts_domain_mapping_is_explicit():
    assert choose_pts_age_ns(10_000_000_000, 9_000_000_000, 9_990_000_000) == (
        "absolute_clock", 10_000_000
    )
    assert choose_pts_age_ns(10_000_000_000, 9_000_000_000, 990_000_000) == (
        "running_time", 10_000_000
    )
    assert choose_pts_age_ns(10_000_000_000, 9_000_000_000, 12_000_000_000) is None


def test_summary_excludes_startup_and_does_not_claim_exposure():
    records = [
        {"python_arrival_ns": 1, "pts_domain": "absolute_clock", "pts_to_pad_ns": 500_000_000},
        {"python_arrival_ns": 2_100_000_001, "pts_domain": "absolute_clock", "pts_to_pad_ns": 10_000_000},
        {"python_arrival_ns": 2_120_000_001, "pts_domain": "absolute_clock", "pts_to_pad_ns": 12_000_000},
    ]
    report = summarize(records, requested_duration_s=3)
    assert report["pts_to_source_pad_p50_ms"] == 11.0
    assert report["source_pad_interval_p50_ms"] == 20.0
    assert report["physical_exposure_time_measured"] is False
