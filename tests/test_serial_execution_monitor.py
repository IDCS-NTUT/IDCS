from tools.serial_execution_monitor import ExecutionAudit


def test_execution_audit_reports_complete_accounting_and_delivery() -> None:
    audit = ExecutionAudit()
    audit.ingest(
        {
            "type": "SerialCommandEventV1",
            "service_epoch": "epoch",
            "sequence": 10,
            "cmd_id": "a",
            "event": "wire_sent",
            "timing": {"event_monotonic_ns": 1_000_000_000},
            "accounting": {"admitted": 1, "terminal": 1, "pending": 0},
        },
        received_ns=1_002_000_000,
    )

    report = audit.report()

    assert report["accounting_complete"] is True
    assert report["outcomes"] == {"wire_sent": 1}
    assert report["delivery_ms"]["max"] == 2.0


def test_execution_audit_detects_sequence_gap_and_duplicate_terminal() -> None:
    audit = ExecutionAudit()
    for sequence in (1, 3):
        audit.ingest(
            {
                "type": "SerialCommandEventV1",
                "service_epoch": "epoch",
                "sequence": sequence,
                "cmd_id": "same",
                "event": "preempted",
                "timing": {},
            },
            received_ns=2_000_000_000,
        )

    report = audit.report()

    assert report["sequence_gaps"] == 1
    assert report["duplicate_terminal_ids"] == 1
