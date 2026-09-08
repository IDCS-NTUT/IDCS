from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_require_auto_rejects_targetless_authorized_observations(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / 'tests/fixtures/control_protocol_shadow_rate_trace.jsonl'
    records = []
    for line in source.read_text(encoding='utf-8').splitlines():
        record = json.loads(line)
        if record.get('type') == 'observation':
            record['observation']['target']['valid'] = False
            record['observation']['safety']['valid'] = True
            record['observation']['safety']['auto_allowed'] = True
        records.append(json.dumps(record))
    trace = tmp_path / 'targetless_auto_trace.jsonl'
    trace.write_text('\n'.join(records) + '\n', encoding='utf-8')

    result = subprocess.run(
        [
            sys.executable,
            str(root / 'tools/validate_control_capture.py'),
            str(trace),
            '--min-observations',
            '1',
            '--min-valid-fraction',
            '0.5',
            '--require-auto',
        ],
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert result.returncode == 2
    assert not report['qualified']
    assert report['target_valid'] == 0
    assert report['auto_allowed'] > 0
    assert report['automatic_tracking_ready'] == 0
    assert report['failures'] == ['insufficient_valid_input_fraction']


def test_scheduler_health_is_required_and_accepted_when_clean(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / 'tests/fixtures/control_protocol_shadow_rate_trace.jsonl'
    base_args = [
        sys.executable,
        str(root / 'tools/validate_control_capture.py'),
        '--min-observations',
        '1',
        '--min-valid-fraction',
        '0',
        '--require-scheduler-health',
    ]

    missing = subprocess.run(
        [*base_args, str(source)],
        capture_output=True,
        text=True,
    )
    missing_report = json.loads(missing.stdout)
    assert missing.returncode == 2
    assert missing_report['failures'] == ['missing_scheduler_summary']

    trace = tmp_path / 'clean_scheduler_trace.jsonl'
    trace.write_text(
        source.read_text(encoding='utf-8')
        + json.dumps(
            {
                'type': 'summary',
                'decode_errors': 0,
                'missed_periods': 0,
                'physical_control_disabled': True,
            }
        )
        + '\n',
        encoding='utf-8',
    )
    clean = subprocess.run(
        [*base_args, str(trace)],
        capture_output=True,
        text=True,
    )
    clean_report = json.loads(clean.stdout)
    assert clean.returncode == 0
    assert clean_report['qualified']
    assert clean_report['scheduler_health']['missed_periods'] == 0
