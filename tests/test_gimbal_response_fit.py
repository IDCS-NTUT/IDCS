import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import fit_gimbal_response as fit


CSV_FIELDS = [
    "axis",
    "accel_byte",
    "profile",
    "setting_id",
    "trial",
    "direction",
    "elapsed_s",
    "cmd_rate_applied_rad_s",
    "angle_rad",
    "omega_rad_s",
    "encoder_dt_s",
    "valid_encoder",
    "omega_valid",
    "send_dropped",
    "missing_reply",
    "limit_blocked",
    "pending_query_count",
    "reply_latency_ms",
]

TRUE_PARAMS = {
    "yaw": (1.8, 0.7, 0.03),
    "pitch": (1.2, 0.9, -0.02),
}
TRUE_DELAY_S = 0.06


def _previous(times, values, query, default=0.0):
    idx = int(np.searchsorted(times, query, side="right")) - 1
    if idx < 0:
        return default
    return float(values[idx])


def _command_for_step(step_idx, trial, direction):
    pattern = [0.0, 0.7, -0.4, 0.9, -0.8, 0.2, -0.6, 0.6, -0.2, 0.8, -0.9, 0.3]
    return direction * pattern[(step_idx + trial) % len(pattern)]


def _synthetic_rows(axis, *, trial, direction, dt=0.02, duration_s=3.0):
    a_u, a_f, bias = TRUE_PARAMS[axis]
    times = np.arange(0.0, duration_s, dt)
    segment_steps = 6
    commands = np.asarray(
        [_command_for_step(idx // segment_steps, trial, direction) for idx in range(times.size)],
        dtype=float,
    )
    theta = 0.0
    omega = 0.0
    rows = []
    for idx, now_s in enumerate(times):
        rows.append(
            {
                "axis": axis,
                "accel_byte": 7 if axis == "yaw" else 9,
                "profile": "prbs",
                "setting_id": 100 + trial,
                "trial": trial,
                "direction": direction,
                "elapsed_s": f"{now_s:.6f}",
                "cmd_rate_applied_rad_s": f"{commands[idx]:.9f}",
                "angle_rad": f"{theta:.9f}",
                "omega_rad_s": f"{omega:.9f}",
                "encoder_dt_s": "" if idx == 0 else f"{dt:.9f}",
                "valid_encoder": "1",
                "omega_valid": "0" if idx == 0 else "1",
                "send_dropped": "0",
                "missing_reply": "0",
                "limit_blocked": "0",
                "pending_query_count": "1",
                "reply_latency_ms": "20.0",
            }
        )
        u_delay = _previous(times, commands, now_s - TRUE_DELAY_S, 0.0)
        omega_dot = (a_u * u_delay) - (a_f * omega) + bias
        theta += dt * omega
        omega += dt * omega_dot
    return rows


def _write_synthetic_csv(path):
    rows = []
    for axis in TRUE_PARAMS:
        for trial in range(6):
            direction = 1 if trial % 2 == 0 else -1
            rows.extend(_synthetic_rows(axis, trial=trial, direction=direction))
    _write_rows(path, rows)


def _write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class GimbalResponseFitTests(unittest.TestCase):
    def _load_fit_fixture(self, tmp):
        csv_path = tmp / "synthetic_sweep.csv"
        _write_synthetic_csv(csv_path)
        samples, counters = fit.load_sweep_samples(csv_path)
        fits = fit.fit_all_axes(
            samples,
            delay_values=fit._delay_grid(0.0, 0.12, 0.02),
            validation_fraction=0.25,
            refine=False,
            theta_residual_weight=0.5,
            load_counters=counters,
        )
        return csv_path, samples, counters, fits

    def test_fit_recovers_synthetic_parameters_and_delay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _csv_path, _samples, _counters, fits = self._load_fit_fixture(Path(temp_dir))

        self.assertEqual({"yaw", "pitch"}, set(fits))
        for axis, truth in TRUE_PARAMS.items():
            with self.subTest(axis=axis):
                axis_fit = fits[axis]
                self.assertAlmostEqual(axis_fit.delay_s, TRUE_DELAY_S, delta=0.021)
                self.assertAlmostEqual(axis_fit.params[0], truth[0], delta=0.2)
                self.assertAlmostEqual(axis_fit.params[1], truth[1], delta=0.2)
                self.assertAlmostEqual(axis_fit.params[2], truth[2], delta=0.08)
                self.assertGreater(axis_fit.validation_metrics["sample_count"], 0)

    def test_train_validation_split_keeps_whole_trials_together(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "synthetic_sweep.csv"
            _write_synthetic_csv(csv_path)
            samples, _counters = fit.load_sweep_samples(csv_path)

        yaw_samples = [sample for sample in samples if sample.axis == "yaw"]
        train, validation, train_keys, validation_keys = fit.split_train_validation(yaw_samples, 0.34)

        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertTrue(set(train_keys).isdisjoint(validation_keys))
        self.assertEqual({sample.group_key for sample in train}, set(train_keys))
        self.assertEqual({sample.group_key for sample in validation}, set(validation_keys))

    def test_invalid_rows_are_excluded_and_counted(self):
        good = {
            "axis": "yaw",
            "accel_byte": "1",
            "profile": "step",
            "setting_id": "0",
            "trial": "0",
            "direction": "1",
            "elapsed_s": "0.0",
            "cmd_rate_applied_rad_s": "0.5",
            "angle_rad": "0.1",
            "omega_rad_s": "0.2",
            "encoder_dt_s": "0.02",
            "valid_encoder": "1",
            "omega_valid": "1",
            "send_dropped": "0",
            "missing_reply": "0",
            "limit_blocked": "0",
            "pending_query_count": "1",
            "reply_latency_ms": "20.0",
        }
        rows = [dict(good)]
        for flag, value in (
            ("limit_blocked", "1"),
            ("valid_encoder", "0"),
            ("send_dropped", "1"),
            ("missing_reply", "1"),
        ):
            row = dict(good)
            row[flag] = value
            rows.append(row)
        row = dict(good)
        row["omega_rad_s"] = "nan"
        rows.append(row)
        for field, value in (
            ("omega_valid", "0"),
            ("encoder_dt_s", "0.001"),
            ("omega_rad_s", "30.0"),
            ("pending_query_count", "5"),
            ("reply_latency_ms", "120.0"),
        ):
            row = dict(good)
            row[field] = value
            rows.append(row)

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "quality.csv"
            _write_rows(csv_path, rows)
            samples, counters = fit.load_sweep_samples(csv_path)

        self.assertEqual(1, len(samples))
        self.assertEqual(11, counters["rows_total"])
        self.assertEqual(1, counters["rows_accepted"])
        self.assertEqual(10, counters["rows_rejected"])
        self.assertEqual(1, counters["rejected_limit_blocked"])
        self.assertEqual(1, counters["rejected_invalid_encoder"])
        self.assertEqual(1, counters["rejected_send_dropped"])
        self.assertEqual(1, counters["rejected_missing_reply"])
        self.assertEqual(1, counters["rejected_nonfinite"])
        self.assertEqual(1, counters["rejected_invalid_omega"])
        self.assertEqual(1, counters["rejected_short_encoder_dt"])
        self.assertEqual(1, counters["rejected_omega_outlier"])
        self.assertEqual(1, counters["rejected_pending_backlog"])
        self.assertEqual(1, counters["rejected_reply_latency"])

    def test_report_json_contains_required_schema_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            csv_path, _samples, counters, fits = self._load_fit_fixture(tmp)
            manifest_path = tmp / "synthetic_sweep.json"
            manifest_path.write_text(
                json.dumps({"format": "idcs.gimbal_response_sweep", "version": 2}),
                encoding="utf-8",
            )
            manifest = fit._load_manifest(manifest_path)
            report = fit.build_report(
                csv_path=csv_path,
                manifest_path=manifest_path,
                manifest=manifest,
                load_counters=counters,
                delay_values=fit._delay_grid(0.0, 0.12, 0.02),
                validation_fraction=0.25,
                refine=False,
                quality_filters={"min_encoder_dt_s": 0.01},
                fits=fits,
            )
            report_path = tmp / "fit_report.json"
            fit._write_report(report_path, report)
            payload = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(fit.REPORT_FORMAT, payload["format"])
        self.assertEqual(1, payload["version"])
        self.assertEqual("idcs.gimbal_response_sweep", payload["source"]["manifest_format"])
        self.assertIn("quality_filters", payload["settings"])
        self.assertEqual(counters["rows_accepted"], payload["load"]["rows_accepted"])
        self.assertIn("yaw", payload["axes"])
        self.assertIn("selected_model", payload["axes"]["yaw"])
        self.assertGreaterEqual(len(payload["axes"]["yaw"]["model_comparison"]), 2)
        self.assertIn("parameters", payload["axes"]["yaw"])
        self.assertIn("delay_s", payload["axes"]["yaw"]["parameters"])
        self.assertIn("validation_metrics", payload["axes"]["yaw"])
        self.assertGreater(payload["axes"]["yaw"]["validation_metrics"]["command_max_abs"], 0.0)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is optional")
    def test_plot_flag_runs_with_noninteractive_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            _csv_path, samples, _counters, fits = self._load_fit_fixture(tmp)
            out_dir = tmp / "plots"
            fit._plot_diagnostics(samples=samples, fits=fits, out_dir=out_dir)

            self.assertTrue((out_dir / "delay_sweep.png").exists())
            self.assertTrue((out_dir / "yaw_replay.png").exists())
            self.assertTrue((out_dir / "yaw_residuals.png").exists())
            self.assertTrue((out_dir / "fit_summary.png").exists())


if __name__ == "__main__":
    unittest.main()
