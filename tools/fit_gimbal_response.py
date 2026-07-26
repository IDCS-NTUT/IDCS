#!/usr/bin/env python3
"""Fit open-loop gimbal response models from response-sweep CSV data.

This is an offline analysis tool. It does not tune controllers, modify runtime
configuration, or publish commands. The input is a CSV produced by
``jetson.tools.gimbal_response_sweep`` plus an optional JSON manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

try:  # pragma: no cover - exercised by tests when scipy is installed
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - optional dependency path
    least_squares = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

REPORT_FORMAT = "idcs.gimbal_response_fit"
REPORT_VERSION = 1
ACTIVE_PHASES = frozenset({"step", "random_step", "random-step", "prbs", "chirp", "sine"})
MODEL_CANDIDATES = (
    "continuous-derivative",
    "discrete-first-order",
    "discrete-first-order-deadband",
    "discrete-first-order-asymmetric",
)


@dataclass(frozen=True)
class SweepSample:
    axis: str
    accel_byte: int
    profile: str
    phase: str
    setting_id: int
    trial: int
    direction: int
    t: float
    u: float
    theta: float
    omega: float
    command_magnitude: float
    source_row: int

    @property
    def group_key(self) -> tuple[str, int, str, int, int, int]:
        return (
            self.axis,
            self.accel_byte,
            self.profile,
            self.setting_id,
            self.trial,
            self.direction,
        )


@dataclass(frozen=True)
class AxisFit:
    axis: str
    params: tuple[float, float, float]
    delay_s: float
    train_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    delay_sweep: list[dict[str, Any]]
    warnings: list[str]
    train_groups: list[tuple[str, int, str, int, int, int]]
    validation_groups: list[tuple[str, int, str, int, int, int]]
    model_name: str = "continuous-derivative"
    candidate_reports: list[dict[str, Any]] = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DiscreteModel:
    name: str
    coeffs: dict[str, float]
    delay_s: float
    train_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    warnings: list[str]
    train_groups: list[tuple[str, int, str, int, int, int]]
    validation_groups: list[tuple[str, int, str, int, int, int]]

    @property
    def params(self) -> tuple[float, float, float]:
        return _discrete_to_continuous_params(self.coeffs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV produced by jetson.tools.gimbal_response_sweep")
    parser.add_argument("--manifest", default=None, help="Optional JSON manifest path")
    parser.add_argument("--out-dir", default="artifacts/gimbal_fit", help="Output directory")
    parser.add_argument("--report-json", default=None, help="Override fit report JSON path")
    parser.add_argument("--plot", action="store_true", help="Write diagnostic plots")
    parser.add_argument("--emit-yaml", action="store_true", help="Print YAML-style parameter snippet")
    parser.add_argument("--delay-start-s", type=float, default=0.0)
    parser.add_argument("--delay-end-s", type=float, default=0.2)
    parser.add_argument("--delay-step-s", type=float, default=0.005)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--no-refine", action="store_true", help="Skip simulation-error refinement")
    parser.add_argument(
        "--min-encoder-dt-s",
        type=float,
        default=0.01,
        help="Reject omega samples derived from shorter encoder intervals; use <=0 to disable.",
    )
    parser.add_argument(
        "--max-abs-omega-rad-s",
        type=float,
        default=5.0,
        help="Reject physically implausible measured omega magnitudes; use <=0 to disable.",
    )
    parser.add_argument(
        "--max-pending-query-count",
        type=int,
        default=2,
        help="Reject rows captured while the encoder query backlog exceeded this count; use <0 to disable.",
    )
    parser.add_argument(
        "--max-reply-latency-ms",
        type=float,
        default=80.0,
        help="Reject rows above this encoder reply latency; use <=0 to disable.",
    )
    parser.add_argument(
        "--theta-residual-weight",
        type=float,
        default=0.5,
        help="Relative weight for theta residuals during refinement",
    )
    return parser.parse_args()


def _float(value: Any, default: float = math.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _is_truthy_number(value: Any, default: int = 0) -> bool:
    if value in ("", None):
        return bool(default)
    return _int(value, default) != 0


def _load_manifest(path: Optional[Path]) -> Optional[Mapping[str, Any]]:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, Mapping) else None


def load_sweep_samples(
    csv_path: Path,
    *,
    min_encoder_dt_s: float = 0.01,
    max_abs_omega_rad_s: float = 5.0,
    max_pending_query_count: int = 2,
    max_reply_latency_ms: float = 80.0,
) -> tuple[list[SweepSample], dict[str, Any]]:
    """Load sweep samples, applying only quality filters needed for fitting."""

    samples: list[SweepSample] = []
    counters = {
        "rows_total": 0,
        "rows_accepted": 0,
        "rows_rejected": 0,
        "rejected_nonfinite": 0,
        "rejected_limit_blocked": 0,
        "rejected_invalid_encoder": 0,
        "rejected_send_dropped": 0,
        "rejected_missing_reply": 0,
        "rejected_invalid_omega": 0,
        "rejected_short_encoder_dt": 0,
        "rejected_omega_outlier": 0,
        "rejected_pending_backlog": 0,
        "rejected_reply_latency": 0,
    }
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader, start=2):
            counters["rows_total"] += 1
            limit_blocked = _is_truthy_number(row.get("limit_blocked"), 0)
            valid_encoder = _is_truthy_number(row.get("valid_encoder"), 1)
            send_dropped = _is_truthy_number(row.get("send_dropped"), 0)
            missing_reply = _is_truthy_number(row.get("missing_reply"), 0)
            if limit_blocked:
                counters["rejected_limit_blocked"] += 1
                counters["rows_rejected"] += 1
                continue
            if not valid_encoder:
                counters["rejected_invalid_encoder"] += 1
                counters["rows_rejected"] += 1
                continue
            if send_dropped:
                counters["rejected_send_dropped"] += 1
                counters["rows_rejected"] += 1
                continue
            if missing_reply:
                counters["rejected_missing_reply"] += 1
                counters["rows_rejected"] += 1
                continue

            omega_valid = row.get("omega_valid")
            if omega_valid not in ("", None) and not _is_truthy_number(omega_valid, 1):
                counters["rejected_invalid_omega"] += 1
                counters["rows_rejected"] += 1
                continue

            t = _float(row.get("elapsed_s"))
            # New sweep files expose the physical integer-RPM command encoded
            # into F6. Older captures retain the nominal applied-rate fallback.
            u = _float(row.get("cmd_rate_encoded_rad_s"))
            if not math.isfinite(u):
                u = _float(row.get("cmd_rate_applied_rad_s"))
            theta = _float(row.get("angle_rad"))
            omega = _float(row.get("omega_rad_s"))
            if not all(math.isfinite(value) for value in (t, u, theta, omega)):
                counters["rejected_nonfinite"] += 1
                counters["rows_rejected"] += 1
                continue
            encoder_dt = _float(row.get("encoder_dt_s"), math.nan)
            if min_encoder_dt_s > 0.0 and math.isfinite(encoder_dt) and encoder_dt < min_encoder_dt_s:
                counters["rejected_short_encoder_dt"] += 1
                counters["rows_rejected"] += 1
                continue
            if max_abs_omega_rad_s > 0.0 and abs(omega) > max_abs_omega_rad_s:
                counters["rejected_omega_outlier"] += 1
                counters["rows_rejected"] += 1
                continue
            pending_query_count = _int(row.get("pending_query_count"), 0)
            if max_pending_query_count >= 0 and pending_query_count > max_pending_query_count:
                counters["rejected_pending_backlog"] += 1
                counters["rows_rejected"] += 1
                continue
            reply_latency_ms = _float(row.get("reply_latency_ms"), math.nan)
            if (
                max_reply_latency_ms > 0.0
                and math.isfinite(reply_latency_ms)
                and reply_latency_ms > max_reply_latency_ms
            ):
                counters["rejected_reply_latency"] += 1
                counters["rows_rejected"] += 1
                continue

            sample = SweepSample(
                axis=str(row.get("axis", "")).strip().lower() or "unknown",
                accel_byte=_int(row.get("accel_byte"), 0),
                profile=str(row.get("profile", "step")).strip().lower() or "step",
                phase=str(row.get("phase", row.get("profile", "step"))).strip().lower() or "step",
                setting_id=_int(row.get("setting_id"), 0),
                trial=_int(row.get("trial"), 0),
                direction=_int(row.get("direction"), 0),
                t=t,
                u=u,
                theta=theta,
                omega=omega,
                command_magnitude=abs(u),
                source_row=row_idx,
            )
            samples.append(sample)
            counters["rows_accepted"] += 1
    return samples, counters


def _group_samples(samples: Sequence[SweepSample]) -> dict[tuple[str, int, str, int, int, int], list[SweepSample]]:
    groups: dict[tuple[str, int, str, int, int, int], list[SweepSample]] = {}
    for sample in samples:
        groups.setdefault(sample.group_key, []).append(sample)
    for group_samples in groups.values():
        group_samples.sort(key=lambda sample: sample.t)
    return groups


def _axis_samples(samples: Sequence[SweepSample], axis: str) -> list[SweepSample]:
    return [sample for sample in samples if sample.axis == axis]


def _active_samples(samples: Sequence[SweepSample], *, min_abs_command: float = 1e-6) -> list[SweepSample]:
    return [
        sample
        for sample in samples
        if sample.phase in ACTIVE_PHASES and abs(sample.u) >= min_abs_command
    ]


def split_train_validation(
    samples: Sequence[SweepSample],
    validation_fraction: float,
) -> tuple[list[SweepSample], list[SweepSample], list[tuple[str, int, str, int, int, int]], list[tuple[str, int, str, int, int, int]]]:
    groups = _group_samples(samples)
    keys = sorted(groups)
    if not keys:
        return [], [], [], []
    if len(keys) == 1:
        return list(groups[keys[0]]), list(groups[keys[0]]), keys, keys
    val_count = max(1, int(math.ceil(len(keys) * max(0.0, min(1.0, validation_fraction)))))
    val_keys = keys[-val_count:]
    train_keys = keys[:-val_count] or keys[:]
    train = [sample for key in train_keys for sample in groups[key]]
    validation = [sample for key in val_keys for sample in groups[key]]
    return train, validation, train_keys, val_keys


def split_active_train_validation(
    samples: Sequence[SweepSample],
    validation_fraction: float,
) -> tuple[list[SweepSample], list[SweepSample], list[tuple[str, int, str, int, int, int]], list[tuple[str, int, str, int, int, int]]]:
    active = _active_samples(samples)
    groups = {
        key: value
        for key, value in _group_samples(active).items()
        if any(abs(sample.u) > 1e-6 for sample in value)
    }
    keys = sorted(groups)
    if not keys:
        return [], [], [], []
    if len(keys) == 1:
        return list(groups[keys[0]]), list(groups[keys[0]]), keys, keys
    val_count = max(1, int(math.ceil(len(keys) * max(0.0, min(1.0, validation_fraction)))))
    val_keys = keys[-val_count:]
    train_keys = keys[:-val_count] or keys[:]
    train = [sample for key in train_keys for sample in groups[key]]
    validation = [sample for key in val_keys for sample in groups[key]]
    return train, validation, train_keys, val_keys


def split_full_trace_by_active_groups(
    samples: Sequence[SweepSample],
    validation_fraction: float,
) -> tuple[list[SweepSample], list[SweepSample], list[tuple[str, int, str, int, int, int]], list[tuple[str, int, str, int, int, int]]]:
    """Split by excited trials, then include the complete trace for those trials."""

    active_groups = {
        key
        for key, value in _group_samples(_active_samples(samples)).items()
        if any(abs(sample.u) > 1e-6 for sample in value)
    }
    if not active_groups:
        return [], [], [], []
    full_groups = _group_samples(samples)
    keys = sorted(key for key in active_groups if key in full_groups)
    if not keys:
        return [], [], [], []
    if len(keys) == 1:
        return list(full_groups[keys[0]]), list(full_groups[keys[0]]), keys, keys
    val_count = max(1, int(math.ceil(len(keys) * max(0.0, min(1.0, validation_fraction)))))
    val_keys = keys[-val_count:]
    train_keys = keys[:-val_count] or keys[:]
    train = [sample for key in train_keys for sample in full_groups[key]]
    validation = [sample for key in val_keys for sample in full_groups[key]]
    return train, validation, train_keys, val_keys


def _sequences(samples: Sequence[SweepSample]) -> list[list[SweepSample]]:
    groups = _group_samples(samples)
    return [groups[key] for key in sorted(groups)]


def _previous_sample(t: np.ndarray, y: np.ndarray, query: float, default: float = 0.0) -> float:
    idx = int(np.searchsorted(t, query, side="right")) - 1
    if idx < 0:
        return default
    return float(y[idx])


def _build_derivative_rows(
    samples: Sequence[SweepSample],
    delay_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    targets: list[float] = []
    for seq in _sequences(samples):
        if len(seq) < 3:
            continue
        t = np.asarray([sample.t for sample in seq], dtype=float)
        u = np.asarray([sample.u for sample in seq], dtype=float)
        omega = np.asarray([sample.omega for sample in seq], dtype=float)
        for idx in range(len(seq) - 1):
            dt = t[idx + 1] - t[idx]
            if dt <= 0.0 or not math.isfinite(float(dt)):
                continue
            u_delay = _previous_sample(t, u, t[idx] - delay_s, 0.0)
            omega_dot = (omega[idx + 1] - omega[idx]) / dt
            if not all(math.isfinite(value) for value in (u_delay, omega[idx], omega_dot)):
                continue
            rows.append([u_delay, -float(omega[idx]), 1.0])
            targets.append(float(omega_dot))
    if not rows:
        return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def _least_squares_params(samples: Sequence[SweepSample], delay_s: float) -> Optional[tuple[float, float, float]]:
    x, y = _build_derivative_rows(samples, delay_s)
    if x.shape[0] < 5:
        return None
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    if not np.all(np.isfinite(beta)):
        return None
    return (float(beta[0]), float(beta[1]), float(beta[2]))


def _simulate_sequence(
    seq: Sequence[SweepSample],
    *,
    params: tuple[float, float, float],
    delay_s: float,
) -> dict[str, np.ndarray]:
    a_u, a_f, bias = params
    t = np.asarray([sample.t for sample in seq], dtype=float)
    u = np.asarray([sample.u for sample in seq], dtype=float)
    theta_meas = np.asarray([sample.theta for sample in seq], dtype=float)
    omega_meas = np.asarray([sample.omega for sample in seq], dtype=float)
    theta = np.zeros_like(theta_meas)
    omega = np.zeros_like(omega_meas)
    if theta.size:
        theta[0] = theta_meas[0]
        omega[0] = omega_meas[0]
    failed = False
    for idx in range(1, len(seq)):
        dt = t[idx] - t[idx - 1]
        if dt <= 0.0 or not math.isfinite(float(dt)):
            theta[idx] = theta[idx - 1]
            omega[idx] = omega[idx - 1]
            continue
        u_delay = _previous_sample(t, u, t[idx - 1] - delay_s, 0.0)
        omega_dot = a_u * u_delay - a_f * omega[idx - 1] + bias
        theta[idx] = theta[idx - 1] + dt * omega[idx - 1]
        omega[idx] = omega[idx - 1] + dt * omega_dot
        if (
            not math.isfinite(float(theta[idx]))
            or not math.isfinite(float(omega[idx]))
            or abs(float(theta[idx])) > 1e6
            or abs(float(omega[idx])) > 1e6
        ):
            theta[idx:] = np.nan
            omega[idx:] = np.nan
            failed = True
            break
    return {
        "t": t,
        "u": u,
        "theta_meas": theta_meas,
        "omega_meas": omega_meas,
        "theta_pred": theta,
        "omega_pred": omega,
        "failed": failed,
    }


def _residual_arrays(
    samples: Sequence[SweepSample],
    *,
    params: tuple[float, float, float],
    delay_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_t: list[np.ndarray] = []
    all_u: list[np.ndarray] = []
    theta_res: list[np.ndarray] = []
    omega_res: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    for seq in _sequences(samples):
        if len(seq) < 2:
            continue
        sim = _simulate_sequence(seq, params=params, delay_s=delay_s)
        valid = np.isfinite(sim["theta_pred"]) & np.isfinite(sim["omega_pred"])
        if not np.any(valid):
            continue
        all_t.append(sim["t"][valid])
        all_u.append(sim["u"][valid])
        theta_res.append((sim["theta_pred"] - sim["theta_meas"])[valid])
        omega_res.append((sim["omega_pred"] - sim["omega_meas"])[valid])
        directions.append(np.asarray([sample.direction for sample in seq], dtype=float)[valid])
    if not theta_res:
        empty = np.zeros((0,), dtype=float)
        return empty, empty, empty, empty, empty
    return (
        np.concatenate(all_t),
        np.concatenate(all_u),
        np.concatenate(theta_res),
        np.concatenate(omega_res),
        np.concatenate(directions),
    )


def _residual_metadata_arrays(
    samples: Sequence[SweepSample],
    *,
    params: tuple[float, float, float],
    delay_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_t: list[np.ndarray] = []
    all_u: list[np.ndarray] = []
    theta_res: list[np.ndarray] = []
    omega_res: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    profiles: list[np.ndarray] = []
    accel_bytes: list[np.ndarray] = []
    for seq in _sequences(samples):
        if len(seq) < 2:
            continue
        sim = _simulate_sequence(seq, params=params, delay_s=delay_s)
        valid = np.isfinite(sim["theta_pred"]) & np.isfinite(sim["omega_pred"])
        if not np.any(valid):
            continue
        all_t.append(sim["t"][valid])
        all_u.append(sim["u"][valid])
        theta_res.append((sim["theta_pred"] - sim["theta_meas"])[valid])
        omega_res.append((sim["omega_pred"] - sim["omega_meas"])[valid])
        directions.append(np.asarray([sample.direction for sample in seq], dtype=float)[valid])
        profiles.append(np.asarray([sample.profile for sample in seq], dtype=object)[valid])
        accel_bytes.append(np.asarray([sample.accel_byte for sample in seq], dtype=int)[valid])
    if not theta_res:
        empty = np.zeros((0,), dtype=float)
        return empty, empty, empty, empty, empty, np.asarray([], dtype=object), np.asarray([], dtype=int)
    return (
        np.concatenate(all_t),
        np.concatenate(all_u),
        np.concatenate(theta_res),
        np.concatenate(omega_res),
        np.concatenate(directions),
        np.concatenate(profiles),
        np.concatenate(accel_bytes),
    )


def _rmse(values: np.ndarray) -> Optional[float]:
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(math.sqrt(float(np.mean(np.square(finite)))))


def _mae(values: np.ndarray) -> Optional[float]:
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.mean(np.abs(finite)))


def _metrics(
    samples: Sequence[SweepSample],
    *,
    params: tuple[float, float, float],
    delay_s: float,
) -> dict[str, Any]:
    _t, u, theta_res, omega_res, directions = _residual_arrays(
        samples,
        params=params,
        delay_s=delay_s,
    )
    direction_bias: dict[str, Optional[float]] = {}
    for direction in (-1.0, 1.0):
        mask = directions == direction
        if np.any(mask):
            direction_bias[str(int(direction))] = float(np.mean(omega_res[mask]))
    return {
        "sample_count": int(theta_res.size),
        "omega_rmse": _rmse(omega_res),
        "theta_rmse": _rmse(theta_res),
        "omega_mae": _mae(omega_res),
        "theta_mae": _mae(theta_res),
        "command_max_abs": None if u.size == 0 else float(np.max(np.abs(u))),
        "direction_omega_bias": direction_bias,
    }


def _score_metric(metrics: Mapping[str, Any]) -> float:
    omega = metrics.get("omega_rmse")
    theta = metrics.get("theta_rmse")
    if omega is None and theta is None:
        return float("inf")
    score = 0.0
    if omega is not None:
        score += float(omega)
    if theta is not None:
        score += 0.25 * float(theta)
    if not math.isfinite(score):
        return float("inf")
    return score


def _delay_grid(start_s: float, end_s: float, step_s: float) -> list[float]:
    if step_s <= 0.0:
        raise ValueError("delay step must be > 0")
    if end_s < start_s:
        raise ValueError("delay end must be >= start")
    values = []
    value = start_s
    while value <= end_s + (0.5 * step_s):
        values.append(round(value, 10))
        value += step_s
    return values


def _u_effective(u: float, coeffs: Mapping[str, float], model_name: str) -> float:
    if model_name == "discrete-first-order-deadband":
        deadband = max(0.0, float(coeffs.get("deadband", 0.0)))
        return math.copysign(max(abs(u) - deadband, 0.0), u)
    if model_name == "discrete-first-order-asymmetric":
        if u >= 0.0:
            deadband = max(0.0, float(coeffs.get("deadband_pos", 0.0)))
            return max(u - deadband, 0.0)
        deadband = max(0.0, float(coeffs.get("deadband_neg", 0.0)))
        return -max(abs(u) - deadband, 0.0)
    return float(u)


def _fit_discrete_coeffs(
    samples: Sequence[SweepSample],
    *,
    delay_s: float,
    model_name: str,
    deadband: float = 0.0,
) -> Optional[dict[str, float]]:
    rows: list[list[float]] = []
    targets: list[float] = []
    dt_values: list[float] = []
    for seq in _sequences(samples):
        if len(seq) < 3:
            continue
        t = np.asarray([sample.t for sample in seq], dtype=float)
        u = np.asarray([sample.u for sample in seq], dtype=float)
        omega = np.asarray([sample.omega for sample in seq], dtype=float)
        for idx in range(len(seq) - 1):
            dt = t[idx + 1] - t[idx]
            if dt <= 0.0 or not math.isfinite(float(dt)):
                continue
            u_delay = _previous_sample(t, u, t[idx] - delay_s, 0.0)
            if model_name == "discrete-first-order-deadband":
                u_eff = math.copysign(max(abs(u_delay) - deadband, 0.0), u_delay)
                row = [float(omega[idx]), u_eff, 1.0]
            elif model_name == "discrete-first-order-asymmetric":
                u_pos = max(u_delay - deadband, 0.0)
                u_neg = max(-u_delay - deadband, 0.0)
                row = [float(omega[idx]), u_pos, u_neg, 1.0]
            else:
                row = [float(omega[idx]), float(u_delay), 1.0]
            target = float(omega[idx + 1])
            if not all(math.isfinite(value) for value in (*row, target)):
                continue
            rows.append(row)
            targets.append(target)
            dt_values.append(float(dt))
    if len(rows) < 5:
        return None
    x = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    if not np.all(np.isfinite(beta)):
        return None
    if model_name == "discrete-first-order-asymmetric":
        coeffs = {
            "c_omega": float(beta[0]),
            "c_u_pos": float(beta[1]),
            "c_u_neg": float(beta[2]),
            "bias": float(beta[3]),
            "deadband_pos": float(deadband),
            "deadband_neg": float(deadband),
            "dt_s": float(np.mean(dt_values)),
        }
    else:
        coeffs = {
            "c_omega": float(beta[0]),
            "c_u": float(beta[1]),
            "bias": float(beta[2]),
            "dt_s": float(np.mean(dt_values)),
        }
        if model_name == "discrete-first-order-deadband":
            coeffs["deadband"] = float(deadband)
    return coeffs


def _simulate_discrete_sequence(
    seq: Sequence[SweepSample],
    *,
    coeffs: Mapping[str, float],
    delay_s: float,
    model_name: str,
) -> dict[str, np.ndarray]:
    t = np.asarray([sample.t for sample in seq], dtype=float)
    u = np.asarray([sample.u for sample in seq], dtype=float)
    theta_meas = np.asarray([sample.theta for sample in seq], dtype=float)
    omega_meas = np.asarray([sample.omega for sample in seq], dtype=float)
    theta = np.zeros_like(theta_meas)
    omega = np.zeros_like(omega_meas)
    if theta.size:
        theta[0] = theta_meas[0]
        omega[0] = omega_meas[0]
    c_omega = float(coeffs.get("c_omega", 1.0))
    bias = float(coeffs.get("bias", 0.0))
    failed = False
    for idx in range(1, len(seq)):
        dt = t[idx] - t[idx - 1]
        if dt <= 0.0 or not math.isfinite(float(dt)):
            theta[idx] = theta[idx - 1]
            omega[idx] = omega[idx - 1]
            continue
        u_delay = _previous_sample(t, u, t[idx - 1] - delay_s, 0.0)
        if model_name == "discrete-first-order-asymmetric":
            u_pos = max(u_delay - max(0.0, float(coeffs.get("deadband_pos", 0.0))), 0.0)
            u_neg = max(-u_delay - max(0.0, float(coeffs.get("deadband_neg", 0.0))), 0.0)
            omega_next = c_omega * omega[idx - 1] + float(coeffs.get("c_u_pos", 0.0)) * u_pos + float(coeffs.get("c_u_neg", 0.0)) * u_neg + bias
        else:
            u_eff = _u_effective(u_delay, coeffs, model_name)
            omega_next = c_omega * omega[idx - 1] + float(coeffs.get("c_u", 0.0)) * u_eff + bias
        theta[idx] = theta[idx - 1] + dt * omega[idx - 1]
        omega[idx] = omega_next
        if (
            not math.isfinite(float(theta[idx]))
            or not math.isfinite(float(omega[idx]))
            or abs(float(theta[idx])) > 1e6
            or abs(float(omega[idx])) > 1e6
        ):
            theta[idx:] = np.nan
            omega[idx:] = np.nan
            failed = True
            break
    return {
        "t": t,
        "u": u,
        "theta_meas": theta_meas,
        "omega_meas": omega_meas,
        "theta_pred": theta,
        "omega_pred": omega,
        "failed": failed,
    }


def _discrete_residual_arrays(
    samples: Sequence[SweepSample],
    *,
    coeffs: Mapping[str, float],
    delay_s: float,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_t: list[np.ndarray] = []
    all_u: list[np.ndarray] = []
    theta_res: list[np.ndarray] = []
    omega_res: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    for seq in _sequences(samples):
        if len(seq) < 2:
            continue
        sim = _simulate_discrete_sequence(seq, coeffs=coeffs, delay_s=delay_s, model_name=model_name)
        valid = np.isfinite(sim["theta_pred"]) & np.isfinite(sim["omega_pred"])
        if not np.any(valid):
            continue
        all_t.append(sim["t"][valid])
        all_u.append(sim["u"][valid])
        theta_res.append((sim["theta_pred"] - sim["theta_meas"])[valid])
        omega_res.append((sim["omega_pred"] - sim["omega_meas"])[valid])
        directions.append(np.asarray([sample.direction for sample in seq], dtype=float)[valid])
    if not theta_res:
        empty = np.zeros((0,), dtype=float)
        return empty, empty, empty, empty, empty
    return (
        np.concatenate(all_t),
        np.concatenate(all_u),
        np.concatenate(theta_res),
        np.concatenate(omega_res),
        np.concatenate(directions),
    )


def _discrete_metrics(
    samples: Sequence[SweepSample],
    *,
    coeffs: Mapping[str, float],
    delay_s: float,
    model_name: str,
) -> dict[str, Any]:
    _t, u, theta_res, omega_res, directions = _discrete_residual_arrays(
        samples,
        coeffs=coeffs,
        delay_s=delay_s,
        model_name=model_name,
    )
    direction_bias: dict[str, Optional[float]] = {}
    for direction in (-1.0, 1.0):
        mask = directions == direction
        if np.any(mask):
            direction_bias[str(int(direction))] = float(np.mean(omega_res[mask]))
    return {
        "sample_count": int(theta_res.size),
        "omega_rmse": _rmse(omega_res),
        "theta_rmse": _rmse(theta_res),
        "omega_mae": _mae(omega_res),
        "theta_mae": _mae(theta_res),
        "command_max_abs": None if u.size == 0 else float(np.max(np.abs(u))),
        "direction_omega_bias": direction_bias,
    }


def _discrete_residual_metadata_arrays(
    samples: Sequence[SweepSample],
    *,
    coeffs: Mapping[str, float],
    delay_s: float,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_t: list[np.ndarray] = []
    all_u: list[np.ndarray] = []
    theta_res: list[np.ndarray] = []
    omega_res: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    profiles: list[np.ndarray] = []
    accel_bytes: list[np.ndarray] = []
    for seq in _sequences(samples):
        if len(seq) < 2:
            continue
        sim = _simulate_discrete_sequence(seq, coeffs=coeffs, delay_s=delay_s, model_name=model_name)
        valid = np.isfinite(sim["theta_pred"]) & np.isfinite(sim["omega_pred"])
        if not np.any(valid):
            continue
        all_t.append(sim["t"][valid])
        all_u.append(sim["u"][valid])
        theta_res.append((sim["theta_pred"] - sim["theta_meas"])[valid])
        omega_res.append((sim["omega_pred"] - sim["omega_meas"])[valid])
        directions.append(np.asarray([sample.direction for sample in seq], dtype=float)[valid])
        profiles.append(np.asarray([sample.profile for sample in seq], dtype=object)[valid])
        accel_bytes.append(np.asarray([sample.accel_byte for sample in seq], dtype=int)[valid])
    if not theta_res:
        empty = np.zeros((0,), dtype=float)
        return empty, empty, empty, empty, empty, np.asarray([], dtype=object), np.asarray([], dtype=int)
    return (
        np.concatenate(all_t),
        np.concatenate(all_u),
        np.concatenate(theta_res),
        np.concatenate(omega_res),
        np.concatenate(directions),
        np.concatenate(profiles),
        np.concatenate(accel_bytes),
    )


def _discrete_to_continuous_params(coeffs: Mapping[str, float]) -> tuple[float, float, float]:
    c_omega = float(coeffs.get("c_omega", 1.0))
    dt_s = max(1e-9, float(coeffs.get("dt_s", 1.0)))
    if 0.0 < c_omega < 1.0:
        a_f = max(1e-9, -math.log(c_omega) / dt_s)
    else:
        a_f = max(1e-9, (1.0 - c_omega) / dt_s)
    c_u = float(coeffs.get("c_u", 0.0))
    if "c_u_pos" in coeffs or "c_u_neg" in coeffs:
        c_u = 0.5 * (float(coeffs.get("c_u_pos", 0.0)) - float(coeffs.get("c_u_neg", 0.0)))
    a_u = c_u * a_f / max(1e-9, 1.0 - c_omega)
    return (float(a_u), float(a_f), float(coeffs.get("bias", 0.0)))


def _fit_discrete_model(
    samples: Sequence[SweepSample],
    *,
    axis: str,
    model_name: str,
    delay_values: Sequence[float],
    validation_fraction: float,
    load_counters: Mapping[str, Any],
) -> Optional[DiscreteModel]:
    axis_data = _axis_samples(samples, axis)
    if len(axis_data) < 8:
        return None
    train, validation, train_groups, validation_groups = split_full_trace_by_active_groups(
        axis_data,
        validation_fraction,
    )
    if not train or not validation:
        return None
    max_u = max(abs(sample.u) for sample in train)
    deadband_values = [0.0]
    if model_name in {"discrete-first-order-deadband", "discrete-first-order-asymmetric"} and max_u > 0.0:
        deadband_values = sorted({0.0, 0.05 * max_u, 0.1 * max_u, 0.2 * max_u})

    best: Optional[tuple[float, dict[str, float], dict[str, Any], dict[str, Any], float]] = None
    for delay_s in delay_values:
        for deadband in deadband_values:
            coeffs = _fit_discrete_coeffs(
                train,
                delay_s=float(delay_s),
                model_name=model_name,
                deadband=float(deadband),
            )
            if coeffs is None:
                continue
            train_metrics = _discrete_metrics(train, coeffs=coeffs, delay_s=float(delay_s), model_name=model_name)
            validation_metrics = _discrete_metrics(validation, coeffs=coeffs, delay_s=float(delay_s), model_name=model_name)
            score = _score_metric(validation_metrics)
            if best is None or score < best[4]:
                best = (float(delay_s), coeffs, train_metrics, validation_metrics, score)
    if best is None:
        return None
    delay_s, coeffs, train_metrics, validation_metrics, _score = best
    params = _discrete_to_continuous_params(coeffs)
    warnings = _warnings_for_fit(
        samples=axis_data,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        params=params,
        load_counters=load_counters,
    )
    c_omega = coeffs.get("c_omega")
    if c_omega is not None and not (-1.0 < float(c_omega) < 1.0):
        warnings.append("discrete pole outside unit circle")
    return DiscreteModel(
        name=model_name,
        coeffs=coeffs,
        delay_s=delay_s,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        warnings=warnings,
        train_groups=train_groups,
        validation_groups=validation_groups,
    )


def _refine_params(
    samples: Sequence[SweepSample],
    *,
    initial: tuple[float, float, float],
    delay_s: float,
    theta_weight: float,
) -> tuple[float, float, float]:
    if least_squares is None:
        return initial

    def residual(vec: np.ndarray) -> np.ndarray:
        params = (float(vec[0]), float(vec[1]), float(vec[2]))
        _t, _u, theta_res, omega_res, _directions = _residual_arrays(
            samples,
            params=params,
            delay_s=delay_s,
        )
        if theta_res.size == 0:
            return np.array([0.0], dtype=float)
        return np.concatenate([omega_res, theta_weight * theta_res])

    lower = np.array([-1e4, 1e-9, -1e4], dtype=float)
    upper = np.array([1e4, 1e4, 1e4], dtype=float)
    x0 = np.array([initial[0], max(initial[1], 1e-9), initial[2]], dtype=float)
    try:
        result = least_squares(residual, x0=x0, bounds=(lower, upper), loss="soft_l1")
    except Exception:
        return initial
    if not result.success or not np.all(np.isfinite(result.x)):
        return initial
    return (float(result.x[0]), float(result.x[1]), float(result.x[2]))


def _warnings_for_fit(
    *,
    samples: Sequence[SweepSample],
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    params: tuple[float, float, float],
    load_counters: Mapping[str, Any],
) -> list[str]:
    warnings: list[str] = []
    u_values = np.asarray([sample.u for sample in samples], dtype=float)
    if u_values.size == 0 or float(np.max(np.abs(u_values))) < 1e-6:
        warnings.append("weak excitation: command magnitude is near zero")
    elif float(np.std(u_values)) < 1e-4:
        warnings.append("weak excitation: command variation is very small")
    total = int(load_counters.get("rows_total", 0))
    rejected = int(load_counters.get("rows_rejected", 0))
    if total > 0 and rejected / total > 0.2:
        warnings.append("high invalid/rejected sample fraction")
    if params[1] <= 0.0:
        warnings.append("non-positive damping estimate")
    train_score = _score_metric(train_metrics)
    validation_score = _score_metric(validation_metrics)
    if math.isfinite(train_score) and validation_score > 3.0 * max(train_score, 1e-9):
        warnings.append("validation error is much larger than train error")
    direction_bias = validation_metrics.get("direction_omega_bias", {})
    if isinstance(direction_bias, Mapping) and "-1" in direction_bias and "1" in direction_bias:
        neg = direction_bias.get("-1")
        pos = direction_bias.get("1")
        if neg is not None and pos is not None and abs(float(neg) - float(pos)) > 0.2:
            warnings.append("direction-dependent residual bias is visible")
    directions = {sample.direction for sample in samples}
    if not ({-1, 1} <= directions):
        warnings.append("only one command direction present")
    return warnings


def fit_axis(
    samples: Sequence[SweepSample],
    *,
    axis: str,
    delay_values: Sequence[float],
    validation_fraction: float,
    refine: bool,
    theta_residual_weight: float,
    load_counters: Mapping[str, Any],
) -> Optional[AxisFit]:
    axis_data = _axis_samples(samples, axis)
    if len(axis_data) < 8:
        return None
    train, validation, train_groups, validation_groups = split_full_trace_by_active_groups(
        axis_data,
        validation_fraction,
    )
    if not train or not validation:
        return None

    sweep_rows: list[dict[str, Any]] = []
    best: Optional[tuple[float, tuple[float, float, float], float]] = None
    for delay_s in delay_values:
        params = _least_squares_params(train, float(delay_s))
        if params is None:
            continue
        train_metrics = _metrics(train, params=params, delay_s=float(delay_s))
        validation_metrics = _metrics(validation, params=params, delay_s=float(delay_s))
        score = _score_metric(validation_metrics)
        sweep_rows.append(
            {
                "delay_s": float(delay_s),
                "train_omega_rmse": train_metrics["omega_rmse"],
                "train_theta_rmse": train_metrics["theta_rmse"],
                "validation_omega_rmse": validation_metrics["omega_rmse"],
                "validation_theta_rmse": validation_metrics["theta_rmse"],
                "score": score,
            }
        )
        if best is None or score < best[2]:
            best = (float(delay_s), params, score)

    if best is None:
        return None
    delay_s, params, _score = best
    if refine:
        params = _refine_params(
            train,
            initial=params,
            delay_s=delay_s,
            theta_weight=theta_residual_weight,
        )
    train_metrics = _metrics(train, params=params, delay_s=delay_s)
    validation_metrics = _metrics(validation, params=params, delay_s=delay_s)
    warnings = _warnings_for_fit(
        samples=axis_data,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        params=params,
        load_counters=load_counters,
    )
    return AxisFit(
        axis=axis,
        params=params,
        delay_s=delay_s,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        delay_sweep=sweep_rows,
        warnings=warnings,
        train_groups=train_groups,
        validation_groups=validation_groups,
        model_name="continuous-derivative",
        candidate_reports=[],
    )


def _discrete_to_axis_fit(fit: DiscreteModel, axis: str, candidate_reports: list[dict[str, Any]]) -> AxisFit:
    return AxisFit(
        axis=axis,
        params=fit.params,
        delay_s=fit.delay_s,
        train_metrics=fit.train_metrics,
        validation_metrics=fit.validation_metrics,
        delay_sweep=[],
        warnings=fit.warnings,
        train_groups=fit.train_groups,
        validation_groups=fit.validation_groups,
        model_name=fit.name,
        candidate_reports=candidate_reports,
    )


def _candidate_report_from_axis_fit(fit: AxisFit) -> dict[str, Any]:
    return {
        "model": fit.model_name,
        "parameters": _fit_to_report(fit)["parameters"],
        "delay_s": fit.delay_s,
        "train_metrics": fit.train_metrics,
        "validation_metrics": fit.validation_metrics,
        "warnings": fit.warnings,
        "score": _score_metric(fit.validation_metrics),
    }


def _candidate_report_from_discrete(fit: DiscreteModel) -> dict[str, Any]:
    a_u, a_f, bias = fit.params
    c_omega = fit.coeffs.get("c_omega")
    return {
        "model": fit.name,
        "coefficients": dict(fit.coeffs),
        "parameters": {
            "a_u": a_u,
            "a_f": a_f,
            "bias": bias,
            "delay_s": fit.delay_s,
            "tau_s": None if a_f <= 0.0 else 1.0 / a_f,
            "dc_gain": None if a_f == 0.0 else a_u / a_f,
            "discrete_pole": c_omega,
        },
        "delay_s": fit.delay_s,
        "train_metrics": fit.train_metrics,
        "validation_metrics": fit.validation_metrics,
        "warnings": fit.warnings,
        "score": _score_metric(fit.validation_metrics),
    }


def fit_axis_model_comparison(
    samples: Sequence[SweepSample],
    *,
    axis: str,
    delay_values: Sequence[float],
    validation_fraction: float,
    refine: bool,
    theta_residual_weight: float,
    load_counters: Mapping[str, Any],
) -> Optional[AxisFit]:
    candidates: list[AxisFit | DiscreteModel] = []
    legacy = fit_axis(
        samples,
        axis=axis,
        delay_values=delay_values,
        validation_fraction=validation_fraction,
        refine=refine,
        theta_residual_weight=theta_residual_weight,
        load_counters=load_counters,
    )
    if legacy is not None:
        candidates.append(legacy)
    for model_name in MODEL_CANDIDATES:
        if model_name == "continuous-derivative":
            continue
        candidate = _fit_discrete_model(
            samples,
            axis=axis,
            model_name=model_name,
            delay_values=delay_values,
            validation_fraction=validation_fraction,
            load_counters=load_counters,
        )
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return None

    def candidate_score(candidate: AxisFit | DiscreteModel) -> float:
        return _score_metric(candidate.validation_metrics)

    reports: list[dict[str, Any]] = []
    for candidate in candidates:
        if isinstance(candidate, AxisFit):
            reports.append(_candidate_report_from_axis_fit(candidate))
        else:
            reports.append(_candidate_report_from_discrete(candidate))
    reports.sort(key=lambda row: float(row["score"]))
    best = min(candidates, key=candidate_score)
    if isinstance(best, AxisFit):
        return AxisFit(
            axis=best.axis,
            params=best.params,
            delay_s=best.delay_s,
            train_metrics=best.train_metrics,
            validation_metrics=best.validation_metrics,
            delay_sweep=best.delay_sweep,
            warnings=best.warnings,
            train_groups=best.train_groups,
            validation_groups=best.validation_groups,
            model_name=best.model_name,
            candidate_reports=reports,
        )
    return _discrete_to_axis_fit(best, axis, reports)


def fit_all_axes(
    samples: Sequence[SweepSample],
    *,
    delay_values: Sequence[float],
    validation_fraction: float,
    refine: bool,
    theta_residual_weight: float,
    load_counters: Mapping[str, Any],
) -> dict[str, AxisFit]:
    axes = sorted({sample.axis for sample in samples})
    fits: dict[str, AxisFit] = {}
    for axis in axes:
        fit = fit_axis_model_comparison(
            samples,
            axis=axis,
            delay_values=delay_values,
            validation_fraction=validation_fraction,
            refine=refine,
            theta_residual_weight=theta_residual_weight,
            load_counters=load_counters,
        )
        if fit is not None:
            fits[axis] = fit
    return fits


def _fit_to_report(fit: AxisFit) -> dict[str, Any]:
    a_u, a_f, bias = fit.params
    tau_s = None if a_f <= 0.0 else 1.0 / a_f
    dc_gain = None if a_f == 0.0 else a_u / a_f
    return {
        "selected_model": fit.model_name,
        "parameters": {
            "a_u": a_u,
            "a_f": a_f,
            "bias": bias,
            "delay_s": fit.delay_s,
            "tau_s": tau_s,
            "dc_gain": dc_gain,
        },
        "train_metrics": fit.train_metrics,
        "validation_metrics": fit.validation_metrics,
        "delay_sweep": fit.delay_sweep,
        "warnings": fit.warnings,
        "train_groups": [list(group) for group in fit.train_groups],
        "validation_groups": [list(group) for group in fit.validation_groups],
        "model_comparison": fit.candidate_reports or [],
    }


def build_report(
    *,
    csv_path: Path,
    manifest_path: Optional[Path],
    manifest: Optional[Mapping[str, Any]],
    load_counters: Mapping[str, Any],
    delay_values: Sequence[float],
    validation_fraction: float,
    refine: bool,
    quality_filters: Mapping[str, Any],
    fits: Mapping[str, AxisFit],
) -> dict[str, Any]:
    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "source": {
            "csv": str(csv_path),
            "manifest": None if manifest_path is None else str(manifest_path),
            "manifest_format": None if manifest is None else manifest.get("format"),
            "manifest_version": None if manifest is None else manifest.get("version"),
        },
        "settings": {
            "delay_values_s": list(delay_values),
            "validation_fraction": validation_fraction,
            "refine": refine,
            "quality_filters": {
                "valid_encoder": True,
                "send_dropped": False,
                "missing_reply": False,
                "limit_blocked": False,
                **dict(quality_filters),
            },
        },
        "load": dict(load_counters),
        "axes": {axis: _fit_to_report(fit) for axis, fit in fits.items()},
    }


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _print_summary(report: Mapping[str, Any]) -> None:
    print(f"Fit report: {report['source']['csv']}")
    axes = report.get("axes", {})
    if not isinstance(axes, Mapping) or not axes:
        print("No axis fits produced.")
        return
    for axis, axis_report in axes.items():
        params = axis_report["parameters"]
        val = axis_report["validation_metrics"]
        print(
            "%s: model=%s a_u=%.6g a_f=%.6g bias=%.6g delay=%.4fs val_omega_rmse=%s val_theta_rmse=%s"
            % (
                str(axis).upper(),
                axis_report.get("selected_model", "unknown"),
                params["a_u"],
                params["a_f"],
                params["bias"],
                params["delay_s"],
                _format_optional(val.get("omega_rmse")),
                _format_optional(val.get("theta_rmse")),
            )
        )
        for warning in axis_report.get("warnings", []):
            print(f"  warning: {warning}")


def _format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


def _emit_yaml_snippet(report: Mapping[str, Any]) -> None:
    axes = report.get("axes", {})
    if not isinstance(axes, Mapping) or not axes:
        return
    print("\n# Open-loop fit parameters (manual review required)")
    print("control:")
    print("  mpc:")
    print("    plant:")
    for axis, axis_report in axes.items():
        params = axis_report["parameters"]
        print(f"      # {axis}: delay_s={params['delay_s']:.6f}, bias={params['bias']:.6f}")
        print(f"      # {axis}_a_u: {params['a_u']:.6f}")
        print(f"      # {axis}_a_f: {params['a_f']:.6f}")


def _plot_diagnostics(
    *,
    samples: Sequence[SweepSample],
    fits: Mapping[str, AxisFit],
    out_dir: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise SystemExit("matplotlib is required for --plot; install .[sysid]") from exc

    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_model_comparison(fits, out_dir, plt)

    summary_rows = []
    for axis, fit in fits.items():
        axis_samples = _axis_samples(samples, axis)
        _plot_raw_command_response(axis, axis_samples, out_dir, plt)
        _plot_replay(axis, axis_samples, fit, out_dir, plt)
        _plot_residuals(axis, axis_samples, fit, out_dir, plt)
        summary_rows.append(
            [
                axis,
                fit.model_name,
                fit.params[0],
                fit.params[1],
                fit.params[2],
                fit.delay_s,
                fit.validation_metrics.get("omega_rmse"),
                fit.validation_metrics.get("theta_rmse"),
            ]
        )
    _plot_summary(summary_rows, out_dir, plt)


def _selected_candidate(fit: AxisFit) -> Optional[Mapping[str, Any]]:
    for candidate in fit.candidate_reports or []:
        if candidate.get("model") == fit.model_name:
            return candidate
    return None


def _simulate_fit_sequence(seq: Sequence[SweepSample], fit: AxisFit) -> dict[str, np.ndarray]:
    candidate = _selected_candidate(fit)
    if fit.model_name.startswith("discrete") and candidate and isinstance(candidate.get("coefficients"), Mapping):
        return _simulate_discrete_sequence(
            seq,
            coeffs=candidate["coefficients"],
            delay_s=fit.delay_s,
            model_name=fit.model_name,
        )
    return _simulate_sequence(seq, params=fit.params, delay_s=fit.delay_s)


def _fit_residual_metadata_arrays(
    samples: Sequence[SweepSample],
    fit: AxisFit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate = _selected_candidate(fit)
    if fit.model_name.startswith("discrete") and candidate and isinstance(candidate.get("coefficients"), Mapping):
        return _discrete_residual_metadata_arrays(
            samples,
            coeffs=candidate["coefficients"],
            delay_s=fit.delay_s,
            model_name=fit.model_name,
        )
    return _residual_metadata_arrays(samples, params=fit.params, delay_s=fit.delay_s)


def _plot_model_comparison(fits: Mapping[str, AxisFit], out_dir: Path, plt: Any) -> None:
    rows: list[tuple[str, str, float, float, Optional[float], Optional[float]]] = []
    for axis, fit in fits.items():
        if fit.candidate_reports:
            for candidate in fit.candidate_reports:
                val = candidate.get("validation_metrics", {})
                rows.append(
                    (
                        axis,
                        str(candidate.get("model", "?")),
                        float(candidate.get("score", float("nan"))),
                        float(candidate.get("delay_s", float("nan"))),
                        val.get("omega_rmse"),
                        val.get("theta_rmse"),
                    )
                )
        else:
            for row in fit.delay_sweep:
                rows.append(
                    (
                        axis,
                        "continuous-derivative",
                        float(row.get("score", float("nan"))),
                        float(row["delay_s"]),
                        row.get("validation_omega_rmse"),
                        row.get("validation_theta_rmse"),
                    )
                )
    if not rows:
        return

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    labels = [f"{axis}\n{model.replace('discrete-', 'disc-')}" for axis, model, *_ in rows]
    scores = [score if math.isfinite(score) else np.nan for _axis, _model, score, *_rest in rows]
    delays = [delay for _axis, _model, _score, delay, *_rest in rows]
    colors = ["tab:blue" if axis == "yaw" else "tab:orange" for axis, *_ in rows]

    axes[0].bar(range(len(rows)), scores, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("validation replay score (log)")
    axes[0].set_title("Model comparison")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(range(len(rows)), delays, color=colors)
    axes[1].set_ylabel("selected delay (s)")
    axes[1].set_xticks(range(len(rows)), labels, rotation=35, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.savefig(out_dir / "model_comparison.png", dpi=160)
    # Keep the historical filename, but make it a useful delay/model summary for non-derivative fits.
    fig.savefig(out_dir / "delay_sweep.png", dpi=160)
    plt.close(fig)


def _plot_raw_command_response(axis: str, samples: Sequence[SweepSample], out_dir: Path, plt: Any) -> None:
    if not samples:
        return
    ordered = sorted(samples, key=lambda sample: sample.t)
    t = np.asarray([sample.t for sample in ordered], dtype=float)
    t = t - float(np.nanmin(t))
    u = np.asarray([sample.u for sample in ordered], dtype=float)
    omega = np.asarray([sample.omega for sample in ordered], dtype=float)
    theta = np.asarray([sample.theta for sample in ordered], dtype=float)
    profile = np.asarray([sample.profile for sample in ordered], dtype=object)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    axes[0].plot(t, u, color="tab:blue", linewidth=1.0, label="applied command")
    axes[0].plot(t, omega, color="tab:red", linewidth=1.0, alpha=0.85, label="measured omega")
    axes[0].set_title(f"{axis.upper()} raw command vs real movement")
    axes[0].set_ylabel("rad/s")
    axes[0].legend(loc="best")

    axes[1].plot(t, theta, color="black", linewidth=1.0, label="measured angle")
    axes[1].set_ylabel("angle rad")
    axes[1].legend(loc="best")

    profile_labels = sorted({str(value) for value in profile})
    for idx, label in enumerate(profile_labels):
        mask = profile == label
        axes[2].plot(t[mask], np.full(int(np.sum(mask)), idx), ".", markersize=2, label=label)
    axes[2].set_yticks(range(len(profile_labels)), profile_labels)
    axes[2].set_ylabel("profile")
    axes[2].set_xlabel("time (s)")
    axes[2].set_title("accepted samples used by fitter")

    for axis_obj in axes:
        axis_obj.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"{axis}_command_response.png", dpi=140)
    plt.close(fig)


def _plot_replay(axis: str, samples: Sequence[SweepSample], fit: AxisFit, out_dir: Path, plt: Any) -> None:
    seqs = _sequences(samples)
    plot_seqs = sorted(seqs, key=len, reverse=True)[:8]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False, constrained_layout=True)
    for idx, seq in enumerate(plot_seqs):
        if len(seq) < 2:
            continue
        sim = _simulate_fit_sequence(seq, fit)
        t0 = sim["t"][0]
        label = f"{seq[0].profile} setting={seq[0].setting_id} trial={seq[0].trial} dir={seq[0].direction}"
        color = f"C{idx % 10}"
        axes[0].plot(sim["t"] - t0, sim["u"], color=color, alpha=0.55, label=label)
        axes[1].plot(sim["t"] - t0, sim["omega_meas"], color=color, alpha=0.35)
        axes[1].plot(sim["t"] - t0, sim["omega_pred"], color=color, linestyle="--", linewidth=1.2)
        axes[2].plot(sim["t"] - t0, sim["theta_meas"], color=color, alpha=0.35)
        axes[2].plot(sim["t"] - t0, sim["theta_pred"], color=color, linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("command rad/s")
    axes[1].set_ylabel("omega rad/s")
    axes[2].set_ylabel("theta rad")
    axes[2].set_xlabel("sequence time (s)")
    axes[0].set_title(f"{axis.upper()} replay: {fit.model_name}, delay={fit.delay_s:.3f}s")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    for axis_obj in axes:
        axis_obj.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"{axis}_replay.png", dpi=140)
    plt.close(fig)


def _plot_residuals(axis: str, samples: Sequence[SweepSample], fit: AxisFit, out_dir: Path, plt: Any) -> None:
    t, u, theta_res, omega_res, directions, profiles, accel_bytes = _fit_residual_metadata_arrays(samples, fit)
    if t.size == 0:
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes = axes.ravel()
    axes[0].plot(t - np.min(t), omega_res, ".", markersize=2)
    axes[0].set_ylabel("omega residual")
    axes[0].set_xlabel("time s")
    axes[1].scatter(np.abs(u), omega_res, c=directions, s=8, cmap="coolwarm")
    axes[1].set_xlabel("|command| rad/s")
    axes[1].set_ylabel("omega residual")
    profile_labels = sorted({str(profile) for profile in profiles})
    if profile_labels:
        axes[2].boxplot([omega_res[profiles == profile] for profile in profile_labels], labels=profile_labels)
    axes[2].set_ylabel("omega residual")
    axes[2].tick_params(axis="x", labelrotation=20)
    accel_labels = sorted({int(accel) for accel in accel_bytes})
    if accel_labels:
        axes[3].boxplot([omega_res[accel_bytes == accel] for accel in accel_labels], labels=[str(v) for v in accel_labels])
    axes[3].set_ylabel("omega residual")
    axes[3].set_xlabel("accel byte")
    axes[4].plot(t - np.min(t), theta_res, ".", markersize=2)
    axes[4].set_xlabel("time s")
    axes[4].set_ylabel("theta residual")
    axes[5].hist(omega_res[np.isfinite(omega_res)], bins=60, color="0.35")
    axes[5].set_xlabel("omega residual")
    axes[5].set_ylabel("count")
    axes[0].set_title(f"{axis.upper()} residuals: {fit.model_name}")
    for axis_obj in axes:
        axis_obj.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"{axis}_residuals.png", dpi=140)
    plt.close(fig)


def _plot_summary(rows: Sequence[Sequence[Any]], out_dir: Path, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(12, max(2.5, 0.5 * len(rows) + 1.5)))
    ax.axis("off")
    labels = ["axis", "model", "a_u", "a_f", "bias", "delay_s", "val omega RMSE", "val theta RMSE"]
    display_rows = [
        [
            row[0],
            row[1],
            f"{row[2]:.5g}",
            f"{row[3]:.5g}",
            f"{row[4]:.5g}",
            f"{row[5]:.4f}",
            _format_optional(row[6]),
            _format_optional(row[7]),
        ]
        for row in rows
    ]
    table = ax.table(cellText=display_rows, colLabels=labels, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(out_dir / "fit_summary.png", dpi=140)
    plt.close(fig)


def run(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv)
    manifest_path = Path(args.manifest) if args.manifest else csv_path.with_suffix(".json")
    if not manifest_path.exists() and not args.manifest:
        manifest_path = None
    out_dir = Path(args.out_dir)
    report_path = Path(args.report_json) if args.report_json else out_dir / "fit_report.json"
    delay_values = _delay_grid(args.delay_start_s, args.delay_end_s, args.delay_step_s)

    quality_filters = {
        "omega_valid": True,
        "min_encoder_dt_s": args.min_encoder_dt_s,
        "max_abs_omega_rad_s": args.max_abs_omega_rad_s,
        "max_pending_query_count": args.max_pending_query_count,
        "max_reply_latency_ms": args.max_reply_latency_ms,
    }
    samples, load_counters = load_sweep_samples(
        csv_path,
        min_encoder_dt_s=args.min_encoder_dt_s,
        max_abs_omega_rad_s=args.max_abs_omega_rad_s,
        max_pending_query_count=args.max_pending_query_count,
        max_reply_latency_ms=args.max_reply_latency_ms,
    )
    manifest = _load_manifest(manifest_path)
    fits = fit_all_axes(
        samples,
        delay_values=delay_values,
        validation_fraction=args.validation_fraction,
        refine=not args.no_refine,
        theta_residual_weight=args.theta_residual_weight,
        load_counters=load_counters,
    )
    report = build_report(
        csv_path=csv_path,
        manifest_path=manifest_path,
        manifest=manifest,
        load_counters=load_counters,
        delay_values=delay_values,
        validation_fraction=args.validation_fraction,
        refine=not args.no_refine,
        quality_filters=quality_filters,
        fits=fits,
    )
    _write_report(report_path, report)
    if args.plot:
        _plot_diagnostics(samples=samples, fits=fits, out_dir=out_dir)
    _print_summary(report)
    print(f"Wrote report: {report_path}")
    if args.emit_yaml:
        _emit_yaml_snippet(report)
    return 0 if fits else 1


def main() -> int:
    args = _parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
