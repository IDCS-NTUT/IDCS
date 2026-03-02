"""Benchmark MPC estimator and solve latency on Jetson.

Example:
    python -m jetson.tools.test_mpc_latency \
        --config configs/dev.yaml \
        --config-extra configs/dev_extra.yaml \
        --iterations 1500 \
        --warmup 300
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from common.config_sync import merge_config_maps, parse_config_text, resolve_active_video_profile
from common.control import ControlConfig, MpcConfig
from jetson.mpc import MpcAxisController, MpcAxisModel, MpcSolverError
from jetson.tools.estimator_variants import EstimatorVariantConfig, available_estimators


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dev.yaml", help="Primary YAML config path")
    parser.add_argument(
        "--config-extra",
        default="configs/dev_extra.yaml",
        help="Secondary YAML config path merged over --config",
    )
    parser.add_argument("--iterations", type=int, default=1500, help="Measured iterations per axis")
    parser.add_argument("--warmup", type=int, default=300, help="Warmup iterations per axis")
    parser.add_argument(
        "--missing-meas-rate",
        type=float,
        default=0.0,
        help="Probability in [0,1] that angle measurement is dropped each tick",
    )
    parser.add_argument(
        "--amplitude-rad",
        type=float,
        default=0.03,
        help="Reference sine amplitude in radians",
    )
    parser.add_argument(
        "--freq-hz",
        type=float,
        default=1.0,
        help="Reference sine frequency in Hz",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mode",
        choices=["full", "estimator"],
        default="full",
        help="Benchmark full MPC step or estimator-only step",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run multiple scenarios and print a comparison table",
    )
    parser.add_argument(
        "--sweep-missing-rates",
        default="0.0,0.02,0.05,0.1",
        help="Comma-separated missing measurement rates for --sweep",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional CSV output path for --sweep summary rows",
    )
    parser.add_argument(
        "--algorithms",
        default="baseline,gated_adaptive,rate_fusion",
        help="Comma-separated estimator algorithms for estimator-mode runs",
    )
    parser.add_argument(
        "--replay-file",
        default=None,
        help="Optional JSONL replay file with fields: axis,u_applied,theta_meas,omega_meas,theta_true,omega_true,dt_s",
    )
    parser.add_argument(
        "--omega-meas-rate",
        type=float,
        default=0.8,
        help="Probability in [0,1] that omega measurement is available in synthetic traces",
    )
    parser.add_argument(
        "--theta-noise-std",
        type=float,
        default=0.003,
        help="Synthetic theta measurement noise sigma (rad)",
    )
    parser.add_argument(
        "--omega-noise-std",
        type=float,
        default=0.04,
        help="Synthetic omega measurement noise sigma (rad/s)",
    )
    return parser.parse_args()


def _read_yaml(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    return parse_config_text(text, str(path))


def _load_control_cfg(config_path: Path, extra_path: Optional[Path]) -> tuple[ControlConfig, MpcConfig]:
    cfg_main = _read_yaml(config_path)
    cfg_merged: Dict[str, Any]
    if extra_path is not None:
        cfg_extra = _read_yaml(extra_path)
        cfg_merged = merge_config_maps(cfg_main, cfg_extra)
    else:
        cfg_merged = dict(cfg_main)

    video_cfg, _ = resolve_active_video_profile(cfg_merged)
    width = int(video_cfg.get("width", 0))
    height = int(video_cfg.get("height", 0))
    if width <= 0 or height <= 0:
        raise SystemExit("video width/height must be positive")

    control_cfg = ControlConfig.from_raw_config(cfg_merged, (width, height))
    if control_cfg.controller != "mpc" or control_cfg.mpc is None:
        raise SystemExit("control.controller must be 'mpc' and include control.mpc section")
    return control_cfg, control_cfg.mpc


def _prepare_sequence(values: Sequence[float], horizon: int) -> tuple[float, ...]:
    if not values:
        return tuple(0.0 for _ in range(horizon))
    last = float(values[-1])
    out = [float(v) for v in values[:horizon]]
    if len(out) < horizon:
        out.extend(last for _ in range(horizon - len(out)))
    return tuple(out)


def _ms(samples_ns: list[int]) -> np.ndarray:
    if not samples_ns:
        return np.asarray([], dtype=float)
    arr = np.asarray(samples_ns, dtype=np.float64)
    return arr / 1e6


def _summary_line(label: str, samples_ns: list[int]) -> str:
    arr_ms = _ms(samples_ns)
    if arr_ms.size == 0:
        return f"{label:<18} n=0"
    p50, p95, p99 = np.percentile(arr_ms, [50, 95, 99])
    mean = float(arr_ms.mean())
    min_v = float(arr_ms.min())
    max_v = float(arr_ms.max())
    stdev = float(arr_ms.std(ddof=0))
    return (
        f"{label:<18} n={arr_ms.size:<5d} mean={mean:8.4f}ms  p50={p50:8.4f}ms  "
        f"p95={p95:8.4f}ms  p99={p99:8.4f}ms  min={min_v:8.4f}ms  max={max_v:8.4f}ms  sd={stdev:8.4f}ms"
    )


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        tok = token.strip()
        if not tok:
            continue
        values.append(float(tok))
    return values


def _parse_str_list(raw: str) -> list[str]:
    values: list[str] = []
    for token in raw.split(","):
        tok = token.strip()
        if tok:
            values.append(tok)
    return values


def _extract_metrics(samples_ns: list[int], budget_ms: float) -> dict[str, float]:
    arr_ms = _ms(samples_ns)
    if arr_ms.size == 0:
        return {
            "count": 0.0,
            "mean_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
            "over_budget": 0.0,
            "over_budget_pct": 0.0,
            "effective_hz": 0.0,
        }
    mean = float(arr_ms.mean())
    p95 = float(np.percentile(arr_ms, 95))
    p99 = float(np.percentile(arr_ms, 99))
    max_v = float(arr_ms.max())
    over_budget = int(np.sum(arr_ms > budget_ms))
    over_budget_pct = (100.0 * over_budget / arr_ms.size) if arr_ms.size else 0.0
    effective_hz = 1000.0 / max(statistics.mean(arr_ms), 1e-9)
    return {
        "count": float(arr_ms.size),
        "mean_ms": mean,
        "p95_ms": p95,
        "p99_ms": p99,
        "max_ms": max_v,
        "over_budget": float(over_budget),
        "over_budget_pct": float(over_budget_pct),
        "effective_hz": float(effective_hz),
    }


def _extract_error_metrics(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mae": 0.0, "rmse": 0.0, "p95": 0.0, "p99": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    mae = float(np.mean(np.abs(arr)))
    rmse = float(np.sqrt(np.mean(arr * arr)))
    p95 = float(np.percentile(np.abs(arr), 95))
    p99 = float(np.percentile(np.abs(arr), 99))
    return {"mae": mae, "rmse": rmse, "p95": p95, "p99": p99}


def _build_synthetic_trace(
    *,
    Ts: float,
    total_steps: int,
    missing_meas_rate: float,
    omega_meas_rate: float,
    theta_noise_std: float,
    omega_noise_std: float,
    seed: int,
) -> list[dict[str, Optional[float]]]:
    rng = np.random.default_rng(seed)
    x_true = np.zeros((3,), dtype=float)
    trace: list[dict[str, Optional[float]]] = []

    for idx in range(total_steps):
        t = idx * Ts
        u_applied = float(0.8 * math.sin(2.0 * math.pi * 0.8 * t) + 0.2 * math.sin(2.0 * math.pi * 0.15 * t))
        u_applied = float(max(-1.0, min(1.0, u_applied)))

        theta = float(x_true[0])
        omega = float(x_true[1])
        d = float(x_true[2])
        theta_next = theta + Ts * omega
        omega_next = omega + Ts * (-0.2 * omega + d + 0.6 * u_applied)
        d_next = d + float(rng.normal(0.0, 0.002))
        x_true[:] = (theta_next, omega_next, d_next)

        theta_meas: Optional[float]
        if float(rng.random()) < missing_meas_rate:
            theta_meas = None
        else:
            theta_meas = float(theta_next + rng.normal(0.0, theta_noise_std))

        omega_meas: Optional[float]
        if float(rng.random()) < omega_meas_rate:
            omega_meas = float(omega_next + rng.normal(0.0, omega_noise_std))
        else:
            omega_meas = None

        trace.append(
            {
                "dt_s": Ts,
                "u_applied": u_applied,
                "theta_true": float(theta_next),
                "omega_true": float(omega_next),
                "theta_meas": theta_meas,
                "omega_meas": omega_meas,
            }
        )

    return trace


def _load_replay_trace(
    *,
    replay_path: Path,
    axis: str,
    default_dt_s: float,
) -> list[dict[str, Optional[float]]]:
    rows: list[dict[str, Optional[float]]] = []
    with replay_path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_no}") from exc
            if not isinstance(payload, Mapping):
                continue
            row_axis = str(payload.get("axis", axis)).strip().lower()
            if row_axis != axis:
                continue
            u_applied = float(payload.get("u_applied", 0.0))
            dt_s = float(payload.get("dt_s", default_dt_s))
            theta_meas_raw = payload.get("theta_meas")
            omega_meas_raw = payload.get("omega_meas")
            theta_true_raw = payload.get("theta_true")
            omega_true_raw = payload.get("omega_true")
            rows.append(
                {
                    "dt_s": dt_s,
                    "u_applied": u_applied,
                    "theta_true": None if theta_true_raw is None else float(theta_true_raw),
                    "omega_true": None if omega_true_raw is None else float(omega_true_raw),
                    "theta_meas": None if theta_meas_raw is None else float(theta_meas_raw),
                    "omega_meas": None if omega_meas_raw is None else float(omega_meas_raw),
                }
            )
    if not rows:
        raise ValueError(f"no replay rows found for axis={axis}")
    return rows


def _run_estimator_variant_benchmark(
    axis: str,
    mpc_cfg: MpcConfig,
    *,
    algorithm: str,
    trace: Sequence[Mapping[str, Optional[float]]],
    warmup: int,
    omega_noise_std: float,
) -> dict[str, Any]:
    registry = available_estimators()
    if algorithm not in registry:
        raise ValueError(f"unknown algorithm: {algorithm}")

    model = MpcAxisModel.from_config(mpc_cfg)
    estimator_cfg = EstimatorVariantConfig(
        A=model.A,
        B=model.B,
        q_theta=float(mpc_cfg.estimator.q_theta),
        q_omega=float(mpc_cfg.estimator.q_omega),
        q_d=float(mpc_cfg.estimator.q_d),
        r_theta=float(mpc_cfg.estimator.r_theta),
        r_omega=max(1e-9, float(omega_noise_std) ** 2),
    )
    estimator = registry[algorithm](estimator_cfg)
    estimator.reset()

    estimator_ns: list[int] = []
    theta_err: list[float] = []
    omega_err: list[float] = []

    for idx, row in enumerate(trace):
        u_applied = float(row.get("u_applied", 0.0) or 0.0)
        theta_meas = row.get("theta_meas")
        omega_meas = row.get("omega_meas")

        t0 = time.perf_counter_ns()
        state = estimator.step(
            u_applied=u_applied,
            theta_meas=None if theta_meas is None else float(theta_meas),
            omega_meas=None if omega_meas is None else float(omega_meas),
        )
        t1 = time.perf_counter_ns()

        if idx >= warmup:
            estimator_ns.append(t1 - t0)
            theta_true = row.get("theta_true")
            omega_true = row.get("omega_true")
            if theta_true is not None and len(state) >= 1:
                theta_err.append(float(state[0]) - float(theta_true))
            if omega_true is not None and len(state) >= 2:
                omega_err.append(float(state[1]) - float(omega_true))

    theta_metrics = _extract_error_metrics(theta_err)
    omega_metrics = _extract_error_metrics(omega_err)

    return {
        "axis": axis,
        "algorithm": algorithm,
        "estimator_ns": estimator_ns,
        "mpc_ns": [],
        "full_ns": list(estimator_ns),
        "theta_err": theta_err,
        "omega_err": omega_err,
        "theta_metrics": theta_metrics,
        "omega_metrics": omega_metrics,
    }


def _execute_once(
    *,
    mode: str,
    algorithm: str,
    control_cfg: ControlConfig,
    mpc_cfg: MpcConfig,
    iterations: int,
    warmup: int,
    missing_meas_rate: float,
    amplitude_rad: float,
    freq_hz: float,
    omega_meas_rate: float,
    theta_noise_std: float,
    omega_noise_std: float,
    replay_file: Optional[Path],
    seed: int,
) -> dict[str, Any]:
    axis_results = []
    start = time.perf_counter()
    for axis_name in ("yaw", "pitch"):
        if mode == "estimator":
            total_steps = warmup + iterations
            if replay_file is not None:
                trace = _load_replay_trace(
                    replay_path=replay_file,
                    axis=axis_name,
                    default_dt_s=float(mpc_cfg.horizon.sample_time_s),
                )
                if len(trace) < total_steps:
                    raise ValueError(
                        f"replay too short for axis={axis_name}: need {total_steps}, got {len(trace)}"
                    )
                trace = trace[:total_steps]
            else:
                trace = _build_synthetic_trace(
                    Ts=float(mpc_cfg.horizon.sample_time_s),
                    total_steps=total_steps,
                    missing_meas_rate=missing_meas_rate,
                    omega_meas_rate=omega_meas_rate,
                    theta_noise_std=theta_noise_std,
                    omega_noise_std=omega_noise_std,
                    seed=seed + (0 if axis_name == "yaw" else 1009),
                )

            result = _run_estimator_variant_benchmark(
                axis_name,
                mpc_cfg,
                algorithm=algorithm,
                trace=trace,
                warmup=warmup,
                omega_noise_std=omega_noise_std,
            )
        else:
            if algorithm != "baseline":
                raise ValueError("full mode currently supports only algorithm=baseline")
            result = _run_axis_benchmark(
                axis_name,
                control_cfg,
                mpc_cfg,
                iterations=iterations,
                warmup=warmup,
                missing_meas_rate=missing_meas_rate,
                amplitude_rad=amplitude_rad,
                freq_hz=freq_hz,
                seed=seed,
            )
        axis_results.append(result)

    elapsed_s = time.perf_counter() - start
    all_est = [x for res in axis_results for x in res["estimator_ns"]]
    all_mpc = [x for res in axis_results for x in res["mpc_ns"]]
    all_full = [x for res in axis_results for x in res["full_ns"]]
    return {
        "axis_results": axis_results,
        "all_est": all_est,
        "all_mpc": all_mpc,
        "all_full": all_full,
        "elapsed_s": elapsed_s,
    }


def _write_sweep_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "algorithm",
        "mode",
        "missing_meas_rate",
        "iterations",
        "warmup",
        "mean_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
        "theta_mae_rad",
        "theta_rmse_rad",
        "theta_p95_abs_rad",
        "omega_mae_rad_s",
        "omega_rmse_rad_s",
        "omega_p95_abs_rad_s",
        "over_budget",
        "over_budget_pct",
        "effective_hz",
        "elapsed_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _run_axis_benchmark(
    axis: str,
    control_cfg: ControlConfig,
    mpc_cfg: MpcConfig,
    *,
    iterations: int,
    warmup: int,
    missing_meas_rate: float,
    amplitude_rad: float,
    freq_hz: float,
    seed: int,
) -> dict[str, Any]:
    controller = MpcAxisController(axis, control_cfg, mpc_cfg)
    controller.reset()

    horizon = int(mpc_cfg.horizon.prediction_horizon)
    Ts = float(mpc_cfg.horizon.sample_time_s)
    rng = np.random.default_rng(seed + (0 if axis == "yaw" else 1009))

    estimator_ns: list[int] = []
    mpc_ns: list[int] = []
    full_ns: list[int] = []

    u_prev = 0.0
    theta_true = 0.0
    omega_true = 0.0

    total = warmup + iterations
    for i in range(total):
        t = i * Ts
        theta_ref_now = amplitude_rad * math.sin(2.0 * math.pi * freq_hz * t)
        omega_ref_now = 2.0 * math.pi * freq_hz * amplitude_rad * math.cos(2.0 * math.pi * freq_hz * t)

        theta_true = theta_true + Ts * omega_true
        omega_true = omega_true + Ts * (-0.2 * omega_true + 0.6 * u_prev)

        meas = theta_true + float(rng.normal(0.0, 0.003))
        if float(rng.random()) < missing_meas_rate:
            meas_value: Optional[float] = None
        else:
            meas_value = float(meas)

        full_t0 = time.perf_counter_ns()
        est_t0 = time.perf_counter_ns()
        xhat = controller.step_estimator(u_prev, meas_value)
        est_t1 = time.perf_counter_ns()

        theta0 = float(xhat[0])
        omega0 = float(omega_ref_now)
        theta_seq_raw = [theta0 + (k * Ts) * omega0 for k in range(horizon)]
        omega_seq_raw = [omega_ref_now for _ in range(horizon)]
        theta_seq = _prepare_sequence(theta_seq_raw, horizon)
        omega_seq = _prepare_sequence(omega_seq_raw, horizon)

        mpc_t0 = time.perf_counter_ns()
        u_cmd, _diag = controller.compute_control(theta_seq, omega_ref_seq=omega_seq)
        mpc_t1 = time.perf_counter_ns()
        full_t1 = time.perf_counter_ns()

        if i >= warmup:
            estimator_ns.append(est_t1 - est_t0)
            mpc_ns.append(mpc_t1 - mpc_t0)
            full_ns.append(full_t1 - full_t0)

        u_prev = float(u_cmd)
        if not math.isfinite(u_prev):
            raise RuntimeError(f"non-finite control output encountered on axis={axis}")

    return {
        "axis": axis,
        "estimator_ns": estimator_ns,
        "mpc_ns": mpc_ns,
        "full_ns": full_ns,
    }


def _run_estimator_only_benchmark(
    axis: str,
    control_cfg: ControlConfig,
    mpc_cfg: MpcConfig,
    *,
    iterations: int,
    warmup: int,
    missing_meas_rate: float,
    seed: int,
) -> dict[str, Any]:
    del control_cfg
    model = MpcAxisModel.from_config(mpc_cfg)
    A = model.A
    B = model.B
    C = model.C
    Q = np.diag(
        [
            float(mpc_cfg.estimator.q_theta),
            float(mpc_cfg.estimator.q_omega),
            float(mpc_cfg.estimator.q_d),
        ]
    )
    R = max(1e-9, float(mpc_cfg.estimator.r_theta))
    x = np.zeros((A.shape[0],), dtype=float)
    P = np.eye(A.shape[0], dtype=float)

    Ts = float(mpc_cfg.horizon.sample_time_s)
    rng = np.random.default_rng(seed + (0 if axis == "yaw" else 1009))

    estimator_ns: list[int] = []
    u_prev = 0.0
    theta_true = 0.0
    omega_true = 0.0

    total = warmup + iterations
    for i in range(total):
        theta_true = theta_true + Ts * omega_true
        omega_true = omega_true + Ts * (-0.2 * omega_true + 0.6 * u_prev)
        meas = theta_true + float(rng.normal(0.0, 0.003))
        meas_value = None if float(rng.random()) < missing_meas_rate else float(meas)

        t0 = time.perf_counter_ns()
        u_vec = np.array([[u_prev]], dtype=float)
        x = A @ x + (B @ u_vec).reshape(x.shape)
        P = A @ P @ A.T + Q
        if meas_value is not None:
            z = float(meas_value)
            innovation = z - float((C @ x.reshape(-1, 1)).item())
            S = float((C @ P @ C.T).item()) + R
            if S > 1e-12:
                K = (P @ C.T) / S
                x = x + (K.flatten() * innovation)
                I = np.eye(A.shape[0], dtype=float)
                P = (I - K @ C) @ P
        state = x
        t1 = time.perf_counter_ns()

        if len(state) >= 2:
            u_prev = float(max(-1.0, min(1.0, -2.0 * state[0] - 0.4 * state[1])))
        else:
            u_prev = 0.0
        if i >= warmup:
            estimator_ns.append(t1 - t0)

    return {
        "axis": axis,
        "estimator_ns": estimator_ns,
        "mpc_ns": [],
        "full_ns": list(estimator_ns),
    }


def main() -> int:
    args = parse_args()

    if args.iterations <= 0:
        raise SystemExit("--iterations must be > 0")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if not (0.0 <= args.missing_meas_rate <= 1.0):
        raise SystemExit("--missing-meas-rate must be in [0, 1]")
    if not (0.0 <= args.omega_meas_rate <= 1.0):
        raise SystemExit("--omega-meas-rate must be in [0, 1]")
    if args.theta_noise_std < 0.0 or args.omega_noise_std < 0.0:
        raise SystemExit("noise std values must be non-negative")

    algorithms = _parse_str_list(args.algorithms)
    if not algorithms:
        raise SystemExit("--algorithms must include at least one estimator name")
    registry = available_estimators()
    unknown = [name for name in algorithms if name not in registry]
    if unknown:
        raise SystemExit(f"unknown estimator algorithms: {unknown}")
    if args.mode == "full":
        algorithms = ["baseline"]
    elif not args.sweep and len(algorithms) > 1:
        raise SystemExit("use --sweep when benchmarking multiple algorithms")

    replay_path = Path(args.replay_file) if args.replay_file else None
    if replay_path is not None and not replay_path.exists():
        raise SystemExit(f"replay file not found: {replay_path}")

    if args.sweep:
        try:
            sweep_rates = _parse_float_list(args.sweep_missing_rates)
        except ValueError as exc:
            raise SystemExit(f"invalid --sweep-missing-rates: {exc}") from exc
        if not sweep_rates:
            raise SystemExit("--sweep-missing-rates must include at least one value")
        for rate in sweep_rates:
            if not (0.0 <= rate <= 1.0):
                raise SystemExit("all sweep missing rates must be in [0, 1]")
    else:
        sweep_rates = []

    config_path = Path(args.config)
    extra_path = Path(args.config_extra) if args.config_extra else None

    try:
        control_cfg, mpc_cfg = _load_control_cfg(config_path, extra_path)
    except Exception as exc:
        print(f"Failed to load config: {exc}")
        return 1

    budget_ms = (1000.0 / control_cfg.loop_hz) if control_cfg.loop_hz else (1000.0 * mpc_cfg.horizon.sample_time_s)

    print("MPC latency benchmark")
    print(f"  config: {config_path}")
    if extra_path is not None:
        print(f"  config-extra: {extra_path}")
    print(f"  axes: yaw,pitch")
    print(
        f"  iterations: {args.iterations}  warmup: {args.warmup}  missing_meas_rate: {args.missing_meas_rate:.3f}  mode: {args.mode}"
    )
    print(
        f"  horizons: Np={mpc_cfg.horizon.prediction_horizon} Nc={mpc_cfg.horizon.control_horizon} Ts={mpc_cfg.horizon.sample_time_s:.6f}s"
    )
    print(f"  control budget per tick: {budget_ms:.4f}ms")
    print(f"  algorithms: {','.join(algorithms)}")
    if replay_path is not None:
        print(f"  replay_file: {replay_path}")
    print("")

    if args.sweep:
        rows: list[dict[str, Any]] = []
        print("[sweep] scenario comparison")
        for alg_idx, algorithm in enumerate(algorithms):
            for rate_idx, rate in enumerate(sweep_rates):
                scenario_name = f"{algorithm}_miss_{rate:.3f}"
                scenario_seed = int(args.seed) + (alg_idx * 100000) + (rate_idx * 1000)
                try:
                    result = _execute_once(
                        mode=args.mode,
                        algorithm=algorithm,
                        control_cfg=control_cfg,
                        mpc_cfg=mpc_cfg,
                        iterations=args.iterations,
                        warmup=args.warmup,
                        missing_meas_rate=float(rate),
                        amplitude_rad=float(args.amplitude_rad),
                        freq_hz=float(args.freq_hz),
                        omega_meas_rate=float(args.omega_meas_rate),
                        theta_noise_std=float(args.theta_noise_std),
                        omega_noise_std=float(args.omega_noise_std),
                        replay_file=replay_path,
                        seed=scenario_seed,
                    )
                except MpcSolverError as exc:
                    print(f"Solver error during scenario={scenario_name}: {exc}")
                    print("Tip: install osqp+scipy or rerun with --mode estimator")
                    return 2
                except Exception as exc:
                    print(f"Scenario failed ({scenario_name}): {exc}")
                    return 2

                metrics = _extract_metrics(result["all_full"], budget_ms)
                theta_err = [x for axis_res in result["axis_results"] for x in axis_res.get("theta_err", [])]
                omega_err = [x for axis_res in result["axis_results"] for x in axis_res.get("omega_err", [])]
                theta_stats = _extract_error_metrics(theta_err)
                omega_stats = _extract_error_metrics(omega_err)
                row = {
                    "scenario": scenario_name,
                    "algorithm": algorithm,
                    "mode": args.mode,
                    "missing_meas_rate": float(rate),
                    "iterations": int(args.iterations),
                    "warmup": int(args.warmup),
                    "mean_ms": metrics["mean_ms"],
                    "p95_ms": metrics["p95_ms"],
                    "p99_ms": metrics["p99_ms"],
                    "max_ms": metrics["max_ms"],
                    "theta_mae_rad": theta_stats["mae"],
                    "theta_rmse_rad": theta_stats["rmse"],
                    "theta_p95_abs_rad": theta_stats["p95"],
                    "omega_mae_rad_s": omega_stats["mae"],
                    "omega_rmse_rad_s": omega_stats["rmse"],
                    "omega_p95_abs_rad_s": omega_stats["p95"],
                    "over_budget": int(metrics["over_budget"]),
                    "over_budget_pct": metrics["over_budget_pct"],
                    "effective_hz": metrics["effective_hz"],
                    "elapsed_s": float(result["elapsed_s"]),
                }
                rows.append(row)
                print(
                    f"- {scenario_name:<26} p95={row['p95_ms']:.4f}ms  theta_mae={row['theta_mae_rad']:.5f}rad  "
                    f"p99={row['p99_ms']:.4f}ms  over_budget={row['over_budget']}"
                )

        print("")
        if args.mode == "estimator":
            print("[sweep] ranked by theta MAE then p95 latency")
            ranked = sorted(rows, key=lambda item: (float(item["theta_mae_rad"]), float(item["p95_ms"])))
        else:
            print("[sweep] ranked by p95 full_step")
            ranked = sorted(rows, key=lambda item: float(item["p95_ms"]))
        for rank, row in enumerate(ranked, start=1):
            if args.mode == "estimator":
                print(
                    f"{rank:>2d}. {row['scenario']:<26} mae={row['theta_mae_rad']:.5f}rad  "
                    f"p95={row['p95_ms']:.4f}ms  max={row['max_ms']:.4f}ms"
                )
            else:
                print(
                    f"{rank:>2d}. {row['scenario']:<26} p95={row['p95_ms']:.4f}ms  "
                    f"mean={row['mean_ms']:.4f}ms  max={row['max_ms']:.4f}ms  over={row['over_budget']}"
                )

        if args.mode == "estimator" and any(row["algorithm"] == "baseline" for row in rows):
            print("\n[sweep] worth-it deltas vs baseline (per missing-rate)")
            baseline_by_rate = {
                float(row["missing_meas_rate"]): row for row in rows if row["algorithm"] == "baseline"
            }
            for row in rows:
                base = baseline_by_rate.get(float(row["missing_meas_rate"]))
                if base is None or row["algorithm"] == "baseline":
                    continue
                base_p95 = max(1e-9, float(base["p95_ms"]))
                base_mae = max(1e-9, float(base["theta_mae_rad"]))
                latency_delta_pct = 100.0 * (float(row["p95_ms"]) - base_p95) / base_p95
                mae_delta_pct = 100.0 * (float(row["theta_mae_rad"]) - base_mae) / base_mae
                verdict = "worth-it" if (mae_delta_pct <= -10.0 and latency_delta_pct <= 30.0) else "not-yet"
                print(
                    f"- {row['scenario']:<26} latency_delta={latency_delta_pct:+.2f}%  "
                    f"mae_delta={mae_delta_pct:+.2f}%  verdict={verdict}"
                )

        if args.csv_out:
            csv_path = Path(args.csv_out)
            _write_sweep_csv(csv_path, rows)
            print(f"\n[sweep] csv written: {csv_path}")

        return 0

    selected_algorithm = algorithms[0]
    try:
        result = _execute_once(
            mode=args.mode,
            algorithm=selected_algorithm,
            control_cfg=control_cfg,
            mpc_cfg=mpc_cfg,
            iterations=args.iterations,
            warmup=args.warmup,
            missing_meas_rate=float(args.missing_meas_rate),
            amplitude_rad=float(args.amplitude_rad),
            freq_hz=float(args.freq_hz),
            omega_meas_rate=float(args.omega_meas_rate),
            theta_noise_std=float(args.theta_noise_std),
            omega_noise_std=float(args.omega_noise_std),
            replay_file=replay_path,
            seed=int(args.seed),
        )
    except MpcSolverError as exc:
        print(f"Solver error while benchmarking: {exc}")
        print("Tip: install osqp+scipy or rerun with --mode estimator")
        return 2
    except Exception as exc:
        print(f"Benchmark failed: {exc}")
        return 2

    axis_results = result["axis_results"]
    elapsed_s = float(result["elapsed_s"])
    all_est = result["all_est"]
    all_mpc = result["all_mpc"]
    all_full = result["all_full"]

    for res in axis_results:
        axis = res["axis"]
        print(f"[{axis}]")
        print(_summary_line("estimator_step", res["estimator_ns"]))
        if args.mode == "full":
            print(_summary_line("mpc_compute", res["mpc_ns"]))
        print(_summary_line("full_step", res["full_ns"]))
        if args.mode == "estimator":
            theta_metrics = res.get("theta_metrics", {})
            omega_metrics = res.get("omega_metrics", {})
            print(
                f"theta_err mae={float(theta_metrics.get('mae', 0.0)):.5f}rad "
                f"rmse={float(theta_metrics.get('rmse', 0.0)):.5f}rad"
            )
            print(
                f"omega_err mae={float(omega_metrics.get('mae', 0.0)):.5f}rad/s "
                f"rmse={float(omega_metrics.get('rmse', 0.0)):.5f}rad/s"
            )
        print("")

    print("[combined]")
    print(_summary_line("estimator_step", all_est))
    if args.mode == "full":
        print(_summary_line("mpc_compute", all_mpc))
    combined_line = _summary_line("full_step", all_full)
    print(combined_line)
    if args.mode == "estimator":
        theta_err = [x for axis_res in axis_results for x in axis_res.get("theta_err", [])]
        omega_err = [x for axis_res in axis_results for x in axis_res.get("omega_err", [])]
        theta_stats = _extract_error_metrics(theta_err)
        omega_stats = _extract_error_metrics(omega_err)
        print(
            f"theta_err combined: mae={theta_stats['mae']:.5f}rad rmse={theta_stats['rmse']:.5f}rad p95_abs={theta_stats['p95']:.5f}rad"
        )
        print(
            f"omega_err combined: mae={omega_stats['mae']:.5f}rad/s rmse={omega_stats['rmse']:.5f}rad/s p95_abs={omega_stats['p95']:.5f}rad/s"
        )

    full_ms = _ms(all_full)
    if full_ms.size:
        p95 = float(np.percentile(full_ms, 95))
        over_budget = int(np.sum(full_ms > budget_ms))
        over_budget_pct = (100.0 * over_budget / full_ms.size) if full_ms.size else 0.0
        effective_hz = 1000.0 / max(statistics.mean(full_ms), 1e-9)
        print("")
        print(f"p95 full_step vs budget: {p95:.4f}ms / {budget_ms:.4f}ms")
        print(f"budget overruns: {over_budget}/{full_ms.size} ({over_budget_pct:.2f}%)")
        print(f"effective compute-only rate: {effective_hz:.2f} Hz")

    print(f"wall-clock benchmark time: {elapsed_s:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
