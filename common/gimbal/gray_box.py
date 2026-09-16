"""Qualified gray-box gimbal plant used by offline and live simulation."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Mapping, Sequence


SUPPORTED_MODEL = "discrete-first-order-asymmetric"


@dataclass(frozen=True)
class AxisPlant:
    """Continuous-time realization of one qualified fitted gimbal axis."""

    axis: str
    a_f: float
    b_pos: float
    b_neg: float
    disturbance: float
    delay_s: float
    fit_dt_s: float
    source_model: str

    def advance(
        self,
        theta: float,
        omega: float,
        command: float,
        dt_s: float,
    ) -> tuple[float, float]:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            return theta, omega
        gain = self.b_pos if command >= 0.0 else self.b_neg
        forcing = gain * command + self.disturbance
        pole = math.exp(-self.a_f * dt_s)
        steady_omega = forcing / self.a_f
        omega_next = pole * omega + (1.0 - pole) * steady_omega
        integral_factor = (1.0 - pole) / self.a_f
        theta_next = (
            theta
            + integral_factor * omega
            + (dt_s - integral_factor) * steady_omega
        )
        return theta_next, omega_next


def _selected_candidate(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = str(entry.get("selected_model", ""))
    candidates = entry.get("model_comparison")
    if not isinstance(candidates, Sequence):
        raise ValueError("fit axis has no model_comparison list")
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate.get("model") == selected:
            return candidate
    raise ValueError(f"selected model {selected!r} is absent from model_comparison")


def _continuous_decay_rate(c_omega: float, fit_dt_s: float) -> float:
    """Match the fitter's stable-pole conversion, including its Euler fallback."""

    if not (-1.0 < c_omega < 1.0 and fit_dt_s > 0.0):
        raise ValueError("invalid discrete pole or sample time")
    if c_omega > 0.0:
        return -math.log(c_omega) / fit_dt_s
    return (1.0 - c_omega) / fit_dt_s


def _validation_fit_candidates(
    reference: str,
    validation_path: Path,
) -> tuple[Path, ...]:
    referenced = Path(reference).expanduser()
    if referenced.is_absolute():
        return (referenced.resolve(),)
    candidates = [referenced.resolve()]
    for parent in validation_path.resolve().parents:
        candidate = (parent / referenced).resolve()
        if candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def load_qualified_plants(
    fit_path: Path | str,
    validation_path: Path | str,
) -> dict[str, AxisPlant]:
    """Load a fit only when its independent validation explicitly qualifies it."""

    fit_path = Path(fit_path).expanduser().resolve()
    validation_path = Path(validation_path).expanduser().resolve()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    qualification = validation.get("qualification")
    if not isinstance(qualification, Mapping) or qualification.get("qualified") is not True:
        raise ValueError("plant validation report is not qualified")
    if validation.get("format") != "idcs.gimbal_frozen_fit_validation":
        raise ValueError("plant validation report has an unsupported format")
    if qualification.get("independent_validation") is not True:
        raise ValueError("plant validation report is not independent")
    validated_fit = validation.get("fit_report")
    if not validated_fit or fit_path not in _validation_fit_candidates(
        str(validated_fit), validation_path
    ):
        raise ValueError("plant validation report does not reference the supplied fit report")

    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    axes = fit.get("axes")
    validation_axes = validation.get("axes")
    if not isinstance(axes, Mapping):
        raise ValueError("fit report has no axes mapping")
    if not isinstance(validation_axes, Mapping):
        raise ValueError("plant validation report has no axes mapping")

    plants: dict[str, AxisPlant] = {}
    for axis in ("yaw", "pitch"):
        entry = axes.get(axis)
        validation_entry = validation_axes.get(axis)
        if not isinstance(entry, Mapping):
            raise ValueError(f"fit report has no {axis} axis")
        if (
            not isinstance(validation_entry, Mapping)
            or validation_entry.get("qualified") is not True
        ):
            raise ValueError(f"plant validation report does not qualify {axis}")
        candidate = _selected_candidate(entry)
        model = str(candidate.get("model", ""))
        if model != SUPPORTED_MODEL:
            raise ValueError(f"unsupported selected plant model for {axis}: {model!r}")
        if str(validation_entry.get("selected_model", "")) != model:
            raise ValueError(f"plant validation model does not match the {axis} fit")
        coeffs = candidate.get("coefficients")
        if not isinstance(coeffs, Mapping):
            raise ValueError(f"selected {axis} model has no coefficients")
        c_omega = float(coeffs["c_omega"])
        fit_dt_s = float(coeffs["dt_s"])
        try:
            a_f = _continuous_decay_rate(c_omega, fit_dt_s)
        except ValueError as exc:
            raise ValueError(
                f"selected {axis} model has invalid discrete pole or sample time"
            ) from exc
        scale = a_f / (1.0 - c_omega)
        plants[axis] = AxisPlant(
            axis=axis,
            a_f=a_f,
            b_pos=float(coeffs["c_u_pos"]) * scale,
            b_neg=-float(coeffs["c_u_neg"]) * scale,
            disturbance=float(coeffs.get("bias", 0.0)) * scale,
            delay_s=float(candidate.get("delay_s", 0.0)),
            fit_dt_s=fit_dt_s,
            source_model=model,
        )
    return plants


@dataclass(frozen=True)
class AxisPlantState:
    position: float
    rate: float


class _AxisRuntime:
    def __init__(self, plant: AxisPlant) -> None:
        self.plant = plant
        self.position = 0.0
        self.rate = 0.0
        self.elapsed_s = 0.0
        self.command_history: Deque[tuple[float, float]] = deque([(0.0, 0.0)])

    def reset(self, position: float, rate: float) -> None:
        self.position = float(position)
        self.rate = float(rate)
        self.elapsed_s = 0.0
        self.command_history.clear()
        self.command_history.append((0.0, 0.0))

    def synchronize(self, position: float, rate: float) -> None:
        self.position = float(position)
        self.rate = float(rate)

    def advance(self, command: float, dt_s: float) -> AxisPlantState:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            return AxisPlantState(self.position, self.rate)
        command = float(command)
        if not math.isfinite(command):
            raise ValueError(f"non-finite {self.plant.axis} plant command")
        self.command_history.append((self.elapsed_s, command))
        delayed_at = self.elapsed_s - self.plant.delay_s
        delayed_command = self.command_history[0][1]
        for command_time, historical_command in reversed(self.command_history):
            if command_time <= delayed_at:
                delayed_command = historical_command
                break
        keep_after = delayed_at - max(self.plant.fit_dt_s, dt_s)
        while (
            len(self.command_history) > 2
            and self.command_history[1][0] < keep_after
        ):
            self.command_history.popleft()
        self.position, self.rate = self.plant.advance(
            self.position,
            self.rate,
            delayed_command,
            dt_s,
        )
        self.elapsed_s += dt_s
        return AxisPlantState(self.position, self.rate)


class QualifiedGimbalPlant:
    """Stateful yaw/pitch simulation backed by one qualified fit report."""

    def __init__(
        self,
        plants: Mapping[str, AxisPlant],
        *,
        fit_report: Path | str | None = None,
        validation_report: Path | str | None = None,
    ) -> None:
        self._yaw = _AxisRuntime(plants["yaw"])
        self._pitch = _AxisRuntime(plants["pitch"])
        self.fit_report = str(Path(fit_report).resolve()) if fit_report else None
        self.validation_report = (
            str(Path(validation_report).resolve()) if validation_report else None
        )

    @classmethod
    def from_reports(
        cls,
        fit_report: Path | str,
        validation_report: Path | str,
    ) -> "QualifiedGimbalPlant":
        plants = load_qualified_plants(fit_report, validation_report)
        return cls(
            plants,
            fit_report=fit_report,
            validation_report=validation_report,
        )

    def reset(
        self,
        *,
        yaw: float,
        pitch: float,
        yaw_rate: float = 0.0,
        pitch_rate: float = 0.0,
    ) -> None:
        self._yaw.reset(yaw, yaw_rate)
        self._pitch.reset(pitch, pitch_rate)

    def synchronize(
        self,
        *,
        yaw: float,
        pitch: float,
        yaw_rate: float,
        pitch_rate: float,
    ) -> None:
        self._yaw.synchronize(yaw, yaw_rate)
        self._pitch.synchronize(pitch, pitch_rate)

    def advance(
        self,
        yaw_command: float,
        pitch_command: float,
        dt_s: float,
    ) -> tuple[AxisPlantState, AxisPlantState]:
        return (
            self._yaw.advance(yaw_command, dt_s),
            self._pitch.advance(pitch_command, dt_s),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "mode": "qualified_gray_box",
            "fit_report": self.fit_report,
            "validation_report": self.validation_report,
            "axes": {
                "yaw": {
                    "model": self._yaw.plant.source_model,
                    "delay_s": self._yaw.plant.delay_s,
                    "fit_dt_s": self._yaw.plant.fit_dt_s,
                },
                "pitch": {
                    "model": self._pitch.plant.source_model,
                    "delay_s": self._pitch.plant.delay_s,
                    "fit_dt_s": self._pitch.plant.fit_dt_s,
                },
            },
        }
