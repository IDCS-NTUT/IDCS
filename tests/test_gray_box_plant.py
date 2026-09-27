from __future__ import annotations

import math
from pathlib import Path

import pytest

from common.gimbal.gray_box import AxisPlant, QualifiedGimbalPlant, load_qualified_plants
from pc.sim_camera import SimCamera


def _plant(axis: str, *, delay_s: float = 0.0) -> AxisPlant:
    return AxisPlant(
        axis=axis,
        a_f=2.0,
        b_pos=2.0,
        b_neg=1.0,
        disturbance=0.0,
        delay_s=delay_s,
        fit_dt_s=0.05,
        source_model="test-asymmetric",
    )


def _gimbal(*, delay_s: float = 0.0) -> QualifiedGimbalPlant:
    return QualifiedGimbalPlant(
        {
            "yaw": _plant("yaw", delay_s=delay_s),
            "pitch": _plant("pitch", delay_s=delay_s),
        }
    )


def test_runtime_uses_directional_gain_and_realized_rate() -> None:
    positive = _gimbal()
    negative = _gimbal()
    positive.reset(yaw=0.0, pitch=0.0)
    negative.reset(yaw=0.0, pitch=0.0)

    positive_yaw, _ = positive.advance(1.0, 0.0, 0.1)
    negative_yaw, _ = negative.advance(-1.0, 0.0, 0.1)

    assert positive_yaw.rate == pytest.approx(1.0 - math.exp(-0.2))
    assert abs(negative_yaw.rate) == pytest.approx(0.5 * positive_yaw.rate)
    assert positive_yaw.position > 0.0
    assert negative_yaw.position < 0.0


def test_runtime_applies_fitted_command_delay() -> None:
    plant = _gimbal(delay_s=0.1)
    plant.reset(yaw=0.0, pitch=0.0)

    first, _ = plant.advance(1.0, 0.0, 0.05)
    second, _ = plant.advance(1.0, 0.0, 0.05)
    third, _ = plant.advance(1.0, 0.0, 0.05)

    assert first.rate == 0.0
    assert second.rate == 0.0
    assert third.rate > 0.0


def test_sim_camera_reports_realized_plant_state_and_resets_from_cam_state() -> None:
    cam = SimCamera(
        width=320,
        height=240,
        renderer_name="cpu",
        debug=False,
        plant_model=_gimbal(),
    )

    cam.apply_control_rates(1.0, -1.0, 0.1)
    moved = cam.get_pose()
    assert 0.0 < moved["pan_rate"] < 1.0
    assert -1.0 < moved["tilt_rate"] < 0.0
    assert moved["pan"] < 0.1

    cam.apply_cam_state(pan=0.4, tilt=-0.2, pan_rate=0.05, tilt_rate=-0.03)
    reset = cam.get_pose()
    assert reset == pytest.approx(
        {"pan": 0.4, "tilt": -0.2, "pan_rate": 0.05, "tilt_rate": -0.03}
    )
    cam.apply_control_rates(0.0, 0.0, 0.1)
    after = cam.get_pose()
    assert after["pan"] > 0.4
    assert 0.0 < after["pan_rate"] < 0.05


def test_repository_fit_is_qualified_and_loadable() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = root / "artifacts/gimbal_fit/controller_sysid_pid_range_wire_20260914"
    plants = load_qualified_plants(
        artifact / "fit_report.json",
        artifact / "independent_validation_report.json",
    )

    assert set(plants) == {"yaw", "pitch"}
    assert plants["yaw"].source_model == "discrete-first-order-asymmetric"
    assert plants["pitch"].delay_s == 0.0


def test_deadband_models_load_and_suppress_small_commands() -> None:
    import pytest

    from common.gimbal.gray_box import AxisPlant

    plant = AxisPlant("pitch", a_f=100.0, b_pos=95.0, b_neg=95.0, disturbance=0.0, delay_s=0.0,
                      fit_dt_s=0.03, source_model="discrete-first-order-deadband",
                      deadband_pos=0.11, deadband_neg=0.11)
    assert plant.effective_command(0.1) == 0.0 and plant.effective_command(-0.05) == 0.0
    assert plant.effective_command(0.5) == pytest.approx(0.39)
    assert plant.effective_command(-0.5) == pytest.approx(-0.39)
    theta, omega = 0.0, 0.0
    for _ in range(100):
        theta, omega = plant.advance(theta, omega, 0.1, 0.01)
    assert theta == 0.0  # inside the deadband: no motion


def test_loader_realizes_the_fitted_deadband_model(tmp_path) -> None:
    import json

    import pytest

    from common.gimbal.gray_box import load_qualified_plants

    coeff = {"bias": 0.0, "c_omega": 0.02, "c_u": 0.9, "deadband": 0.11, "dt_s": 0.03}
    axes = {a: {"selected_model": "discrete-first-order-deadband",
                "model_comparison": [{"model": "discrete-first-order-deadband", "delay_s": 0.0,
                                      "coefficients": coeff}]} for a in ("yaw", "pitch")}
    fit = tmp_path / "fit_report.json"
    fit.write_text(json.dumps({"axes": axes}))
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps({
        "format": "idcs.gimbal_frozen_fit_validation", "fit_report": str(fit),
        "qualification": {"qualified": True, "independent_validation": True},
        "axes": {a: {"qualified": True, "selected_model": "discrete-first-order-deadband"} for a in ("yaw", "pitch")},
    }))
    plants = load_qualified_plants(fit, validation)
    assert plants["pitch"].deadband_pos == plants["pitch"].deadband_neg == pytest.approx(0.11)
    assert plants["pitch"].b_pos == pytest.approx(plants["pitch"].b_neg)
