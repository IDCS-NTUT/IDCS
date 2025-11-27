import time
import unittest

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from common.control import ControlDebugOverlayConfig
from common.schemas import ControlCmd, MpcAxisDiagnostic

from pc.ui import MpcDebugOverlay  # noqa: E402  (cv2 import guarded above)


class MpcDebugOverlayTests(unittest.TestCase):
    def test_overlay_records_history_and_renders(self) -> None:
        cfg = ControlDebugOverlayConfig(
            enabled=True,
            history_window_s=0.5,
            opacity=0.9,
            bar_height_px=24,
            show_terms=("theta", "omega"),
        )
        overlay = MpcDebugOverlay(cfg)

        cmd = ControlCmd(
            frame_id=1,
            src_ts_ms=0,
            cmd_ts_ms=1,
            target_ok=True,
            target_uv=(0.0, 0.0),
            err_uv=(0.0, 0.0),
            err_rad=(0.0, 0.0),
            pan_rate_cmd=0.0,
            tilt_rate_cmd=0.0,
            controller_mode="mpc",
            mpc={
                "yaw": MpcAxisDiagnostic(
                    status="optimal",
                    cost=0.5,
                    u0=0.2,
                    terms={"theta": 0.3, "omega": 0.1},
                )
            },
        )

        now = time.time()
        overlay.ingest(cmd, now)
        self.assertGreaterEqual(len(overlay._history["yaw"]), 1)  # type: ignore[attr-defined]

        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        overlay.render(frame, now + 0.1)
        self.assertGreaterEqual(len(overlay._history["yaw"]), 1)  # type: ignore[attr-defined]

        overlay.render(frame, now + 1.0)
        self.assertEqual(len(overlay._history["yaw"]), 0)  # type: ignore[attr-defined]

    def test_overlay_handles_signed_terms(self) -> None:
        cfg = ControlDebugOverlayConfig(
            enabled=True,
            history_window_s=1.0,
            opacity=0.9,
            bar_height_px=24,
            show_terms=("theta", "theta_linear", "slew_linear"),
        )
        overlay = MpcDebugOverlay(cfg)

        cmd = ControlCmd(
            frame_id=2,
            src_ts_ms=0,
            cmd_ts_ms=1,
            target_ok=True,
            target_uv=(0.0, 0.0),
            err_uv=(0.0, 0.0),
            err_rad=(0.0, 0.0),
            pan_rate_cmd=0.0,
            tilt_rate_cmd=0.0,
            controller_mode="mpc",
            mpc={
                "yaw": MpcAxisDiagnostic(
                    status="optimal",
                    cost=1.2,
                    u0=-0.1,
                    terms={"theta": -0.5, "theta_linear": 0.25, "slew_linear": -0.35},
                )
            },
        )

        now = time.time()
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        overlay.ingest(cmd, now)
        overlay.render(frame, now + 0.05)

        max_total = overlay._max_total("yaw")  # type: ignore[attr-defined]
        self.assertGreater(max_total, 0.0)
        self.assertAlmostEqual(max_total, 1.10, places=2)

    def test_overlay_flips_theta_sign(self) -> None:
        cfg = ControlDebugOverlayConfig(
            enabled=True,
            history_window_s=1.0,
            opacity=0.9,
            bar_height_px=24,
            show_terms=("theta", "omega"),
        )
        overlay = MpcDebugOverlay(cfg)

        cmd = ControlCmd(
            frame_id=3,
            src_ts_ms=0,
            cmd_ts_ms=1,
            target_ok=True,
            target_uv=(0.0, 0.0),
            err_uv=(0.0, 0.0),
            err_rad=(0.0, 0.0),
            pan_rate_cmd=0.0,
            tilt_rate_cmd=0.0,
            controller_mode="mpc",
            mpc={
                "yaw": MpcAxisDiagnostic(
                    status="optimal",
                    cost=0.7,
                    u0=0.05,
                    terms={"theta": 0.4, "omega": -0.2},
                )
            },
        )

        now = time.time()
        overlay.ingest(cmd, now)

        sample = overlay._latest_sample("yaw")  # type: ignore[attr-defined]
        assert sample is not None
        self.assertAlmostEqual(sample.terms["theta"], -0.4)
        self.assertAlmostEqual(sample.terms["omega"], -0.2)
