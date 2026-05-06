import unittest

from common.threat_calc import TargetThreatState


class TargetThreatStateTests(unittest.TestCase):
    def test_update_allows_position_only_defaults(self) -> None:
        state = TargetThreatState(
            target_id=7,
            zone_radii={"warning": 10.0},
            asset_xy=(0.0, 0.0),
        )

        metrics = state.update(current_xy=(3.0, 4.0), current_time=1.5)

        self.assertEqual(metrics["center_x"], 0.0)
        self.assertEqual(metrics["center_y"], 0.0)
        self.assertEqual(metrics["bbox_width"], 0.0)
        self.assertEqual(metrics["bbox_height"], 0.0)
        self.assertEqual(metrics["confidence"], 1.0)
        self.assertEqual(metrics["distance_to_asset"], 5.0)

    def test_update_rejects_partial_bbox_with_non_positive_size(self) -> None:
        state = TargetThreatState(
            target_id=8,
            zone_radii={"warning": 10.0},
            asset_xy=(0.0, 0.0),
        )

        with self.assertRaises(ValueError):
            state.update(
                current_xy=(1.0, 1.0),
                current_time=2.0,
                bbox_x=10.0,
                bbox_y=20.0,
                bbox_width=0.0,
                bbox_height=5.0,
            )

    def test_update_accumulates_zone_dwell_using_frame_delta(self) -> None:
        state = TargetThreatState(
            target_id=9,
            zone_radii={"warning": 10.0},
            asset_xy=(0.0, 0.0),
        )

        first = state.update(current_xy=(3.0, 4.0), current_time=1.0)
        second = state.update(current_xy=(3.0, 4.0), current_time=2.0)
        third = state.update(current_xy=(3.0, 4.0), current_time=3.0)

        self.assertEqual(first["time_inside_zone"], 0.0)
        self.assertEqual(second["time_inside_zone"], 1.0)
        self.assertEqual(third["time_inside_zone"], 2.0)
        self.assertEqual(state.get_total_zone_time("warning"), 2.0)
