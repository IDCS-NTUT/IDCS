import unittest

from common.schemas import Box
from jetson.multi_target_tracker import (
    active_track_id_matches,
    assign_active_track_id_to_spatial_match,
    update_active_track_hit_streak,
)


class TrackIdentityHelpersTest(unittest.TestCase):
    def test_missing_observed_id_does_not_clear_active_id(self) -> None:
        hits, active_id = update_active_track_hit_streak(
            active_track_id=7,
            observed_track_id=None,
            hits=3,
        )

        self.assertEqual(hits, 0)
        self.assertEqual(active_id, 7)

    def test_hit_streak_latches_and_resets_on_real_id_changes(self) -> None:
        hits, active_id = update_active_track_hit_streak(
            active_track_id=None,
            observed_track_id=5,
            hits=0,
        )
        self.assertEqual((hits, active_id), (1, 5))

        hits, active_id = update_active_track_hit_streak(
            active_track_id=active_id,
            observed_track_id=5,
            hits=hits,
        )
        self.assertEqual((hits, active_id), (2, 5))

        hits, active_id = update_active_track_hit_streak(
            active_track_id=active_id,
            observed_track_id=6,
            hits=hits,
        )
        self.assertEqual((hits, active_id), (1, 6))

    def test_active_match_requires_a_real_observed_id(self) -> None:
        self.assertFalse(active_track_id_matches(7, None))
        self.assertFalse(active_track_id_matches(7, 8))
        self.assertTrue(active_track_id_matches(7, 7))
        self.assertTrue(active_track_id_matches(None, 7))

    def test_spatial_assignment_can_override_wrong_heartbeat_id(self) -> None:
        boxes = [
            Box(x=0.34, y=0.38, w=0.08, h=0.08, conf=0.9, cls="drone", track_id=22),
            Box(x=0.70, y=0.55, w=0.10, h=0.10, conf=0.8, cls="drone", track_id=7),
        ]

        assigned = assign_active_track_id_to_spatial_match(
            boxes,
            active_track_id=7,
            img_w=1280,
            img_h=720,
            last_target_uv=(486.4, 302.4),
            last_target_box_xyxy=(435.2, 273.6, 537.6, 331.2),
            gate_px=0.0,
            allow_replace=True,
        )

        self.assertTrue(assigned)
        self.assertEqual(boxes[0].track_id, 7)
        self.assertIsNone(boxes[1].track_id)

    def test_spatial_assignment_preserves_existing_ids_without_replace(self) -> None:
        boxes = [
            Box(x=0.34, y=0.38, w=0.08, h=0.08, conf=0.9, cls="drone", track_id=22),
        ]

        assigned = assign_active_track_id_to_spatial_match(
            boxes,
            active_track_id=7,
            img_w=1280,
            img_h=720,
            last_target_uv=(486.4, 302.4),
            last_target_box_xyxy=(435.2, 273.6, 537.6, 331.2),
            gate_px=0.0,
            allow_replace=False,
        )

        self.assertFalse(assigned)
        self.assertEqual(boxes[0].track_id, 22)


if __name__ == "__main__":
    unittest.main()
