from pathlib import Path

from common.config_sync import merge_config_maps, parse_config_text
from common.control import ControlConfig
from common.schemas import Box, DetectionMsg
from jetson.deepstream.target_selection import DeepStreamTargetSelector, normalize_message_class_labels


def test_person_only_frame_is_unselected_without_loading_learned_runtime():
    selector = DeepStreamTargetSelector.from_paths(
        [
            Path("configs/network.yaml"),
            Path("configs/perception.yaml"),
            Path("configs/control.yaml"),
            Path("configs/system.yaml"),
        ]
    )
    message = DetectionMsg(
        frame_id=1,
        src_ts_ms=1,
        rx_ts_ms=2,
        infer_ts_ms=3,
        img_w=1280,
        img_h=720,
        boxes=[Box(x=0.4, y=0.4, w=0.1, h=0.2, cls="1", conf=0.9, track_id=7)],
    )

    selector.select(message, now_s=1.0)

    assert message.boxes[0].cls == "person"
    assert message.target_idx is None
    assert message.target_track_id is None
    assert selector._planner is None


def test_person_sim_override_preserves_the_enabled_learned_policy():
    paths = [
        Path("configs/network.yaml"),
        Path("configs/perception.yaml"),
        Path("configs/control.yaml"),
        Path("configs/system.yaml"),
        Path("configs/deepstream_person_sim_validation.yaml"),
    ]
    config = merge_config_maps(
        *(parse_config_text(path.read_text(encoding="utf-8"), str(path)) for path in paths)
    )

    control = ControlConfig.from_raw_config(config, (1280, 720))

    assert control.swarm_eval.enabled
    assert control.swarm_eval.excluded_target_classes == ("drone",)
    assert control.swarm_eval.learned_model.enabled
    assert control.swarm_eval.learned_model.backend == "torch"
    assert control.swarm_eval.learned_model.max_update_rate_hz == 10.0


def test_label_normalization_is_available_before_async_policy_results():
    message = DetectionMsg(
        frame_id=1, src_ts_ms=1, rx_ts_ms=2, infer_ts_ms=3, img_w=1280, img_h=720,
        boxes=[Box(x=0, y=0, w=.1, h=.1, cls="1", conf=.9)],
    )

    normalize_message_class_labels(message, {"0": "drone", "1": "person"})

    assert message.boxes[0].cls == "person"
