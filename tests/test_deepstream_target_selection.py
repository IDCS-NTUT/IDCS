import queue
from pathlib import Path
from types import SimpleNamespace

from common.config import load_config_bundle
from common.control import ControlConfig
from common.schemas import Box, DetectionMsg
from common.synthetic_perception import load_synthetic_scenario, snapshot_at
from jetson.deepstream import async_target_selection as async_module
from jetson.deepstream.async_target_selection import AsyncDeepStreamTargetSelector
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
    config = load_config_bundle(paths).mutable_copy()

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


def test_v2_selection_uses_guaranteed_synthetic_track_without_model_runtime():
    paths = [
        Path("configs/network.yaml"),
        Path("configs/perception.yaml"),
        Path("configs/control.yaml"),
        Path("configs/system.yaml"),
    ]
    config = load_config_bundle(paths).data
    calls = []

    class DeterministicPlanner:
        def update_and_select(self, message, **kwargs):
            calls.append((message, kwargs))
            return SimpleNamespace(chosen_box_index=0)

    selector = DeepStreamTargetSelector(
        config,
        planner_factory=lambda _control: DeterministicPlanner(),
    )
    scenario = load_synthetic_scenario(
        Path("tests/fixtures/synthetic_tracking_v1.json")
    )
    source = snapshot_at(scenario, 2)

    result = selector.select_snapshot(source, now_s=12.5)

    assert source.selection is None
    assert result.selection is not None
    assert result.selection.track_id == 41
    assert result.selection.source_frame_id == 2
    assert result.selection.selected_time_ns == 12_500_000_000
    assert calls[0][0].boxes[0].track_id == 41
    assert calls[0][0].boxes[0].cls == "drone"


def test_async_selector_passes_one_hashed_config_snapshot_to_worker(monkeypatch):
    class FakeQueue:
        def __init__(self):
            self.values = []

        def put_nowait(self, value):
            self.values.append(value)

        def get_nowait(self):
            raise queue.Empty

    class FakeProcess:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon
            self.alive = False

        def start(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            self.alive = False

        def terminate(self):
            self.alive = False

    class FakeContext:
        def __init__(self):
            self.process = None

        def Queue(self, maxsize):
            return FakeQueue()

        def Process(self, **kwargs):
            self.process = FakeProcess(**kwargs)
            return self.process

    context = FakeContext()
    monkeypatch.setattr(async_module.mp, "get_context", lambda _method: context)
    selector = AsyncDeepStreamTargetSelector([
        Path("configs/network.yaml"),
        Path("configs/perception.yaml"),
        Path("configs/control.yaml"),
        Path("configs/system.yaml"),
    ])

    assert context.process is not None
    config_snapshot, digest, _requests, _results = context.process.args
    assert isinstance(config_snapshot, dict)
    assert len(digest) == 64
    assert selector.report()["config_digest"] == digest

    selector.close()
