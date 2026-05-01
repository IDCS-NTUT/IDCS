import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

if torch is not None:
    from jetson.swarm_policy_model import create_swarm_policy_model


@unittest.skipUnless(torch is not None, "torch is not installed")
class SwarmPolicyModelTests(unittest.TestCase):
    def test_forward_masks_invalid_targets(self) -> None:
        selector = create_swarm_policy_model(
            target_feature_size=10,
            global_feature_size=4,
            hidden_size=16,
            context_size=8,
            device=torch.device("cpu"),
        )
        target_features = torch.zeros((2, 4, 10), dtype=torch.float32)
        global_features = torch.zeros((2, 4), dtype=torch.float32)
        target_mask = torch.tensor(
            [[True, True, False, False], [True, False, False, False]],
            dtype=torch.bool,
        )

        logits, value_preds, class_logits = selector.forward(
            target_features,
            global_features,
            target_mask,
        )

        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(tuple(value_preds.shape), (2, 4))
        self.assertEqual(tuple(class_logits.shape), (2, 4, 3))
        self.assertLess(float(logits[0, 2].item()), -1e20)
        self.assertLess(float(logits[1, 1].item()), -1e20)
        self.assertGreater(float(value_preds[0, 2].item()), 1e20)
        self.assertGreater(float(value_preds[1, 1].item()), 1e20)

    def test_predict_action_numpy_returns_valid_index(self) -> None:
        selector = create_swarm_policy_model(
            target_feature_size=10,
            global_feature_size=4,
            hidden_size=16,
            context_size=8,
            device=torch.device("cpu"),
        )
        target_features = np.zeros((1, 3, 10), dtype=np.float32)
        global_features = np.zeros((1, 4), dtype=np.float32)
        target_mask = np.array([[True, True, False]], dtype=bool)

        actions, probs, value_preds = selector.predict_action_numpy(
            target_features,
            global_features,
            target_mask,
        )

        self.assertEqual(actions.shape, (1,))
        self.assertIn(int(actions[0]), {0, 1})
        self.assertEqual(probs.shape, (1, 3))
        self.assertEqual(value_preds.shape, (1, 3))
        self.assertAlmostEqual(float(probs[0, 2]), 0.0, places=6)
        logits, _, class_probs = selector.predict_outputs_numpy(
            target_features,
            global_features,
            target_mask,
        )
        self.assertEqual(logits.shape, (1, 3))
        self.assertEqual(class_probs.shape, (1, 3, 3))

    def test_predict_action_numpy_supports_value_rerank(self) -> None:
        selector = create_swarm_policy_model(
            target_feature_size=10,
            global_feature_size=4,
            hidden_size=16,
            context_size=8,
            device=torch.device("cpu"),
        )
        target_features = np.zeros((1, 3, 10), dtype=np.float32)
        global_features = np.zeros((1, 4), dtype=np.float32)
        target_mask = np.array([[True, True, False]], dtype=bool)

        actions, probs, value_preds = selector.predict_action_numpy(
            target_features,
            global_features,
            target_mask,
            use_value_rerank=True,
            rerank_topk=2,
        )

        self.assertEqual(actions.shape, (1,))
        self.assertIn(int(actions[0]), {0, 1})
        self.assertEqual(probs.shape, (1, 3))
        self.assertEqual(value_preds.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
