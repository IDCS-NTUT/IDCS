"""Tests for threat inference and evaluation modules."""

import sys
from pathlib import Path
import numpy as np

# Add repo to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from jetson.threat_inference import ThreatInferenceEngine, ThreatMetricsWindow


def test_threat_inference_engine():
    """Test ONNX model loading and inference."""
    model_path = Path(__file__).resolve().parents[1] / "models" / "threat_model.onnx"
    
    if not model_path.exists():
        print(f"⊘ SKIP: Model not found at {model_path}")
        return True
    
    try:
        engine = ThreatInferenceEngine(model_path)
    except ImportError:
        print("⊘ SKIP: ONNX Runtime not available")
        return True
    
    # Test single sample inference
    features = np.random.randn(1, 176).astype(np.float32)
    class_id, confidence, probs = engine.predict(features)
    
    assert 0 <= class_id <= 2, f"Invalid class_id: {class_id}"
    assert 0 <= confidence <= 1, f"Invalid confidence: {confidence}"
    assert probs.shape == (3,), f"Invalid probs shape: {probs.shape}"
    assert np.isclose(np.sum(probs), 1.0), f"Probs don't sum to 1: {np.sum(probs)}"
    
    print("✓ Single sample inference")
    
    # Test batch inference
    features_batch = np.random.randn(5, 176).astype(np.float32)
    class_ids, confidences, probs_batch = engine.predict_batch(features_batch)
    
    assert class_ids.shape == (5,), f"Invalid class_ids shape: {class_ids.shape}"
    assert confidences.shape == (5,), f"Invalid confidences shape: {confidences.shape}"
    assert probs_batch.shape == (5, 3), f"Invalid probs_batch shape: {probs_batch.shape}"
    
    print("✓ Batch inference")
    
    # Test threatening probability
    threaten_prob = engine.predict_threaten_probability(features)
    assert 0 <= threaten_prob <= 1, f"Invalid threatening probability: {threaten_prob}"
    
    print("✓ Threatening probability")
    
    # Test input shape variants
    # [16, 11] -> should be reshaped to [1, 176]
    features_windowed = np.random.randn(16, 11).astype(np.float32)
    class_id, confidence, probs = engine.predict(features_windowed)
    assert 0 <= class_id <= 2, f"Failed to handle [16, 11] shape"
    
    print("✓ Windowed input shape handling")
    
    return True


def test_threat_metrics_window():
    """Test metrics window buffer."""
    window = ThreatMetricsWindow(max_window_size=16)
    
    assert window.size() == 0, "Window should start empty"
    assert not window.is_full(), "Empty window should not be full"
    
    # Add frames
    for i in range(10):
        metrics = {
            "center_x": 0.5 + i * 0.01,
            "center_y": 0.5,
            "bbox_w": 0.1,
            "bbox_h": 0.2,
            "velocity_x": 0.5,
            "velocity_y": 0.0,
            "confidence": 0.95,
            "distance": 5.0 + i,
            "distance_rate": -0.5,
            "zone_level": 2.0,
            "dwell_time": i * 0.033,
        }
        window.add_frame_metrics(metrics)
    
    assert window.size() == 10, f"Window size should be 10, got {window.size()}"
    assert not window.is_full(), "Window should not be full at 10 frames"
    
    print("✓ Add frames")
    
    # Get features
    features = window.get_features()
    assert features is not None, "Features should not be None"
    assert features.shape == (10, 11), f"Invalid features shape: {features.shape}"
    assert np.isfinite(features).all(), "Features contain non-finite values"
    
    print("✓ Get features")
    
    # Fill window
    for i in range(10, 16):
        metrics = {
            "center_x": 0.5 + i * 0.01,
            "center_y": 0.5,
            "bbox_w": 0.1,
            "bbox_h": 0.2,
            "velocity_x": 0.5,
            "velocity_y": 0.0,
            "confidence": 0.95,
            "distance": 5.0 + i,
            "distance_rate": -0.5,
            "zone_level": 2.0,
            "dwell_time": i * 0.033,
        }
        window.add_frame_metrics(metrics)
    
    assert window.size() == 16, f"Window size should be 16, got {window.size()}"
    assert window.is_full(), "Window should be full at 16 frames"
    
    print("✓ Fill window")
    
    # Test overflow (should remove oldest)
    metrics = {
        "center_x": 0.7,
        "center_y": 0.6,
        "bbox_w": 0.1,
        "bbox_h": 0.2,
        "velocity_x": 0.5,
        "velocity_y": 0.0,
        "confidence": 0.95,
        "distance": 25.0,
        "distance_rate": -0.5,
        "zone_level": 2.0,
        "dwell_time": 0.5,
    }
    window.add_frame_metrics(metrics)
    
    assert window.size() == 16, f"Window should stay at max size, got {window.size()}"
    features = window.get_features()
    # Last feature should be the new one
    assert features[-1, 0] == 0.7, "Oldest frame not removed"
    
    print("✓ Overflow handling")
    
    # Clear
    window.clear()
    assert window.size() == 0, "Window should be empty after clear"
    
    print("✓ Clear")
    
    return True


def test_threat_evaluator():
    """Test threat evaluator."""
    try:
        from jetson.threat_evaluator import ThreatEvaluator
    except ImportError:
        print("⊘ SKIP: Cannot import threat_evaluator")
        return True
    
    # Mock box class
    class MockBox:
        def __init__(self, track_id, x, y, w, h, conf, distance_m=None):
            self.track_id = track_id
            self.x = x
            self.y = y
            self.w = w
            self.h = h
            self.conf = conf
            self.distance_m = distance_m
            # Threat fields will be set by evaluator
            self.threat_level = None
            self.threat_confidence = None
            self.threat_score_benign = None
            self.threat_score_suspicious = None
            self.threat_score_threatening = None
    
    # Create evaluator (no model)
    evaluator = ThreatEvaluator(
        model_engine=None,
        defended_asset_xy=(0.0, 0.0),
        enable_rule_based=True,
    )
    
    # Create mock boxes
    boxes = [
        MockBox(track_id=1, x=0.4, y=0.4, w=0.1, h=0.2, conf=0.95, distance_m=5.0),
        MockBox(track_id=2, x=0.5, y=0.5, w=0.08, h=0.15, conf=0.90, distance_m=10.0),
    ]
    
    # Update threat scores
    threat_scores = evaluator.update(boxes, frame_w=1920, frame_h=1080)
    
    assert len(threat_scores) == 2, f"Should have 2 threat scores, got {len(threat_scores)}"
    assert 1 in threat_scores, "Track 1 should have threat score"
    assert 2 in threat_scores, "Track 2 should have threat score"
    
    print("✓ Update threat scores")
    
    # Apply scores to boxes
    evaluator.apply_threat_scores(boxes, threat_scores)
    
    assert boxes[0].threat_level is not None, "Track 1 should have threat_level"
    assert boxes[1].threat_level is not None, "Track 2 should have threat_level"
    
    print("✓ Apply threat scores to boxes")
    
    # Get summary
    summary = evaluator.get_threat_status_summary()
    assert summary["total_tracks"] == 2, f"Should have 2 total tracks, got {summary['total_tracks']}"
    
    print("✓ Get threat status summary")
    
    # Reset track
    evaluator.reset_track(1)
    assert 1 not in evaluator.track_threat_states, "Track 1 should be reset"
    
    print("✓ Reset track")
    
    # Clear all
    evaluator.clear_all()
    assert len(evaluator.track_threat_states) == 0, "All tracks should be cleared"
    
    print("✓ Clear all")
    
    return True


if __name__ == "__main__":
    try:
        test_threat_inference_engine()
        print()
        test_threat_metrics_window()
        print()
        test_threat_evaluator()
        print("\n✓ All threat module tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
