"""
Integration test for src/eval/vision_grounding_eval.py

Verifies:
  - VisionGroundingEval.evaluate() returns a dict with correct keys and
    values in [0, 1] when run over 3 prediction/gt pairs with boxes,
    text, and counts.
  - BENCHMARK_REGISTRY["vision_grounding_eval"] is VisionGroundingEval.
"""

from __future__ import annotations

from src.eval import BENCHMARK_REGISTRY
from src.eval.vision_grounding_eval import VisionGroundingEval, VisionGroundingEvalConfig

# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_vision_grounding_eval_end_to_end() -> None:
    """Full pipeline: construct evaluator, run evaluate(), check output."""
    config = VisionGroundingEvalConfig(iou_threshold=0.5)
    evaluator = VisionGroundingEval(config=config)

    predictions = [
        {
            "boxes": [(10, 10, 50, 50), (60, 60, 100, 100)],
            "text": "a cat sitting on a mat",
            "count": 2,
        },
        {
            "boxes": [(0, 0, 30, 30)],
            "text": "hello world",
            "count": 1,
        },
        {
            "boxes": [(5, 5, 25, 25), (40, 40, 80, 80), (90, 10, 120, 40)],
            "text": "three objects detected",
            "count": 3,
        },
    ]

    ground_truths = [
        {
            "boxes": [(10, 10, 50, 50), (65, 65, 105, 105)],  # one exact, one slight shift
            "text": "a cat sitting on a mat",
            "count": 2,
        },
        {
            "boxes": [(0, 0, 30, 30)],  # exact match
            "text": "hello word",  # one char off
            "count": 2,  # off by one
        },
        {
            "boxes": [
                (5, 5, 25, 25),
                (40, 40, 80, 80),
                (200, 200, 250, 250),
            ],  # two match, one miss
            "text": "three objects detected",
            "count": 3,
        },
    ]

    result = evaluator.evaluate(predictions, ground_truths)

    # --- Correct output shape ---
    assert isinstance(result, dict), "evaluate() must return a dict"
    assert "f1" in result, "result must contain 'f1'"
    assert "ned" in result, "result must contain 'ned'"
    assert "count_acc" in result, "result must contain 'count_acc'"

    # --- Values in [0, 1] ---
    assert 0.0 <= result["f1"] <= 1.0, f"f1 out of range: {result['f1']}"
    assert 0.0 <= result["ned"] <= 1.0, f"ned out of range: {result['ned']}"
    assert 0.0 <= result["count_acc"] <= 1.0, f"count_acc out of range: {result['count_acc']}"


def test_benchmark_registry_wired() -> None:
    """BENCHMARK_REGISTRY must map 'vision_grounding_eval' to VisionGroundingEval."""
    assert "vision_grounding_eval" in BENCHMARK_REGISTRY, (
        "BENCHMARK_REGISTRY missing key 'vision_grounding_eval'"
    )
    assert BENCHMARK_REGISTRY["vision_grounding_eval"] is VisionGroundingEval, (
        "BENCHMARK_REGISTRY['vision_grounding_eval'] must be VisionGroundingEval class"
    )
