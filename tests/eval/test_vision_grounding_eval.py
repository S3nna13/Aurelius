"""
Unit tests for src/eval/vision_grounding_eval.py

Covers:
  - box_iou: identical, no-overlap, partial
  - soft_iou_f1: perfect, no-match, empty pred, empty gt
  - normalized_edit_distance: identical, both-empty, known (kitten/sitting),
    empty pred
  - count_accuracy: exact, off-by-one, zero gt
  - evaluate: aggregation smoke test
"""

from __future__ import annotations

import pytest

from src.eval.vision_grounding_eval import VisionGroundingEval, VisionGroundingEvalConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def evaluator() -> VisionGroundingEval:
    return VisionGroundingEval()


@pytest.fixture
def strict_evaluator() -> VisionGroundingEval:
    """Evaluator with a higher IoU threshold (0.9) to force failures."""
    return VisionGroundingEval(config=VisionGroundingEvalConfig(iou_threshold=0.9))


# ---------------------------------------------------------------------------
# 1. box_iou — identical boxes
# ---------------------------------------------------------------------------


def test_box_iou_identical(evaluator: VisionGroundingEval) -> None:
    box = (0.0, 0.0, 4.0, 4.0)
    assert evaluator.box_iou(box, box) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. box_iou — no overlap
# ---------------------------------------------------------------------------


def test_box_iou_no_overlap(evaluator: VisionGroundingEval) -> None:
    box_a = (0.0, 0.0, 2.0, 2.0)
    box_b = (3.0, 3.0, 5.0, 5.0)
    assert evaluator.box_iou(box_a, box_b) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. box_iou — partial overlap with known value
#    Two 2×2 boxes: A=(0,0,2,2), B=(1,1,3,3)
#    Intersection = 1×1 = 1; Union = 4+4-1 = 7; IoU = 1/7 ≈ 0.1428…
# ---------------------------------------------------------------------------


def test_box_iou_partial(evaluator: VisionGroundingEval) -> None:
    box_a = (0.0, 0.0, 2.0, 2.0)
    box_b = (1.0, 1.0, 3.0, 3.0)
    expected = 1.0 / 7.0
    assert evaluator.box_iou(box_a, box_b) == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# 4. soft_iou_f1 — perfect match (pred == gt)
# ---------------------------------------------------------------------------


def test_soft_iou_f1_perfect(evaluator: VisionGroundingEval) -> None:
    boxes = [(0, 0, 10, 10), (20, 20, 30, 30)]
    result = evaluator.soft_iou_f1(boxes, boxes)
    assert result["f1"] == pytest.approx(1.0, abs=1e-6)
    assert result["precision"] == pytest.approx(1.0, abs=1e-6)
    assert result["recall"] == pytest.approx(1.0, abs=1e-6)
    assert result["tp"] == 2


# ---------------------------------------------------------------------------
# 5. soft_iou_f1 — no matching boxes (non-overlapping)
# ---------------------------------------------------------------------------


def test_soft_iou_f1_no_match(evaluator: VisionGroundingEval) -> None:
    pred = [(0, 0, 1, 1)]
    gt = [(100, 100, 200, 200)]
    result = evaluator.soft_iou_f1(pred, gt)
    assert result["f1"] == pytest.approx(0.0, abs=1e-6)
    assert result["tp"] == 0


# ---------------------------------------------------------------------------
# 6. soft_iou_f1 — empty predictions
# ---------------------------------------------------------------------------


def test_soft_iou_f1_empty_pred(evaluator: VisionGroundingEval) -> None:
    gt = [(0, 0, 10, 10)]
    result = evaluator.soft_iou_f1([], gt)
    assert result["f1"] == pytest.approx(0.0, abs=1e-6)
    assert result["precision"] == pytest.approx(0.0, abs=1e-6)
    assert result["recall"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 7. soft_iou_f1 — empty ground truths
# ---------------------------------------------------------------------------


def test_soft_iou_f1_empty_gt(evaluator: VisionGroundingEval) -> None:
    pred = [(0, 0, 10, 10)]
    result = evaluator.soft_iou_f1(pred, [])
    assert result["f1"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 8. normalized_edit_distance — identical strings
# ---------------------------------------------------------------------------


def test_ned_identical(evaluator: VisionGroundingEval) -> None:
    assert evaluator.normalized_edit_distance("hello", "hello") == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 9. normalized_edit_distance — both empty
# ---------------------------------------------------------------------------


def test_ned_empty_both(evaluator: VisionGroundingEval) -> None:
    # edit_distance("","")=0, denom=max(0,0,1)=1, NED=1-0/1=1.0
    assert evaluator.normalized_edit_distance("", "") == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 10. normalized_edit_distance — "kitten" vs "sitting"
#     edit_distance = 3, max_len = 7, NED = 1 - 3/7 ≈ 0.5714
# ---------------------------------------------------------------------------


def test_ned_known(evaluator: VisionGroundingEval) -> None:
    ned = evaluator.normalized_edit_distance("kitten", "sitting")
    expected = 1.0 - 3.0 / 7.0
    assert ned == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# 11. normalized_edit_distance — empty pred, non-empty gt
#     pred="", gt="abc" → edit_distance=3, denom=3, NED=1-3/3=0.0
# ---------------------------------------------------------------------------


def test_ned_empty_pred(evaluator: VisionGroundingEval) -> None:
    assert evaluator.normalized_edit_distance("", "abc") == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 12. count_accuracy — exact match
# ---------------------------------------------------------------------------


def test_count_accuracy_exact(evaluator: VisionGroundingEval) -> None:
    assert evaluator.count_accuracy(5, 5) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 13. count_accuracy — off by one (gt=5, pred=4)
#     1 - |4-5| / max(5,1) = 1 - 1/5 = 0.8
# ---------------------------------------------------------------------------


def test_count_accuracy_off_by_one(evaluator: VisionGroundingEval) -> None:
    assert evaluator.count_accuracy(4, 5) == pytest.approx(0.8, abs=1e-6)


# ---------------------------------------------------------------------------
# 14. count_accuracy — zero gt
#     gt=0, pred=0 → 1 - 0/1 = 1.0 (no crash)
#     gt=0, pred=3 → max(0, 1 - 3/1) = 0.0 (no crash)
# ---------------------------------------------------------------------------


def test_count_accuracy_zero_gt(evaluator: VisionGroundingEval) -> None:
    assert evaluator.count_accuracy(0, 0) == pytest.approx(1.0, abs=1e-6)
    assert evaluator.count_accuracy(3, 0) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 15. evaluate — aggregates over a list of 2 samples
# ---------------------------------------------------------------------------


def test_evaluate_aggregates(evaluator: VisionGroundingEval) -> None:
    predictions = [
        {"boxes": [(0, 0, 10, 10)], "text": "cat", "count": 3},
        {"boxes": [(5, 5, 15, 15)], "text": "dog", "count": 2},
    ]
    ground_truths = [
        {"boxes": [(0, 0, 10, 10)], "text": "cat", "count": 3},
        {"boxes": [(100, 100, 200, 200)], "text": "dog", "count": 4},
    ]
    result = evaluator.evaluate(predictions, ground_truths)
    assert "f1" in result
    assert "ned" in result
    assert "count_acc" in result
    assert 0.0 <= result["f1"] <= 1.0
    assert 0.0 <= result["ned"] <= 1.0
    assert 0.0 <= result["count_acc"] <= 1.0


# ---------------------------------------------------------------------------
# Additional edge-case tests (taking total to 16)
# ---------------------------------------------------------------------------


# 16a. soft_iou_f1 — both empty → perfect score by convention (n_pred==0 and n_gt==0)
def test_soft_iou_f1_both_empty(evaluator: VisionGroundingEval) -> None:
    result = evaluator.soft_iou_f1([], [])
    assert result["f1"] == pytest.approx(1.0, abs=1e-6)


# 16b. box_iou — touching edges (share a boundary line → zero area intersection)
def test_box_iou_touching_edges(evaluator: VisionGroundingEval) -> None:
    box_a = (0.0, 0.0, 2.0, 2.0)
    box_b = (2.0, 0.0, 4.0, 2.0)  # shares right/left edge, zero-width intersection
    assert evaluator.box_iou(box_a, box_b) == pytest.approx(0.0, abs=1e-6)
