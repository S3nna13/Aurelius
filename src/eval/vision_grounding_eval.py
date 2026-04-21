"""
vision_grounding_eval.py — Vision Grounding Evaluation metrics for Aurelius.

Implements three metrics from Kimi K2.5 §4 (arXiv:2602.02276):
  1. Soft-IoU F1   — for object detection / grounding boxes
  2. Normalized Edit Distance (NED) — for OCR text quality
  3. Absolute Difference accuracy — for counting tasks

Pure PyTorch + pure Python only. No transformers, scipy, sklearn, PIL, cv2,
difflib, or editdistance packages are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VisionGroundingEvalConfig:
    """Configuration for VisionGroundingEval.

    Attributes:
        iou_threshold: Minimum IoU for a predicted box to count as a true
            positive against a ground-truth box (default 0.5).
    """
    iou_threshold: float = 0.5


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class VisionGroundingEval:
    """Evaluator for vision grounding tasks.

    Supports three evaluation modalities per sample:
      - Bounding-box detection via Soft-IoU F1
      - OCR text quality via Normalized Edit Distance
      - Counting accuracy via Absolute Difference score
    """

    def __init__(self, config: Optional[VisionGroundingEvalConfig] = None) -> None:
        self.config = config if config is not None else VisionGroundingEvalConfig()

    # ------------------------------------------------------------------
    # Box IoU
    # ------------------------------------------------------------------

    def box_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute Intersection-over-Union between two boxes.

        Args:
            box1: (x1, y1, x2, y2) — top-left and bottom-right corners.
            box2: (x1, y1, x2, y2)

        Returns:
            IoU in [0, 1].
        """
        x1_a, y1_a, x2_a, y2_a = float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3])
        x1_b, y1_b, x2_b, y2_b = float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3])

        # Intersection
        ix1 = max(x1_a, x1_b)
        iy1 = max(y1_a, y1_b)
        ix2 = min(x2_a, x2_b)
        iy2 = min(y2_a, y2_b)

        inter_w = max(0.0, ix2 - ix1)
        inter_h = max(0.0, iy2 - iy1)
        inter_area = inter_w * inter_h

        if inter_area == 0.0:
            return 0.0

        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)
        union_area = area_a + area_b - inter_area

        if union_area <= 0.0:
            return 0.0

        return inter_area / union_area

    # ------------------------------------------------------------------
    # Soft-IoU F1
    # ------------------------------------------------------------------

    def soft_iou_f1(
        self,
        pred_boxes: List[Tuple],
        gt_boxes: List[Tuple],
    ) -> Dict[str, float]:
        """Compute Soft-IoU F1 for a single sample.

        For each predicted box, find the best-matching ground-truth box by IoU.
        A predicted box is a true positive if that IoU >= iou_threshold.
        Each ground-truth box can be matched at most once.

        Args:
            pred_boxes: List of (x1, y1, x2, y2) predicted boxes.
            gt_boxes:   List of (x1, y1, x2, y2) ground-truth boxes.

        Returns:
            dict with keys "precision", "recall", "f1", "tp" (int).
        """
        n_pred = len(pred_boxes)
        n_gt = len(gt_boxes)

        if n_pred == 0 and n_gt == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0}

        if n_pred == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0}

        if n_gt == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0}

        threshold = self.config.iou_threshold
        matched_gt: set = set()
        tp = 0

        for pb in pred_boxes:
            best_iou = 0.0
            best_j = -1
            for j, gb in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = self.box_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= threshold and best_j >= 0:
                tp += 1
                matched_gt.add(best_j)

        precision = tp / n_pred
        recall = tp / n_gt
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

        return {"precision": precision, "recall": recall, "f1": f1, "tp": tp}

    # ------------------------------------------------------------------
    # Levenshtein edit distance (from scratch)
    # ------------------------------------------------------------------

    @staticmethod
    def _levenshtein(s: str, t: str) -> int:
        """Standard DP Levenshtein edit distance.

        Implemented from scratch — no difflib, no editdistance package.
        """
        m, n = len(s), len(t)
        # Use two rows to save memory
        prev = list(range(n + 1))
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            curr[0] = i
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    curr[j] = prev[j - 1]
                else:
                    curr[j] = 1 + min(prev[j - 1], prev[j], curr[j - 1])
            prev, curr = curr, prev

        return prev[n]

    # ------------------------------------------------------------------
    # Normalized Edit Distance
    # ------------------------------------------------------------------

    def normalized_edit_distance(self, pred: str, gt: str) -> float:
        """Compute Normalized Edit Distance (NED).

        NED = 1 - edit_distance(pred, gt) / max(len(pred), len(gt), 1)

        Returns a value in [0, 1] where 1.0 = perfect match.

        Args:
            pred: Predicted string.
            gt:   Ground-truth string.
        """
        denom = max(len(pred), len(gt), 1)
        ed = self._levenshtein(pred, gt)
        return 1.0 - ed / denom

    # ------------------------------------------------------------------
    # Counting accuracy
    # ------------------------------------------------------------------

    def count_accuracy(self, pred_count: int, gt_count: int) -> float:
        """Compute counting accuracy as 1 - normalised absolute difference.

        count_diff = max(0, 1 - abs(pred_count - gt_count) / max(gt_count, 1))

        Returns 1.0 for exact match, decreases linearly with error.

        Args:
            pred_count: Predicted integer count.
            gt_count:   Ground-truth integer count.
        """
        denom = max(gt_count, 1)
        diff = abs(pred_count - gt_count)
        return max(0.0, 1.0 - diff / denom)

    # ------------------------------------------------------------------
    # Aggregate evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
    ) -> Dict[str, float]:
        """Evaluate a list of prediction / ground-truth pairs.

        Each element in *predictions* and *ground_truths* may contain any
        subset of the optional keys:
          - "boxes"  : list of (x1, y1, x2, y2) tuples
          - "text"   : str
          - "count"  : int

        Metrics are computed only for samples where both prediction and
        ground truth provide the relevant key, then averaged.

        Args:
            predictions:   List of per-sample prediction dicts.
            ground_truths: List of per-sample ground-truth dicts.

        Returns:
            dict with keys "f1", "ned", "count_acc" — each the mean over
            all samples that contributed to the metric (or 0.0 if none did).
        """
        f1_scores: List[float] = []
        ned_scores: List[float] = []
        count_scores: List[float] = []

        for pred, gt in zip(predictions, ground_truths):
            # --- boxes ---
            if "boxes" in pred and "boxes" in gt:
                result = self.soft_iou_f1(pred["boxes"], gt["boxes"])
                f1_scores.append(result["f1"])

            # --- text ---
            if "text" in pred and "text" in gt:
                ned = self.normalized_edit_distance(pred["text"], gt["text"])
                ned_scores.append(ned)

            # --- count ---
            if "count" in pred and "count" in gt:
                acc = self.count_accuracy(pred["count"], gt["count"])
                count_scores.append(acc)

        def _mean(lst: List[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        return {
            "f1": _mean(f1_scores),
            "ned": _mean(ned_scores),
            "count_acc": _mean(count_scores),
        }


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------

from src.eval import BENCHMARK_REGISTRY  # noqa: E402

BENCHMARK_REGISTRY["vision_grounding_eval"] = VisionGroundingEval
