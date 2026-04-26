"""Process Reward Model (PRM) evaluation — step-level metrics.

PRMs assign rewards to intermediate reasoning steps rather than only final
answers.  This module provides:
  - Step verification accuracy
  - Step-level precision / recall / F1
  - Pearson correlation between aggregate PRM score and final answer correctness
  - First-error detection rate
  - Expected Calibration Error (ECE) across deciles of predicted reward
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Config & data classes
# ---------------------------------------------------------------------------


@dataclass
class PRMEvalConfig:
    """Configuration for ProcessRewardEval."""

    correct_threshold: float = 0.5  # step reward > this → predicted correct
    iou_threshold: float = 0.5  # for step boundary matching (reserved)


@dataclass
class StepPrediction:
    """Prediction and ground-truth label for a single reasoning step."""

    step_idx: int
    predicted_reward: float  # PRM score for this step
    is_correct_pred: bool  # predicted_reward > threshold
    is_correct_gt: bool  # ground-truth correctness for this step
    step_text: str = ""


@dataclass
class SolutionEval:
    """All step predictions for a single problem solution."""

    problem_id: str
    steps: list[StepPrediction]
    final_answer_correct: bool
    prm_final_score: float  # aggregate PRM score (e.g. min or product)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


class ProcessRewardEval:
    """Evaluate a Process Reward Model at the step level."""

    def __init__(self, config: PRMEvalConfig | None = None) -> None:
        self.config = config if config is not None else PRMEvalConfig()

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def step_accuracy(self, solutions: list[SolutionEval]) -> float:
        """Fraction of steps where is_correct_pred == is_correct_gt."""
        total = 0
        correct = 0
        for sol in solutions:
            for step in sol.steps:
                total += 1
                if step.is_correct_pred == step.is_correct_gt:
                    correct += 1
        if total == 0:
            return 0.0
        return correct / total

    def step_precision_recall_f1(self, solutions: list[SolutionEval]) -> dict:
        """Treating step correctness prediction as binary classification.

        Positive class = predicted correct (is_correct_pred=True).

        Returns
        -------
        dict with keys "precision", "recall", "f1".
        """
        tp = fp = fn = 0
        for sol in solutions:
            for step in sol.steps:
                if step.is_correct_pred and step.is_correct_gt:
                    tp += 1
                elif step.is_correct_pred and not step.is_correct_gt:
                    fp += 1
                elif not step.is_correct_pred and step.is_correct_gt:
                    fn += 1
                # tn: not counted

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    def final_answer_correlation(self, solutions: list[SolutionEval]) -> float:
        """Pearson correlation between prm_final_score and final_answer_correct.

        Returns a value in [-1, 1].  Returns 0.0 when there is no variance in
        either variable (division by zero guard).
        """
        if not solutions:
            return 0.0

        scores = torch.tensor([s.prm_final_score for s in solutions], dtype=torch.float64)
        labels = torch.tensor(
            [float(s.final_answer_correct) for s in solutions], dtype=torch.float64
        )

        scores_mean = scores.mean()
        labels_mean = labels.mean()

        s_diff = scores - scores_mean
        l_diff = labels - labels_mean

        numerator = (s_diff * l_diff).sum()
        denominator = torch.sqrt((s_diff**2).sum() * (l_diff**2).sum())

        if denominator.item() == 0.0:
            return 0.0

        return float(numerator / denominator)

    def first_error_detection(self, solutions: list[SolutionEval]) -> dict:
        """For each solution, check if the first incorrect step is flagged.

        A solution contributes only if it contains at least one step where
        is_correct_gt=False.

        Returns
        -------
        dict with keys:
            "detection_rate"  – fraction of applicable solutions where the first
                                incorrect step was flagged (is_correct_pred=False).
                                None if no solution has an incorrect step.
            "mean_position"   – mean 0-based index of the first incorrect step
                                across applicable solutions. 0.0 if none.
        """
        total_applicable = 0
        detected = 0
        positions: list[int] = []

        for sol in solutions:
            # find first step with is_correct_gt=False
            first_error: StepPrediction | None = None
            for step in sol.steps:
                if not step.is_correct_gt:
                    first_error = step
                    break

            if first_error is None:
                # No errors in this solution — not applicable
                continue

            total_applicable += 1
            positions.append(first_error.step_idx)
            if not first_error.is_correct_pred:
                detected += 1

        if total_applicable == 0:
            return {"detection_rate": None, "mean_position": 0.0}

        detection_rate = detected / total_applicable
        mean_position = sum(positions) / len(positions)
        return {"detection_rate": detection_rate, "mean_position": mean_position}

    def calibration_error(self, solutions: list[SolutionEval]) -> float:
        """Expected Calibration Error (ECE) across deciles of predicted_reward.

        Steps are grouped into 10 equal-width bins of predicted_reward in [0, 1].
        For each non-empty bin: |mean_predicted_reward - fraction_correct_gt|.
        Returns mean over non-empty bins.  Returns 0.0 when there are no steps.
        """
        # Collect all steps
        all_steps: list[StepPrediction] = []
        for sol in solutions:
            all_steps.extend(sol.steps)

        if not all_steps:
            return 0.0

        n_bins = 10
        # bins: index 0 → [0, 0.1), …, 9 → [0.9, 1.0]
        bin_pred_sums: list[float] = [0.0] * n_bins
        bin_gt_sums: list[float] = [0.0] * n_bins
        bin_counts: list[int] = [0] * n_bins

        for step in all_steps:
            # Clamp to [0, 1] for binning; values outside this range still
            # land in the edge bins.
            r = step.predicted_reward
            bin_idx = min(int(r * n_bins), n_bins - 1)
            # For predicted rewards < 0 fall into bin 0
            bin_idx = max(bin_idx, 0)
            bin_pred_sums[bin_idx] += r
            bin_gt_sums[bin_idx] += float(step.is_correct_gt)
            bin_counts[bin_idx] += 1

        errors: list[float] = []
        for i in range(n_bins):
            if bin_counts[i] == 0:
                continue
            mean_pred = bin_pred_sums[i] / bin_counts[i]
            mean_gt = bin_gt_sums[i] / bin_counts[i]
            errors.append(abs(mean_pred - mean_gt))

        if not errors:
            return 0.0
        return sum(errors) / len(errors)

    def evaluate(self, solutions: list[SolutionEval]) -> dict:
        """Run all metrics and return a combined report.

        Returns
        -------
        dict with keys:
            step_accuracy, precision, recall, f1,
            final_answer_correlation, detection_rate, mean_position,
            calibration_error
        """
        acc = self.step_accuracy(solutions)
        prf = self.step_precision_recall_f1(solutions)
        corr = self.final_answer_correlation(solutions)
        fed = self.first_error_detection(solutions)
        ece = self.calibration_error(solutions)

        return {
            "step_accuracy": acc,
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"],
            "final_answer_correlation": corr,
            "detection_rate": fed["detection_rate"],
            "mean_position": fed["mean_position"],
            "calibration_error": ece,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from src.eval import BENCHMARK_REGISTRY  # noqa: E402

BENCHMARK_REGISTRY.setdefault("process_reward_eval", ProcessRewardEval)
