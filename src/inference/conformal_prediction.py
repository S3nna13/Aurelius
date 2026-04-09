"""Conformal prediction for LLMs: prediction sets with coverage guarantees and RAPS."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ConformalConfig:
    alpha: float = 0.1        # target miscoverage rate (1-alpha = target coverage)
    method: str = "aps"       # "aps" | "raps" | "lac"
    raps_lambda: float = 0.01
    raps_k_reg: int = 5


def compute_softmax_scores(logits: Tensor) -> Tensor:
    """Apply softmax to logits to get probabilities.

    Args:
        logits: (..., C) raw logit tensor

    Returns:
        (..., C) probability tensor summing to 1 along last dim
    """
    return F.softmax(logits, dim=-1)


def compute_aps_scores(probs: Tensor, labels: Tensor) -> Tensor:
    """Compute APS (Adaptive Prediction Sets) nonconformity scores.

    For each sample, sort class probabilities in descending order and compute
    the cumulative sum up to and including the rank of the true label.

    Args:
        probs:  (N, C) probability tensor
        labels: (N,)  integer true class indices

    Returns:
        (N,) APS nonconformity scores
    """
    N = probs.shape[0]

    # Sort probabilities descending; sorted_idx[i] gives class order for sample i
    sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)

    # Find rank of each true label in the sorted order
    # rank_of_label[i] = position (0-indexed) of labels[i] in sorted_idx[i]
    labels_expanded = labels.unsqueeze(1).expand_as(sorted_idx)  # (N, C)
    ranks = (sorted_idx == labels_expanded).nonzero(as_tuple=False)  # (N, 2)
    # ranks[:, 0] are sample indices, ranks[:, 1] are the rank positions
    rank_positions = ranks[:, 1]  # (N,)

    # Cumsum of sorted probs; score = cumsum up to and including the true label rank
    cumsum = torch.cumsum(sorted_probs, dim=1)  # (N, C)
    scores = cumsum[torch.arange(N), rank_positions]  # (N,)

    return scores


def compute_lac_scores(probs: Tensor, labels: Tensor) -> Tensor:
    """Compute LAC (Least Ambiguous Classifier) nonconformity scores.

    Score = 1 - P(true label), so a perfectly confident correct prediction
    gives a score near 0.

    Args:
        probs:  (N, C) probability tensor
        labels: (N,)  integer true class indices

    Returns:
        (N,) scores in [0, 1]
    """
    N = probs.shape[0]
    true_probs = probs[torch.arange(N), labels]  # (N,)
    return 1.0 - true_probs


def compute_raps_scores(
    probs: Tensor,
    labels: Tensor,
    lambda_reg: float,
    k_reg: int,
) -> Tensor:
    """Compute RAPS (Regularized Adaptive Prediction Sets) nonconformity scores.

    Adds a regularization penalty to APS scores for labels that appear at high
    rank (i.e., low probability classes), encouraging smaller prediction sets.

    Score = aps_score + lambda_reg * max(0, rank - k_reg)

    Args:
        probs:      (N, C) probability tensor
        labels:     (N,)  integer true class indices
        lambda_reg: regularization weight
        k_reg:      rank threshold below which no penalty is applied

    Returns:
        (N,) RAPS scores (>= corresponding APS scores)
    """
    N = probs.shape[0]

    sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)

    labels_expanded = labels.unsqueeze(1).expand_as(sorted_idx)
    ranks = (sorted_idx == labels_expanded).nonzero(as_tuple=False)
    rank_positions = ranks[:, 1].float()  # (N,) 0-indexed

    cumsum = torch.cumsum(sorted_probs, dim=1)
    aps_scores = cumsum[torch.arange(N), rank_positions.long()]  # (N,)

    penalty = lambda_reg * torch.clamp(rank_positions - k_reg, min=0.0)
    return aps_scores + penalty


def calibrate_threshold(cal_scores: Tensor, alpha: float) -> float:
    """Compute conformal threshold from calibration nonconformity scores.

    Uses finite-sample correction: q_level = ceil((N+1)*(1-alpha)) / N

    Args:
        cal_scores: (N,) nonconformity scores on calibration set
        alpha:      target miscoverage rate

    Returns:
        Threshold float such that coverage >= 1-alpha on new data
    """
    N = cal_scores.shape[0]
    q_level = math.ceil((N + 1) * (1 - alpha)) / N
    q_level = min(q_level, 1.0)  # clamp to valid quantile range
    threshold = torch.quantile(cal_scores, q_level).item()
    return float(threshold)


def construct_prediction_set(
    probs: Tensor,
    threshold: float,
    method: str = "aps",
) -> list[list[int]]:
    """Build prediction sets for each sample.

    APS / RAPS: include classes in descending probability order until the
    cumulative sum meets or exceeds the threshold.

    LAC: include all classes where (1 - prob) <= threshold, i.e., prob >= 1 - threshold.

    Args:
        probs:     (N, C) probability tensor
        threshold: conformal threshold from calibrate_threshold
        method:    "aps", "raps", or "lac"

    Returns:
        List of N lists, each containing the class indices in the prediction set
    """
    N, C = probs.shape
    prediction_sets: list[list[int]] = []

    if method == "lac":
        for i in range(N):
            p = probs[i]
            pred_set = [c for c in range(C) if (1.0 - p[c].item()) <= threshold]
            prediction_sets.append(pred_set)
    else:
        # APS / RAPS: greedy cumulative inclusion
        sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=1)  # (N, C)

        for i in range(N):
            pred_set: list[int] = []
            for rank in range(C):
                class_idx = sorted_idx[i, rank].item()
                pred_set.append(int(class_idx))
                if cumsum[i, rank].item() >= threshold:
                    break
            prediction_sets.append(pred_set)

    return prediction_sets


class ConformalPredictor:
    """Conformal predictor that calibrates on held-out data and builds
    prediction sets with guaranteed marginal coverage.

    Usage:
        cfg = ConformalConfig(alpha=0.1, method="aps")
        predictor = ConformalPredictor(cfg)
        predictor.calibrate(cal_probs, cal_labels)
        sets = predictor.predict(test_probs)
        rate = predictor.coverage_rate(sets, test_labels)
    """

    def __init__(self, config: ConformalConfig) -> None:
        self.config = config
        self.threshold: float | None = None

    def calibrate(self, cal_probs: Tensor, cal_labels: Tensor) -> float:
        """Compute nonconformity scores and set the conformal threshold.

        Args:
            cal_probs:  (N, C) softmax probabilities on calibration set
            cal_labels: (N,)  integer true labels

        Returns:
            The calibrated threshold (also stored in self.threshold)
        """
        method = self.config.method
        if method == "aps":
            scores = compute_aps_scores(cal_probs, cal_labels)
        elif method == "raps":
            scores = compute_raps_scores(
                cal_probs,
                cal_labels,
                self.config.raps_lambda,
                self.config.raps_k_reg,
            )
        elif method == "lac":
            scores = compute_lac_scores(cal_probs, cal_labels)
        else:
            raise ValueError(f"Unknown method: {method!r}. Choose 'aps', 'raps', or 'lac'.")

        self.threshold = calibrate_threshold(scores, self.config.alpha)
        return self.threshold

    def predict(self, test_probs: Tensor) -> list[list[int]]:
        """Construct prediction sets for test samples.

        Must call calibrate() first.

        Args:
            test_probs: (M, C) softmax probabilities for M test samples

        Returns:
            List of M prediction sets (lists of class indices)
        """
        if self.threshold is None:
            raise RuntimeError("Call calibrate() before predict().")
        return construct_prediction_set(test_probs, self.threshold, self.config.method)

    def coverage_rate(
        self,
        prediction_sets: list[list[int]],
        true_labels: Tensor,
    ) -> float:
        """Compute empirical coverage: fraction of samples where true label is in set.

        Args:
            prediction_sets: list of N prediction sets
            true_labels:     (N,) integer true class indices

        Returns:
            Coverage rate in [0, 1]
        """
        covered = sum(
            1
            for pred_set, label in zip(prediction_sets, true_labels.tolist())
            if int(label) in pred_set
        )
        return covered / len(prediction_sets)
