"""Conformal Prediction for Language Model Token Sets.

Provides coverage-guaranteed token set prediction for language models.
Instead of argmax decoding, returns a set S such that:
    P(true_token ∈ S) >= 1 - alpha

Uses RAPS (Regularized Adaptive Prediction Sets) for efficiency.

References:
    Angelopoulos et al. 2020 (RAPS) — https://arxiv.org/abs/2009.14193
    Kumar et al. 2023 (LLM conformal) — various works
"""

import math

import torch
from torch import Tensor


class CalibrationSet:
    """Stores nonconformity scores for conformal calibration.

    Nonconformity score = 1 - P(true_token), so a low score means the model
    was confident and correct; a high score means it assigned low probability
    to the true token.
    """

    def __init__(self) -> None:
        self.scores: list[float] = []
        self.n: int = 0

    def add(self, probs: Tensor, true_token_id: int) -> None:
        """Add a calibration example.

        Args:
            probs: (V,) softmax probabilities over the vocabulary.
            true_token_id: Index of the ground-truth next token.
        """
        score = float(1.0 - probs[true_token_id].item())
        self.scores.append(score)
        self.n += 1

    def quantile(self, alpha: float) -> float:
        """Return the conformal quantile for miscoverage level alpha.

        Uses the finite-sample corrected quantile:
            q = ceil((n+1)(1-alpha)) / n  — i.e. the ceil((n+1)(1-alpha))-th
            order statistic.

        Args:
            alpha: Desired miscoverage rate (e.g. 0.1 for 90% coverage).

        Returns:
            Conformal threshold tau.  Returns 1.0 if no calibration data.
        """
        if self.n == 0:
            return 1.0

        level = math.ceil((self.n + 1) * (1.0 - alpha))
        # Clamp to valid index range [1, n]
        level = max(1, min(level, self.n))
        sorted_scores = sorted(self.scores)
        return float(sorted_scores[level - 1])


class RAPSCalibrationSet:
    """Regularized Adaptive Prediction Sets calibration (RAPS).

    Adds a regularization term that penalises large prediction sets, making
    them more efficient (smaller) while preserving coverage.
    """

    def __init__(self, k_reg: int = 5, lambda_reg: float = 0.01) -> None:
        """
        Args:
            k_reg: Regularization begins after this many tokens in sorted order.
            lambda_reg: Penalty per additional token beyond k_reg.
        """
        self.k_reg = k_reg
        self.lambda_reg = lambda_reg
        self.scores: list[float] = []
        self.n: int = 0

    def add(self, probs: Tensor, true_token_id: int) -> None:
        """Add a calibration example using the RAPS nonconformity score.

        The RAPS score for sample i with true label y_i is:
            s_i = cumsum[rank(y_i)] + lambda * max(0, rank(y_i) - k_reg)

        where cumsum[r] is the sum of the r+1 largest probabilities (0-indexed
        rank r), i.e. we include the true token in the cumsum.

        Args:
            probs: (V,) softmax probabilities over the vocabulary.
            true_token_id: Index of the ground-truth next token.
        """
        # Sort descending to get rank ordering
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)

        # Find 0-indexed rank of the true token in the sorted order
        # sorted_idx[rank] == true_token_id
        rank_tensor = (sorted_idx == true_token_id).nonzero(as_tuple=True)[0]
        rank = int(rank_tensor[0].item())

        # Cumulative sum up to and *including* the true token (rank is 0-indexed)
        cumsum = float(sorted_probs[: rank + 1].sum().item())

        # RAPS regularization term
        reg = self.lambda_reg * max(0, rank - self.k_reg)
        score = cumsum + reg

        self.scores.append(score)
        self.n += 1

    def quantile(self, alpha: float) -> float:
        """Return the conformal quantile for miscoverage level alpha.

        Identical finite-sample correction as CalibrationSet.quantile.
        """
        if self.n == 0:
            return 1.0

        level = math.ceil((self.n + 1) * (1.0 - alpha))
        level = max(1, min(level, self.n))
        sorted_scores = sorted(self.scores)
        return float(sorted_scores[level - 1])


class ConformalTokenSet:
    """Prediction-set constructor for a fixed conformal threshold tau.

    At inference time, after calibration has produced tau, wrap this class
    around your model's softmax output to get a token set with guaranteed
    coverage.
    """

    def __init__(self, tau: float) -> None:
        """
        Args:
            tau: Conformal threshold (output of CalibrationSet.quantile or
                 RAPSCalibrationSet.quantile).
        """
        self.tau = tau

    def predict_set(self, probs: Tensor) -> list[int]:
        """Return the prediction set for a single position.

        Tokens are included greedily from highest probability downward until
        the cumulative probability exceeds tau.  The token that first pushes
        the cumsum above tau is *included* (so coverage is respected).

        Args:
            probs: (V,) softmax probabilities.

        Returns:
            List of token ids in descending probability order.
        """
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = 0.0
        result: list[int] = []
        for i in range(len(sorted_probs)):
            cumsum += float(sorted_probs[i].item())
            result.append(int(sorted_idx[i].item()))
            if cumsum > self.tau:
                break
        return result

    def set_size_stats(self, probs_batch: Tensor) -> dict[str, float]:
        """Compute set-size statistics over a batch.

        Args:
            probs_batch: (B, V) softmax probabilities.

        Returns:
            Dictionary with keys 'mean_set_size', 'max_set_size', 'min_set_size'.
        """
        sizes = [len(self.predict_set(probs_batch[i])) for i in range(probs_batch.shape[0])]
        return {
            "mean_set_size": float(sum(sizes)) / len(sizes),
            "max_set_size": int(max(sizes)),
            "min_set_size": int(min(sizes)),
        }


class ConformalLMDecoder:
    """End-to-end conformal LM decoding pipeline.

    Usage::

        decoder = ConformalLMDecoder(alpha=0.1, use_raps=True)
        tau = decoder.calibrate(cal_probs, cal_tokens)
        token_set = decoder.predict(query_probs)
        cov = decoder.coverage_estimate(test_probs, test_tokens)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        use_raps: bool = False,
        k_reg: int = 5,
        lambda_reg: float = 0.01,
    ) -> None:
        """
        Args:
            alpha: Miscoverage level (e.g. 0.1 → 90% coverage guarantee).
            use_raps: If True, use RAPS; otherwise use plain LAC scores.
            k_reg: RAPS regularization start rank.
            lambda_reg: RAPS regularization penalty per rank beyond k_reg.
        """
        self.alpha = alpha
        if use_raps:
            self.calibration_set: CalibrationSet | RAPSCalibrationSet = RAPSCalibrationSet(
                k_reg=k_reg, lambda_reg=lambda_reg
            )
        else:
            self.calibration_set = CalibrationSet()
        self.tau: float | None = None

    def calibrate(self, probs_list: list[Tensor], true_tokens: list[int]) -> float:
        """Run calibration and store the resulting threshold tau.

        Args:
            probs_list: List of (V,) probability tensors (one per calibration token).
            true_tokens: List of ground-truth token ids aligned with probs_list.

        Returns:
            Conformal threshold tau = calibration_set.quantile(alpha).
        """
        for probs, token in zip(probs_list, true_tokens):
            self.calibration_set.add(probs, token)
        self.tau = self.calibration_set.quantile(self.alpha)
        return self.tau

    def predict(self, probs: Tensor) -> list[int]:
        """Return the prediction set for a single token position.

        Args:
            probs: (V,) softmax probabilities.

        Returns:
            List of token ids whose cumulative probability exceeds tau.

        Raises:
            RuntimeError: If calibrate() has not been called yet.
        """
        if self.tau is None:
            raise RuntimeError("Not calibrated")
        token_set = ConformalTokenSet(self.tau)
        return token_set.predict_set(probs)

    def coverage_estimate(self, probs_list: list[Tensor], true_tokens: list[int]) -> float:
        """Empirically estimate coverage on a held-out set.

        Args:
            probs_list: List of (V,) probability tensors.
            true_tokens: Corresponding ground-truth token ids.

        Returns:
            Fraction of positions where the true token is in the prediction set.
        """
        if not probs_list:
            return 0.0
        covered = sum(
            1 for probs, token in zip(probs_list, true_tokens) if token in self.predict(probs)
        )
        return covered / len(probs_list)
