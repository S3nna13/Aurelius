"""Reward model ensemble with uncertainty calibration and agreement scoring (v2).

Adds:
- EnsembleConfig: aggregation modes including trimmed_mean, median, min, max, mean
- RewardEnsemble: accepts a list of pre-built nn.Module reward models
- RewardCalibrator: linear least-squares calibration to human scores
- RewardAgreementFilter: uncertainty-aware sample and pair filtering
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# EnsembleConfig
# ---------------------------------------------------------------------------

@dataclass
class EnsembleConfig:
    """Configuration for the reward ensemble (v2)."""

    n_models: int = 5
    aggregation: str = "mean"          # "mean" | "median" | "min" | "max" | "trimmed_mean"
    trimmed_mean_ratio: float = 0.1    # fraction to trim from each end
    uncertainty_threshold: float = 0.5


# ---------------------------------------------------------------------------
# RewardEnsemble
# ---------------------------------------------------------------------------

class RewardEnsemble(nn.Module):
    """Ensemble of independent reward models.

    Each model in *models* must accept ``(B, d_model)`` and return ``(B,)``
    (or ``(B, 1)`` — the module squeezes automatically).

    Args:
        models: List of K ``nn.Module`` reward scorers.
        config: :class:`EnsembleConfig` hyperparameters.
    """

    def __init__(self, models: List[nn.Module], config: EnsembleConfig) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        self.config = config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stack_outputs(self, x: Tensor) -> Tensor:
        """Run every model and stack results.

        Args:
            x: ``(B, d_model)`` input features.

        Returns:
            ``(K, B)`` per-model rewards.
        """
        outputs = []
        for model in self.models:
            out = model(x)           # (B,) or (B, 1)
            if out.dim() == 2:
                out = out.squeeze(-1)
            outputs.append(out)
        return torch.stack(outputs, dim=0)  # (K, B)

    def _trimmed_mean(self, rewards: Tensor, ratio: float) -> Tensor:
        """Compute trimmed mean along the model dimension (dim=0).

        Sort each column, discard the bottom and top *ratio* fraction, then
        take the arithmetic mean of the remaining values.

        Args:
            rewards: ``(K, B)`` rewards from K models.
            ratio:   Fraction of models to drop from each tail (e.g. 0.1).

        Returns:
            ``(B,)`` trimmed-mean reward.
        """
        K = rewards.shape[0]
        n_trim = max(0, int(K * ratio))

        sorted_rewards, _ = torch.sort(rewards, dim=0)  # (K, B)

        if 2 * n_trim >= K:
            # Nothing left after trimming — fall back to full mean
            return sorted_rewards.mean(dim=0)

        trimmed = sorted_rewards[n_trim: K - n_trim, :]  # (K - 2*n_trim, B)
        return trimmed.mean(dim=0)

    def _aggregate(self, rewards: Tensor) -> Tensor:
        """Aggregate ``(K, B)`` rewards to ``(B,)`` per :attr:`config.aggregation`."""
        mode = self.config.aggregation
        if mode == "mean":
            return rewards.mean(dim=0)
        elif mode == "median":
            return rewards.median(dim=0).values
        elif mode == "min":
            return rewards.min(dim=0).values
        elif mode == "max":
            return rewards.max(dim=0).values
        elif mode == "trimmed_mean":
            return self._trimmed_mean(rewards, self.config.trimmed_mean_ratio)
        else:
            raise ValueError(
                f"Unknown aggregation '{mode}'. "
                "Choose from 'mean', 'median', 'min', 'max', 'trimmed_mean'."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Aggregate reward over all models.

        Args:
            x: ``(B, d_model)`` input features.

        Returns:
            ``(B,)`` aggregated scalar reward.
        """
        rewards = self._stack_outputs(x)   # (K, B)
        return self._aggregate(rewards)    # (B,)

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict mean and standard deviation of reward across models.

        Args:
            x: ``(B, d_model)`` input features.

        Returns:
            Tuple ``(mean_reward, std_reward)`` both shape ``(B,)``.
        """
        rewards = self._stack_outputs(x)   # (K, B)
        mean_reward = rewards.mean(dim=0)  # (B,)
        if rewards.shape[0] > 1:
            std_reward = rewards.std(dim=0, correction=0)
        else:
            std_reward = torch.zeros_like(mean_reward)
        return mean_reward, std_reward

    def agreement_score(self, x: Tensor) -> Tensor:
        """Pairwise rank agreement across all model pairs.

        For each prompt pair ``(i, j)`` in the batch, counts the fraction of
        model pairs that agree on whether ``x[i]`` is better than ``x[j]``
        (i.e., ``r_k[i] > r_k[j]``).  The returned score for sample ``i``
        is the mean agreement when ``i`` is the *first* member of every pair
        ``(i, j)`` with ``j != i``.

        Args:
            x: ``(B, d_model)`` input features.

        Returns:
            ``(B,)`` agreement score in ``[0, 1]``.
        """
        rewards = self._stack_outputs(x)  # (K, B)
        K, B = rewards.shape

        if B <= 1:
            return torch.ones(B, device=x.device, dtype=x.dtype)

        # For each ordered pair (i, j), determine how many model *pairs* agree
        # on the sign of r[i] - r[j].  We summarise per sample i.
        # signs[k, i, j] = sign(r_k[i] - r_k[j])  -- 1 or -1 (or 0)
        # agreement(i,j) = fraction of model pairs (k, l) where sign_k == sign_l
        # Then agreement_score[i] = mean over j != i of agreement(i, j)

        # Build pairwise sign matrix: (K, B, B)
        r_i = rewards.unsqueeze(2)   # (K, B, 1)
        r_j = rewards.unsqueeze(1)   # (K, 1, B)
        signs = torch.sign(r_i - r_j)  # (K, B, B)  -- values in {-1, 0, 1}

        if K == 1:
            # Only one model: trivially full agreement
            scores = torch.ones(B, device=x.device, dtype=x.dtype)
            return scores

        # Fraction of model pairs (k, l) with k < l that agree on each (i, j)
        # = mean over model pairs of (signs[k,i,j] == signs[l,i,j])
        # Efficient: use outer product check
        # agree_kl[k, l, i, j] = (signs[k,i,j] == signs[l,i,j])
        # We compare all K×K, exclude diagonal, then average.
        signs_a = signs.unsqueeze(1)  # (K, 1, B, B)
        signs_b = signs.unsqueeze(0)  # (1, K, B, B)
        agree = (signs_a == signs_b).float()  # (K, K, B, B)

        # Mask diagonal (same model)
        eye = torch.eye(K, device=x.device, dtype=torch.bool)
        eye = eye.unsqueeze(-1).unsqueeze(-1)  # (K, K, 1, 1)
        agree = agree.masked_fill(eye, 0.0)

        n_pairs = K * (K - 1)  # off-diagonal entries
        # Sum over model pairs → (B, B), then average
        pair_agree = agree.sum(dim=(0, 1)) / max(n_pairs, 1)  # (B, B)

        # Per-sample: average over j != i
        # Zero out diagonal
        diag_mask = torch.eye(B, device=x.device, dtype=torch.bool)
        pair_agree = pair_agree.masked_fill(diag_mask, 0.0)
        scores = pair_agree.sum(dim=1) / max(B - 1, 1)  # (B,)
        return scores


# ---------------------------------------------------------------------------
# RewardCalibrator
# ---------------------------------------------------------------------------

class RewardCalibrator:
    """Linearly calibrate raw reward scores to align with human scores.

    Fits ``calibrated = a * raw + b`` via ordinary least squares, then applies
    the transform on new data.
    """

    def __init__(self) -> None:
        self._a: float = 1.0
        self._b: float = 0.0
        self._fitted: bool = False

    def fit(self, raw_scores: Tensor, human_scores: Tensor) -> None:
        """Fit linear coefficients ``a`` and ``b`` via least squares.

        Args:
            raw_scores:   ``(N,)`` model-produced reward scores.
            human_scores: ``(N,)`` corresponding human preference scores.
        """
        raw = raw_scores.float().view(-1)
        human = human_scores.float().view(-1)
        N = raw.shape[0]

        # Build design matrix [raw, 1] of shape (N, 2)
        ones = torch.ones(N, device=raw.device, dtype=raw.dtype)
        A = torch.stack([raw, ones], dim=1)  # (N, 2)

        # Least-squares: (A^T A)^{-1} A^T human
        AtA = A.t().mm(A)   # (2, 2)
        Ath = A.t().mv(human)  # (2,)

        # Solve via pseudo-inverse for numerical stability
        coeffs = torch.linalg.lstsq(AtA, Ath.unsqueeze(1)).solution.squeeze(1)  # (2,)
        self._a = coeffs[0].item()
        self._b = coeffs[1].item()
        self._fitted = True

    def calibrate(self, raw_scores: Tensor) -> Tensor:
        """Apply the fitted linear transform.

        Args:
            raw_scores: ``(N,)`` raw reward scores.

        Returns:
            ``(N,)`` calibrated scores.
        """
        return self._a * raw_scores.float() + self._b

    def calibration_error(self, raw_scores: Tensor, targets: Tensor) -> float:
        """Root-mean-squared error between calibrated scores and targets.

        Args:
            raw_scores: ``(N,)`` raw reward scores.
            targets:    ``(N,)`` ground-truth scores.

        Returns:
            RMSE as a Python float.
        """
        calibrated = self.calibrate(raw_scores)
        rmse = torch.sqrt(((calibrated - targets.float()) ** 2).mean())
        return rmse.item()


# ---------------------------------------------------------------------------
# RewardAgreementFilter
# ---------------------------------------------------------------------------

class RewardAgreementFilter:
    """Filter samples and pairs based on ensemble agreement / uncertainty.

    Args:
        config: :class:`EnsembleConfig` — uses :attr:`~EnsembleConfig.uncertainty_threshold`.
    """

    def __init__(self, config: EnsembleConfig) -> None:
        self.config = config

    def filter_by_agreement(
        self,
        x: Tensor,
        ensemble: RewardEnsemble,
    ) -> Tuple[Tensor, Tensor]:
        """Keep samples whose cross-model std is below the uncertainty threshold.

        Args:
            x:        ``(B, d)`` input features.
            ensemble: Fitted :class:`RewardEnsemble`.

        Returns:
            Tuple ``(kept_x, kept_mask)`` where *kept_x* is ``(M, d)`` and
            *kept_mask* is a boolean ``(B,)`` tensor.
        """
        with torch.no_grad():
            _, std = ensemble.predict(x)  # (B,)

        kept_mask = std <= self.config.uncertainty_threshold  # (B,) bool
        kept_x = x[kept_mask]
        return kept_x, kept_mask

    def high_confidence_pairs(
        self,
        x_w: Tensor,
        x_l: Tensor,
        ensemble: RewardEnsemble,
        margin: float = 0.1,
    ) -> Tensor:
        """Select pairs where the ensemble is confident the winner beats the loser.

        Accepts pair *i* when:
        - ``mean_r_w[i] - mean_r_l[i] > margin``
        - ``std_r_w[i] <= uncertainty_threshold``
        - ``std_r_l[i] <= uncertainty_threshold``

        Args:
            x_w:      ``(B, d)`` winning-side features.
            x_l:      ``(B, d)`` losing-side features.
            ensemble: :class:`RewardEnsemble` to score features.
            margin:   Minimum reward margin required for acceptance.

        Returns:
            ``LongTensor`` of accepted pair indices, shape ``(M,)``.
        """
        with torch.no_grad():
            mean_w, std_w = ensemble.predict(x_w)  # (B,)
            mean_l, std_l = ensemble.predict(x_l)  # (B,)

        threshold = self.config.uncertainty_threshold
        margin_ok = (mean_w - mean_l) > margin       # (B,) bool
        low_unc_w = std_w <= threshold               # (B,) bool
        low_unc_l = std_l <= threshold               # (B,) bool
        accepted = margin_ok & low_unc_w & low_unc_l

        return accepted.nonzero(as_tuple=False).squeeze(1).long()
