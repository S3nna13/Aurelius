"""
Uncertainty estimation for LLMs.

Implements MC Dropout, Deep Ensembles, conformal calibration, entropy-based
metrics, temperature calibration, and an aggregate benchmark utility.

Pure stdlib + torch — no third-party dependencies.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPS = 1e-8


def _safe_log(x: Tensor) -> Tensor:
    return torch.log(x.clamp(min=_EPS))


def _entropy_from_probs(probs: Tensor) -> Tensor:
    """Shannon entropy H = -sum(p * log p) along last dim."""
    return -(probs * _safe_log(probs)).sum(dim=-1)


# ---------------------------------------------------------------------------
# MC Dropout Estimator
# ---------------------------------------------------------------------------


class MCDropoutEstimator:
    """Uncertainty via multiple dropout-enabled stochastic forward passes.

    Parameters
    ----------
    model:
        Any nn.Module whose forward(input_ids) returns logits (B, T, V).
    n_samples:
        Number of stochastic forward passes.
    dropout_rate:
        Metadata; the model must already contain Dropout layers.
    """

    def __init__(self, model: nn.Module, n_samples: int = 20, dropout_rate: float = 0.1) -> None:
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

    def enable_dropout(self) -> None:
        """Put model in train mode (activates dropout) but freeze BN/LN stats."""
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                module.eval()

    def predict(self, input_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run n_samples stochastic forward passes and aggregate.

        Returns
        -------
        mean_logits : (B, T, V)
        var_logits  : (B, T, V)  epistemic uncertainty
        entropy     : (B, T)     entropy of mean predictive distribution
        """
        self.enable_dropout()
        samples: list[Tensor] = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self.model(input_ids)  # (B, T, V)
                samples.append(logits)

        logits_stack = torch.stack(samples, dim=0)  # (n_samples, B, T, V)
        mean_logits = logits_stack.mean(dim=0)  # (B, T, V)
        var_logits = logits_stack.var(dim=0)  # (B, T, V)

        mean_probs = F.softmax(mean_logits, dim=-1)  # (B, T, V)
        entropy = _entropy_from_probs(mean_probs)  # (B, T)

        return mean_logits, var_logits, entropy

    def predictive_entropy(self, logits_samples: Tensor) -> Tensor:
        """Total uncertainty: entropy of the mean predictive distribution.

        Parameters
        ----------
        logits_samples : (n_samples, B, T, V)

        Returns
        -------
        Tensor of shape (B, T)
        """
        mean_probs = F.softmax(logits_samples, dim=-1).mean(dim=0)  # (B, T, V)
        return _entropy_from_probs(mean_probs)  # (B, T)

    def mutual_information(self, logits_samples: Tensor) -> Tensor:
        """Epistemic uncertainty = predictive_entropy - mean(entropy per sample).

        Parameters
        ----------
        logits_samples : (n_samples, B, T, V)

        Returns
        -------
        Tensor of shape (B, T), non-negative.
        """
        pred_ent = self.predictive_entropy(logits_samples)  # (B, T)

        per_sample_probs = F.softmax(logits_samples, dim=-1)  # (n_s, B, T, V)
        per_sample_ent = _entropy_from_probs(per_sample_probs)  # (n_s, B, T)
        mean_cond_ent = per_sample_ent.mean(dim=0)  # (B, T)

        return (pred_ent - mean_cond_ent).clamp(min=0.0)


# ---------------------------------------------------------------------------
# Deep Ensemble
# ---------------------------------------------------------------------------


class DeepEnsemble:
    """Uncertainty via an ensemble of independently trained models.

    Parameters
    ----------
    models:
        Either a list of nn.Module instances, or a single nn.Module class
        when n_models is also given.
    n_models:
        If provided and ``models`` is callable (not a list), instantiate
        ``n_models`` independent copies.
    """

    def __init__(
        self,
        models: list[nn.Module] | nn.Module,
        n_models: int | None = None,
    ) -> None:
        if isinstance(models, list):
            self.models: list[nn.Module] = models
        elif n_models is not None and callable(models):
            self.models = [models() for _ in range(n_models)]
        else:
            raise ValueError("Pass either a list of models, or a model factory + n_models.")

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Run all ensemble members and aggregate.

        Returns
        -------
        mean_logits : (B, T, V)  mean of member logits
        uncertainty : (B, T)     std over member softmax probabilities
        """
        all_logits: list[Tensor] = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                all_logits.append(model(input_ids))

        stacked = torch.stack(all_logits, dim=0)  # (M, B, T, V)
        mean_logits = stacked.mean(dim=0)  # (B, T, V)

        probs = F.softmax(stacked, dim=-1)  # (M, B, T, V)
        uncertainty = probs.std(dim=0).mean(dim=-1)  # (B, T)

        return mean_logits, uncertainty

    def calibrated_uncertainty(self, logits: Tensor, labels: Tensor) -> tuple[float, float]:
        """ECE and MCE via equal-width confidence binning.

        Parameters
        ----------
        logits : (B, T, V)
        labels : (B, T)     ground-truth token indices

        Returns
        -------
        ece : float in [0, 1]
        mce : float in [0, 1]
        """
        probs = F.softmax(logits, dim=-1)  # (B, T, V)
        confidence, predicted = probs.max(dim=-1)  # (B, T)
        correct = predicted.eq(labels).float()  # (B, T)

        confidence_flat = confidence.reshape(-1)
        correct_flat = correct.reshape(-1)

        n_bins = 10
        bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
        ece_sum = 0.0
        mce = 0.0
        n_total = float(confidence_flat.numel())

        for i in range(n_bins):
            lo, hi = bin_edges[i].item(), bin_edges[i + 1].item()
            if i == n_bins - 1:
                mask = (confidence_flat >= lo) & (confidence_flat <= hi)
            else:
                mask = (confidence_flat >= lo) & (confidence_flat < hi)
            if mask.sum().item() == 0:
                continue
            bin_conf = confidence_flat[mask].mean().item()
            bin_acc = correct_flat[mask].mean().item()
            bin_size = float(mask.sum().item())
            gap = abs(bin_conf - bin_acc)
            ece_sum += gap * bin_size / n_total
            if gap > mce:
                mce = gap

        return float(ece_sum), float(mce)


# ---------------------------------------------------------------------------
# Entropy Thresholder
# ---------------------------------------------------------------------------


class EntropyThresholder:
    """Filter unreliable predictions based on token-level entropy."""

    def __init__(self, threshold: float = 2.0) -> None:
        self.threshold = threshold

    def is_uncertain(self, entropy: Tensor) -> Tensor:
        """Bool mask: True where entropy > threshold.

        Parameters
        ----------
        entropy : (B, T)

        Returns
        -------
        Tensor bool (B, T)
        """
        return entropy > self.threshold

    def filter_predictions(self, logits: Tensor, entropy: Tensor) -> dict:
        """Partition positions into confident / uncertain.

        Parameters
        ----------
        logits  : (B, T, V)
        entropy : (B, T)

        Returns
        -------
        dict with keys:
            'confident'              : bool Tensor (B, T)
            'uncertain'              : bool Tensor (B, T)
            'mean_confident_entropy' : float
            'coverage'               : float
        """
        uncertain_mask = self.is_uncertain(entropy)
        confident_mask = ~uncertain_mask

        n_total = float(entropy.numel())
        n_confident = float(confident_mask.sum().item())

        if n_confident > 0:
            mean_confident_entropy = float(entropy[confident_mask].mean().item())
        else:
            mean_confident_entropy = 0.0

        coverage = n_confident / n_total if n_total > 0 else 0.0

        return {
            "confident": confident_mask,
            "uncertain": uncertain_mask,
            "mean_confident_entropy": mean_confident_entropy,
            "coverage": coverage,
        }


# ---------------------------------------------------------------------------
# Temperature Calibration
# ---------------------------------------------------------------------------


class TemperatureCalibration:
    """Post-hoc temperature scaling to calibrate model confidence."""

    def __init__(self) -> None:
        self._optimal_temperature: float | None = None

    def fit(
        self,
        logits: Tensor,
        labels: Tensor,
        n_iters: int = 100,
    ) -> float:
        """Grid-search for T in [0.1, 10.0] that minimises NLL.

        Parameters
        ----------
        logits : (B, T, V) or (N, V)
        labels : (B, T)    or (N,)

        Returns
        -------
        optimal_temperature : float
        """
        log_temps = torch.linspace(math.log(0.1), math.log(10.0), n_iters)
        temps = log_temps.exp()

        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_labels = labels.reshape(-1).long()

        best_nll = float("inf")
        best_temp = 1.0

        for t in temps:
            t_val = float(t.item())
            scaled = flat_logits / t_val
            nll = F.cross_entropy(scaled, flat_labels).item()
            if nll < best_nll:
                best_nll = nll
                best_temp = t_val

        self._optimal_temperature = best_temp
        return best_temp

    def calibrate(self, logits: Tensor, temperature: float) -> Tensor:
        """Divide logits by temperature (same shape as input).

        Parameters
        ----------
        logits      : (..., V)
        temperature : scalar > 0
        """
        return logits / temperature

    def expected_calibration_error(
        self,
        probs: Tensor,
        labels: Tensor,
        n_bins: int = 10,
    ) -> float:
        """ECE via equal-width confidence bins.

        Parameters
        ----------
        probs  : (B, T, V) or (N, V)
        labels : (B, T)    or (N,)

        Returns
        -------
        ece : float in [0, 1]
        """
        flat_probs = probs.reshape(-1, probs.size(-1))
        flat_labels = labels.reshape(-1).long()

        confidence, predicted = flat_probs.max(dim=-1)
        correct = predicted.eq(flat_labels).float()

        bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = float(confidence.numel())

        for i in range(n_bins):
            lo, hi = bin_edges[i].item(), bin_edges[i + 1].item()
            if i == n_bins - 1:
                mask = (confidence >= lo) & (confidence <= hi)
            else:
                mask = (confidence >= lo) & (confidence < hi)
            if mask.sum().item() == 0:
                continue
            bin_conf = confidence[mask].mean().item()
            bin_acc = correct[mask].mean().item()
            ece += abs(bin_conf - bin_acc) * float(mask.sum().item()) / n_total

        return float(ece)


# ---------------------------------------------------------------------------
# Uncertainty Benchmark
# ---------------------------------------------------------------------------


class UncertaintyBenchmark:
    """Aggregate uncertainty evaluation utilities."""

    def __init__(self) -> None:
        pass

    def auroc_uncertainty(
        self,
        uncertainty_scores: Tensor,
        is_wrong: Tensor,
    ) -> float:
        """AUROC for detecting incorrect predictions via uncertainty.

        Parameters
        ----------
        uncertainty_scores : (N,) float  — higher means more uncertain
        is_wrong           : (N,) bool   — True when prediction is incorrect

        Returns
        -------
        auroc : float in [0, 1]
        """
        scores = uncertainty_scores.reshape(-1).float()
        labels = is_wrong.reshape(-1).float()

        n = scores.numel()
        if n == 0:
            return 0.5

        n_pos = float(labels.sum().item())
        n_neg = float(n - n_pos)

        if n_pos == 0 or n_neg == 0:
            return 1.0 if n_pos > 0 else 0.0

        sorted_idx = torch.argsort(scores, descending=True)
        sorted_labels = labels[sorted_idx]

        tpr = torch.cumsum(sorted_labels, dim=0) / n_pos
        fpr = torch.cumsum(1.0 - sorted_labels, dim=0) / n_neg

        tpr = torch.cat([torch.zeros(1), tpr])
        fpr = torch.cat([torch.zeros(1), fpr])

        auroc = float(torch.trapz(tpr, fpr).abs().item())
        return float(min(max(auroc, 0.0), 1.0))

    def brier_score(self, probs: Tensor, labels: Tensor) -> float:
        """Mean squared error between predicted probs and one-hot labels.

        Parameters
        ----------
        probs  : (B, T, V) or (N, V)
        labels : (B, T)    or (N,)

        Returns
        -------
        brier : float in [0, 1]  (normalised by 2)
        """
        flat_probs = probs.reshape(-1, probs.size(-1)).float()
        flat_labels = labels.reshape(-1).long()

        one_hot = torch.zeros_like(flat_probs)
        one_hot.scatter_(1, flat_labels.unsqueeze(1), 1.0)

        raw = ((flat_probs - one_hot) ** 2).sum(dim=-1).mean().item()
        return float(raw / 2.0)
