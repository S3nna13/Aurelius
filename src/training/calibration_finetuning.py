"""
Calibration Fine-Tuning
=======================
Techniques to make a language model's output probabilities better reflect
true likelihoods.

Classes:
    LabelSmoothingLoss       - Label-smoothed cross-entropy (KL form)
    TemperatureScalingTrainer - Post-hoc scalar temperature calibration
    VectorScalingTrainer      - Per-class temperature + bias calibration
    FocalLoss                 - Focal loss for hard-example weighting
    MixupTrainer              - Mixup data augmentation for calibration
    CalibrationBenchmark      - ECE / MCE / Brier score / reliability diagram
    CalibrationConfig         - Dataclass of default hyper-parameters
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# CalibrationConfig
# ---------------------------------------------------------------------------

@dataclass
class CalibrationConfig:
    """Default hyper-parameters for calibration fine-tuning."""
    smoothing: float = 0.1
    gamma: float = 2.0
    alpha: float = 0.2
    n_bins: int = 15
    lr: float = 0.01
    n_steps: int = 100


# ---------------------------------------------------------------------------
# LabelSmoothingLoss
# ---------------------------------------------------------------------------

class LabelSmoothingLoss(nn.Module):
    """Label-smoothed cross-entropy loss.

    Smooth target distribution:
        q(k) = (1 - eps) * one_hot(k) + eps / V

    Loss = KL(q || p_theta) up to a constant
         = -sum_k q(k) * log_softmax(logits)[k]
    """

    def __init__(
        self,
        vocab_size: int,
        smoothing: float = 0.1,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits:  [B, T, V]  raw (un-normalised) scores
            targets: [B, T]     integer class indices

        Returns:
            scalar loss
        """
        B, T, V = logits.shape
        eps = self.smoothing

        log_probs = F.log_softmax(logits, dim=-1)           # [B, T, V]

        # Build smooth target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, eps / V)
            mask = targets != self.ignore_index             # [B, T]
            valid_targets = targets.clone()
            valid_targets[~mask] = 0                        # avoid index error
            smooth_targets.scatter_(
                2,
                valid_targets.unsqueeze(-1),                # [B, T, 1]
                (1.0 - eps) + eps / V,
            )

        # KL loss per position: -sum_k q(k) * log p_theta(k)
        loss_per_token = -(smooth_targets * log_probs).sum(dim=-1)  # [B, T]

        # Mask out ignored positions
        if self.ignore_index is not None:
            loss_per_token = loss_per_token * mask.float()
            n_valid = mask.float().sum().clamp(min=1.0)
            return loss_per_token.sum() / n_valid

        return loss_per_token.mean()


# ---------------------------------------------------------------------------
# TemperatureScalingTrainer
# ---------------------------------------------------------------------------

class TemperatureScalingTrainer:
    """Post-hoc temperature scaling for multi-class classifiers.

    A single scalar T is learned on a held-out validation set so that
    the NLL of scaled_logits(logits) = logits / T is minimised.
    """

    def __init__(self, model: nn.Module, lr: float = 0.01) -> None:
        self.model = model
        self.lr = lr
        self.temperature = nn.Parameter(torch.ones(1))

    # ------------------------------------------------------------------
    def calibrate(
        self,
        logits: Tensor,
        labels: Tensor,
        n_steps: int = 100,
    ) -> float:
        """Learn temperature to minimise NLL.

        Args:
            logits: [N, C]
            labels: [N]   integer class indices
            n_steps: gradient steps

        Returns:
            Final positive temperature (float).
        """
        self.temperature = nn.Parameter(torch.ones(1, dtype=logits.dtype))
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=self.lr, max_iter=n_steps
        )

        logits_detached = logits.detach()
        labels_detached = labels.detach()

        def closure() -> Tensor:
            optimizer.zero_grad()
            scaled = logits_detached / self.temperature.clamp(min=1e-6)
            loss = F.cross_entropy(scaled, labels_detached)
            loss.backward()
            return loss

        optimizer.step(closure)

        return float(self.temperature.clamp(min=1e-6).item())

    # ------------------------------------------------------------------
    def scaled_logits(self, logits: Tensor) -> Tensor:
        """Divide logits by learned temperature."""
        T = self.temperature.clamp(min=1e-6).to(logits.device)
        return logits / T

    # ------------------------------------------------------------------
    @staticmethod
    def ece(
        probs: Tensor,
        labels: Tensor,
        n_bins: int = 15,
    ) -> float:
        """Expected Calibration Error.

        ECE = sum_b (|B_b| / N) * |acc_b - conf_b|

        Args:
            probs:  [N, C]  softmax probabilities
            labels: [N]     integer ground-truth class indices
            n_bins: number of equal-width confidence bins

        Returns:
            ECE in [0, 1].
        """
        confidences, predictions = probs.max(dim=1)    # [N]
        accuracies = predictions.eq(labels).float()    # [N]
        N = float(labels.size(0))

        bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
        ece_val = 0.0
        for i in range(n_bins):
            lo, hi = bin_boundaries[i].item(), bin_boundaries[i + 1].item()
            if i == n_bins - 1:
                in_bin = (confidences >= lo) & (confidences <= hi)
            else:
                in_bin = (confidences >= lo) & (confidences < hi)
            n_in_bin = in_bin.float().sum().item()
            if n_in_bin > 0:
                acc_bin = accuracies[in_bin].mean().item()
                conf_bin = confidences[in_bin].mean().item()
                ece_val += (n_in_bin / N) * abs(acc_bin - conf_bin)

        return float(ece_val)


# ---------------------------------------------------------------------------
# VectorScalingTrainer
# ---------------------------------------------------------------------------

class VectorScalingTrainer:
    """Per-class temperature (W) and bias (b) calibration.

    scaled_logits = W * logits + b   (element-wise along class axis)
    """

    def __init__(self, n_classes: int, lr: float = 0.01) -> None:
        self.n_classes = n_classes
        self.lr = lr
        self.W = nn.Parameter(torch.ones(n_classes))
        self.b = nn.Parameter(torch.zeros(n_classes))

    # ------------------------------------------------------------------
    def calibrate(
        self,
        logits: Tensor,
        labels: Tensor,
        n_steps: int = 100,
    ) -> None:
        """Learn W and b to minimise NLL.

        Args:
            logits: [N, C]
            labels: [N]
            n_steps: gradient steps
        """
        self.W = nn.Parameter(torch.ones(self.n_classes, dtype=logits.dtype))
        self.b = nn.Parameter(torch.zeros(self.n_classes, dtype=logits.dtype))
        optimizer = torch.optim.Adam([self.W, self.b], lr=self.lr)

        logits_detached = logits.detach()
        labels_detached = labels.detach()

        for _ in range(n_steps):
            optimizer.zero_grad()
            scaled = logits_detached * self.W + self.b
            loss = F.cross_entropy(scaled, labels_detached)
            loss.backward()
            optimizer.step()

    # ------------------------------------------------------------------
    def scaled_logits(self, logits: Tensor) -> Tensor:
        """Apply per-class scaling: W * logits + b."""
        W = self.W.to(logits.device)
        b = self.b.to(logits.device)
        return logits * W + b


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for sequence modelling.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Down-weights easy (high-confidence) examples so training focuses on
    hard tokens.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Tensor] = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)   # [C] or None
        self.ignore_index = ignore_index

    # ------------------------------------------------------------------
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits:  [B, T, V]
            targets: [B, T]

        Returns:
            scalar focal loss
        """
        B, T, V = logits.shape

        # Flatten to [B*T, V] and [B*T]
        flat_logits = logits.reshape(-1, V)
        flat_targets = targets.reshape(-1)

        mask = flat_targets != self.ignore_index
        valid_logits = flat_logits[mask]
        valid_targets = flat_targets[mask]

        if valid_targets.numel() == 0:
            return logits.sum() * 0.0

        log_probs = F.log_softmax(valid_logits, dim=-1)     # [N, V]
        probs = log_probs.exp()                             # [N, V]

        # Gather probability of the correct class
        log_pt = log_probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)  # [N]
        pt = probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)          # [N]

        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(valid_targets.device)[valid_targets]
            focal_weight = focal_weight * alpha_t

        loss = -(focal_weight * log_pt)
        return loss.mean()


# ---------------------------------------------------------------------------
# MixupTrainer
# ---------------------------------------------------------------------------

class MixupTrainer:
    """Mixup data augmentation for calibration.

    Mixup interpolates between pairs of training examples in embedding
    space and computes the interpolated cross-entropy loss.
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.2,
        lr: float = 1e-4,
    ) -> None:
        self.model = model
        self.alpha = alpha
        self.lr = lr
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    def mixup_batch(
        self,
        input_ids_a: Tensor,
        input_ids_b: Tensor,
    ) -> Tuple[Tensor, float]:
        """Mix two batches in embedding space.

        Args:
            input_ids_a: [B, T] integer token ids
            input_ids_b: [B, T] integer token ids

        Returns:
            mixed_embed: [B, T, d]  interpolated embeddings
            lam:         mixing coefficient in [0, 1]
        """
        if self.alpha > 0:
            dist = torch.distributions.Beta(
                torch.tensor(self.alpha), torch.tensor(self.alpha)
            )
            lam = float(dist.sample().item())
        else:
            lam = 1.0

        embed = self.model.get_input_embeddings()   # nn.Embedding

        emb_a = embed(input_ids_a).detach()         # [B, T, d]
        emb_b = embed(input_ids_b).detach()         # [B, T, d]
        mixed_embed = lam * emb_a + (1.0 - lam) * emb_b
        return mixed_embed, lam

    # ------------------------------------------------------------------
    def train_step(
        self,
        model: nn.Module,
        input_ids_a: Tensor,
        input_ids_b: Tensor,
        labels_a: Tensor,
        labels_b: Tensor,
    ) -> Tensor:
        """One mixup training step.

        Loss = lam * CE(logits, labels_a) + (1 - lam) * CE(logits, labels_b)

        Args:
            model:       the language model
            input_ids_a: [B, T]
            input_ids_b: [B, T]
            labels_a:    [B, T]
            labels_b:    [B, T]

        Returns:
            scalar loss Tensor
        """
        self.optimizer.zero_grad()

        mixed_embed, lam = self.mixup_batch(input_ids_a, input_ids_b)

        # Forward pass using mixed embeddings (inputs_embeds path)
        logits = model(inputs_embeds=mixed_embed)   # [B, T, V]

        B, T, V = logits.shape
        flat_logits = logits.reshape(-1, V)
        flat_a = labels_a.reshape(-1)
        flat_b = labels_b.reshape(-1)

        loss_a = F.cross_entropy(flat_logits, flat_a, ignore_index=-100)
        loss_b = F.cross_entropy(flat_logits, flat_b, ignore_index=-100)
        loss = lam * loss_a + (1.0 - lam) * loss_b

        loss.backward()
        self.optimizer.step()
        return loss.detach()


# ---------------------------------------------------------------------------
# CalibrationBenchmark
# ---------------------------------------------------------------------------

class CalibrationBenchmark:
    """Diagnostic metrics for classifier calibration."""

    # ------------------------------------------------------------------
    @staticmethod
    def _bin_stats(
        probs: Tensor,
        labels: Tensor,
        n_bins: int,
    ) -> Tuple[List[float], List[float], List[int]]:
        """Shared helper that computes per-bin accuracy / confidence / count."""
        confidences, predictions = probs.max(dim=1)
        accuracies = predictions.eq(labels).float()
        N = labels.size(0)

        bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
        bin_accs: List[float] = []
        bin_confs: List[float] = []
        bin_counts: List[int] = []

        for i in range(n_bins):
            lo = bin_boundaries[i].item()
            hi = bin_boundaries[i + 1].item()
            in_bin = (confidences >= lo) & (confidences < hi)
            # Include hi edge in last bin
            if i == n_bins - 1:
                in_bin = (confidences >= lo) & (confidences <= hi)
            n_in = int(in_bin.float().sum().item())
            bin_counts.append(n_in)
            if n_in > 0:
                bin_accs.append(accuracies[in_bin].mean().item())
                bin_confs.append(confidences[in_bin].mean().item())
            else:
                bin_accs.append(0.0)
                bin_confs.append(0.0)

        return bin_accs, bin_confs, bin_counts

    # ------------------------------------------------------------------
    def reliability_diagram_data(
        self,
        probs: Tensor,
        labels: Tensor,
        n_bins: int = 10,
    ) -> Dict[str, object]:
        """Compute data needed to draw a reliability diagram.

        Args:
            probs:  [N, C]  softmax probabilities
            labels: [N]     ground-truth integer class indices
            n_bins: number of bins

        Returns:
            dict with keys "bin_accs", "bin_confs", "bin_counts"
        """
        bin_accs, bin_confs, bin_counts = self._bin_stats(probs, labels, n_bins)
        return {
            "bin_accs": bin_accs,
            "bin_confs": bin_confs,
            "bin_counts": bin_counts,
        }

    # ------------------------------------------------------------------
    def mce(
        self,
        probs: Tensor,
        labels: Tensor,
        n_bins: int = 15,
    ) -> float:
        """Maximum Calibration Error.

        MCE = max_b |acc_b - conf_b|

        Args:
            probs:  [N, C]
            labels: [N]
            n_bins: number of bins

        Returns:
            MCE in [0, 1].
        """
        bin_accs, bin_confs, bin_counts = self._bin_stats(probs, labels, n_bins)
        max_err = 0.0
        for n, acc, conf in zip(bin_counts, bin_accs, bin_confs):
            if n > 0:
                max_err = max(max_err, abs(acc - conf))
        return float(max_err)

    # ------------------------------------------------------------------
    @staticmethod
    def brier_score(probs: Tensor, labels: Tensor) -> float:
        """Brier score: mean squared error of predicted probabilities.

        BS = (1/N) sum_i sum_k (p_{i,k} - y_{i,k})^2

        Args:
            probs:  [N, C]
            labels: [N]

        Returns:
            Brier score (lower is better, perfect = 0).
        """
        N, C = probs.shape
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        return float(((probs - one_hot) ** 2).sum(dim=1).mean().item())
