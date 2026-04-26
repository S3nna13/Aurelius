"""Uncertainty quantification for LLMs: MC-Dropout, entropy, epistemic/aleatoric split."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty estimation routines."""

    n_mc_samples: int = 10
    dropout_rate: float = 0.1
    temperature: float = 1.0
    top_k: int = 0
    use_ensemble: bool = False


# ---------------------------------------------------------------------------
# Core entropy helper
# ---------------------------------------------------------------------------


def _entropy(probs: Tensor) -> Tensor:
    """Shannon entropy over the last dimension of *probs*.

    Args:
        probs: (..., vocab) probability tensor.

    Returns:
        (...,) non-negative entropy values.
    """
    return -(probs * torch.log(probs + 1e-9)).sum(dim=-1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_predictive_entropy(probs: Tensor) -> Tensor:
    """Compute Shannon entropy of a probability distribution.

    Args:
        probs: (B, vocab) or (B, T, vocab) tensor of probabilities.

    Returns:
        (B,) or (B, T) entropy values.
    """
    return _entropy(probs)


def compute_mutual_information(probs_samples: Tensor) -> Tensor:
    """Compute mutual information (epistemic uncertainty) from MC samples.

    MI = H(E[p]) - E[H(p)]

    Args:
        probs_samples: (S, B, vocab) tensor -- S Monte Carlo samples.

    Returns:
        (B,) mutual information values (non-negative).
    """
    mean_probs = probs_samples.mean(dim=0)  # (B, vocab)
    h_mean = _entropy(mean_probs)  # (B,)
    h_samples = _entropy(probs_samples)  # (S, B)
    mean_h = h_samples.mean(dim=0)  # (B,)
    return torch.clamp(h_mean - mean_h, min=0.0)


def compute_token_uncertainty(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """Per-token predictive entropy after temperature scaling.

    Args:
        logits: (B, T, vocab) raw logits.
        temperature: positive scalar; values > 1 flatten the distribution.

    Returns:
        (B, T) per-token entropy.
    """
    scaled = logits / max(temperature, 1e-8)
    probs = F.softmax(scaled, dim=-1)
    return _entropy(probs)


def compute_sequence_confidence(log_probs: Tensor) -> Tensor:
    """Sequence-level confidence as exp(mean log-prob).

    Args:
        log_probs: (T,) per-token log probabilities (must be <= 0).

    Returns:
        Scalar confidence in (0, 1].
    """
    return torch.exp(log_probs.mean())


def detect_uncertainty_threshold(uncertainties: Tensor, threshold: float) -> Tensor:
    """Flag sequences whose mean uncertainty exceeds *threshold*.

    Args:
        uncertainties: (B, T) or (B,) per-token (or per-sequence) uncertainty.
        threshold: scalar decision boundary.

    Returns:
        (B,) bool tensor -- True where mean uncertainty > threshold.
    """
    if uncertainties.dim() == 1:
        mean_u = uncertainties
    else:
        mean_u = uncertainties.mean(dim=-1)
    return mean_u > threshold


def compute_epistemic_aleatoric_split(
    probs_samples: Tensor,
) -> tuple[Tensor, Tensor]:
    """Decompose total uncertainty into epistemic and aleatoric components.

    total     = H(E[p])            (B,)
    aleatoric = E[H(p)]            (B,)
    epistemic = total - aleatoric  (B,)

    Args:
        probs_samples: (S, B, vocab) MC sample probabilities.

    Returns:
        (epistemic, aleatoric) each of shape (B,).
    """
    mean_probs = probs_samples.mean(dim=0)  # (B, vocab)
    total = _entropy(mean_probs)  # (B,)
    aleatoric = _entropy(probs_samples).mean(dim=0)  # (B,)
    epistemic = torch.clamp(total - aleatoric, min=0.0)
    return epistemic, aleatoric


# ---------------------------------------------------------------------------
# MC-Dropout estimator
# ---------------------------------------------------------------------------


class MCDropoutEstimator:
    """Wraps an nn.Module to perform MC-Dropout uncertainty estimation."""

    def __init__(self, model: nn.Module, config: UncertaintyConfig) -> None:
        self.model = model
        self.config = config

    def enable_dropout(self) -> None:
        """Set model to eval mode but re-enable all Dropout sub-modules."""
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()

    def sample(self, input_ids: Tensor) -> Tensor:
        """Collect n_mc_samples forward passes with dropout active.

        Args:
            input_ids: (B, T) token indices.

        Returns:
            (n_mc_samples, B, T, vocab) logit tensor.
        """
        self.enable_dropout()
        samples = []
        with torch.no_grad():
            for _ in range(self.config.n_mc_samples):
                logits = self.model(input_ids)  # (B, T, vocab)
                samples.append(logits)
        return torch.stack(samples, dim=0)  # (S, B, T, vocab)

    def estimate(self, input_ids: Tensor) -> dict[str, Tensor]:
        """Full uncertainty estimate for input_ids.

        Returns a dict with:
            mean_probs         : (B, T, vocab)
            predictive_entropy : (B, T)
            mutual_information : (B, T)
        """
        logit_samples = self.sample(input_ids)  # (S, B, T, vocab)
        S, B, T, V = logit_samples.shape

        scaled = logit_samples / max(self.config.temperature, 1e-8)
        prob_samples = F.softmax(scaled, dim=-1)  # (S, B, T, vocab)

        mean_probs = prob_samples.mean(dim=0)  # (B, T, vocab)
        predictive_entropy = _entropy(mean_probs)  # (B, T)

        # Per-token mutual information: reshape to (S, B*T, vocab)
        prob_flat = prob_samples.view(S, B * T, V)
        mi_flat = compute_mutual_information(prob_flat)  # (B*T,)
        mutual_information = mi_flat.view(B, T)

        return {
            "mean_probs": mean_probs,
            "predictive_entropy": predictive_entropy,
            "mutual_information": mutual_information,
        }
