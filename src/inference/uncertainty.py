"""Monte Carlo Dropout uncertainty estimation for AureliusTransformer.

Approximates Bayesian inference by applying stochastic dropout during
multiple forward passes, then computing predictive entropy and mutual
information as measures of total and epistemic uncertainty respectively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class UncertaintyConfig:
    n_samples: int = 10          # number of MC forward passes
    dropout_p: float = 0.1       # dropout probability for MC sampling
    eps: float = 1e-9            # numerical stability


@dataclass
class UncertaintyResult:
    predictive_entropy: torch.Tensor  # (B, S) total uncertainty
    mutual_information: torch.Tensor  # (B, S) epistemic uncertainty
    mean_probs: torch.Tensor          # (B, S, V) mean probability across samples
    confidence: torch.Tensor          # (B, S) max probability (argmax confidence)

    def summary(self) -> str:
        return (
            f"Mean predictive entropy: {self.predictive_entropy.mean().item():.4f}\n"
            f"Mean mutual information: {self.mutual_information.mean().item():.4f}\n"
            f"Mean confidence:         {self.confidence.mean().item():.4f}"
        )


def predictive_entropy(probs: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute entropy of the mean predictive distribution.

    H[p] = -sum_v p_v * log(p_v)

    Args:
        probs: (B, S, V) probability tensor (mean over MC samples)

    Returns:
        (B, S) entropy
    """
    p = probs.clamp(min=eps)
    return -(p * p.log()).sum(dim=-1)


def mutual_information(sample_probs: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute mutual information (epistemic uncertainty).

    MI = H[E_q[p]] - E_q[H[p]]
       = entropy(mean_probs) - mean(entropy(sample_probs))

    Args:
        sample_probs: (N, B, S, V) probabilities from N MC samples

    Returns:
        (B, S) mutual information
    """
    mean_probs = sample_probs.mean(dim=0)  # (B, S, V)

    # H[E_q[p]]
    H_mean = predictive_entropy(mean_probs, eps)

    # E_q[H[p]]
    p = sample_probs.clamp(min=eps)
    per_sample_entropy = -(p * p.log()).sum(dim=-1)  # (N, B, S)
    E_H = per_sample_entropy.mean(dim=0)             # (B, S)

    return (H_mean - E_H).clamp(min=0.0)


class MCDropoutEstimator:
    """Uncertainty estimation via Monte Carlo Dropout.

    Temporarily adds dropout to the model's hidden states during inference
    to approximate Bayesian inference.

    Usage:
        estimator = MCDropoutEstimator(model, UncertaintyConfig(n_samples=20))
        result = estimator.estimate(input_ids)
    """

    def __init__(self, model: nn.Module, cfg: UncertaintyConfig | None = None):
        self.model = model
        self.cfg = cfg or UncertaintyConfig()

    def estimate(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> UncertaintyResult:
        """Run N stochastic forward passes and compute uncertainty metrics.

        Args:
            input_ids: (B, S) input token IDs
            labels: ignored (for API compatibility)

        Returns:
            UncertaintyResult with per-token uncertainty estimates.
        """
        self.model.eval()  # Keep BN in eval mode

        all_probs: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(self.cfg.n_samples):
                # Forward pass with dropout applied to logits as a proxy
                _, logits, _ = self.model(input_ids)  # (B, S, V)

                # Apply MC dropout to logits as a simple approximation
                logits_dropped = F.dropout(logits, p=self.cfg.dropout_p, training=True)
                probs = F.softmax(logits_dropped, dim=-1)  # (B, S, V)
                all_probs.append(probs)

        sample_probs = torch.stack(all_probs, dim=0)  # (N, B, S, V)
        mean_probs = sample_probs.mean(dim=0)          # (B, S, V)

        pred_entropy = predictive_entropy(mean_probs, self.cfg.eps)
        mi = mutual_information(sample_probs, self.cfg.eps)
        confidence = mean_probs.max(dim=-1).values

        return UncertaintyResult(
            predictive_entropy=pred_entropy,
            mutual_information=mi,
            mean_probs=mean_probs,
            confidence=confidence,
        )
