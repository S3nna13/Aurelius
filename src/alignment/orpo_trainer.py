"""Aurelius — ORPO Trainer: Odds Ratio Preference Optimization.

Hong et al. (2024) "ORPO: Monolithic Preference Optimization without Reference
Model" — https://arxiv.org/abs/2403.07691

ORPO eliminates the need for a reference model entirely by combining a standard
SFT (NLL) loss on chosen responses with an odds-ratio preference penalty:

    L_ORPO = L_SFT + lambda_or * L_OR

    where:
        L_SFT    = -mean(log p(y_w | x))      # NLL on chosen
        odds(y)  = p(y|x) / (1 - p(y|x))
        log_OR   = log_odds(y_w) - log_odds(y_l)
        L_OR     = -mean(logsigmoid(log_OR))
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ORPOConfig:
    """Configuration for ORPOTrainer.

    Attributes:
        lambda_or: Weight applied to the odds-ratio loss component.
        eps: Small constant for numerical stability.
    """

    lambda_or: float = 0.1
    eps: float = 1e-8


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


@dataclass
class ORPOBatch:
    """Per-token log-probs and masks for a chosen/rejected batch.

    Attributes:
        chosen_log_probs:   ``[B, T_w]`` per-token log-probs for chosen.
        rejected_log_probs: ``[B, T_l]`` per-token log-probs for rejected.
        chosen_mask:        ``[B, T_w]`` binary attention mask (1 = valid token).
        rejected_mask:      ``[B, T_l]`` binary attention mask (1 = valid token).
    """

    chosen_log_probs: torch.Tensor
    rejected_log_probs: torch.Tensor
    chosen_mask: torch.Tensor
    rejected_mask: torch.Tensor


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class ORPOTrainer:
    """ORPO — Odds Ratio Preference Optimization (no reference model).

    Usage::

        config  = ORPOConfig(lambda_or=0.1)
        trainer = ORPOTrainer(config)

        batch = ORPOBatch(
            chosen_log_probs=...,
            rejected_log_probs=...,
            chosen_mask=...,
            rejected_mask=...,
        )
        out = trainer.total_loss(batch)
        out["loss"].backward()
    """

    def __init__(self, config: ORPOConfig | None = None) -> None:
        self.config = config if config is not None else ORPOConfig()

    # ------------------------------------------------------------------
    # Core primitives
    # ------------------------------------------------------------------

    def sequence_log_prob(
        self,
        log_probs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean per-token log-prob over valid (unmasked) tokens.

        Args:
            log_probs: ``[B, T]`` per-token log-probabilities.
            mask:      ``[B, T]`` binary mask; 1 = valid token, 0 = padding.

        Returns:
            ``[B]`` mean log-prob for each sequence in the batch.
        """
        # Sum log-probs over valid positions, divide by valid token count
        masked = log_probs * mask
        valid_counts = mask.sum(dim=-1).clamp(min=1)  # avoid div-by-zero
        return masked.sum(dim=-1) / valid_counts

    def log_odds(self, seq_log_prob: torch.Tensor) -> torch.Tensor:
        """Compute log-odds from per-sequence mean log-probabilities.

        odds(y|x)      = exp(log_p) / (1 - exp(log_p) + eps)
        log_odds(y|x)  = log(odds + eps)

        Args:
            seq_log_prob: ``[B]`` mean log-prob per sequence (values < 0).

        Returns:
            ``[B]`` log-odds tensor.
        """
        eps = self.config.eps
        prob = seq_log_prob.exp()
        odds = prob / (1.0 - prob + eps)
        return torch.log(odds + eps)

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def sft_loss(
        self,
        chosen_log_probs: torch.Tensor,
        chosen_mask: torch.Tensor,
    ) -> torch.Tensor:
        """SFT (NLL) loss over all valid chosen-response tokens.

        L_SFT = -mean_{batch, tokens}(chosen_log_probs * mask)

        Args:
            chosen_log_probs: ``[B, T_w]`` per-token log-probs for chosen.
            chosen_mask:      ``[B, T_w]`` binary mask.

        Returns:
            Scalar loss tensor.
        """
        # Mean over all valid token positions across the entire batch
        total_log_prob = (chosen_log_probs * chosen_mask).sum()
        total_tokens = chosen_mask.sum().clamp(min=1)
        return -(total_log_prob / total_tokens)

    def odds_ratio_loss(self, batch: ORPOBatch) -> torch.Tensor:
        """Odds-ratio preference loss (L_OR).

        Computes:
            log_p_w  = sequence_log_prob(chosen)
            log_p_l  = sequence_log_prob(rejected)
            log_OR   = log_odds(log_p_w) - log_odds(log_p_l)
            L_OR     = -mean(logsigmoid(log_OR))

        Args:
            batch: ORPOBatch with chosen/rejected log-probs and masks.

        Returns:
            Scalar loss tensor.
        """
        log_p_w = self.sequence_log_prob(batch.chosen_log_probs, batch.chosen_mask)
        log_p_l = self.sequence_log_prob(batch.rejected_log_probs, batch.rejected_mask)

        log_or = self.log_odds(log_p_w) - self.log_odds(log_p_l)
        return -F.logsigmoid(log_or).mean()

    # ------------------------------------------------------------------
    # Combined loss
    # ------------------------------------------------------------------

    def total_loss(self, batch: ORPOBatch) -> dict[str, torch.Tensor]:
        """Compute the combined ORPO loss.

        L_ORPO = L_SFT + lambda_or * L_OR

        Args:
            batch: ORPOBatch.

        Returns:
            Dict with keys:
            - ``"loss"``            — combined total loss (scalar).
            - ``"sft_loss"``        — SFT component (scalar).
            - ``"or_loss"``         — odds-ratio component (scalar).
            - ``"log_odds_ratio"``  — mean log-OR across the batch (scalar).
        """
        cfg = self.config

        sft = self.sft_loss(batch.chosen_log_probs, batch.chosen_mask)
        or_loss = self.odds_ratio_loss(batch)

        # Compute log-OR for logging (reuse computations)
        log_p_w = self.sequence_log_prob(batch.chosen_log_probs, batch.chosen_mask)
        log_p_l = self.sequence_log_prob(batch.rejected_log_probs, batch.rejected_mask)
        log_or = self.log_odds(log_p_w) - self.log_odds(log_p_l)

        combined = sft + cfg.lambda_or * or_loss

        return {
            "loss": combined,
            "sft_loss": sft,
            "or_loss": or_loss,
            "log_odds_ratio": log_or.mean(),
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self, batch: ORPOBatch) -> dict[str, float]:
        """Compute diagnostic statistics for a batch (no-grad).

        Args:
            batch: ORPOBatch.

        Returns:
            Dict with keys:
            - ``"chosen_mean_logp"``     — mean sequence log-prob for chosen.
            - ``"rejected_mean_logp"``   — mean sequence log-prob for rejected.
            - ``"log_odds_ratio_mean"``  — mean log-OR across batch.
            - ``"reward_accuracy"``      — fraction of pairs where log-OR > 0
                                          (chosen preferred over rejected).
        """
        with torch.no_grad():
            log_p_w = self.sequence_log_prob(batch.chosen_log_probs, batch.chosen_mask)
            log_p_l = self.sequence_log_prob(batch.rejected_log_probs, batch.rejected_mask)
            log_or = self.log_odds(log_p_w) - self.log_odds(log_p_l)
            reward_acc = (log_or > 0).float().mean().item()

        return {
            "chosen_mean_logp": log_p_w.mean().item(),
            "rejected_mean_logp": log_p_l.mean().item(),
            "log_odds_ratio_mean": log_or.mean().item(),
            "reward_accuracy": reward_acc,
        }
