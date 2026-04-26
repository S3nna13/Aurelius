"""Language modeling loss functions for the Aurelius LLM project.

Implements multiple loss variants for language model training:
- Standard cross-entropy with ignore_index masking
- Label-smoothed cross-entropy
- Focal loss (Lin et al. 2017)
- Token-weighted loss
- Z-loss auxiliary regularization (Zoph et al. 2022, ST-MoE)
- Perplexity computation
- LMLoss nn.Module combining the above based on LossConfig
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class LossConfig:
    """Configuration for LMLoss module.

    Attributes:
        label_smoothing: Label smoothing factor in [0, 1). 0 disables smoothing.
        focal_gamma:     Focal loss gamma. Used only when label_smoothing == 0.
                         0 disables focal loss (falls back to plain CE).
        ignore_index:    Token id to exclude from loss computation (e.g. padding).
        reduction:       How to aggregate token losses. "mean" or "sum".
    """

    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    ignore_index: int = -100
    reduction: str = "mean"


# ---------------------------------------------------------------------------
# cross_entropy_loss
# ---------------------------------------------------------------------------


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Standard cross-entropy loss for language modeling.

    Args:
        logits:       (B, T, V) raw model logits.
        labels:       (B, T) long tensor of target token ids.
        ignore_index: Positions with this label are excluded from the loss.
        reduction:    "mean" returns a scalar, "none" returns per-token losses (B*T,).

    Returns:
        Scalar (reduction="mean") or 1-D tensor of per-token losses (reduction="none").
    """
    V = logits.shape[-1]
    logits_flat = logits.reshape(-1, V)  # (B*T, V)
    labels_flat = labels.reshape(-1)  # (B*T,)

    return F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=ignore_index,
        reduction=reduction,
    )


# ---------------------------------------------------------------------------
# label_smoothed_loss
# ---------------------------------------------------------------------------


def label_smoothed_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float = 0.1,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy with label smoothing.

    Combines hard-label CE and a uniform-distribution penalty:
        loss = (1 - s) * CE_hard  +  s * (-mean(log_softmax))

    The uniform component encourages the model to spread probability mass
    more evenly, preventing over-confident predictions.

    Args:
        logits:       (B, T, V) raw model logits.
        labels:       (B, T) long tensor of target token ids.
        smoothing:    Label smoothing factor s in [0, 1).
        ignore_index: Positions with this label are excluded.

    Returns:
        Scalar mean loss over all valid (non-ignored) positions.
    """
    V = logits.shape[-1]
    logits_flat = logits.reshape(-1, V)  # (N, V)
    labels_flat = labels.reshape(-1)  # (N,)

    valid_mask = labels_flat != ignore_index

    if not valid_mask.any():
        return logits_flat.new_zeros(())

    # Built-in label_smoothing handles the math correctly:
    # loss = (1-s)*CE_hard + s * (-mean(log_softmax))
    loss_per_token = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=ignore_index,
        label_smoothing=smoothing,
        reduction="none",
    )

    n_valid = valid_mask.sum().clamp(min=1)
    return loss_per_token[valid_mask].sum() / n_valid


# ---------------------------------------------------------------------------
# focal_loss
# ---------------------------------------------------------------------------


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Focal loss for language modeling (Lin et al. 2017).

    Down-weights easy examples (high p_t) so training concentrates on
    hard / uncertain tokens:
        FL(p_t) = (1 - p_t)^gamma * CE

    Args:
        logits:       (B, T, V) raw model logits.
        labels:       (B, T) long tensor of target token ids.
        gamma:        Focusing parameter. gamma=0 reduces to standard CE.
        ignore_index: Positions with this label are excluded.

    Returns:
        Scalar mean loss over all valid positions.
    """
    V = logits.shape[-1]
    logits_flat = logits.reshape(-1, V)
    labels_flat = labels.reshape(-1)

    valid_mask = labels_flat != ignore_index

    if not valid_mask.any():
        return logits_flat.new_zeros(())

    # Per-token CE (unmasked positions will have 0 CE due to ignore_index)
    ce = F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index, reduction="none")

    # p_t = probability assigned to the correct class = exp(-CE)
    p_t = torch.exp(-ce)

    # Focal modulating factor
    focal_weight = (1.0 - p_t) ** gamma

    loss = focal_weight * ce  # (N,)

    n_valid = valid_mask.sum().clamp(min=1)
    return loss[valid_mask].sum() / n_valid


# ---------------------------------------------------------------------------
# token_weighted_loss
# ---------------------------------------------------------------------------


def token_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Per-token weighted cross-entropy.

    Computes a weighted average of per-token CE losses using a caller-supplied
    weight tensor. Positions with ignore_index labels are excluded regardless
    of their weight value.

    Args:
        logits:       (B, T, V) raw model logits.
        labels:       (B, T) long tensor of target token ids.
        weights:      (B, T) float tensor of per-token weights.
        ignore_index: Positions with this label are excluded.

    Returns:
        Scalar weighted-mean loss, or 0 if all positions are masked / zero-weight.
    """
    V = logits.shape[-1]
    logits_flat = logits.reshape(-1, V)  # (N, V)
    labels_flat = labels.reshape(-1)  # (N,)
    weights_flat = weights.reshape(-1).to(logits.dtype)  # (N,)

    valid_mask = labels_flat != ignore_index

    if not valid_mask.any():
        return logits_flat.new_zeros(())

    ce = F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index, reduction="none")

    # Zero out ignored positions in weight so they don't contribute
    weights_valid = weights_flat * valid_mask.to(weights_flat.dtype)

    weight_sum = weights_valid.sum().clamp(min=1e-9)

    if weight_sum <= 1e-9:
        return logits_flat.new_zeros(())

    return (ce * weights_valid).sum() / weight_sum


# ---------------------------------------------------------------------------
# z_loss
# ---------------------------------------------------------------------------


def z_loss(
    logits: torch.Tensor,
    coef: float = 1e-4,
) -> torch.Tensor:
    """Z-loss auxiliary regularization (Zoph et al. 2022, ST-MoE paper).

    Penalises large logit values to keep the softmax numerically stable and
    prevent logit explosion during training:
        z_loss = coef * mean(log(sum(exp(logits)))^2)

    Args:
        logits: (B, T, V) or any shape ending in V — raw model logits.
        coef:   Regularization coefficient (default 1e-4).

    Returns:
        Scalar z-loss value.
    """
    # log(sum(exp(x))) = logsumexp; shape (...,) after reducing last dim
    log_z = torch.logsumexp(logits, dim=-1)  # (B, T)
    return coef * (log_z**2).mean()


# ---------------------------------------------------------------------------
# compute_ppl_from_loss
# ---------------------------------------------------------------------------


def compute_ppl_from_loss(loss: torch.Tensor) -> torch.Tensor:
    """Compute perplexity from a (mean) cross-entropy loss.

    Perplexity = exp(CE_loss).

    Args:
        loss: Scalar or batch tensor of mean CE losses.

    Returns:
        Tensor of same shape as loss containing perplexity values.
    """
    return torch.exp(loss)


# ---------------------------------------------------------------------------
# LMLoss — nn.Module combining the above
# ---------------------------------------------------------------------------


class LMLoss(nn.Module):
    """Combined language modeling loss module.

    Selects the appropriate base loss function based on LossConfig:
    - label_smoothing > 0  → label_smoothed_loss
    - label_smoothing == 0 and focal_gamma > 0 → focal_loss
    - otherwise → cross_entropy_loss

    Always returns a dict with:
        "loss"    — combined scalar loss (base loss; z-loss is not included here
                    but can be added externally via z_loss())
        "ce_loss" — the raw CE component for logging
        "ppl"     — perplexity derived from ce_loss

    Args:
        config: LossConfig instance controlling loss behaviour.
    """

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute loss.

        Args:
            logits: (B, T, V) raw model logits.
            labels: (B, T) long tensor of target token ids.

        Returns:
            Dict with keys "loss", "ce_loss", "ppl".
        """
        cfg = self.config

        # Always compute plain CE for logging / ppl
        ce = cross_entropy_loss(
            logits,
            labels,
            ignore_index=cfg.ignore_index,
            reduction=cfg.reduction,
        )

        # Choose the primary loss function
        if cfg.label_smoothing > 0.0:
            primary_loss = label_smoothed_loss(
                logits,
                labels,
                smoothing=cfg.label_smoothing,
                ignore_index=cfg.ignore_index,
            )
        elif cfg.focal_gamma > 0.0:
            primary_loss = focal_loss(
                logits,
                labels,
                gamma=cfg.focal_gamma,
                ignore_index=cfg.ignore_index,
            )
        else:
            primary_loss = ce

        ppl = compute_ppl_from_loss(ce)

        return {
            "loss": primary_loss,
            "ce_loss": ce,
            "ppl": ppl,
        }
