"""Token importance scoring for curriculum training: identify informative tokens for selective training."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TokenImportanceConfig:
    """Configuration for token importance scoring and curriculum training.

    Attributes:
        scoring_method: Method to score tokens — "loss" | "gradient" | "attention" | "random".
        top_k_fraction: Fraction of tokens to keep per sequence (0 < f <= 1).
        min_tokens: Minimum number of tokens to keep per sequence regardless of fraction.
        smooth_alpha: EMA smoothing coefficient for running importance statistics.
        update_freq: Number of training steps between full importance recalculations.
    """

    scoring_method: str = "loss"
    top_k_fraction: float = 0.5
    min_tokens: int = 4
    smooth_alpha: float = 0.1
    update_freq: int = 100


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def score_tokens_by_loss(
    model: nn.Module,
    input_ids: Tensor,
) -> Tensor:
    """Compute per-token cross-entropy loss as an importance score.

    Performs a forward pass through the model (without labels so no aggregated
    loss is computed), then manually computes per-token cross-entropy by
    shifting input/target by one position (causal language modelling).

    Args:
        model: An ``AureliusTransformer`` (or compatible) module whose forward
               returns ``(loss, logits, past_key_values)``.
        input_ids: ``(B, T)`` integer token ids.

    Returns:
        per_token_loss: ``(B, T-1)`` float tensor — higher values indicate
        tokens that are harder (more important) for the model to predict.
    """
    with torch.no_grad():
        _loss, logits, _pkv = model(input_ids)

    # logits: (B, T, V)
    # Predict token t+1 from position t → shift by one
    # targets: positions 1..T-1, predictions: positions 0..T-2
    shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
    shift_targets = input_ids[:, 1:].contiguous()   # (B, T-1)

    B, T_minus_1, V = shift_logits.shape

    per_token_loss = F.cross_entropy(
        shift_logits.view(B * T_minus_1, V),
        shift_targets.view(B * T_minus_1),
        reduction="none",
    ).view(B, T_minus_1)  # (B, T-1)

    return per_token_loss


def score_tokens_by_attention(
    logits: Tensor,
) -> Tensor:
    """Compute token difficulty via logit entropy as an importance proxy.

    High entropy means the model is uncertain — the token is harder to predict
    and therefore more informative for training.

    Args:
        logits: ``(B, T, V)`` raw (unnormalised) output logits.

    Returns:
        entropy: ``(B, T)`` float tensor — higher = more uncertain / important.
    """
    # Entropy: -sum(p * log p) = -sum(softmax * log_softmax)
    log_probs = F.log_softmax(logits, dim=-1)   # (B, T, V)
    probs = torch.exp(log_probs)                # (B, T, V)
    entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)
    return entropy


# ---------------------------------------------------------------------------
# Token selection
# ---------------------------------------------------------------------------


def select_important_tokens(
    scores: Tensor,
    config: TokenImportanceConfig,
) -> Tensor:
    """Select the most important tokens per sequence based on importance scores.

    For "random" scoring method the selection is random; otherwise the
    top-scoring tokens are selected.

    Args:
        scores: ``(B, T)`` float importance scores — higher = more important.
        config: ``TokenImportanceConfig`` controlling selection behaviour.

    Returns:
        mask: ``(B, T)`` boolean tensor — ``True`` marks an important token.
    """
    B, T = scores.shape
    k = max(int(T * config.top_k_fraction), config.min_tokens)
    k = min(k, T)  # cannot select more than T tokens

    if config.scoring_method == "random":
        # Random selection: use torch.rand to produce random scores
        rand_scores = torch.rand(B, T, device=scores.device, dtype=scores.dtype)
        topk_indices = rand_scores.topk(k, dim=1).indices
    else:
        topk_indices = scores.topk(k, dim=1).indices  # (B, k)

    mask = torch.zeros(B, T, dtype=torch.bool, device=scores.device)
    mask.scatter_(1, topk_indices, True)
    return mask


# ---------------------------------------------------------------------------
# Masked label construction
# ---------------------------------------------------------------------------


def build_masked_labels(
    labels: Tensor,
    mask: Tensor,
) -> Tensor:
    """Build a label tensor that ignores unimportant tokens in the loss.

    Positions where ``mask`` is ``False`` are set to ``-100``, which is the
    conventional ``ignore_index`` in PyTorch's cross-entropy loss.

    Args:
        labels: ``(B, T)`` integer token ids.
        mask: ``(B, T)`` boolean tensor — ``True`` means *keep this token*.

    Returns:
        masked_labels: ``(B, T)`` labels with ``-100`` at unimportant positions.
    """
    masked = labels.clone()
    masked[~mask] = -100
    return masked


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


class TokenImportanceCurriculum:
    """Progressive curriculum that gradually widens the set of trained tokens.

    At step 0 only ``top_k_fraction`` of tokens are selected (the most
    informative ones).  The fraction linearly anneals to 1.0 (all tokens)
    by ``total_steps``.

    Additionally, maintains a running exponential moving average (EMA) of the
    mean importance score across batches, which can be used externally for
    monitoring or adaptive scheduling.

    Args:
        config: ``TokenImportanceConfig`` supplying defaults.
        total_steps: Total number of training steps over which to anneal.
    """

    def __init__(self, config: TokenImportanceConfig, total_steps: int) -> None:
        self.config = config
        self.total_steps = total_steps
        self._running_mean: float = 0.0

    def get_fraction(self, step: int) -> float:
        """Return the token-keep fraction at the given training step.

        Linearly anneals from ``config.top_k_fraction`` at step 0 to 1.0 at
        step ``>= total_steps``.

        Args:
            step: Current training step (0-indexed).

        Returns:
            Fraction in ``[top_k_fraction, 1.0]``.
        """
        if self.total_steps <= 0:
            return 1.0
        t = min(step / self.total_steps, 1.0)
        return self.config.top_k_fraction + t * (1.0 - self.config.top_k_fraction)

    def update_running_stats(self, new_scores: Tensor) -> None:
        """EMA update of running mean importance score.

        Args:
            new_scores: ``(B, T)`` (or any shape) importance scores for the
                        current batch.
        """
        batch_mean = new_scores.mean().item()
        alpha = self.config.smooth_alpha
        self._running_mean = (1.0 - alpha) * self._running_mean + alpha * batch_mean

    def get_running_mean(self) -> float:
        """Return the current EMA running mean of importance scores."""
        return self._running_mean


# ---------------------------------------------------------------------------
# Selective trainer
# ---------------------------------------------------------------------------


class SelectiveTrainer:
    """Wraps a model and applies selective token training.

    At each training step, tokens are scored by per-token loss, the most
    important ones are selected (respecting the current curriculum fraction),
    and a masked cross-entropy loss is computed only over those positions.

    Args:
        model: An ``AureliusTransformer``-compatible module.
        optimizer: A ``torch.optim.Optimizer`` instance.
        config: ``TokenImportanceConfig`` controlling selection behaviour.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        config: TokenImportanceConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self._step = 0

    def train_step(self, input_ids: Tensor, labels: Tensor) -> dict[str, float]:
        """Perform one selective training step.

        1. Score tokens by per-token loss (no grad).
        2. Select important tokens using the configured fraction.
        3. Build masked labels (``-100`` for unimportant tokens).
        4. Forward pass + cross-entropy on masked labels only.
        5. Backward pass + optimizer step.

        Args:
            input_ids: ``(B, T)`` token indices.
            labels: ``(B, T)`` target token indices.

        Returns:
            dict with keys:
                ``"loss"``: scalar training loss (float).
                ``"n_active_tokens"``: total number of active (non-masked) tokens.
                ``"fraction_active"``: fraction of label tokens that were active.
        """
        self.model.train()

        # 1. Score tokens (no gradient through scoring)
        token_scores = score_tokens_by_loss(self.model, input_ids)  # (B, T-1)

        # Pad scores to full sequence length so mask aligns with labels
        B, T = input_ids.shape
        # token_scores is (B, T-1); prepend a zero score for the first position
        # so that it aligns with the labels tensor which has length T.
        padded_scores = torch.cat(
            [torch.zeros(B, 1, device=token_scores.device, dtype=token_scores.dtype),
             token_scores],
            dim=1,
        )  # (B, T)

        # 2. Select important tokens
        mask = select_important_tokens(padded_scores, self.config)  # (B, T) bool

        # 3. Build masked labels
        masked_labels = build_masked_labels(labels, mask)  # (B, T)

        # 4. Forward pass + loss
        self.optimizer.zero_grad()
        _loss_out, logits, _pkv = self.model(input_ids)  # logits: (B, T, V)

        B_l, T_l, V = logits.shape
        loss = F.cross_entropy(
            logits.view(B_l * T_l, V),
            masked_labels.view(B_l * T_l),
            ignore_index=-100,
        )

        # 5. Backward + optimizer step
        loss.backward()
        self.optimizer.step()

        self._step += 1

        # Compute stats
        n_active = int(mask.sum().item())
        fraction_active = n_active / (B * T)

        return {
            "loss": loss.item(),
            "n_active_tokens": n_active,
            "fraction_active": fraction_active,
        }
