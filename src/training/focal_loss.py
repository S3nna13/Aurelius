"""Focal Loss and hard example mining losses for language model training.

Implements Focal Loss (Lin et al. 2017, "Focal Loss for Dense Object Detection"),
label-smoothed focal loss, Poly Loss (Leng et al. 2022), and supporting utilities
including an adaptive gamma scheduler and a drop-in focal-loss trainer wrapper.

Key idea: focal loss down-weights easy examples (high p_t) and concentrates
gradient signal on hard examples (low p_t).

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

With gamma=0 this reduces to standard cross-entropy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Core focal loss function
# ---------------------------------------------------------------------------


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | torch.Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute focal loss for classification.

    Handles both (N, C) and (B, T, V) shaped logits by flattening to (N, C).

    Args:
        logits:       (N, C) or (B, T, V) raw model logits.
        targets:      (N,) or (B, T) long tensor of class indices.
        gamma:        Focusing parameter. 0 = standard CE, 2 = standard focal.
        alpha:        Per-class weight. None = no weighting, scalar float applies
                      uniformly, Tensor of shape (C,) applies per-class weights.
        ignore_index: Positions with this target value are excluded from loss.
        reduction:    "mean" | "sum" | "none"

    Returns:
        Scalar loss (reduction="mean"/"sum") or per-token losses (reduction="none").

    Algorithm:
        1. Flatten to (N, C) if needed
        2. CE = F.cross_entropy(logits, targets, reduction='none', ignore_index)
        3. p_t = exp(-CE)            — probability of correct class
        4. focal_weight = (1-p_t)^gamma
        5. If alpha provided: apply per-class alpha weighting
        6. loss = focal_weight * CE
        7. Apply reduction over non-ignored positions
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")

    # Flatten (B, T, V) -> (N, C) and (B, T) -> (N,)
    C = logits.shape[-1]
    logits_flat = logits.reshape(-1, C)
    targets_flat = targets.reshape(-1)

    # Step 2: Per-token CE loss (unreduced), ignoring ignore_index positions
    ce = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=ignore_index)

    # Step 3: p_t = exp(-CE)  (probability of the correct class)
    p_t = torch.exp(-ce)

    # Step 4: Focal modulating factor
    focal_weight = (1.0 - p_t) ** gamma

    # Step 5: Per-class alpha weighting
    if alpha is not None:
        if isinstance(alpha, torch.Tensor):
            # alpha is (C,) — gather per-token alpha based on target class
            alpha_tensor = alpha.to(logits_flat.device, dtype=logits_flat.dtype)
            # For ignored positions keep weight 1 (loss is already 0 there)
            valid_mask = targets_flat != ignore_index
            alpha_t = torch.ones_like(targets_flat, dtype=logits_flat.dtype)
            valid_targets = targets_flat.clone()
            valid_targets[~valid_mask] = 0  # safe index — these will be overwritten anyway
            alpha_t[valid_mask] = alpha_tensor[valid_targets[valid_mask]]
        else:
            # Scalar alpha
            alpha_t = float(alpha)

        focal_weight = focal_weight * alpha_t

    # Step 6: Weighted loss
    loss = focal_weight * ce

    # Step 7: Reduction (only over non-ignored positions)
    if reduction == "none":
        return loss

    valid_mask = targets_flat != ignore_index
    if reduction == "mean":
        n_valid = valid_mask.sum().clamp(min=1)
        return loss[valid_mask].sum() / n_valid
    else:  # "sum"
        return loss[valid_mask].sum()


# ---------------------------------------------------------------------------
# Label-smoothed focal loss
# ---------------------------------------------------------------------------


def label_smoothed_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    smoothing: float = 0.1,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Focal loss combined with label smoothing.

    Smooth targets: (1-smoothing)*one_hot + smoothing/C
    Then apply focal weighting using the smoothed CE as a proxy for confidence.

    Args:
        logits:       (N, C) or (B, T, V) raw model logits.
        targets:      (N,) or (B, T) long tensor of class indices.
        gamma:        Focusing parameter.
        smoothing:    Label smoothing factor in [0, 1).
        ignore_index: Positions with this target value are excluded.

    Returns:
        Scalar mean loss.
    """
    C = logits.shape[-1]
    logits_flat = logits.reshape(-1, C)
    targets_flat = targets.reshape(-1)

    valid_mask = targets_flat != ignore_index

    if not valid_mask.any():
        return logits_flat.new_zeros(())

    # Standard CE (unsmoothed) used to compute p_t for focal weighting
    ce_hard = F.cross_entropy(
        logits_flat, targets_flat, reduction="none", ignore_index=ignore_index
    )
    p_t = torch.exp(-ce_hard)
    focal_weight = (1.0 - p_t) ** gamma

    # Smoothed CE: mix one-hot with uniform distribution
    # CE_smooth = (1-s)*CE_hard + s * mean(log_softmax over all classes)
    F.log_softmax(logits_flat, dim=-1)  # (N, C)
    # Hard CE contribution: -(1-s) * log_prob[target]
    ce_smooth = F.cross_entropy(
        logits_flat,
        targets_flat,
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=smoothing,
    )

    loss = focal_weight * ce_smooth
    n_valid = valid_mask.sum().clamp(min=1)
    return loss[valid_mask].sum() / n_valid


# ---------------------------------------------------------------------------
# Poly Loss
# ---------------------------------------------------------------------------


def poly_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Poly Loss (Leng et al. 2022): CE + epsilon * (1 - p_t).

    A simple first-order polynomial approximation that often outperforms CE
    on classification tasks.

        poly1 = CE + epsilon * (1 - p_t)

    Args:
        logits:       (N, C) or (B, T, V) raw model logits.
        targets:      (N,) or (B, T) long tensor of class indices.
        epsilon:      Polynomial coefficient (default 1.0).
        ignore_index: Positions with this target value are excluded.

    Returns:
        Scalar mean loss.
    """
    C = logits.shape[-1]
    logits_flat = logits.reshape(-1, C)
    targets_flat = targets.reshape(-1)

    valid_mask = targets_flat != ignore_index

    ce = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=ignore_index)
    p_t = torch.exp(-ce)

    loss = ce + epsilon * (1.0 - p_t)

    n_valid = valid_mask.sum().clamp(min=1)
    return loss[valid_mask].sum() / n_valid


# ---------------------------------------------------------------------------
# Adaptive gamma scheduler
# ---------------------------------------------------------------------------


class AdaptiveGammaScheduler:
    """Dynamically adjust focal loss gamma based on training progress.

    Starts with gamma=0 (standard CE) and linearly ramps up to target_gamma
    over warmup_steps. This prevents the model from focusing too hard on
    potentially noisy examples early in training.

        gamma(step) = target_gamma * min(1.0, step / warmup_steps)

    Args:
        target_gamma:  Final gamma value after warmup (default 2.0).
        warmup_steps:  Number of steps to reach target_gamma (default 1000).
    """

    def __init__(self, target_gamma: float = 2.0, warmup_steps: int = 1000) -> None:
        self.target_gamma = target_gamma
        self.warmup_steps = warmup_steps
        self.current_gamma = 0.0

    def step(self, current_step: int) -> float:
        """Update internal state and return current gamma for given step."""
        ratio = min(1.0, current_step / max(self.warmup_steps, 1))
        self.current_gamma = self.target_gamma * ratio
        return self.current_gamma

    def get_gamma(self) -> float:
        """Return current gamma without advancing the scheduler."""
        return self.current_gamma


# ---------------------------------------------------------------------------
# FocalLossTrainer
# ---------------------------------------------------------------------------


class FocalLossTrainer:
    """Trainer that replaces standard cross-entropy with focal loss.

    Wraps an AureliusTransformer and optimizer, performing forward passes,
    focal loss computation, backward pass, and optimizer step. Tracks
    easy/hard token statistics to monitor training dynamics.

    Args:
        model:          AureliusTransformer (or any nn.Module whose forward
                        returns (loss_or_none, logits, kv_cache)).
        optimizer:      torch.optim.Optimizer.
        gamma:          Focal loss focusing parameter (default 2.0).
        easy_threshold: Tokens with p_t above this are classified as "easy"
                        (default 0.8).
        max_seq_len:    Maximum sequence length (default 512).
        ignore_index:   Label value to ignore (default -100).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        gamma: float = 2.0,
        easy_threshold: float = 0.8,
        max_seq_len: int = 512,
        ignore_index: int = -100,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.easy_threshold = easy_threshold
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """Perform one training step with focal loss.

        Args:
            input_ids: (B, T) long tensor of input token ids.
            labels:    (B, T) long tensor of target token ids
                       (ignore_index for positions to skip).

        Returns:
            dict with keys:
                'loss':             scalar float — mean focal loss.
                'easy_token_ratio': float in [0, 1] — fraction of valid tokens
                                    with p_t > easy_threshold.
                'mean_pt':          float — mean p_t over valid tokens.
                'gamma':            float — current gamma value used.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass — AureliusTransformer returns (loss, logits, kv_cache)
        _, logits, _ = self.model(input_ids)

        # Compute focal loss
        loss = focal_loss(
            logits,
            labels,
            gamma=self.gamma,
            ignore_index=self.ignore_index,
            reduction="mean",
        )

        loss.backward()
        self.optimizer.step()

        # ---- Compute diagnostic statistics (no grad needed) ----
        with torch.no_grad():
            C = logits.shape[-1]
            logits_flat = logits.reshape(-1, C)
            targets_flat = labels.reshape(-1)
            valid_mask = targets_flat != self.ignore_index

            if valid_mask.any():
                ce = F.cross_entropy(
                    logits_flat, targets_flat, reduction="none", ignore_index=self.ignore_index
                )
                p_t = torch.exp(-ce)
                valid_p_t = p_t[valid_mask]
                easy_token_ratio = (valid_p_t > self.easy_threshold).float().mean().item()
                mean_pt = valid_p_t.mean().item()
            else:
                easy_token_ratio = 0.0
                mean_pt = 0.0

        return {
            "loss": loss.item(),
            "easy_token_ratio": easy_token_ratio,
            "mean_pt": mean_pt,
            "gamma": self.gamma,
        }
