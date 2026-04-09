"""Advanced DPO variants: DPO, IPO, and SLiC losses for preference learning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DPOAdvancedConfig:
    """Configuration for advanced DPO-family preference optimization."""

    beta: float = 0.1
    loss_type: str = "dpo"          # "dpo" | "ipo" | "slic"
    label_smoothing: float = 0.0    # label smoothing for DPO
    slic_delta: float = 1.0         # SLiC margin parameter
    reference_free: bool = False    # if True, use 0 as ref log-probs


# ---------------------------------------------------------------------------
# Log-prob computation
# ---------------------------------------------------------------------------

def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sequence sum of log-probabilities.

    Args:
        model: Policy model. Forward returns (_, logits, _).
        input_ids: Shape (B, seq_len).
        labels: Shape (B, seq_len). Positions with -100 are masked (padding).

    Returns:
        Shape (B,) — sum of log-probs over non-masked positions.
    """
    _, logits, _ = model(input_ids)  # (B, seq_len, vocab_size)

    # Shift: logits at position t predict token at position t+1
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, seq_len-1, vocab_size)

    # Shift labels to align with shifted logits
    shifted_labels = labels[:, 1:]  # (B, seq_len-1)

    # Mask out padding positions (label == -100)
    pad_mask = (shifted_labels != -100).float()  # (B, seq_len-1)

    # Replace -100 with 0 so gather doesn't fail; masked positions are zeroed out anyway
    gather_labels = shifted_labels.clone()
    gather_labels[shifted_labels == -100] = 0

    token_lp = log_probs.gather(2, gather_labels.unsqueeze(-1)).squeeze(-1)  # (B, seq_len-1)

    return (token_lp * pad_mask).sum(dim=-1)  # (B,)


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------

def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct Preference Optimization loss (Rafailov et al. 2023).

    Args:
        policy_chosen_logps: Shape (B,) — policy log-probs for chosen sequences.
        policy_rejected_logps: Shape (B,) — policy log-probs for rejected sequences.
        ref_chosen_logps: Shape (B,) — reference log-probs for chosen sequences.
        ref_rejected_logps: Shape (B,) — reference log-probs for rejected sequences.
        beta: Temperature / KL penalty coefficient.
        label_smoothing: Label smoothing coefficient in [0, 1).

    Returns:
        (loss_scalar, reward_margin) where reward_margin = chosen_reward - rejected_reward.
    """
    chosen_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_ratio = policy_rejected_logps - ref_rejected_logps

    h = beta * (chosen_ratio - rejected_ratio)

    # Primary DPO loss
    loss = -F.logsigmoid(h)

    if label_smoothing > 0.0:
        smooth_loss = -F.logsigmoid(-h)
        loss = (1.0 - label_smoothing) * loss + label_smoothing * smooth_loss

    chosen_rewards = beta * chosen_ratio.detach()
    rejected_rewards = beta * rejected_ratio.detach()

    return loss.mean(), (chosen_rewards - rejected_rewards).mean()


# ---------------------------------------------------------------------------
# IPO loss
# ---------------------------------------------------------------------------

def ipo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identity Preference Optimization loss (Azar et al. 2024).

    Squared hinge loss that avoids the overoptimization issue of DPO by
    targeting h = 1/(2*beta) rather than driving it to infinity.

    Args:
        policy_chosen_logps: Shape (B,).
        policy_rejected_logps: Shape (B,).
        ref_chosen_logps: Shape (B,).
        ref_rejected_logps: Shape (B,).
        beta: Regularization coefficient.

    Returns:
        (loss_scalar, reward_margin) where reward_margin = mean of h.detach().
    """
    chosen_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_ratio = policy_rejected_logps - ref_rejected_logps

    h = chosen_ratio - rejected_ratio  # (B,)

    target = 1.0 / (2.0 * beta)
    loss = (h - target) ** 2

    return loss.mean(), h.detach().mean()


# ---------------------------------------------------------------------------
# SLiC loss
# ---------------------------------------------------------------------------

def slic_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequence Likelihood Calibration hinge loss (Zhao et al. 2023).

    Reference model logps are not used; loss is purely based on policy
    log-prob margin between chosen and rejected sequences.

    Args:
        policy_chosen_logps: Shape (B,).
        policy_rejected_logps: Shape (B,).
        ref_chosen_logps: Shape (B,) — unused.
        ref_rejected_logps: Shape (B,) — unused.
        delta: Margin parameter.

    Returns:
        (loss_scalar, mean margin) where margin = policy_chosen - policy_rejected.
    """
    margin = policy_chosen_logps - policy_rejected_logps  # (B,)
    loss = torch.clamp(delta - margin, min=0.0)
    return loss.mean(), margin.detach().mean()


# ---------------------------------------------------------------------------
# DPOAdvancedTrainer
# ---------------------------------------------------------------------------

class DPOAdvancedTrainer:
    """Trainer for DPO, IPO, and SLiC preference optimization variants.

    Args:
        policy_model: The trainable policy model.
        ref_model: Frozen reference model (can be None if reference_free=True).
        config: DPOAdvancedConfig specifying loss type and hyperparameters.
        optimizer: PyTorch optimizer for policy_model parameters.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module | None,
        config: DPOAdvancedConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.config = config
        self.optimizer = optimizer

    def _build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Build labels by shifting input_ids by 1 (next-token prediction)."""
        labels = input_ids.clone()
        # Shift: labels[t] = input_ids[t+1]; last position has no target → -100
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        return labels

    def train_step(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> dict[str, float]:
        """Run a single training step.

        Args:
            chosen_ids: Shape (B, seq_len) — chosen token sequences.
            rejected_ids: Shape (B, seq_len) — rejected token sequences.

        Returns:
            Dict with keys: "loss", "reward_margin", "chosen_reward", "rejected_reward".
        """
        self.policy_model.train()

        chosen_labels = self._build_labels(chosen_ids)
        rejected_labels = self._build_labels(rejected_ids)

        # Policy log-probs
        policy_chosen_logps = compute_log_probs(self.policy_model, chosen_ids, chosen_labels)
        policy_rejected_logps = compute_log_probs(self.policy_model, rejected_ids, rejected_labels)

        # Reference log-probs
        if self.config.reference_free:
            ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
            ref_rejected_logps = torch.zeros_like(policy_rejected_logps)
        else:
            with torch.no_grad():
                ref_chosen_logps = compute_log_probs(self.ref_model, chosen_ids, chosen_labels)
                ref_rejected_logps = compute_log_probs(self.ref_model, rejected_ids, rejected_labels)

        # Dispatch to the appropriate loss function
        cfg = self.config
        if cfg.loss_type == "dpo":
            loss, reward_margin = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=cfg.beta,
                label_smoothing=cfg.label_smoothing,
            )
            chosen_reward = (cfg.beta * (policy_chosen_logps - ref_chosen_logps).detach()).mean()
            rejected_reward = (cfg.beta * (policy_rejected_logps - ref_rejected_logps).detach()).mean()

        elif cfg.loss_type == "ipo":
            loss, reward_margin = ipo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=cfg.beta,
            )
            chosen_reward = (policy_chosen_logps - ref_chosen_logps).detach().mean()
            rejected_reward = (policy_rejected_logps - ref_rejected_logps).detach().mean()

        elif cfg.loss_type == "slic":
            loss, reward_margin = slic_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                delta=cfg.slic_delta,
            )
            chosen_reward = policy_chosen_logps.detach().mean()
            rejected_reward = policy_rejected_logps.detach().mean()

        else:
            raise ValueError(f"Unknown loss_type: {cfg.loss_type!r}. Choose 'dpo', 'ipo', or 'slic'.")

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "reward_margin": reward_margin.item(),
            "chosen_reward": chosen_reward.item(),
            "rejected_reward": rejected_reward.item(),
        }
