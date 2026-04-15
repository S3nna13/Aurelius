"""Direct Preference Optimization (DPO) trainer.

Trains language models directly from preference pairs without a separate
reward model by optimizing the implicit reward defined by the policy.

References:
    Rafailov et al. 2023, "Direct Preference Optimization: Your Language Model
    is Secretly a Reward Model"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DPOConfig:
    """Configuration for DPO training."""

    beta: float = 0.1          # KL penalty coefficient
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"  # "sigmoid" or "ipo"
    reference_free: bool = False  # if True, ref log-probs are treated as 0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_log_probs(logits: Tensor, labels: Tensor) -> Tensor:
    """Compute sum of log-probabilities for label tokens.

    Args:
        logits: (B, T, V) model logits.
        labels: (B, T) token ids; positions with -100 are masked out.

    Returns:
        (B,) summed log-prob per sequence.
    """
    B, T, V = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)

    # Gather log-probs for the actual tokens
    # Clamp -100 to 0 for gather (will be masked out below)
    safe_labels = labels.clone()
    safe_labels[safe_labels == -100] = 0
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, T)

    # Mask padding
    mask = (labels != -100).float()
    return (token_log_probs * mask).sum(dim=-1)  # (B,)


def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    ref_chosen_logps: Tensor,
    ref_rejected_logps: Tensor,
    config: DPOConfig,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute DPO loss and implicit rewards.

    Args:
        policy_chosen_logps: (B,) log-probs of chosen sequences under policy.
        policy_rejected_logps: (B,) log-probs of rejected sequences under policy.
        ref_chosen_logps: (B,) log-probs under reference model.
        ref_rejected_logps: (B,) log-probs under reference model.
        config: DPOConfig.

    Returns:
        (loss, chosen_rewards, rejected_rewards) — all tensors.
    """
    chosen_rewards = config.beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = config.beta * (policy_rejected_logps - ref_rejected_logps)

    reward_diff = chosen_rewards - rejected_rewards

    if config.loss_type == "sigmoid":
        loss = -F.logsigmoid(reward_diff - config.label_smoothing).mean()
    elif config.loss_type == "ipo":
        # IPO: (reward_diff - 1/(2*beta))^2
        loss = ((reward_diff - 1.0 / (2.0 * config.beta)) ** 2).mean()
    else:
        raise ValueError(f"Unknown loss_type '{config.loss_type}'. Use 'sigmoid' or 'ipo'.")

    return loss, chosen_rewards, rejected_rewards


def ipo_loss(
    chosen_logps: Tensor,
    rejected_logps: Tensor,
    ref_chosen_logps: Tensor,
    ref_rejected_logps: Tensor,
    beta: float,
) -> Tensor:
    """Standalone IPO loss (Identity Preference Optimization).

    Args:
        chosen_logps: (B,) policy log-probs for chosen.
        rejected_logps: (B,) policy log-probs for rejected.
        ref_chosen_logps: (B,) reference log-probs for chosen.
        ref_rejected_logps: (B,) reference log-probs for rejected.
        beta: KL penalty coefficient.

    Returns:
        Scalar IPO loss.
    """
    chosen_rewards = beta * (chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (rejected_logps - ref_rejected_logps)
    return ((chosen_rewards - rejected_rewards - 1.0 / (2.0 * beta)) ** 2).mean()


def compute_reward_margin(chosen_rewards: Tensor, rejected_rewards: Tensor) -> float:
    """Mean margin between chosen and rejected rewards."""
    return (chosen_rewards - rejected_rewards).mean().item()


def compute_reward_accuracy(chosen_rewards: Tensor, rejected_rewards: Tensor) -> float:
    """Fraction of pairs where chosen reward > rejected reward."""
    return (chosen_rewards > rejected_rewards).float().mean().item()


# ---------------------------------------------------------------------------
# DPO Trainer
# ---------------------------------------------------------------------------

class DPOTrainer:
    """Direct Preference Optimization trainer.

    Args:
        policy_model_fn: Callable taking (B, T) token_ids, returning (B, T, V) logits.
        ref_model_fn: Reference model callable with same signature.
        optimizer: PyTorch optimizer for policy parameters.
        config: DPOConfig.
    """

    def __init__(
        self,
        policy_model_fn: Callable[[Tensor], Tensor],
        ref_model_fn: Callable[[Tensor], Tensor],
        optimizer,
        config: Optional[DPOConfig] = None,
    ) -> None:
        self.policy_model_fn = policy_model_fn
        self.ref_model_fn = ref_model_fn
        self.optimizer = optimizer
        self.config = config or DPOConfig()

    def _compute_all_logps(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
        labels_chosen: Tensor,
        labels_rejected: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute policy and reference log-probs for both sequences."""
        policy_chosen = compute_log_probs(self.policy_model_fn(chosen_ids), labels_chosen)
        policy_rejected = compute_log_probs(self.policy_model_fn(rejected_ids), labels_rejected)

        if self.config.reference_free:
            ref_chosen = torch.zeros_like(policy_chosen)
            ref_rejected = torch.zeros_like(policy_rejected)
        else:
            with torch.no_grad():
                ref_chosen = compute_log_probs(self.ref_model_fn(chosen_ids), labels_chosen)
                ref_rejected = compute_log_probs(self.ref_model_fn(rejected_ids), labels_rejected)

        return policy_chosen, policy_rejected, ref_chosen, ref_rejected

    def compute_rewards(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
        labels_chosen: Tensor,
        labels_rejected: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute all four log-probs and implicit rewards."""
        pc, pr, rc, rr = self._compute_all_logps(chosen_ids, rejected_ids, labels_chosen, labels_rejected)
        chosen_rewards = self.config.beta * (pc - rc)
        rejected_rewards = self.config.beta * (pr - rr)
        return {
            "policy_chosen_logps": pc,
            "policy_rejected_logps": pr,
            "ref_chosen_logps": rc,
            "ref_rejected_logps": rr,
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards,
        }

    def train_step(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
        labels_chosen: Tensor,
        labels_rejected: Tensor,
    ) -> Dict[str, float]:
        """One training step. Returns metrics dict."""
        self.optimizer.zero_grad()

        pc, pr, rc, rr = self._compute_all_logps(chosen_ids, rejected_ids, labels_chosen, labels_rejected)
        loss, chosen_rewards, rejected_rewards = dpo_loss(pc, pr, rc, rr, self.config)

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "reward_margin": compute_reward_margin(chosen_rewards, rejected_rewards),
            "reward_accuracy": compute_reward_accuracy(chosen_rewards, rejected_rewards),
        }

    def evaluate(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
        labels_chosen: Tensor,
        labels_rejected: Tensor,
    ) -> Dict[str, float]:
        """Evaluation step (no gradient). Returns same metrics as train_step."""
        with torch.no_grad():
            pc, pr, rc, rr = self._compute_all_logps(chosen_ids, rejected_ids, labels_chosen, labels_rejected)
            loss, chosen_rewards, rejected_rewards = dpo_loss(pc, pr, rc, rr, self.config)

        return {
            "loss": loss.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "reward_margin": compute_reward_margin(chosen_rewards, rejected_rewards),
            "reward_accuracy": compute_reward_accuracy(chosen_rewards, rejected_rewards),
        }
