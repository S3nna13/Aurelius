"""Online DPO (Guo et al. 2024): on-the-fly preference pair generation and DPO training.

Algorithm per prompt:
  1. Sample K completions from the current policy.
  2. Score each completion with a reward signal.
  3. Select the top-scoring completion as "chosen" and the bottom as "rejected".
  4. Compute standard DPO loss against a frozen reference model.

All implementation uses pure native PyTorch — no transformers, trl, or peft.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OnlineDPOConfig:
    """Hyperparameters for Online DPO."""

    beta: float = 0.1
    """KL regularisation coefficient (inverse temperature for DPO reward)."""

    n_completions: int = 4
    """Number of completions to sample per prompt (K)."""

    temperature: float = 1.0
    """Sampling temperature applied to policy logits."""

    top_k_pairs: int = 1
    """How many (chosen, rejected) pairs to extract per prompt."""


# ---------------------------------------------------------------------------
# CompletionSampler
# ---------------------------------------------------------------------------

class CompletionSampler:
    """Sample completions and compute their log-probabilities from logit tensors.

    Designed to work with pre-computed logit tensors rather than autoregressive
    generation, making it easy to unit-test without a full generative model.

    Args:
        vocab_size: Size of the vocabulary.
        temperature: Sampling temperature. Must be > 0.
    """

    def __init__(self, vocab_size: int, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.vocab_size = vocab_size
        self.temperature = temperature

    def sample(self, logits: Tensor, n_samples: int) -> Tensor:
        """Sample n_samples completions for each item in the batch.

        Each of the B items in the batch is replicated n_samples times, and
        tokens are sampled independently per position from the temperature-scaled
        categorical distribution.

        Args:
            logits: Shape (B, T, V) — per-position logits from the policy.
            n_samples: Number of completions to draw per batch item (K).

        Returns:
            LongTensor of shape (B * n_samples, T) — sampled token ids.
            The first n_samples rows correspond to batch item 0, the next
            n_samples rows to batch item 1, etc.
        """
        B, T, V = logits.shape

        # Repeat each batch item n_samples times: (B, T, V) -> (B*K, T, V)
        # interleave so that rows [0..K-1] are for batch item 0, etc.
        logits_rep = logits.repeat_interleave(n_samples, dim=0)  # (B*K, T, V)

        # Scale by temperature
        scaled = logits_rep / self.temperature  # (B*K, T, V)

        # Sample independently at each position
        # Flatten to (B*K*T, V), multinomial, then reshape
        flat = scaled.reshape(-1, V)  # (B*K*T, V)
        probs = F.softmax(flat, dim=-1)  # (B*K*T, V)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B*K*T,)
        token_ids = sampled.reshape(B * n_samples, T)  # (B*K, T)

        return token_ids

    def log_probs_of(self, logits: Tensor, token_ids: Tensor) -> Tensor:
        """Compute per-sequence sum of log probabilities.

        Sums the log-probability of each token position (mean over T), with
        positions where token_ids == -1 treated as padding and masked out.

        Args:
            logits: Shape (B, T, V) — policy logits.
            token_ids: LongTensor of shape (B, T) — target token ids.
                       Positions with value -1 are treated as padding.

        Returns:
            Tensor of shape (B,) — mean log-probability over non-padding positions.
        """
        B, T, V = logits.shape

        # Compute log-softmax over vocab dimension
        log_p = F.log_softmax(logits, dim=-1)  # (B, T, V)

        # Build padding mask: True where token is NOT padding
        pad_mask = token_ids != -1  # (B, T), bool

        # Clamp ids for gather (pad positions get id=0, will be masked out)
        safe_ids = token_ids.clone()
        safe_ids[~pad_mask] = 0  # won't contribute due to masking

        # Gather log-probs of actual tokens: (B, T)
        token_log_p = log_p.gather(2, safe_ids.unsqueeze(-1)).squeeze(-1)  # (B, T)

        # Zero out padding positions
        token_log_p = token_log_p * pad_mask.float()

        # Mean over non-padding positions per sequence
        n_valid = pad_mask.float().sum(dim=1).clamp(min=1)  # (B,)
        seq_log_p = token_log_p.sum(dim=1) / n_valid  # (B,)

        return seq_log_p


# ---------------------------------------------------------------------------
# OnlinePairBuilder
# ---------------------------------------------------------------------------

class OnlinePairBuilder:
    """Build (chosen, rejected) index pairs from per-completion reward scores.

    Given a flat reward vector of length B*K (K completions per prompt), reshapes
    to (B, K), finds the argmax (chosen) and argmin (rejected) per prompt, and
    returns global indices into the flat vector.

    Args:
        config: OnlineDPOConfig — uses config.n_completions as default group_size.
    """

    def __init__(self, config: OnlineDPOConfig) -> None:
        self.config = config

    def build_pairs(
        self,
        rewards: Tensor,
        group_size: int | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """Extract (chosen_indices, rejected_indices) from flat rewards.

        Args:
            rewards: Shape (B * K,) — scalar reward for each completion.
                     The first K entries correspond to batch item 0, etc.
            group_size: Number of completions per prompt (K). Defaults to
                        config.n_completions if None.

        Returns:
            Tuple of two LongTensors, each of shape (B,):
              - chosen_indices:   flat indices of the highest-reward completion.
              - rejected_indices: flat indices of the lowest-reward completion.
        """
        K = group_size if group_size is not None else self.config.n_completions
        total = rewards.shape[0]
        if total % K != 0:
            raise ValueError(
                f"rewards length {total} is not divisible by group_size {K}"
            )
        B = total // K

        rewards_2d = rewards.reshape(B, K)  # (B, K)

        chosen_local = rewards_2d.argmax(dim=1)   # (B,) local indices in [0, K)
        rejected_local = rewards_2d.argmin(dim=1)  # (B,)

        # Convert to global indices in the flat (B*K,) tensor
        offsets = torch.arange(B, device=rewards.device) * K  # (B,)
        chosen_indices = offsets + chosen_local    # (B,)
        rejected_indices = offsets + rejected_local  # (B,)

        return chosen_indices, rejected_indices


# ---------------------------------------------------------------------------
# OnlineDPOLoss
# ---------------------------------------------------------------------------

class OnlineDPOLoss(nn.Module):
    """Standard DPO loss for a single (chosen, rejected) pair per batch item.

    Loss = mean(-log_sigmoid(beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))))

    Args:
        beta: KL penalty coefficient.
    """

    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        pi_chosen: Tensor,
        pi_rejected: Tensor,
        ref_chosen: Tensor,
        ref_rejected: Tensor,
    ) -> Tuple[Tensor, Dict]:
        """Compute DPO loss and associated metrics.

        Args:
            pi_chosen:    Shape (B,) — log-probs of chosen under the policy.
            pi_rejected:  Shape (B,) — log-probs of rejected under the policy.
            ref_chosen:   Shape (B,) — log-probs of chosen under the reference.
            ref_rejected: Shape (B,) — log-probs of rejected under the reference.

        Returns:
            Tuple of (loss, metrics):
              - loss: scalar Tensor.
              - metrics: dict with keys "loss", "accuracy", "reward_chosen",
                         "reward_rejected", "margin".
        """
        # Implicit reward under the policy relative to reference
        reward_chosen = self.beta * (pi_chosen - ref_chosen)     # (B,)
        reward_rejected = self.beta * (pi_rejected - ref_rejected)  # (B,)

        # DPO contrastive logit
        logits = reward_chosen - reward_rejected  # (B,)

        # Loss: -log σ(logit)
        per_sample = -F.logsigmoid(logits)  # (B,)
        loss = per_sample.mean()            # scalar

        accuracy = (logits > 0).float().mean().item()
        reward_chosen_mean = reward_chosen.mean().item()
        reward_rejected_mean = reward_rejected.mean().item()
        margin = reward_chosen_mean - reward_rejected_mean

        metrics: Dict = {
            "loss": loss.item(),
            "accuracy": accuracy,
            "reward_chosen": reward_chosen_mean,
            "reward_rejected": reward_rejected_mean,
            "margin": margin,
        }

        return loss, metrics


# ---------------------------------------------------------------------------
# OnlineDPOTrainer
# ---------------------------------------------------------------------------

class OnlineDPOTrainer:
    """Orchestrates one Online DPO update step.

    Args:
        policy_model: The trainable policy (nn.Module).
        ref_model: Frozen reference model (nn.Module).
        optimizer: PyTorch optimizer bound to policy_model.parameters().
        config: OnlineDPOConfig.
        loss_fn: OnlineDPOLoss instance.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: OnlineDPOConfig,
        loss_fn: OnlineDPOLoss,
    ) -> None:
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.config = config
        self.loss_fn = loss_fn

    def freeze_ref(self) -> None:
        """Freeze all parameters of the reference model."""
        for param in self.ref_model.parameters():
            param.requires_grad_(False)

    def online_step(
        self,
        pi_chosen: Tensor,
        pi_rejected: Tensor,
        ref_chosen: Tensor,
        ref_rejected: Tensor,
    ) -> Dict:
        """Perform one backward pass and optimizer step, return metrics.

        Args:
            pi_chosen:    Shape (B,) — policy log-probs for chosen completions.
            pi_rejected:  Shape (B,) — policy log-probs for rejected completions.
            ref_chosen:   Shape (B,) — reference log-probs for chosen (no grad).
            ref_rejected: Shape (B,) — reference log-probs for rejected (no grad).

        Returns:
            Dict with keys: "loss", "accuracy", "reward_chosen",
            "reward_rejected", "margin".
        """
        self.optimizer.zero_grad()
        loss, metrics = self.loss_fn(pi_chosen, pi_rejected, ref_chosen, ref_rejected)
        loss.backward()
        self.optimizer.step()
        return metrics
