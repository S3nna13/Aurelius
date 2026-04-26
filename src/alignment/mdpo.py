"""Mirror Descent Policy Optimization (MDPO).

Online RL alignment via mirror descent updates with KL-proximal regularization.
Based on Tomar et al. 2020 and Ziegler et al. 2019 style KL-constrained updates.

Reference:
    - Tomar et al. (2020): Mirror Descent Policy Optimization
    - Ziegler et al. (2019): Fine-Tuning Language Models from Human Preferences
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MDPOConfig:
    """Configuration for MDPO training."""

    kl_coef: float = 0.1  # KL penalty coefficient beta
    lr: float = 1e-5  # learning rate
    n_steps: int = 4  # gradient steps per batch
    max_grad_norm: float = 1.0  # gradient clipping
    reward_scale: float = 1.0  # scale rewards before applying
    entropy_coef: float = 0.01  # entropy bonus coefficient
    max_seq_len: int = 64


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class MDPOBatch:
    """A single MDPO training batch."""

    input_ids: torch.Tensor  # (B, T) prompt + response token ids
    rewards: torch.Tensor  # (B,) scalar reward per sequence
    ref_log_probs: torch.Tensor  # (B, T) log probs under reference policy
    prompt_len: int  # length of prompt prefix (not trained on)


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------


def sequence_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    prompt_len: int = 0,
) -> torch.Tensor:
    """Compute per-token log probabilities for response tokens only.

    Returns (B, T-prompt_len-1) log probs of each response token.
    We compute logprob of token[t+1] given token[0..t].

    Args:
        model: Policy model.  If it returns a tuple, logits = out[1].
        input_ids: (B, T) integer token ids.
        prompt_len: Number of prompt tokens to skip at the start.

    Returns:
        Tensor of shape (B, T - prompt_len - 1) with log probabilities
        for each response token (all values <= 0).
    """
    with torch.no_grad():
        out = model(input_ids)
        logits = out[1] if isinstance(out, tuple) else out  # (B, T, V)

    log_probs_all = F.log_softmax(logits, dim=-1)  # (B, T, V)

    # We want logprob of token[t+1] given context token[0..t],
    # for t in [prompt_len, ..., T-2] (response token positions).
    # Context logits: positions [prompt_len, ..., T-2]  => log_probs_all[:, prompt_len:-1, :]
    # Target token ids: positions [prompt_len+1, ..., T-1] => input_ids[:, prompt_len+1:]
    # This yields T - prompt_len - 1 response tokens, matching the spec.
    start = prompt_len
    ctx_log_probs = log_probs_all[:, start:-1, :]  # (B, S, V)
    target_ids = input_ids[:, start + 1 :]  # (B, S)

    token_log_probs = ctx_log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(
        -1
    )  # (B, S)

    return token_log_probs


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def mdpo_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    kl_coef: float = 0.1,
    entropy_coef: float = 0.01,
) -> tuple[torch.Tensor, dict]:
    """MDPO objective: maximize (reward - beta * KL) + entropy bonus.

    loss = -mean(rewards) + kl_coef * KL(policy || ref) - entropy_coef * H(policy)

    where KL = mean over (B, S) of (log_probs - ref_log_probs)
    and H = -mean(log_probs)

    Args:
        log_probs: (B, S) log probs from current policy.
        ref_log_probs: (B, S) log probs from reference policy.
        rewards: (B,) scalar reward per sequence.
        kl_coef: KL penalty coefficient beta.
        entropy_coef: Entropy bonus coefficient.

    Returns:
        (scalar_loss, metrics_dict) where metrics_dict has keys:
        'loss', 'reward', 'kl', 'entropy'
    """
    # KL divergence term (mean over batch and sequence positions)
    kl = (log_probs - ref_log_probs).mean()

    # Entropy: H(pi) = -E[log pi]
    entropy = -log_probs.mean()

    # Mean reward
    reward = rewards.mean()

    # Minimize negative of (reward - beta*KL + entropy_coef*H)
    loss = -reward + kl_coef * kl - entropy_coef * entropy

    metrics: dict = {
        "loss": loss.item(),
        "reward": reward.item(),
        "kl": kl.item(),
        "entropy": entropy.item(),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class MDPOTrainer:
    """Online MDPO trainer with KL-proximal regularization."""

    def __init__(self, model: nn.Module, ref_model: nn.Module, config: MDPOConfig):
        self.model = model
        self.ref_model = ref_model
        self.config = config
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    def train_step(self, batch: MDPOBatch) -> dict:
        """Single MDPO update step.  Runs config.n_steps gradient steps.

        Returns:
            metrics dict from the final gradient step.
        """
        self.model.train()
        metrics: dict = {}

        for _ in range(self.config.n_steps):
            out = self.model(batch.input_ids)
            logits = out[1] if isinstance(out, tuple) else out  # (B, T, V)
            log_probs_all = F.log_softmax(logits, dim=-1)

            prompt_len = batch.prompt_len
            start = prompt_len
            ctx_log_probs = log_probs_all[:, start:-1, :]
            target_ids = batch.input_ids[:, start + 1 :]
            log_probs = ctx_log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

            scaled_rewards = batch.rewards * self.config.reward_scale

            loss, metrics = mdpo_loss(
                log_probs=log_probs,
                ref_log_probs=batch.ref_log_probs,
                rewards=scaled_rewards,
                kl_coef=self.config.kl_coef,
                entropy_coef=self.config.entropy_coef,
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        return metrics

    def make_batch(
        self,
        input_ids: torch.Tensor,
        rewards: torch.Tensor,
        prompt_len: int,
    ) -> MDPOBatch:
        """Construct an MDPOBatch by computing ref_log_probs from ref_model.

        Args:
            input_ids: (B, T) token ids (prompt + response).
            rewards: (B,) scalar reward per sequence.
            prompt_len: number of prompt prefix tokens.

        Returns:
            MDPOBatch with ref_log_probs from the frozen reference model.
        """
        self.ref_model.model_eval_mode = True
        self.ref_model.eval()
        ref_log_probs = sequence_log_probs(self.ref_model, input_ids, prompt_len=prompt_len)

        return MDPOBatch(
            input_ids=input_ids,
            rewards=rewards,
            ref_log_probs=ref_log_probs,
            prompt_len=prompt_len,
        )
