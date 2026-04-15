"""PPO (Proximal Policy Optimization) trainer for RLHF.

Implements the core PPO components:
- PPOConfig: hyperparameters for PPO training
- compute_gae: Generalized Advantage Estimation
- compute_policy_loss: Clipped surrogate objective
- compute_value_loss: Clipped value function loss
- compute_entropy: Entropy bonus
- PPOLoss: Combined loss module
- PPOTrainer: Full training loop
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """Hyperparameters for PPO training."""
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.1
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4
    mini_batch_size: int = 8


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE-Lambda).

    Args:
        rewards: (T,) rewards at each timestep
        values:  (T,) baseline value estimates V(s_t)
        dones:   (T,) episode termination flags (1.0 = done)
        gamma:   discount factor
        lam:     GAE lambda

    Returns:
        (advantages, returns) each of shape (T,)
        advantages[t] = delta_t + gamma*lam*delta_{t+1} + ...
        returns = advantages + values
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T, dtype=rewards.dtype, device=rewards.device)
    dones_float = dones.float()

    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = 0.0
        else:
            next_val = float(values[t + 1]) * (1.0 - float(dones_float[t]))

        delta = float(rewards[t]) + gamma * next_val - float(values[t])
        gae = delta + gamma * lam * (1.0 - float(dones_float[t])) * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def compute_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
) -> torch.Tensor:
    """PPO clipped surrogate policy loss.

    Args:
        log_probs:     (T,) or (B,) log-probs under current policy
        old_log_probs: (T,) or (B,) log-probs under old policy (no grad)
        advantages:    (T,) or (B,) advantage estimates
        clip_eps:      clipping epsilon

    Returns:
        Scalar policy loss (to be minimized — negative of the PPO objective).
    """
    ratio = torch.exp(log_probs - old_log_probs)
    loss_unclipped = ratio * advantages
    loss_clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    # PPO objective: maximize E[min(ratio*A, clip*A)]  → minimize its negation
    return -torch.min(loss_unclipped, loss_clipped).mean()


def compute_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float,
) -> torch.Tensor:
    """Clipped value function loss.

    Args:
        values:     (T,) current value estimates (with grad)
        old_values: (T,) value estimates from rollout collection (no grad)
        returns:    (T,) GAE returns (targets)
        clip_eps:   clipping epsilon for value function

    Returns:
        Scalar value loss (non-negative).
    """
    values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
    loss_unclipped = (values - returns).pow(2)
    loss_clipped = (values_clipped - returns).pow(2)
    return torch.max(loss_unclipped, loss_clipped).mean()


def compute_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """Entropy approximation from scalar log-probs.

    Uses H ≈ -mean(log_probs) as a proxy for entropy (when log_probs are
    per-token summed log-probs or per-sample log-probs).

    Args:
        log_probs: (T,) or (B,) log-probabilities

    Returns:
        Scalar entropy estimate (non-negative proxy).
    """
    return -log_probs.mean()


# ---------------------------------------------------------------------------
# Combined loss module
# ---------------------------------------------------------------------------

class PPOLoss(nn.Module):
    """Combines policy loss, value loss, and entropy bonus into a single loss.

    Args:
        config: PPOConfig with loss coefficients
    """

    def __init__(self, config: PPOConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all PPO losses.

        Args:
            log_probs:     (B,) current policy log-probs
            old_log_probs: (B,) old policy log-probs (detached)
            advantages:    (B,) GAE advantages
            values:        (B,) current value estimates
            old_values:    (B,) old value estimates (detached)
            returns:       (B,) GAE returns (targets)

        Returns:
            Dict with keys: 'total_loss', 'policy_loss', 'value_loss', 'entropy'
        """
        cfg = self.config

        policy_loss = compute_policy_loss(log_probs, old_log_probs, advantages, cfg.clip_eps)
        value_loss = compute_value_loss(values, old_values, returns, cfg.value_clip_eps)
        entropy = compute_entropy(log_probs)

        total_loss = (
            policy_loss
            + cfg.value_coef * value_loss
            - cfg.entropy_coef * entropy
        )

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        }


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """PPO training loop for RLHF.

    Args:
        policy_model: Model that returns (_, logits, _) where logits is (B, T, V)
        value_model:  Model that returns (_, values, _) where values is (B, T, 1)
        ref_model:    Reference (frozen) model for KL penalty
        optimizer:    PyTorch optimizer for policy and value parameters
        config:       PPOConfig
    """

    def __init__(
        self,
        policy_model: nn.Module,
        value_model: nn.Module,
        ref_model: nn.Module,
        optimizer,
        config: PPOConfig,
    ) -> None:
        self.policy_model = policy_model
        self.value_model = value_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.config = config
        self.ppo_loss = PPOLoss(config)

    def get_log_probs(self, model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute summed log-probs per sequence.

        Runs a forward pass of ``model`` and collects the log-probability of
        each token (shifted by one), then sums across the sequence dimension.

        Args:
            model:     Model with forward(input_ids) -> (_, logits, _),
                       logits shape (B, T, V)
            input_ids: (B, T) token ids

        Returns:
            (B,) summed log-probs per sequence (non-positive).
        """
        _, logits, _ = model(input_ids)  # (B, T, V)
        B, T, V = logits.shape

        if T < 2:
            # Degenerate case: no next-token to predict
            return torch.zeros(B, device=logits.device)

        # Predict token t+1 from position t: logits[:, :-1, :] → target ids[:, 1:]
        shift_logits = logits[:, :-1, :]      # (B, T-1, V)
        shift_targets = input_ids[:, 1:]      # (B, T-1)

        log_probs_all = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)
        # Gather log-prob of the actual next token
        token_lp = log_probs_all.gather(
            2, shift_targets.unsqueeze(-1)
        ).squeeze(-1)  # (B, T-1)

        # Sum across time → (B,)  (sum of non-positive values → non-positive)
        return token_lp.sum(dim=-1)

    def compute_kl_divergence(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean per-sample KL divergence approximation.

        KL ≈ mean(log_probs - ref_log_probs)

        Args:
            log_probs:     (B,) current policy log-probs
            ref_log_probs: (B,) reference policy log-probs

        Returns:
            Scalar KL estimate.
        """
        return (log_probs - ref_log_probs).mean()

    def ppo_step(
        self,
        input_ids: torch.Tensor,
        rewards: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> Dict[str, float]:
        """Run a single PPO update step.

        Treats each token position as a timestep for GAE computation.
        Runs ppo_epochs mini-epoch updates.

        Args:
            input_ids:     (B, T) token ids
            rewards:       (B,) scalar reward per sequence (broadcast to tokens)
            old_log_probs: (B,) log-probs from rollout collection (detached)

        Returns:
            Dict with float metrics: 'total_loss', 'policy_loss', 'value_loss',
            'entropy', 'kl'.
        """
        cfg = self.config
        B, T = input_ids.shape
        device = input_ids.device

        # ---------- collect reference log-probs (frozen) ----------
        with torch.no_grad():
            ref_log_probs = self.get_log_probs(self.ref_model, input_ids)  # (B,)

        # ---------- collect value estimates (no grad for GAE targets) ----------
        with torch.no_grad():
            _, v_out, _ = self.value_model(input_ids)  # (B, T, 1)
            # Mean across token dimension → (B,)
            values_for_gae = v_out.squeeze(-1).mean(dim=-1)  # (B,)

        # ---------- GAE: treat each sequence as a single-step episode ----------
        # For simplicity: use per-sequence rewards and values (one "timestep" per seq)
        dones = torch.ones(B, device=device)  # each sequence ends after one step
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values_for_gae,
            dones=dones,
            gamma=cfg.gamma,
            lam=cfg.lam,
        )
        advantages = advantages.detach()
        returns = returns.detach()
        old_vals_detached = values_for_gae.detach()

        # ---------- PPO epochs ----------
        total_metrics: Dict[str, float] = {
            "total_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
        }
        n_updates = 0

        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(B, device=device)

            for start in range(0, B, cfg.mini_batch_size):
                mb_idx = perm[start : start + cfg.mini_batch_size]
                if mb_idx.numel() == 0:
                    continue

                mb_ids = input_ids[mb_idx]             # (mb, T)
                mb_rewards = rewards[mb_idx]            # (mb,) — not used directly; GAE already done
                mb_adv = advantages[mb_idx]             # (mb,)
                mb_ret = returns[mb_idx]                # (mb,)
                mb_old_lp = old_log_probs[mb_idx]       # (mb,)
                mb_old_vals = old_vals_detached[mb_idx] # (mb,)

                self.optimizer.zero_grad()

                # New policy log-probs
                new_log_probs = self.get_log_probs(self.policy_model, mb_ids)  # (mb,)

                # New value estimates
                _, v_out_new, _ = self.value_model(mb_ids)  # (mb, T, 1)
                new_values = v_out_new.squeeze(-1).mean(dim=-1)  # (mb,)

                # PPO losses
                loss_dict = self.ppo_loss(
                    log_probs=new_log_probs,
                    old_log_probs=mb_old_lp.detach(),
                    advantages=mb_adv,
                    values=new_values,
                    old_values=mb_old_vals,
                    returns=mb_ret,
                )

                # KL penalty (add to total loss)
                kl = self.compute_kl_divergence(
                    new_log_probs.detach(), ref_log_probs[mb_idx].detach()
                )

                total = loss_dict["total_loss"] + cfg.kl_coef * kl
                total.backward()

                nn.utils.clip_grad_norm_(
                    list(self.policy_model.parameters()) + list(self.value_model.parameters()),
                    cfg.max_grad_norm,
                )
                self.optimizer.step()

                total_metrics["total_loss"] += loss_dict["total_loss"].item()
                total_metrics["policy_loss"] += loss_dict["policy_loss"].item()
                total_metrics["value_loss"] += loss_dict["value_loss"].item()
                total_metrics["entropy"] += loss_dict["entropy"].item()
                total_metrics["kl"] += kl.item()
                n_updates += 1

        denom = max(n_updates, 1)
        return {k: v / denom for k, v in total_metrics.items()}
