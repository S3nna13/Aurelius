"""PPO (Proximal Policy Optimization) for RLHF — v2.

Implements KL-penalized PPO with a value network (critic), GAE advantage
estimation, and KL-divergence penalty against a frozen reference policy.

Classes:
    ValueHead         - scalar per-token value estimates from hidden states
    GAEComputation    - Generalized Advantage Estimation
    PPOLoss           - clipped surrogate + value + entropy + KL losses
    PPOBuffer         - experience replay buffer for PPO
    PPOTrainer        - full RLHF PPO training loop
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# ValueHead
# ---------------------------------------------------------------------------


class ValueHead(nn.Module):
    """Scalar value estimator on top of transformer hidden states.

    Architecture: Linear(d_model, d_model//2) -> GELU -> Linear(d_model//2, 1)
    Output is squeezed to (B, T).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        mid = max(1, d_model // 2)
        self.fc1 = nn.Linear(d_model, mid)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mid, 1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Compute per-token value estimates.

        Args:
            hidden_states: (B, T, D) transformer hidden states.

        Returns:
            values: (B, T) scalar value estimates.
        """
        x = self.fc1(hidden_states)
        x = self.act(x)
        x = self.fc2(x)
        return x.squeeze(-1)


# ---------------------------------------------------------------------------
# GAEComputation
# ---------------------------------------------------------------------------


class GAEComputation:
    """Generalized Advantage Estimation (GAE-Lambda).

    Computes advantages and returns from per-step rewards, values, and done
    flags using the standard GAE formula (Schulman et al., 2015).
    """

    def __init__(self, gamma: float = 0.99, lam: float = 0.95) -> None:
        self.gamma = gamma
        self.lam = lam

    def compute(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute GAE advantages and discounted returns.

        Args:
            rewards: (B, T) per-step rewards.
            values:  (B, T) value estimates from the critic.
            dones:   (B, T) episode-done flags (1.0=done, 0.0=not done).

        Returns:
            advantages: (B, T)
            returns:    (B, T)  = advantages + values (value targets).
        """
        B, T = rewards.shape
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
            else:
                next_value = values[:, t + 1] * (1.0 - dones[:, t])

            delta = rewards[:, t] + self.gamma * next_value - values[:, t]
            gae = delta + self.gamma * self.lam * (1.0 - dones[:, t]) * gae
            advantages[:, t] = gae

        returns = advantages + values
        return advantages, returns


# ---------------------------------------------------------------------------
# PPOLoss
# ---------------------------------------------------------------------------


class PPOLoss(nn.Module):
    """Clipped PPO surrogate loss with value, entropy, and KL components.

    Args:
        clip_eps:  PPO clipping epsilon (default 0.2).
        vf_coeff:  value-function loss coefficient (default 0.5).
        ent_coeff: entropy bonus coefficient (default 0.01).
        kl_coeff:  KL-divergence penalty coefficient (default 0.1).
    """

    def __init__(
        self,
        clip_eps: float = 0.2,
        vf_coeff: float = 0.5,
        ent_coeff: float = 0.01,
        kl_coeff: float = 0.1,
    ) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.vf_coeff = vf_coeff
        self.ent_coeff = ent_coeff
        self.kl_coeff = kl_coeff

    def forward(
        self,
        log_probs: Tensor,
        old_log_probs: Tensor,
        advantages: Tensor,
        values: Tensor,
        returns: Tensor,
        ref_log_probs: Tensor,
    ) -> dict[str, Tensor]:
        """Compute PPO loss components.

        All inputs are (B, T) tensors.

        Returns:
            dict with keys: total, policy_loss, value_loss, entropy_loss,
                            kl_loss, clip_fraction.
        """
        # policy loss (clipped surrogate)
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        # clip fraction (diagnostic, no grad)
        with torch.no_grad():
            clipped = (ratio < 1.0 - self.clip_eps) | (ratio > 1.0 + self.clip_eps)
            clip_fraction = clipped.float().mean()

        # value loss
        value_loss = self.vf_coeff * F.mse_loss(values, returns.detach())

        # entropy loss (encourage exploration; entropy approx = -log_prob)
        entropy_loss = -self.ent_coeff * torch.mean(-log_probs)

        # KL divergence penalty (stay near reference policy)
        kl_loss = self.kl_coeff * torch.mean(log_probs - ref_log_probs)

        total = policy_loss + value_loss + entropy_loss + kl_loss

        return {
            "total": total,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "kl_loss": kl_loss,
            "clip_fraction": clip_fraction,
        }


# ---------------------------------------------------------------------------
# PPOBuffer
# ---------------------------------------------------------------------------


class PPOBuffer:
    """Fixed-capacity experience buffer for PPO.

    Stores full episode trajectories and returns them stacked as a batch.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._data: list[dict[str, Tensor]] = []

    def add(
        self,
        input_ids: Tensor,
        log_probs: Tensor,
        values: Tensor,
        rewards: Tensor,
        dones: Tensor,
    ) -> None:
        """Store one episode's tensors."""
        self._data.append(
            {
                "input_ids": input_ids.detach(),
                "log_probs": log_probs.detach(),
                "values": values.detach(),
                "rewards": rewards.detach(),
                "dones": dones.detach(),
            }
        )

    def get_batch(self) -> dict[str, Tensor]:
        """Return all stored episodes stacked along dim=0."""
        if not self._data:
            raise RuntimeError("PPOBuffer is empty.")
        keys = list(self._data[0].keys())
        return {k: torch.cat([ep[k] for ep in self._data], dim=0) for k in keys}

    def clear(self) -> None:
        """Empty the buffer."""
        self._data.clear()

    def is_full(self) -> bool:
        """Return True when the buffer holds at least capacity episodes."""
        return len(self._data) >= self.capacity


# ---------------------------------------------------------------------------
# PPOTrainer
# ---------------------------------------------------------------------------


class PPOTrainer:
    """Full RLHF PPO training loop.

    Args:
        policy_model:     callable (input_ids) -> (logits, hidden_states).
                          logits: (B, T, V), hidden_states: (B, T, D).
        ref_model:        same signature, frozen (no grad).
        value_head:       ValueHead instance.
        reward_fn:        callable (sequences: Tensor) -> rewards (B,).
        optimizer_policy: optimizer for policy_model parameters.
        optimizer_value:  optimizer for value_head parameters.
        ppo_epochs:       number of PPO update epochs per train_epoch call.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        value_head: ValueHead,
        reward_fn: Callable[[Tensor], Tensor],
        optimizer_policy: torch.optim.Optimizer,
        optimizer_value: torch.optim.Optimizer,
        ppo_epochs: int = 4,
    ) -> None:
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.value_head = value_head
        self.reward_fn = reward_fn
        self.optimizer_policy = optimizer_policy
        self.optimizer_value = optimizer_value
        self.ppo_epochs = ppo_epochs

        self.ppo_loss_fn = PPOLoss()
        self.gae = GAEComputation()

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(self, input_ids: Tensor, max_new_tokens: int = 16) -> dict[str, Tensor]:
        """Generate sequences, collect log-probs, values, and rewards.

        Args:
            input_ids:      (B, T_prompt) prompt token ids.
            max_new_tokens: number of tokens to generate.

        Returns:
            dict with keys: sequences, log_probs, values, rewards.
            sequences: (B, T_prompt + max_new_tokens)
            log_probs:  (B, max_new_tokens)
            values:     (B, max_new_tokens)
            rewards:    (B,) scalar reward per sequence
        """
        self.policy_model.eval()
        self.value_head.eval()

        input_ids.shape[0]
        generated = input_ids.clone()

        collected_log_probs: list[Tensor] = []
        collected_values: list[Tensor] = []

        for _ in range(max_new_tokens):
            logits, hidden_states = self.policy_model(generated)

            last_logits = logits[:, -1, :]
            last_hidden = hidden_states[:, -1:, :]

            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            log_prob_dist = F.log_softmax(last_logits, dim=-1)
            token_log_prob = log_prob_dist.gather(1, next_token).squeeze(1)

            value = self.value_head(last_hidden).squeeze(-1)
            if value.dim() > 1:
                value = value.squeeze(-1)

            collected_log_probs.append(token_log_prob)
            collected_values.append(value)

            generated = torch.cat([generated, next_token], dim=1)

        log_probs = torch.stack(collected_log_probs, dim=1)
        values = torch.stack(collected_values, dim=1)
        rewards = self.reward_fn(generated)

        return {
            "sequences": generated,
            "log_probs": log_probs,
            "values": values,
            "rewards": rewards,
        }

    # ------------------------------------------------------------------
    # train_epoch
    # ------------------------------------------------------------------

    def train_epoch(self, input_ids: Tensor) -> dict[str, float]:
        """Collect a rollout then perform ppo_epochs of PPO updates.

        Args:
            input_ids: (B, T_prompt) prompt token ids.

        Returns:
            dict: avg_total_loss, policy_loss, value_loss, kl_loss, mean_reward.
        """
        rollout_data = self.rollout(input_ids)
        sequences = rollout_data["sequences"]
        old_log_probs = rollout_data["log_probs"]
        old_values = rollout_data["values"]
        rewards_scalar = rollout_data["rewards"]

        B, T_new = old_log_probs.shape
        T_prompt = sequences.shape[1] - T_new

        # Broadcast scalar reward to per-token (reward at last step)
        rewards_per_step = torch.zeros(B, T_new, device=old_log_probs.device)
        rewards_per_step[:, -1] = rewards_scalar

        dones = torch.zeros(B, T_new, device=old_log_probs.device)
        dones[:, -1] = 1.0

        advantages, returns = self.gae.compute(rewards_per_step, old_values, dones)

        # Reference log-probs (frozen)
        with torch.no_grad():
            ref_logits, _ = self.ref_model(sequences)
            gen_tokens = sequences[:, T_prompt:]
            ref_lp_all = F.log_softmax(ref_logits[:, T_prompt - 1 : -1, :], dim=-1)
            ref_log_probs = ref_lp_all.gather(2, gen_tokens.unsqueeze(-1)).squeeze(-1)

        total_losses: list[float] = []
        pol_losses: list[float] = []
        val_losses: list[float] = []
        kl_losses: list[float] = []

        self.policy_model.train()
        self.value_head.train()

        for _ in range(self.ppo_epochs):
            logits, hidden_states = self.policy_model(sequences)
            gen_tokens = sequences[:, T_prompt:]

            curr_lp_all = F.log_softmax(logits[:, T_prompt - 1 : -1, :], dim=-1)
            curr_log_probs = curr_lp_all.gather(2, gen_tokens.unsqueeze(-1)).squeeze(-1)

            curr_values = self.value_head(hidden_states[:, T_prompt - 1 : -1, :])

            loss_dict = self.ppo_loss_fn(
                log_probs=curr_log_probs,
                old_log_probs=old_log_probs.detach(),
                advantages=advantages.detach(),
                values=curr_values,
                returns=returns.detach(),
                ref_log_probs=ref_log_probs,
            )

            total_loss = loss_dict["total"]
            self.optimizer_policy.zero_grad()
            self.optimizer_value.zero_grad()
            total_loss.backward()
            self.optimizer_policy.step()
            self.optimizer_value.step()

            total_losses.append(total_loss.item())
            pol_losses.append(loss_dict["policy_loss"].item())
            val_losses.append(loss_dict["value_loss"].item())
            kl_losses.append(loss_dict["kl_loss"].item())

        return {
            "avg_total_loss": sum(total_losses) / len(total_losses),
            "policy_loss": sum(pol_losses) / len(pol_losses),
            "value_loss": sum(val_losses) / len(val_losses),
            "kl_loss": sum(kl_losses) / len(kl_losses),
            "mean_reward": rewards_scalar.mean().item(),
        }
