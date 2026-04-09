"""Complete PPO training loop for RLHF: rollout collection, GAE advantages, clipped policy gradient."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """Hyperparameters for PPO training."""

    clip_epsilon: float = 0.2          # PPO policy clipping parameter
    value_clip: float = 0.2            # value function clipping parameter
    entropy_coeff: float = 0.01        # entropy bonus coefficient
    value_coeff: float = 0.5           # value loss coefficient
    gamma: float = 0.99                # discount factor
    gae_lambda: float = 0.95           # GAE lambda
    n_epochs: int = 4                  # PPO epochs per rollout
    minibatch_size: int = 4            # minibatch size for PPO updates
    max_grad_norm: float = 0.5         # gradient clipping norm
    normalize_advantages: bool = True  # whether to normalize advantages
    kl_target: float = 0.01           # adaptive KL penalty target


# ---------------------------------------------------------------------------
# GAE advantages
# ---------------------------------------------------------------------------

def compute_gae_advantages(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    config: PPOConfig,
) -> tuple[Tensor, Tensor]:
    """Compute Generalized Advantage Estimation (GAE) advantages and returns.

    Args:
        rewards: (T,) rewards at each timestep
        values:  (T,) value estimates V(s_t)
        dones:   (T,) float/bool, 1.0 if episode ended at t
        config:  PPOConfig with gamma and gae_lambda

    Returns:
        (advantages, returns) both (T,)
        advantages[t] = delta_t + gamma * lambda * advantages[t+1]
        delta_t = rewards[t] + gamma * values[t+1] * (1-dones[t]) - values[t]
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T, device=rewards.device, dtype=rewards.dtype)
    dones_float = dones.float()

    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = 0.0
        else:
            next_val = values[t + 1] * (1.0 - dones_float[t])

        delta = rewards[t] + config.gamma * next_val - values[t]
        gae = delta + config.gamma * config.gae_lambda * (1.0 - dones_float[t]) * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO losses
# ---------------------------------------------------------------------------

def ppo_policy_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    clip_epsilon: float,
) -> Tensor:
    """Clipped PPO policy loss.

    Args:
        log_probs:     log probs under current policy
        old_log_probs: log probs under old policy (from rollout)
        advantages:    same shape as log_probs
        clip_epsilon:  clipping parameter epsilon

    Returns:
        Scalar loss = -mean(min(r * A, clip(r, 1-eps, 1+eps) * A))
    """
    r = (log_probs - old_log_probs).exp()
    loss_unclipped = r * advantages
    loss_clipped = r.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    loss = -torch.min(loss_unclipped, loss_clipped).mean()
    return loss


def ppo_value_loss(
    values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    value_clip: float,
) -> Tensor:
    """Clipped value function loss.

    Args:
        values:     current value estimates
        old_values: value estimates from rollout collection
        returns:    discounted returns (targets)
        value_clip: clipping range epsilon

    Returns:
        Scalar loss = mean(max(mse(v, ret), mse(clip(v, old_v-eps, old_v+eps), ret)))
    """
    v_clipped = old_values + (values - old_values).clamp(-value_clip, value_clip)
    loss_unclipped = (values - returns) ** 2
    loss_clipped = (v_clipped - returns) ** 2
    loss = torch.max(loss_unclipped, loss_clipped).mean()
    return loss


def compute_entropy_bonus(log_probs: Tensor) -> Tensor:
    """Entropy proxy: mean negative log probability.

    Args:
        log_probs: log probabilities

    Returns:
        Scalar = -mean(log_probs)
    """
    return -log_probs.mean()


# ---------------------------------------------------------------------------
# Rollout dataclass
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    """Data collected during a policy rollout."""

    input_ids: Tensor    # (B, T) generated tokens
    log_probs: Tensor    # (B, T) log probs from policy at rollout time
    values: Tensor       # (B, T) value estimates at rollout time
    rewards: Tensor      # (B,) scalar rewards per sample
    advantages: Tensor   # (B, T) computed GAE advantages
    returns: Tensor      # (B, T) discounted returns


# ---------------------------------------------------------------------------
# PPOTrainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """Complete PPO training loop for RLHF.

    Collects rollouts from a policy, computes GAE advantages, and performs
    multiple epochs of clipped policy gradient updates.
    """

    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        value_fn: nn.Module,
        reward_fn: Callable,
        policy_optimizer,
        value_optimizer,
        config: PPOConfig,
    ) -> None:
        self.policy = policy
        self.ref_policy = ref_policy
        self.value_fn = value_fn
        self.reward_fn = reward_fn
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.config = config

    def collect_rollout(self, prompt_ids: Tensor, max_new_tokens: int) -> Rollout:
        """Collect a rollout by generating responses from the policy.

        Args:
            prompt_ids:     (B, prompt_len) prompt token ids
            max_new_tokens: number of new tokens to generate per sample

        Returns:
            Rollout with (B, T) sequence fields and (B,) rewards
        """
        cfg = self.config
        B = prompt_ids.shape[0]
        T = max_new_tokens

        self.policy.eval()
        self.value_fn.eval()

        all_input_ids = []
        all_log_probs = []
        all_rewards = []

        with torch.no_grad():
            for b in range(B):
                prompt = prompt_ids[b:b+1]  # (1, prompt_len)

                # Generate tokens autoregressively (greedy)
                generated_ids_list = []
                log_probs_list = []
                cur_ids = prompt
                past_key_values = None

                for _ in range(T):
                    _, logits, past_key_values = self.policy(
                        cur_ids, past_key_values=past_key_values
                    )
                    next_logits = logits[:, -1, :]  # (1, vocab_size)
                    log_probs_step = F.log_softmax(next_logits, dim=-1)

                    # Greedy decoding
                    next_token = next_logits.argmax(dim=-1, keepdim=True)  # (1, 1)
                    token_log_prob = log_probs_step.gather(1, next_token)   # (1, 1)

                    generated_ids_list.append(next_token)
                    log_probs_list.append(token_log_prob)
                    cur_ids = next_token

                gen_ids = torch.cat(generated_ids_list, dim=1)  # (1, T)
                gen_lp = torch.cat(log_probs_list, dim=1)        # (1, T)

                reward = self.reward_fn(gen_ids[0].tolist())

                all_input_ids.append(gen_ids)
                all_log_probs.append(gen_lp)
                all_rewards.append(float(reward))

        # Stack across batch dimension
        input_ids = torch.cat(all_input_ids, dim=0)  # (B, T)
        log_probs = torch.cat(all_log_probs, dim=0)  # (B, T)
        rewards_tensor = torch.tensor(
            all_rewards, dtype=torch.float32, device=prompt_ids.device
        )  # (B,)

        # Estimate values: get policy logits then apply value_fn linear head
        # Spec: "forward -> mean-pool logits -> Linear head"
        with torch.no_grad():
            _, policy_logits, _ = self.policy(input_ids)
            # policy_logits: (B, T, vocab_size)
            # Mean-pool over vocab dim: (B, T, vocab_size) -> (B, T, vocab_size)
            # Then apply value_fn (nn.Linear(vocab_size, 1)) -> (B, T, 1) -> squeeze -> (B, T)
            val_out = self.value_fn(policy_logits)
            if isinstance(val_out, tuple):
                val_logits = val_out[1] if len(val_out) >= 2 else val_out[0]
            else:
                val_logits = val_out
            # val_logits: (B, T, 1) -> squeeze last dim -> (B, T)
            if val_logits.dim() == 3 and val_logits.shape[-1] == 1:
                values = val_logits.squeeze(-1)
            else:
                values = val_logits.mean(dim=-1)

        # Compute GAE advantages per sample
        all_advantages = []
        all_returns = []
        for b in range(B):
            # Scalar reward broadcast to last token only
            seq_rewards = torch.zeros(T, device=prompt_ids.device)
            seq_rewards[-1] = rewards_tensor[b]

            dones = torch.zeros(T, device=prompt_ids.device)
            dones[-1] = 1.0

            adv, ret = compute_gae_advantages(seq_rewards, values[b], dones, cfg)
            all_advantages.append(adv.unsqueeze(0))
            all_returns.append(ret.unsqueeze(0))

        advantages = torch.cat(all_advantages, dim=0)  # (B, T)
        returns = torch.cat(all_returns, dim=0)         # (B, T)

        if cfg.normalize_advantages:
            mean = advantages.mean()
            std = advantages.std() + 1e-8
            advantages = (advantages - mean) / std

        return Rollout(
            input_ids=input_ids,
            log_probs=log_probs,
            values=values,
            rewards=rewards_tensor,
            advantages=advantages,
            returns=returns,
        )

    def ppo_step(self, rollout: Rollout) -> dict[str, float]:
        """Perform PPO update epochs over collected rollout data.

        Args:
            rollout: Rollout collected by collect_rollout

        Returns:
            Dict with keys "policy_loss", "value_loss", "entropy", "kl"
        """
        cfg = self.config
        B, T = rollout.input_ids.shape

        # Flatten for minibatch processing
        flat_input_ids = rollout.input_ids.reshape(B * T)
        old_log_probs = rollout.log_probs.reshape(B * T).detach()
        old_values = rollout.values.reshape(B * T).detach()
        advantages = rollout.advantages.reshape(B * T).detach()
        returns = rollout.returns.reshape(B * T).detach()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_updates = 0

        self.policy.train()
        self.value_fn.train()

        N = B * T
        indices = torch.arange(N)

        for _ in range(cfg.n_epochs):
            perm = torch.randperm(N)
            shuffled = indices[perm]

            for start in range(0, N, cfg.minibatch_size):
                mb_idx = shuffled[start: start + cfg.minibatch_size]
                if len(mb_idx) == 0:
                    continue

                mb_input_ids = flat_input_ids[mb_idx].unsqueeze(0)  # (1, mb_size)
                mb_old_lp = old_log_probs[mb_idx]
                mb_old_val = old_values[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]

                # Policy forward
                _, logits, _ = self.policy(mb_input_ids)
                # logits: (1, mb_size, vocab_size)
                log_probs_all = F.log_softmax(logits, dim=-1)

                mb_tokens = flat_input_ids[mb_idx]  # (mb_size,)
                mb_lp = log_probs_all[0].gather(
                    1, mb_tokens.unsqueeze(-1)
                ).squeeze(-1)  # (mb_size,)

                p_loss = ppo_policy_loss(mb_lp, mb_old_lp, mb_adv, cfg.clip_epsilon)
                entropy = compute_entropy_bonus(mb_lp)

                # Value forward: logits already computed above, apply value_fn linear head
                # value_fn is nn.Linear(vocab_size, 1); logits: (1, mb_size, vocab_size)
                val_out = self.value_fn(logits)
                if isinstance(val_out, tuple):
                    val_logits = val_out[1] if len(val_out) >= 2 else val_out[0]
                else:
                    val_logits = val_out
                # val_logits: (1, mb_size, 1) -> squeeze -> (mb_size,)
                if val_logits.dim() == 3 and val_logits.shape[-1] == 1:
                    curr_values = val_logits.squeeze(-1).squeeze(0)  # (mb_size,)
                else:
                    curr_values = val_logits.mean(dim=-1).squeeze(0)  # (mb_size,)

                v_loss = ppo_value_loss(curr_values, mb_old_val, mb_ret, cfg.value_clip)

                total_loss = p_loss + cfg.value_coeff * v_loss - cfg.entropy_coeff * entropy

                with torch.no_grad():
                    kl = (mb_old_lp - mb_lp).mean()

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_fn.parameters(), cfg.max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()

                total_policy_loss += p_loss.item()
                total_value_loss += v_loss.item()
                total_entropy += entropy.item()
                total_kl += kl.item()
                n_updates += 1

        if n_updates == 0:
            n_updates = 1

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "kl": total_kl / n_updates,
        }

    def train_step(self, prompt_ids: Tensor) -> dict[str, float]:
        """Perform a full PPO train step: collect rollout + PPO update.

        Args:
            prompt_ids: (B, prompt_len) prompt token ids

        Returns:
            Combined metrics dict from ppo_step
        """
        rollout = self.collect_rollout(prompt_ids, max_new_tokens=8)
        metrics = self.ppo_step(rollout)
        return metrics
