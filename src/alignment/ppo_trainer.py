"""PPO trainer for RLHF: clip-ratio policy gradient with value function baseline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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

    clip_ratio: float = 0.2  # epsilon for clipping ratio
    vf_coeff: float = 0.5  # value function loss coefficient
    entropy_coeff: float = 0.01  # entropy bonus coefficient
    n_epochs: int = 4  # PPO epochs per rollout
    n_rollout_steps: int = 8  # tokens to sample per prompt
    gamma: float = 1.0  # discount factor
    gae_lambda: float = 0.95  # GAE lambda
    temperature: float = 1.0
    max_grad_norm: float = 1.0


# ---------------------------------------------------------------------------
# Value head
# ---------------------------------------------------------------------------


class ValueHead(nn.Module):
    """Scalar value function head on top of backbone hidden states."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1, bias=True)

    def forward(self, hidden: Tensor) -> Tensor:
        """hidden (B, T, d_model) -> values (B, T)"""
        return self.linear(hidden).squeeze(-1)


# ---------------------------------------------------------------------------
# GAE advantages
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: Tensor,  # (T,)
    values: Tensor,  # (T,)
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute GAE advantages and returns.

    Returns (advantages (T,), returns (T,)).
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T, device=rewards.device, dtype=rewards.dtype)

    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = 0.0
        else:
            next_val = values[t + 1].item()

        delta = rewards[t].item() + gamma * next_val - values[t].item()
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO losses
# ---------------------------------------------------------------------------


def ppo_policy_loss(
    log_probs: Tensor,  # (B, T) — current policy log probs
    old_log_probs: Tensor,  # (B, T) — old policy log probs (from rollout)
    advantages: Tensor,  # (B, T)
    clip_ratio: float,
) -> tuple[Tensor, dict]:
    """Clipped PPO policy loss.

    ratio = exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = clamp(ratio, 1-clip_ratio, 1+clip_ratio) * advantages
    loss = -mean(min(surr1, surr2))

    Returns (loss, metrics) where metrics has:
        'policy_loss': float
        'clip_fraction': float — fraction of ratios clipped
        'mean_ratio': float
    """
    ratio = (log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    loss = -torch.min(surr1, surr2).mean()

    # Metrics
    clipped = (ratio < (1.0 - clip_ratio)) | (ratio > (1.0 + clip_ratio))
    clip_fraction = clipped.float().mean().item()
    mean_ratio = ratio.mean().item()

    metrics = {
        "policy_loss": loss.item(),
        "clip_fraction": clip_fraction,
        "mean_ratio": mean_ratio,
    }
    return loss, metrics


def ppo_value_loss(values: Tensor, returns: Tensor) -> Tensor:
    """MSE value function loss. Returns scalar."""
    return F.mse_loss(values, returns)


def entropy_bonus(log_probs: Tensor) -> Tensor:
    """Entropy of categorical distribution. Returns scalar.

    H = -mean(sum_a pi(a) log pi(a)) ~= -mean(log_probs) for sampled actions.
    """
    return -log_probs.mean()


# ---------------------------------------------------------------------------
# PPOTrainer
# ---------------------------------------------------------------------------


class PPOTrainer:
    """RLHF PPO trainer."""

    def __init__(
        self,
        policy: nn.Module,
        ref_model: nn.Module,
        reward_fn: Callable[[Tensor], float],
        config: PPOConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.policy = policy
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.config = config
        self.optimizer = optimizer

        # Freeze reference model
        for p in ref_model.parameters():
            p.requires_grad_(False)

        # Value head — trained alongside policy
        self.value_head = ValueHead(policy.config.d_model)

        # Add value head params to optimizer
        optimizer.add_param_group({"params": list(self.value_head.parameters())})

        # Storage for hidden states captured via forward hook
        self._hidden_states: list[Tensor] = []
        self._hook_handle = None

    def _register_hidden_hook(self) -> None:
        """Register a forward hook on the policy's final norm to capture hidden states."""
        if self._hook_handle is not None:
            return

        # Find the final norm layer (named 'norm')
        norm_module = getattr(self.policy, "norm", None)
        if norm_module is None:
            # Fallback: look for any RMSNorm or LayerNorm
            for name, module in self.policy.named_modules():
                if "norm" in name.lower() and "." not in name:
                    norm_module = module
                    break

        if norm_module is not None:

            def hook_fn(module, input, output):
                self._hidden_states.append(output.detach())

            self._hook_handle = norm_module.register_forward_hook(hook_fn)

    def _remove_hidden_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def _get_hidden_states_and_logits(
        self, input_ids: Tensor, past_key_values=None
    ) -> tuple[Tensor, Tensor, list]:
        """Run policy forward, capturing hidden states via hook.

        Returns (hidden_states, logits, present_key_values).
        """
        self._hidden_states.clear()
        self._register_hidden_hook()
        _, logits, pkv = self.policy(input_ids, past_key_values=past_key_values)
        if self._hidden_states:
            hidden = self._hidden_states[-1]
        else:
            # Fallback: use embedding-dim slice of logits (shouldn't happen normally)
            hidden = logits
        return hidden, logits, pkv

    def collect_rollout(self, prompt_ids: Tensor) -> dict:
        """Sample n_rollout_steps tokens from policy.

        Returns rollout dict with: 'tokens', 'log_probs', 'values', 'rewards'
        """
        cfg = self.config
        B = prompt_ids.shape[0]
        T = cfg.n_rollout_steps

        self.policy.eval()
        self.value_head.eval()

        all_tokens = []  # list of (B, 1) tensors
        all_log_probs = []  # list of (B,) tensors
        all_values = []  # list of (B,) tensors

        with torch.no_grad():
            cur_ids = prompt_ids
            past_key_values = None

            for step in range(T):
                hidden, logits, past_key_values = self._get_hidden_states_and_logits(
                    cur_ids, past_key_values
                )
                # logits: (B, seq_len, vocab_size) — use last token's logits
                last_logits = logits[:, -1, :]  # (B, vocab_size)
                last_hidden = hidden[:, -1:, :]  # (B, 1, d_model)

                # Scale by temperature
                scaled_logits = last_logits / cfg.temperature
                log_probs_all = F.log_softmax(scaled_logits, dim=-1)  # (B, vocab_size)

                # Sample next token
                probs = log_probs_all.exp()
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
                token_log_prob = log_probs_all.gather(1, next_token).squeeze(1)  # (B,)

                # Value estimate from last hidden state
                # value_head: (B, 1, d_model) -> (B, 1) -> (B,)
                val = self.value_head(last_hidden).squeeze(1)  # (B,)

                all_tokens.append(next_token)
                all_log_probs.append(token_log_prob)
                all_values.append(val)

                cur_ids = next_token  # next step input is just the new token

        # Stack: (B, T)
        tokens = torch.cat(all_tokens, dim=1)  # (B, T)
        log_probs = torch.stack(all_log_probs, dim=1)  # (B, T)
        values = torch.stack(all_values, dim=1)  # (B, T)

        # Compute rewards for each sequence
        rewards_list = []
        for b in range(B):
            r = self.reward_fn(tokens[b])  # pass (T,) tensor
            rewards_list.append(float(r))
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=prompt_ids.device)  # (B,)

        return {
            "tokens": tokens,
            "log_probs": log_probs,
            "values": values,
            "rewards": rewards,
        }

    def ppo_update(self, rollout: dict) -> dict:
        """Run n_epochs of PPO updates on the rollout.

        Returns dict with: 'policy_loss', 'value_loss', 'entropy', 'total_loss'
        """
        cfg = self.config
        tokens = rollout["tokens"]  # (B, T)
        old_log_probs = rollout["log_probs"].detach()  # (B, T)
        old_values = rollout["values"].detach()  # (B, T)
        rewards = rollout["rewards"]  # (B,)

        B, T = tokens.shape

        # Compute GAE per sequence
        all_advantages = []
        all_returns = []
        for b in range(B):
            # Broadcast scalar reward to last timestep only
            seq_rewards = torch.zeros(T, device=rewards.device)
            seq_rewards[-1] = rewards[b]

            adv, ret = compute_gae(seq_rewards, old_values[b], cfg.gamma, cfg.gae_lambda)
            all_advantages.append(adv)
            all_returns.append(ret)

        advantages = torch.stack(all_advantages, dim=0)  # (B, T)
        returns = torch.stack(all_returns, dim=0)  # (B, T)

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_total_loss = 0.0

        self.policy.train()
        self.value_head.train()

        for _ in range(cfg.n_epochs):
            # Forward pass: run policy over generated tokens to get current log probs + values
            self._hidden_states.clear()
            self._register_hidden_hook()
            _, logits, _ = self.policy(tokens)
            hidden = self._hidden_states[-1] if self._hidden_states else logits

            # logits: (B, T, vocab_size)
            log_probs_all = F.log_softmax(logits, dim=-1)  # (B, T, vocab_size)

            # Gather log probs for the sampled tokens
            curr_log_probs = log_probs_all.gather(2, tokens.unsqueeze(-1)).squeeze(-1)  # (B, T)

            # Value head over all hidden states
            curr_values = self.value_head(hidden)  # (B, T)

            p_loss, _ = ppo_policy_loss(curr_log_probs, old_log_probs, advantages, cfg.clip_ratio)
            v_loss = ppo_value_loss(curr_values, returns)
            ent = entropy_bonus(curr_log_probs)

            total_loss = p_loss + cfg.vf_coeff * v_loss - cfg.entropy_coeff * ent

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value_head.parameters()),
                cfg.max_grad_norm,
            )
            self.optimizer.step()

            total_policy_loss += p_loss.item()
            total_value_loss += v_loss.item()
            total_entropy += ent.item()
            total_total_loss += total_loss.item()

        n = cfg.n_epochs
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "total_loss": total_total_loss / n,
        }

    def train_step(self, prompt_ids: Tensor) -> dict:
        """collect_rollout + ppo_update.

        Returns combined metrics dict.
        """
        rollout = self.collect_rollout(prompt_ids)
        metrics = self.ppo_update(rollout)
        metrics["mean_reward"] = rollout["rewards"].mean().item()
        return metrics
