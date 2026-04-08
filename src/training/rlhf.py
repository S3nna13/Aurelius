"""Online PPO-based RLHF training loop.

Pipeline per step:
1. Generate rollouts: model.generate() for each prompt in batch
2. Score rollouts: reward_model(prompt + response) -> scalar reward
3. Compute GAE advantages from rewards and value estimates
4. Update policy: ppo_loss with clipped surrogate objective
"""
from __future__ import annotations

from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from src.alignment.gae import compute_gae, normalize_advantages


@dataclass
class RLHFConfig:
    n_rollouts_per_prompt: int = 4     # candidates generated per prompt
    max_new_tokens: int = 64           # max response length
    temperature: float = 1.0           # generation temperature
    top_p: float = 0.9                 # nucleus sampling
    ppo_epochs: int = 4                # PPO gradient steps per rollout batch
    mini_batch_size: int = 8           # sequences per gradient step
    lr: float = 1e-5
    kl_coef: float = 0.1               # KL penalty coefficient
    gamma: float = 1.0                 # GAE gamma
    lam: float = 0.95                  # GAE lambda
    clip_eps: float = 0.2              # PPO clip epsilon
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0


@dataclass
class RLHFBatch:
    """One batch of rollout data."""
    prompt_ids: list[torch.Tensor]       # each (S_prompt,)
    response_ids: list[torch.Tensor]     # each (S_resp,)
    rewards: torch.Tensor                # (N,) scalar per sequence
    old_log_probs: torch.Tensor          # (N, S_resp) per-token log probs
    values: torch.Tensor                 # (N, S_resp) per-token value estimates
    advantages: torch.Tensor             # (N, S_resp) GAE advantages
    returns: torch.Tensor                # (N, S_resp) GAE returns


class RLHFTrainer:
    """Online PPO-based RLHF trainer.

    Args:
        policy: ValueHead-wrapped AureliusTransformer (has .backbone + .value_head)
        reward_model: RewardModel for scoring rollouts
        cfg: Training configuration
    """

    def __init__(
        self,
        policy: "ValueHead",        # ValueHead instance
        reward_model: nn.Module,    # RewardModel instance
        cfg: RLHFConfig | None = None,
    ) -> None:
        self.policy = policy
        self.reward_model = reward_model
        self.cfg = cfg or RLHFConfig()
        self.optimizer = AdamW(
            [p for p in policy.parameters() if p.requires_grad],
            lr=self.cfg.lr,
        )

    @torch.no_grad()
    def collect_rollouts(
        self,
        prompts: list[torch.Tensor],  # list of (S_prompt,) tensors
    ) -> RLHFBatch:
        """Generate responses, score with reward model, compute GAE."""
        self.policy.eval()
        self.reward_model.eval()

        device = next(self.policy.parameters()).device
        cfg = self.cfg

        all_prompt_ids: list[torch.Tensor] = []
        all_response_ids: list[torch.Tensor] = []
        all_rewards: list[torch.Tensor] = []
        all_log_probs: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []

        for prompt_ids in prompts:
            prompt_ids = prompt_ids.to(device)
            S_prompt = prompt_ids.shape[0]

            # Step 1: Generate n_rollouts_per_prompt responses
            batch_prompt = prompt_ids.unsqueeze(0).expand(cfg.n_rollouts_per_prompt, -1)
            generated = self.policy.backbone.generate(
                batch_prompt,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )
            # Extract response tokens: (n_rollouts, S_resp)
            responses = generated[:, S_prompt:]

            for i in range(cfg.n_rollouts_per_prompt):
                resp_ids = responses[i]  # (S_resp,)
                S_resp = resp_ids.shape[0]

                # Step 2: Score (prompt + response) with reward model
                full_ids = torch.cat([prompt_ids, resp_ids]).unsqueeze(0)  # (1, S_total)
                reward_scalar = self.reward_model(full_ids)  # (1,)

                # Step 3: Compute per-token log-probs from policy backbone
                _, logits, _ = self.policy.backbone(full_ids)  # (1, S_total, V)
                # Positions [S_prompt-1 .. S_prompt+S_resp-2] predict response tokens
                scoring_logits = logits[0, S_prompt - 1 : S_prompt + S_resp - 1, :]  # (S_resp, V)
                log_probs_all = F.log_softmax(scoring_logits, dim=-1)  # (S_resp, V)
                token_log_probs = log_probs_all.gather(
                    1, resp_ids.unsqueeze(1)
                ).squeeze(1)  # (S_resp,)

                # Step 4: Compute value estimates from ValueHead
                _, _, values_full = self.policy(full_ids)  # (1, S_total)
                # Value estimates at positions that predict response tokens
                token_values = values_full[0, S_prompt - 1 : S_prompt + S_resp - 1]  # (S_resp,)

                all_prompt_ids.append(prompt_ids.cpu())
                all_response_ids.append(resp_ids.cpu())
                all_rewards.append(reward_scalar.squeeze(0).cpu())
                all_log_probs.append(token_log_probs.cpu())
                all_values.append(token_values.cpu())

        # Stack rewards: (N,)
        rewards_tensor = torch.stack(all_rewards)  # (N,)

        # Step 5: Compute GAE per sequence
        all_advantages: list[torch.Tensor] = []
        all_returns_list: list[torch.Tensor] = []

        for seq_values, seq_reward in zip(all_values, all_rewards):
            S_resp = seq_values.shape[0]
            seq_rewards = seq_reward.expand(S_resp)  # broadcast scalar to (S_resp,)
            dones = torch.zeros(S_resp)
            dones[-1] = 1.0  # episode ends at last token

            advantages, returns = compute_gae(
                seq_rewards,
                seq_values,
                dones,
                gamma=cfg.gamma,
                lam=cfg.lam,
            )
            all_advantages.append(advantages)
            all_returns_list.append(returns)

        # Pad sequences to common length for batching
        max_resp_len = max(t.shape[0] for t in all_log_probs)

        def pad_to(tensors: list[torch.Tensor], length: int, value: float = 0.0) -> torch.Tensor:
            out = torch.full((len(tensors), length), value)
            for i, t in enumerate(tensors):
                out[i, : t.shape[0]] = t
            return out

        old_log_probs = pad_to(all_log_probs, max_resp_len, 0.0)          # (N, S_resp)
        values_tensor = pad_to(all_values, max_resp_len, 0.0)             # (N, S_resp)
        advantages_tensor = pad_to(all_advantages, max_resp_len, 0.0)    # (N, S_resp)
        returns_tensor = pad_to(all_returns_list, max_resp_len, 0.0)     # (N, S_resp)

        return RLHFBatch(
            prompt_ids=all_prompt_ids,
            response_ids=all_response_ids,
            rewards=rewards_tensor,
            old_log_probs=old_log_probs,
            values=values_tensor,
            advantages=advantages_tensor,
            returns=returns_tensor,
        )

    def train_step(self, batch: RLHFBatch) -> dict[str, float]:
        """Run PPO update on collected rollout batch.

        Returns dict of metrics: policy_loss, value_loss, entropy, kl.
        """
        self.policy.train()
        cfg = self.cfg
        device = next(self.policy.parameters()).device

        N = len(batch.prompt_ids)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_updates = 0

        for _epoch in range(cfg.ppo_epochs):
            # Shuffle indices each epoch
            perm = torch.randperm(N).tolist()

            for start in range(0, N, cfg.mini_batch_size):
                mb_indices = perm[start : start + cfg.mini_batch_size]
                if not mb_indices:
                    continue

                self.optimizer.zero_grad()

                mb_policy_loss = torch.tensor(0.0, device=device)
                mb_value_loss = torch.tensor(0.0, device=device)
                mb_entropy = torch.tensor(0.0, device=device)
                mb_kl = torch.tensor(0.0, device=device)
                mb_count = 0

                for idx in mb_indices:
                    prompt_ids = batch.prompt_ids[idx].to(device)
                    resp_ids = batch.response_ids[idx].to(device)
                    S_prompt = prompt_ids.shape[0]
                    S_resp = resp_ids.shape[0]

                    old_lp = batch.old_log_probs[idx, :S_resp].to(device)   # (S_resp,)
                    advantages = batch.advantages[idx, :S_resp].to(device)  # (S_resp,)
                    returns = batch.returns[idx, :S_resp].to(device)        # (S_resp,)

                    # Forward through ValueHead to get new logits and values
                    full_ids = torch.cat([prompt_ids, resp_ids]).unsqueeze(0)  # (1, S_total)
                    _, logits, new_values_full = self.policy(full_ids)          # logits (1, S, V), values (1, S)

                    scoring_logits = logits[0, S_prompt - 1 : S_prompt + S_resp - 1, :]  # (S_resp, V)
                    new_values = new_values_full[0, S_prompt - 1 : S_prompt + S_resp - 1]  # (S_resp,)

                    log_probs_all = F.log_softmax(scoring_logits, dim=-1)  # (S_resp, V)
                    new_lp = log_probs_all.gather(
                        1, resp_ids.unsqueeze(1)
                    ).squeeze(1)  # (S_resp,)

                    # PPO clipped surrogate objective
                    ratio = (new_lp - old_lp).exp()  # (S_resp,)
                    loss_unclipped = -ratio * advantages
                    loss_clipped = -ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
                    policy_loss = torch.max(loss_unclipped, loss_clipped).mean()

                    # KL penalty: old_log_probs - new_log_probs
                    kl = (old_lp - new_lp).mean()

                    # Value loss: MSE
                    value_loss = 0.5 * (new_values - returns).pow(2).mean()

                    # Entropy bonus
                    probs = log_probs_all.exp()
                    entropy = -(probs * log_probs_all).sum(dim=-1).mean()

                    # Total loss per sequence
                    seq_loss = (
                        policy_loss
                        + cfg.kl_coef * kl
                        + cfg.value_loss_coef * value_loss
                        - cfg.entropy_coef * entropy
                    )

                    mb_policy_loss = mb_policy_loss + policy_loss.detach()
                    mb_value_loss = mb_value_loss + value_loss.detach()
                    mb_entropy = mb_entropy + entropy.detach()
                    mb_kl = mb_kl + kl.detach()
                    mb_count += 1

                    seq_loss.backward()

                # Clip grad norm and step optimizer
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                if mb_count > 0:
                    total_policy_loss += (mb_policy_loss / mb_count).item()
                    total_value_loss += (mb_value_loss / mb_count).item()
                    total_entropy += (mb_entropy / mb_count).item()
                    total_kl += (mb_kl / mb_count).item()
                    n_updates += 1

        denom = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "entropy": total_entropy / denom,
            "kl": total_kl / denom,
        }

    def step(
        self,
        prompts: list[torch.Tensor],
    ) -> dict[str, float]:
        """collect_rollouts + train_step in one call."""
        batch = self.collect_rollouts(prompts)
        return self.train_step(batch)
