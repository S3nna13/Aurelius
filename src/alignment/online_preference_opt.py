"""Online Preference Optimization (OPO).

Combines RLHF online data collection with DPO-style updates. Based on:
- Guo et al. 2024 "Direct Language Model Alignment from Online AI Feedback"
- Yuan et al. 2024 self-play preference optimization

Algorithm:
  For each prompt:
    1. Sample n_samples_per_prompt completions from the current policy.
    2. Score each response with a reward model.
    3. Create preference pairs (best vs worst response).
    4. Apply DPO loss + SFT regularization on chosen responses.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OPOConfig:
    """Configuration for Online Preference Optimization."""

    beta: float = 0.1               # DPO temperature (KL penalty coefficient)
    lr: float = 1e-5
    batch_size: int = 4
    n_steps: int = 4                # optimization steps per batch
    max_new_tokens: int = 16        # tokens to generate for self-play
    temperature: float = 1.0        # generation temperature
    top_p: float = 0.9
    reward_model_weight: float = 1.0
    sft_weight: float = 0.1         # SFT regularization coefficient
    max_seq_len: int = 64


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OnlinePair:
    """An online-collected preference pair."""

    prompt_ids: Tensor        # (T_p,)
    chosen_ids: Tensor        # (T_r,) higher-reward response
    rejected_ids: Tensor      # (T_r,) lower-reward response
    chosen_reward: float
    rejected_reward: float


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_response(
    model: nn.Module,
    prompt_ids: Tensor,      # (T_p,)
    max_new_tokens: int = 16,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> Tensor:
    """Autoregressive generation with top-p (nucleus) sampling.

    Args:
        model: Language model. Forward returns (loss, logits, kv) or logits.
        prompt_ids: 1-D token id tensor of shape (T_p,).
        max_new_tokens: Number of new tokens to generate.
        temperature: Sampling temperature; 1.0 = unscaled, lower = sharper.
        top_p: Nucleus sampling cumulative probability threshold.

    Returns:
        Tensor of shape (max_new_tokens,) with generated token ids.
    """
    # Add batch dimension: (T_p,) -> (1, T_p)
    current = prompt_ids.unsqueeze(0)
    generated: List[Tensor] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(current)
            logits = out[1] if isinstance(out, tuple) else out  # (1, seq, vocab)
            next_logits = logits[0, -1, :]  # (vocab,)

            if temperature == 0.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # (1,)
            else:
                scaled = next_logits / temperature
                probs = F.softmax(scaled, dim=-1)  # (vocab,)

                # Top-p filtering
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    # Remove tokens with cumulative probability above threshold
                    # Shift right so the first token above threshold is kept
                    sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
                    sorted_probs = sorted_probs.masked_fill(
                        sorted_indices_to_remove, 0.0
                    )
                    # Renormalize
                    sorted_probs = sorted_probs / (sorted_probs.sum() + 1e-10)
                    # Sample from filtered distribution
                    sample_idx = torch.multinomial(sorted_probs, num_samples=1)  # (1,)
                    next_token = sorted_indices[sample_idx]  # (1,)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)  # (1,)

            generated.append(next_token)
            current = torch.cat([current, next_token.unsqueeze(0)], dim=1)  # (1, seq+1)

    return torch.cat(generated, dim=0)  # (max_new_tokens,)


# ---------------------------------------------------------------------------
# Online data collection
# ---------------------------------------------------------------------------

def collect_online_pairs(
    model: nn.Module,
    reward_model: nn.Module,
    prompts: List[Tensor],   # list of (T_p,) prompt tensors
    config: OPOConfig,
    n_samples_per_prompt: int = 2,
) -> List[OnlinePair]:
    """For each prompt, generate n_samples responses, score with reward model,
    create preference pairs (best vs worst).

    Args:
        model: Policy model used for generation.
        reward_model: Reward model; forward(token_ids: (B, T)) -> (B,) rewards.
        prompts: List of 1-D prompt tensors.
        config: OPOConfig with generation parameters.
        n_samples_per_prompt: Number of responses to sample per prompt.

    Returns:
        List of OnlinePairs, one per prompt.
    """
    model.eval()
    reward_model.eval()
    pairs: List[OnlinePair] = []

    with torch.no_grad():
        for prompt_ids in prompts:
            responses: List[Tensor] = []
            rewards: List[float] = []

            for _ in range(n_samples_per_prompt):
                resp = generate_response(
                    model,
                    prompt_ids,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                )
                responses.append(resp)

                # Score with reward model: combine prompt + response
                full_ids = torch.cat([prompt_ids, resp], dim=0).unsqueeze(0)  # (1, T)
                reward_val = float(reward_model(full_ids)[0].item())
                rewards.append(reward_val)

            best_idx = int(max(range(len(rewards)), key=lambda i: rewards[i]))
            worst_idx = int(min(range(len(rewards)), key=lambda i: rewards[i]))

            pairs.append(
                OnlinePair(
                    prompt_ids=prompt_ids,
                    chosen_ids=responses[best_idx],
                    rejected_ids=responses[worst_idx],
                    chosen_reward=rewards[best_idx],
                    rejected_reward=rewards[worst_idx],
                )
            )

    return pairs


# ---------------------------------------------------------------------------
# Log-prob helpers
# ---------------------------------------------------------------------------

def _compute_sequence_log_probs(
    model: nn.Module,
    prompt_ids: Tensor,   # (1, T_p)
    response_ids: Tensor, # (1, T_r)
) -> Tensor:
    """Compute sum of log probs for response tokens given prompt.

    Returns scalar tensor per item in batch dimension (1,).
    """
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)  # (1, T_p + T_r)
    prompt_len = prompt_ids.shape[1]

    out = model(full_ids)
    logits = out[1] if isinstance(out, tuple) else out  # (1, seq, vocab)

    # Shift: logits[i] predicts token[i+1]
    log_probs_all = F.log_softmax(logits[:, :-1, :], dim=-1)  # (1, seq-1, vocab)

    # Response tokens start at prompt_len in full_ids.
    # In shifted view, position (prompt_len-1) predicts response token 0.
    resp_start = prompt_len - 1
    T_r = response_ids.shape[1]
    log_probs_resp = log_probs_all[:, resp_start:resp_start + T_r, :]  # (1, T_r, vocab)

    token_lp = log_probs_resp.gather(
        2, response_ids.unsqueeze(-1)
    ).squeeze(-1)  # (1, T_r)

    return token_lp.sum(dim=-1)  # (1,)


# ---------------------------------------------------------------------------
# OPO Loss
# ---------------------------------------------------------------------------

def opo_loss(
    model: nn.Module,
    ref_model: nn.Module,
    batch: List[OnlinePair],
    beta: float = 0.1,
    sft_weight: float = 0.1,
) -> tuple:
    """Online DPO loss with SFT regularization on chosen responses.

    L = L_DPO(chosen, rejected) + sft_weight * L_SFT(chosen)

    L_DPO = -logsigmoid(beta * ((log pi(y_c|x) - log ref(y_c|x)) -
                                 (log pi(y_r|x) - log ref(y_r|x))))
    L_SFT = -mean(log pi(y_c|x))  (negative log likelihood on chosen)

    Args:
        model: Trainable policy model.
        ref_model: Frozen reference model.
        batch: List of OnlinePairs.
        beta: DPO temperature.
        sft_weight: SFT regularization coefficient.

    Returns:
        (loss, metrics) where metrics has keys:
        'dpo_loss', 'sft_loss', 'total_loss', 'reward_margin'
    """
    dpo_losses: List[Tensor] = []
    sft_losses: List[Tensor] = []
    chosen_rewards_list: List[float] = []
    rejected_rewards_list: List[float] = []

    for pair in batch:
        prompt_b = pair.prompt_ids.unsqueeze(0)      # (1, T_p)
        chosen_b = pair.chosen_ids.unsqueeze(0)      # (1, T_r)
        rejected_b = pair.rejected_ids.unsqueeze(0)  # (1, T_r)

        # Policy log probs (with grad)
        pi_chosen_lp = _compute_sequence_log_probs(model, prompt_b, chosen_b)      # (1,)
        pi_rejected_lp = _compute_sequence_log_probs(model, prompt_b, rejected_b)  # (1,)

        # Reference log probs (no grad)
        with torch.no_grad():
            ref_chosen_lp = _compute_sequence_log_probs(
                ref_model, prompt_b, chosen_b
            )    # (1,)
            ref_rejected_lp = _compute_sequence_log_probs(
                ref_model, prompt_b, rejected_b
            )  # (1,)

        # DPO implicit rewards
        chosen_reward_t = beta * (pi_chosen_lp - ref_chosen_lp)
        rejected_reward_t = beta * (pi_rejected_lp - ref_rejected_lp)

        dpo_logit = chosen_reward_t - rejected_reward_t
        dpo_l = -F.logsigmoid(dpo_logit)
        dpo_losses.append(dpo_l)

        # SFT loss on chosen: negative log likelihood
        sft_l = -pi_chosen_lp
        sft_losses.append(sft_l)

        chosen_rewards_list.append(chosen_reward_t.item())
        rejected_rewards_list.append(rejected_reward_t.item())

    dpo_loss_val = torch.cat(dpo_losses).mean()
    sft_loss_val = torch.cat(sft_losses).mean()
    total_loss = dpo_loss_val + sft_weight * sft_loss_val

    mean_chosen_reward = sum(chosen_rewards_list) / len(chosen_rewards_list)
    mean_rejected_reward = sum(rejected_rewards_list) / len(rejected_rewards_list)
    reward_margin = mean_chosen_reward - mean_rejected_reward

    metrics = {
        "dpo_loss": dpo_loss_val.item(),
        "sft_loss": sft_loss_val.item(),
        "total_loss": total_loss.item(),
        "reward_margin": reward_margin,
    }

    return total_loss, metrics


# ---------------------------------------------------------------------------
# OPOTrainer
# ---------------------------------------------------------------------------

class OPOTrainer:
    """Online Preference Optimization Trainer.

    Combines online data collection (self-play via reward model) with
    DPO-style policy updates and SFT regularization.

    Args:
        model: Trainable policy model (AureliusTransformer).
        ref_model: Frozen reference model (deep copy of initial policy).
        reward_model: Reward model used to score generated responses.
        config: OPOConfig.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        config: OPOConfig,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.config = config

        # Freeze reference model and reward model
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        for p in self.reward_model.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.lr,
        )

    def collect_and_train(
        self,
        prompts: List[Tensor],
    ) -> dict:
        """Full OPO step: collect online pairs + DPO update.

        Steps:
          1. Collect online preference pairs via self-play + reward model.
          2. Run n_steps of DPO + SFT gradient updates.

        Args:
            prompts: List of 1-D prompt tensors.

        Returns:
            Metrics dict with 'dpo_loss', 'sft_loss', 'total_loss', 'reward_margin'.
        """
        # Step 1: Collect online pairs
        pairs = collect_online_pairs(
            model=self.model,
            reward_model=self.reward_model,
            prompts=prompts,
            config=self.config,
        )

        if not pairs:
            return {
                "dpo_loss": 0.0,
                "sft_loss": 0.0,
                "total_loss": 0.0,
                "reward_margin": 0.0,
            }

        # Step 2: Gradient updates
        self.model.train()
        self.ref_model.eval()

        last_metrics: dict = {}
        for _ in range(self.config.n_steps):
            self.optimizer.zero_grad()

            loss, metrics = opo_loss(
                model=self.model,
                ref_model=self.ref_model,
                batch=pairs,
                beta=self.config.beta,
                sft_weight=self.config.sft_weight,
            )

            loss.backward()
            self.optimizer.step()
            last_metrics = metrics

        return last_metrics
