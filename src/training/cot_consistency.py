"""Chain-of-Thought Consistency training.

Trains models to produce consistent intermediate reasoning steps that converge
to the same answer across multiple sampled chains (Self-Consistency / STaR).

References:
    Wang et al. 2022 - Self-Consistency Improves CoT Reasoning
    Zelikman et al. 2022 - STaR: Bootstrapping Reasoning with Reasoning
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# ChainOfThoughtSampler
# ---------------------------------------------------------------------------


class ChainOfThoughtSampler:
    """Sample multiple reasoning chains autoregressively with temperature."""

    def __init__(
        self,
        model: nn.Module,
        n_chains: int = 5,
        temperature: float = 0.8,
    ) -> None:
        self.model = model
        self.n_chains = n_chains
        self.temperature = temperature

    def sample_chains(
        self,
        input_ids: Tensor,
        max_reasoning_tokens: int = 32,
        max_answer_tokens: int = 8,
    ) -> tuple[list[Tensor], list[float]]:
        """Sample n_chains sequences autoregressively with temperature scaling.

        Args:
            input_ids: (seq_len,) prompt token ids (1-D).
            max_reasoning_tokens: max tokens to generate for reasoning.
            max_answer_tokens: max tokens to generate for answer.

        Returns:
            chains: list of n_chains 1-D tensors (generated tokens only).
            log_probs: list of n_chains per-chain sum log-prob floats.
        """
        device = input_ids.device
        max_new = max_reasoning_tokens + max_answer_tokens
        chains: list[Tensor] = []
        log_probs: list[float] = []

        self.model.train(False)
        with torch.no_grad():
            for _ in range(self.n_chains):
                generated: list[int] = []
                chain_log_prob: float = 0.0
                context = input_ids.clone()

                for _ in range(max_new):
                    logits = self.model(context.unsqueeze(0))  # (1, T, V)
                    # Support both (1, T, V) and (1, V) shaped outputs
                    if logits.dim() == 3:
                        next_logits = logits[0, -1, :]  # (V,)
                    else:
                        next_logits = logits[0]  # (V,)

                    # Temperature-scaled sampling
                    scaled = next_logits / max(self.temperature, 1e-8)
                    probs = F.softmax(scaled, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # (1,)
                    token_id = int(next_token[0])

                    log_p = float(F.log_softmax(scaled, dim=-1)[token_id])
                    chain_log_prob += log_p
                    generated.append(token_id)
                    context = torch.cat([context, next_token], dim=0)

                chains.append(torch.tensor(generated, dtype=torch.long, device=device))
                log_probs.append(chain_log_prob)

        return chains, log_probs


# ---------------------------------------------------------------------------
# AnswerExtractor
# ---------------------------------------------------------------------------


class AnswerExtractor:
    """Extract final answer tokens from a reasoning chain."""

    def __init__(self, answer_separator_id: int = 2) -> None:
        self.separator_id = answer_separator_id

    def extract(self, chain: Tensor) -> tuple[Tensor, Tensor]:
        """Split chain at the first occurrence of separator_id.

        If separator found at position i:
            reasoning = chain[:i]
            answer    = chain[i+1:]
        If separator not found:
            reasoning = chain[:-1]
            answer    = chain[-1:]

        Args:
            chain: 1-D tensor of token ids.

        Returns:
            (reasoning, answer) — both 1-D tensors.
        """
        if chain.numel() == 0:
            empty = torch.tensor([], dtype=torch.long, device=chain.device)
            return empty, empty

        sep_mask = chain == self.separator_id
        sep_positions = sep_mask.nonzero(as_tuple=False)
        if sep_positions.numel() > 0:
            idx = int(sep_positions[0])
            reasoning = chain[:idx]
            answer = chain[idx + 1:]
            # Guard against empty answer when separator is the last token
            if answer.numel() == 0:
                answer = chain[-1:]
                reasoning = chain[:-1]
        else:
            reasoning = chain[:-1]
            answer = chain[-1:]

        return reasoning, answer

    def majority_vote(self, answers: list[Tensor]) -> Tensor:
        """Return the most frequently occurring answer sequence.

        Args:
            answers: list of 1-D tensors (possibly different lengths).

        Returns:
            The most common answer tensor. Ties broken by first occurrence.
        """
        if not answers:
            raise ValueError("answers list must be non-empty")

        counts: Counter = Counter()
        first_seen: dict[tuple, Tensor] = {}
        for ans in answers:
            key = tuple(ans.tolist())
            counts[key] += 1
            if key not in first_seen:
                first_seen[key] = ans

        most_common_key = counts.most_common(1)[0][0]
        return first_seen[most_common_key]


# ---------------------------------------------------------------------------
# ConsistencyReward
# ---------------------------------------------------------------------------


class ConsistencyReward:
    """Reward chains that agree with the majority answer."""

    def __init__(
        self,
        consistency_weight: float = 1.0,
        length_penalty: float = 0.01,
    ) -> None:
        self.consistency_weight = consistency_weight
        self.length_penalty = length_penalty

    def compute(
        self,
        chains: list[Tensor],
        log_probs: list[float],
    ) -> list[float]:
        """Compute per-chain rewards based on agreement with the majority answer.

        reward_i = consistency_weight * (1.0 if answer_i == majority else 0.0)
                   - length_penalty * len(chain_i)

        If all chains agree: all rewards set to 1.0 (no discriminative signal).

        Args:
            chains: list of n_chains 1-D generated token tensors.
            log_probs: list of n_chains per-chain sum log-probs (unused here).

        Returns:
            rewards: list of n_chains float rewards.
        """
        extractor = AnswerExtractor()
        answers = [extractor.extract(c)[1] for c in chains]
        majority = extractor.majority_vote(answers)
        majority_key = tuple(majority.tolist())

        # When all chains agree there is no gradient signal needed
        all_agree = all(tuple(a.tolist()) == majority_key for a in answers)
        if all_agree:
            return [1.0] * len(chains)

        rewards: list[float] = []
        for chain, ans in zip(chains, answers):
            match = 1.0 if tuple(ans.tolist()) == majority_key else 0.0
            reward = self.consistency_weight * match - self.length_penalty * len(chain)
            # Clamp to [0, 1] so rewards remain interpretable as probabilities
            reward = max(0.0, min(1.0, reward))
            rewards.append(reward)

        return rewards


# ---------------------------------------------------------------------------
# CoTConsistencyLoss
# ---------------------------------------------------------------------------


class CoTConsistencyLoss(nn.Module):
    """REINFORCE-based loss for consistency training.

    loss = -mean(log_probs * advantages)
    advantages = rewards - baseline_value
    """

    def __init__(self, baseline: str = "mean") -> None:
        super().__init__()
        if baseline not in ("mean", "none"):
            raise ValueError(f"baseline must be 'mean' or 'none', got {baseline!r}")
        self.baseline = baseline

    def forward(self, log_probs: Tensor, rewards: Tensor) -> Tensor:
        """Compute REINFORCE loss.

        Args:
            log_probs: (N,) per-chain sum log-probs (differentiable).
            rewards:   (N,) per-chain rewards.

        Returns:
            Scalar loss tensor.
        """
        if self.baseline == "mean":
            baseline_val = rewards.mean()
        else:
            baseline_val = torch.zeros(1, device=rewards.device, dtype=rewards.dtype)

        advantages = (rewards - baseline_val).detach()
        loss = -(log_probs * advantages).mean()
        return loss


# ---------------------------------------------------------------------------
# STaRTrainer
# ---------------------------------------------------------------------------


class STaRTrainer:
    """Self-Taught Reasoner (Zelikman et al. 2022) style trainer.

    Alternates between:
    - rationalize_step: supervised training on chains that produced correct answers.
    - consistency_step: REINFORCE training to maximise self-consistency.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        n_chains: int = 4,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.n_chains = n_chains
        self.sampler = ChainOfThoughtSampler(model, n_chains=n_chains)
        self.extractor = AnswerExtractor()
        self.consistency_loss_fn = CoTConsistencyLoss(baseline="mean")

    def rationalize_step(
        self,
        input_ids: Tensor,
        correct_answer_ids: Tensor,
    ) -> dict:
        """Sample chains, keep those with correct answers, train on them via CE.

        Args:
            input_ids: (seq_len,) prompt token ids.
            correct_answer_ids: (answer_len,) ground-truth answer token ids.

        Returns:
            dict: n_correct (int), n_total (int), loss (scalar tensor or int 0).
        """
        chains, _ = self.sampler.sample_chains(input_ids)

        correct_key = tuple(correct_answer_ids.tolist())
        correct_chains: list[Tensor] = []
        for chain in chains:
            _, ans = self.extractor.extract(chain)
            if tuple(ans.tolist()) == correct_key:
                correct_chains.append(chain)

        n_correct = len(correct_chains)
        n_total = len(chains)

        if n_correct == 0:
            return {"n_correct": 0, "n_total": n_total, "loss": 0}

        # Supervised CE on (prompt + chain) next-token prediction
        self.model.train()
        total_loss = torch.zeros(1, device=input_ids.device)

        for chain in correct_chains:
            seq = torch.cat([input_ids, chain], dim=0)  # (T,)
            if seq.numel() < 2:
                continue

            inp = seq[:-1].unsqueeze(0)   # (1, T-1)
            tgt = seq[1:]                  # (T-1,)

            logits = self.model(inp)       # (1, T-1, V)
            if logits.dim() == 3:
                logits = logits[0]         # (T-1, V)

            loss = F.cross_entropy(logits, tgt)
            total_loss = total_loss + loss

        avg_loss = total_loss / n_correct
        self.optimizer.zero_grad()
        avg_loss.backward()
        self.optimizer.step()

        return {"n_correct": n_correct, "n_total": n_total, "loss": avg_loss}

    def consistency_step(self, input_ids: Tensor) -> dict:
        """Sample chains and apply REINFORCE with consistency reward.

        Args:
            input_ids: (seq_len,) prompt token ids.

        Returns:
            dict: consistency_loss (scalar tensor), mean_reward (float),
                  n_agreeing (int).
        """
        device = input_ids.device

        # Sample chains without gradients first (for reward computation)
        chains, _ = self.sampler.sample_chains(input_ids)

        # Re-compute log-probs with gradients for the policy gradient update
        self.model.train()
        diff_log_probs: list[Tensor] = []
        for chain in chains:
            seq = torch.cat([input_ids, chain], dim=0)
            if seq.numel() < 2:
                diff_log_probs.append(torch.zeros(1, device=device).squeeze())
                continue

            inp = seq[:-1].unsqueeze(0)  # (1, T-1)
            tgt = seq[1:]                # (T-1,)

            logits = self.model(inp)     # (1, T-1, V)
            if logits.dim() == 3:
                logits = logits[0]       # (T-1, V)

            log_p_seq = F.log_softmax(logits, dim=-1)

            # Isolate log-probs for generated chain tokens only
            n_prompt = input_ids.numel()
            chain_offset = n_prompt - 1   # tgt starts at seq[1]
            chain_log_p = log_p_seq[chain_offset:, :]  # (chain_len, V)
            chain_tgt = tgt[chain_offset:]              # (chain_len,)

            if chain_tgt.numel() == 0:
                diff_log_probs.append(torch.zeros(1, device=device).squeeze())
                continue

            selected = chain_log_p.gather(1, chain_tgt.unsqueeze(1)).squeeze(1)
            diff_log_probs.append(selected.sum())

        log_probs_tensor = torch.stack(diff_log_probs)  # (N,)

        # Rewards are computed without gradients
        reward_fn = ConsistencyReward()
        rewards_list = reward_fn.compute(chains, [float(lp.detach()) for lp in diff_log_probs])
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float, device=device)

        loss = self.consistency_loss_fn(log_probs_tensor, rewards_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        answers = [self.extractor.extract(c)[1] for c in chains]
        majority = self.extractor.majority_vote(answers)
        majority_key = tuple(majority.tolist())
        n_agreeing = sum(1 for a in answers if tuple(a.tolist()) == majority_key)
        mean_reward = float(rewards_tensor.mean())

        return {
            "consistency_loss": loss,
            "mean_reward": mean_reward,
            "n_agreeing": n_agreeing,
        }
