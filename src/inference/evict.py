"""EVICT: Adaptive Verification for MoE Speculative Decoding.

Truncates draft tree before verification, retaining only cost-effective prefix.
Uses drafter signals to estimate candidate benefit combined with
offline-profiled verification cost. Training-free, lossless.

Paper: arXiv:2605.00342 — Pan et al.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class DraftCandidate(NamedTuple):
    token_ids: Tensor
    log_probs: Tensor
    draft_score: float
    branch_id: int


class ExpertActivationProfile:
    """Offline-profiled per-token expert activation cost."""

    def __init__(self, vocab_size: int, n_experts: int) -> None:
        self.prob = torch.zeros(vocab_size, n_experts)

    def profile_token(self, token_id: int) -> float:
        return self.prob[token_id].sum().item()

    def estimate_verification_cost(self, token_ids: Tensor) -> Tensor:
        return torch.tensor([self.profile_token(t) for t in token_ids])


class EVICTVerifier:
    """Adaptive verification truncation for MoE speculative decoding.

    Key idea: truncate draft tree before target verification, keeping
    only the cost-effective prefix where candidate benefit > verification cost.
    """

    def __init__(self, expert_profile: ExpertActivationProfile,
                 draft_score_threshold: float = 0.1) -> None:
        self.expert_profile = expert_profile
        self.threshold = draft_score_threshold

    def plan_verification(self, candidates: list[DraftCandidate],
                         max_verification_tokens: int) -> list[int]:
        """Select which draft tokens to actually verify.

        Returns list of token indices to keep (prefix of draft sequence).
        Uses:
          benefit = cumulative draft_score
          cost = cumulative expert activation count
        Truncates when marginal_benefit / marginal_cost drops below threshold.
        """
        if not candidates:
            return []

        selected = []
        cum_benefit = 0.0
        cum_cost = 0.0

        for cand in candidates:
            token_cost = self.expert_profile.profile_token(cand.token_ids.item())
            marginal_benefit = cand.draft_score
            marginal_cost = token_cost

            if marginal_cost > 0:
                efficiency = marginal_benefit / marginal_cost
                if efficiency >= self.threshold and len(selected) < max_verification_tokens:
                    selected.append(cand.branch_id)
                    cum_benefit += marginal_benefit
                    cum_cost += marginal_cost

        return selected

    def truncate_draft_tree(self, draft_tree: list[DraftCandidate]) -> list[DraftCandidate]:
        """Truncate draft tree to cost-effective prefix only."""
        verified = self.plan_verification(draft_tree, max_verification_tokens=64)
        return [c for c in draft_tree if c.branch_id in verified]


def evict_speedup_ratio(
    original_time: float,
    evict_time: float,
    original_tokens: int,
    evict_tokens: int,
) -> float:
    """Compute effective speedup from EVICT truncation."""
    return (original_time / evict_time) * (evict_tokens / original_tokens)


class EVICTScheduler:
    """SGLang-compatible adaptive scheduling with EVICT truncation."""

    def __init__(self, verifier: EVICTVerifier) -> None:
        self.verifier = verifier

    def schedule_batch(self, batch_drafts: list[list[DraftCandidate]],
                       max_tokens: int) -> list[list[DraftCandidate]]:
        """Schedule verification for a batch of draft trees."""
        scheduled = []
        budget = max_tokens // len(batch_drafts) if batch_drafts else 0
        for drafts in batch_drafts:
            truncated = self.verifier.truncate_draft_tree(drafts)
            scheduled.append(truncated[:budget])
        return scheduled


__all__ = ["EVICTVerifier", "EVICTScheduler", "DraftCandidate", "ExpertActivationProfile", "evict_speedup_ratio"]