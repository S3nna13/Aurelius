"""EVICT: Adaptive Verification for MoE Speculative Decoding.

Training-free, hyperparameter-free method that truncates draft tree before verification
and retains only cost-effective prefixes. Uses drafter signals to estimate candidate benefit
and combines with offline-profiled verification cost.

Paper: arXiv:2605.00342
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor


class DraftCandidate(NamedTuple):
    tokens: Tensor
    log_probs: Tensor
    seq_len: int
    benefit_score: float = 0.0


class EVICTVerifier:
    """Adaptive tree truncation for MoE speculative decoding."""

    def __init__(
        self,
        expert_cost_profile: dict[int, float] | None = None,
        min_prefix_len: int = 2,
    ) -> None:
        self.expert_cost_profile = expert_cost_profile or {}
        self.min_prefix_len = min_prefix_len

    def estimate_expert_activation_cost(self, hidden: Tensor, gate_logits: Tensor) -> float:
        if not self.expert_cost_profile:
            return 1.0
        topk = F.softmax(gate_logits, dim=-1).topk(2, dim=-1)
        cost = sum(self.expert_cost_profile.get(int(k), 1.0) for k in topk.indices[0])
        return cost / 2.0

    def score_candidate(self, candidate: DraftCandidate, draft_logits: Tensor) -> float:
        entropy = -(F.softmax(draft_logits, dim=-1) * F.log_softmax(draft_logits, dim=-1)).sum(-1)
        confidence = draft_logits.max(-1)[0]
        depth_bonus = candidate.seq_len / 10.0
        return (confidence.mean() - 0.5 * entropy.mean() + 0.1 * depth_bonus).item()

    def truncate_tree(
        self,
        candidates: list[DraftCandidate],
        draft_logits: list[Tensor],
        target_budget: int,
    ) -> list[DraftCandidate]:
        benefit_scores = [
            self.score_candidate(cand, draft_logits[i]) for i, cand in enumerate(candidates)
        ]

        sorted_candidates = sorted(
            [(c, score, c.seq_len) for c, score in zip(candidates, benefit_scores)],
            key=lambda x: x[1] / max(x[2], 1),
            reverse=True,
        )

        kept = []
        total_len = 0
        for cand, score, seq_len in sorted_candidates:
            if total_len + seq_len <= target_budget and len(kept) < len(candidates):
                kept.append(cand)
                total_len += seq_len

        if len(kept) < self.min_prefix_len:
            kept = candidates[: self.min_prefix_len]

        return kept

    def verify_batch(
        self,
        kept_candidates: list[DraftCandidate],
        target_logits: Tensor,
    ) -> tuple[Tensor, list[bool]]:
        if not kept_candidates:
            return torch.empty(0, dtype=torch.long, device=target_logits.device), []

        accepted = []
        for cand in kept_candidates:
            draft_logp = cand.log_probs[-1]
            target_logp = target_logits[cand.seq_len - 1]
            accepted.append((target_logp - draft_logp).exp().item() > 0.5)
        accepted_tokens = torch.stack([c.tokens for c in kept_candidates])
        return accepted_tokens, accepted


__all__ = ["EVICTVerifier", "DraftCandidate"]
