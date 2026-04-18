from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ALConfig:
    strategy: str = "entropy"
    n_select: int = 10
    seed: int = 42


class ActiveLearner:
    def __init__(self, config: ALConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)

    def least_confidence(self, probs: Tensor) -> Tensor:
        max_probs, _ = probs.max(dim=-1)
        return 1.0 - max_probs

    def margin_score(self, probs: Tensor) -> Tensor:
        sorted_probs, _ = probs.sort(dim=-1, descending=True)
        return sorted_probs[:, 0] - sorted_probs[:, 1]

    def entropy_score(self, probs: Tensor) -> Tensor:
        return -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

    def coreset_score(
        self,
        embeddings: Tensor,
        selected_embeddings: Optional[Tensor],
    ) -> Tensor:
        if selected_embeddings is None or selected_embeddings.shape[0] == 0:
            return torch.ones(embeddings.shape[0], dtype=embeddings.dtype)

        # Pairwise squared L2 distances: (N, M)
        diffs = embeddings.unsqueeze(1) - selected_embeddings.unsqueeze(0)
        dists = (diffs ** 2).sum(dim=-1).sqrt()
        min_dists, _ = dists.min(dim=-1)
        return min_dists

    def select(
        self,
        probs: Tensor,
        embeddings: Optional[Tensor] = None,
        already_selected_mask: Optional[Tensor] = None,
    ) -> Tensor:
        strategy = self.config.strategy
        n = self.config.n_select

        if strategy == "least_confidence":
            scores = self.least_confidence(probs)
        elif strategy == "margin":
            scores = self.margin_score(probs)
        elif strategy == "coreset":
            if embeddings is None:
                raise ValueError("embeddings required for coreset strategy")
            if already_selected_mask is not None:
                sel_emb = embeddings[already_selected_mask]
            else:
                sel_emb = torch.zeros(0, embeddings.shape[-1], dtype=embeddings.dtype)
            scores = self.coreset_score(embeddings, sel_emb)
        else:
            scores = self.entropy_score(probs)

        if already_selected_mask is not None:
            scores = scores.clone()
            scores[already_selected_mask] = -float("inf")

        _, indices = scores.topk(n, largest=True, sorted=True)
        return indices.long()

    def update_pool(self, pool_ids: List[int], selected_ids: List[int]) -> List[int]:
        selected_set = set(selected_ids)
        return [pid for pid in pool_ids if pid not in selected_set]
