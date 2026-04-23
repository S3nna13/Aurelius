"""Attention flow analysis: rollout, head importance, circuit tracing."""

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor


@dataclass
class AttentionFlowConfig:
    n_layers: int
    n_heads: int
    add_residual: bool = True


class AttentionRollout:
    """Compute attention rollout across layers."""

    def __init__(self, config: AttentionFlowConfig):
        self.config = config

    def compute_rollout(self, attn_matrices: List[Tensor]) -> Tensor:
        """Compute attention rollout.

        Args:
            attn_matrices: list of (n_heads, seq_len, seq_len) tensors, one per layer.

        Returns:
            (seq_len, seq_len) rollout tensor.
        """
        # Mean over heads -> (seq_len, seq_len) per layer
        averaged = [a.mean(dim=0) for a in attn_matrices]  # list of (S, S)

        if self.config.add_residual:
            processed = []
            for a in averaged:
                seq_len = a.size(0)
                identity = torch.eye(seq_len, dtype=a.dtype, device=a.device)
                a = a + identity
                # Normalize rows
                row_sums = a.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                a = a / row_sums
                processed.append(a)
        else:
            processed = averaged

        # Matrix multiply across layers: rollout = A_L @ ... @ A_1
        rollout = processed[0]
        for a in processed[1:]:
            rollout = a @ rollout

        return rollout

    def token_relevance(self, rollout: Tensor, target_idx: int) -> Tensor:
        """Return the relevance scores for a target token.

        Args:
            rollout: (seq_len, seq_len) rollout tensor.
            target_idx: index of the target token.

        Returns:
            1D tensor of length seq_len.
        """
        return rollout[target_idx]


class HeadImportance:
    """Score attention heads by importance."""

    def __init__(self, n_layers: int, n_heads: int):
        self.n_layers = n_layers
        self.n_heads = n_heads

    def score_by_gradient(
        self, attn_weights: List[Tensor], gradients: List[Tensor]
    ) -> Tensor:
        """Compute importance scores via gradient * attention.

        Args:
            attn_weights: list of (n_heads, seq_len, seq_len) per layer.
            gradients: list of (n_heads, seq_len, seq_len) per layer.

        Returns:
            (n_layers, n_heads) importance scores.
        """
        scores = []
        for w, g in zip(attn_weights, gradients):
            # Element-wise multiply, mean over seq dims -> (n_heads,)
            head_scores = (w * g).abs().mean(dim=(-2, -1))
            scores.append(head_scores)
        return torch.stack(scores, dim=0)  # (n_layers, n_heads)

    def top_heads(
        self, scores: Tensor, k: int = 5
    ) -> List[Tuple[int, int, float]]:
        """Return top-k heads by importance score.

        Args:
            scores: (n_layers, n_heads) tensor.
            k: number of top heads to return.

        Returns:
            list of (layer_idx, head_idx, score) tuples sorted descending.
        """
        n_layers, n_heads = scores.shape
        flat = scores.reshape(-1)
        topk_vals, topk_indices = torch.topk(flat, k=min(k, flat.numel()))
        results = []
        for idx, val in zip(topk_indices.tolist(), topk_vals.tolist()):
            layer_idx = idx // n_heads
            head_idx = idx % n_heads
            results.append((layer_idx, head_idx, float(val)))
        return results
