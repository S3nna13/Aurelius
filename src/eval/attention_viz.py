"""Attention pattern numerical analysis.

Characterizes attention weight distributions without rendering:
- Concentration: how focused/diffuse is attention?
- Recency bias: does attention prefer recent tokens?
- Sink tokens: are specific positions getting excess attention?
- Head diversity: how different are attention heads from each other?

No matplotlib or visualization dependencies — pure tensor analytics.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor


class AttentionConcentration:
    """Measures how focused or diffuse attention distributions are."""

    def __init__(self) -> None:
        pass

    def gini_coefficient(self, attn: Tensor) -> Tensor:
        """Gini coefficient per (batch, head, query position).

        Args:
            attn: (B, H, T_q, T_k) attention weights.

        Returns:
            (B, H, T_q) Gini coefficients in [0, 1].
        """
        # attn: (B, H, T_q, T_k)
        B, H, T_q, T_k = attn.shape
        n = T_k

        # Sort ascending along T_k dimension
        x, _ = attn.sort(dim=-1)  # (B, H, T_q, T_k)

        # Indices 1..n
        i = torch.arange(1, n + 1, dtype=attn.dtype, device=attn.device)  # (T_k,)

        # G = (2 * sum(i * x_i) / (n * sum(x_i))) - (n+1)/n
        sum_ix = (i * x).sum(dim=-1)   # (B, H, T_q)
        sum_x = x.sum(dim=-1)          # (B, H, T_q)

        gini = (2.0 * sum_ix / (n * (sum_x + 1e-10))) - (n + 1.0) / n
        return gini

    def top_k_fraction(self, attn: Tensor, k: int = 5) -> Tensor:
        """Fraction of total attention weight in top-k keys per query.

        Args:
            attn: (B, H, T_q, T_k) attention weights.
            k: number of top keys to consider.

        Returns:
            (B, H, T_q) fraction in [0, 1].
        """
        k = min(k, attn.shape[-1])
        top_vals, _ = attn.topk(k, dim=-1)   # (B, H, T_q, k)
        top_sum = top_vals.sum(dim=-1)        # (B, H, T_q)
        total = attn.sum(dim=-1)              # (B, H, T_q)
        return top_sum / (total + 1e-10)

    def entropy(self, attn: Tensor) -> Tensor:
        """-sum(p * log(p + 1e-10)) per query position.

        Args:
            attn: (B, H, T_q, T_k) attention weights.

        Returns:
            (B, H, T_q) entropy values.
        """
        return -(attn * torch.log(attn + 1e-10)).sum(dim=-1)


class RecencyBiasAnalyzer:
    """Quantifies whether attention patterns prefer more recent tokens."""

    def __init__(self) -> None:
        pass

    def relative_position_weights(self, attn: Tensor) -> Tensor:
        """Mean attention weight as a function of backward offset.

        For each diagonal offset d (0=same pos, 1=one back, ...),
        averages attention over all valid (b, h, q) where k = q - d.

        Args:
            attn: (B, H, T, T) self-attention weights.

        Returns:
            (T,) mean attention weight for each backward offset 0..T-1.
        """
        B, H, T, _ = attn.shape
        result = attn.new_zeros(T)
        counts = attn.new_zeros(T)

        for d in range(T):
            # Valid query positions: d..T-1, corresponding key = q - d
            q_indices = torch.arange(d, T, device=attn.device)
            k_indices = q_indices - d
            # Gather: attn[b, h, q, k] for all b, h and valid (q,k) pairs
            # attn[:, :, q_indices, k_indices] -> (B, H, num_valid)
            vals = attn[:, :, q_indices, k_indices]  # (B, H, num_valid)
            result[d] = vals.mean()
            counts[d] = 1.0

        return result

    def recency_score(self, attn: Tensor, window: int = 5) -> float:
        """Fraction of attention in the last `window` tokens relative to all past tokens.

        For each (b, h, q): sum(attn[b,h,q, max(0,q-window):q+1]) / sum(attn[b,h,q, :q+1])

        Args:
            attn: (B, H, T, T) self-attention weights.
            window: number of recent positions to include.

        Returns:
            Scalar float in [0, 1].
        """
        B, H, T, _ = attn.shape
        total_recency = 0.0
        count = 0

        for q in range(T):
            past_end = q + 1                      # keys 0..q inclusive
            window_start = max(0, q - window + 1) # last `window` keys ending at q

            past_sum = attn[:, :, q, :past_end].sum(dim=-1)          # (B, H)
            window_sum = attn[:, :, q, window_start:past_end].sum(dim=-1)  # (B, H)

            ratio = (window_sum / (past_sum + 1e-8)).mean().item()
            total_recency += ratio
            count += 1

        return total_recency / max(count, 1)


class SinkTokenDetector:
    """Identifies token positions that attract disproportionate attention."""

    def __init__(self, sink_threshold: float = 0.1) -> None:
        """
        Args:
            sink_threshold: fraction of total attention to declare a position a sink.
        """
        self.sink_threshold = sink_threshold

    def detect_sinks(self, attn: Tensor) -> Tensor:
        """Bool mask of key positions that are attention sinks.

        Args:
            attn: (B, H, T_q, T_k) attention weights.

        Returns:
            (T_k,) bool tensor; True where mean received attention > sink_threshold.
        """
        mean_received = attn.mean(dim=(0, 1, 2))  # (T_k,)
        return mean_received > self.sink_threshold

    def sink_positions(self, attn: Tensor) -> List[int]:
        """List of position indices that are attention sinks.

        Args:
            attn: (B, H, T_q, T_k) attention weights.

        Returns:
            List of integer position indices.
        """
        mask = self.detect_sinks(attn)
        return mask.nonzero(as_tuple=False).squeeze(-1).tolist()

    def sink_attention_fraction(self, attn: Tensor) -> float:
        """Total fraction of attention directed to sink tokens.

        Args:
            attn: (B, H, T_q, T_k) attention weights.

        Returns:
            Scalar float in [0, 1].
        """
        mask = self.detect_sinks(attn)          # (T_k,)
        sink_total = attn[..., mask].sum().item()
        grand_total = attn.sum().item()
        return sink_total / (grand_total + 1e-10)


class HeadDiversityAnalyzer:
    """Measures how distinct different attention heads are from each other."""

    def __init__(self) -> None:
        pass

    def inter_head_kl(self, attn: Tensor) -> Tensor:
        """Pairwise KL divergence between attention heads.

        Computes mean attention over batch dimension first, then for each
        pair of heads (h1, h2) computes KL(h1 || h2) averaged over queries.

        Args:
            attn: (B, H, T_q, T_k) attention weights.

        Returns:
            (H, H) matrix of mean KL divergences.
        """
        # Average over batch: (H, T_q, T_k)
        mean_attn = attn.mean(dim=0)
        H, T_q, T_k = mean_attn.shape

        # Add small epsilon for numerical stability
        p = mean_attn + 1e-10  # (H, T_q, T_k)

        kl_matrix = attn.new_zeros(H, H)
        for h1 in range(H):
            for h2 in range(H):
                # KL(p[h1] || p[h2]) = sum(p[h1] * log(p[h1] / p[h2]))
                # averaged over T_q
                kl_per_query = (p[h1] * (torch.log(p[h1]) - torch.log(p[h2]))).sum(dim=-1)
                kl_matrix[h1, h2] = kl_per_query.mean()

        return kl_matrix

    def head_specialization(self, attn: Tensor) -> Tensor:
        """KL divergence of each head from the mean head.

        Args:
            attn: (B, H, T_q, T_k) attention weights.

        Returns:
            (H,) mean KL of each head from the mean head, averaged over B and T_q.
        """
        B, H, T_q, T_k = attn.shape

        mean_head = attn.mean(dim=1, keepdim=True)  # (B, 1, T_q, T_k)

        # Add epsilon
        p = attn + 1e-10          # (B, H, T_q, T_k)
        q = mean_head + 1e-10     # (B, 1, T_q, T_k)

        # KL(p || q) per (b, h, q)
        kl = (p * (torch.log(p) - torch.log(q))).sum(dim=-1)  # (B, H, T_q)

        # Average over B and T_q -> (H,)
        return kl.mean(dim=(0, 2))

    def effective_heads(self, attn: Tensor, threshold: float = 0.1) -> int:
        """Number of heads with specialization score above threshold.

        Args:
            attn: (B, H, T_q, T_k) attention weights.
            threshold: minimum KL specialization score to count as effective.

        Returns:
            Integer count of effective heads (at least 1).
        """
        scores = self.head_specialization(attn)  # (H,)
        count = int((scores > threshold).sum().item())
        return max(count, 1)
